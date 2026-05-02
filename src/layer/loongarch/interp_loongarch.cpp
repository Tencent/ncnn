// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "interp_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

#include "interp_bicubic.h"
#include "interp_bilinear.h"

#if __loongarch_sx
#include "interp_bicubic_pack4.h"
#include "interp_bilinear_pack4.h"
#if __loongarch_asx
#include "interp_bicubic_pack8.h"
#include "interp_bilinear_pack8.h"
#endif // __loongarch_asx
#endif // __loongarch_sx

Interp_loongarch::Interp_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Interp_loongarch::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_blobs[0].elembits() == 16)
        return forward_bf16s(bottom_blobs, top_blobs, opt);
#endif

    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if (!size_expr.empty())
    {
        std::vector<Mat> bottom_blob_shapes(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            bottom_blob_shapes[i] = bottom_blobs[i].shape();
        }
        eval_size_expr(bottom_blob_shapes, outw, outh);
    }

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                __m256 _v = (__m256)__lasx_xvld((const float*)bottom_blob + q * 8, 0);
                top_blob_c.fill(_v);
            }

            return 0;
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                __m128 _v = (__m128)__lsx_vld((const float*)bottom_blob + q * 4, 0);
                top_blob_c.fill(_v);
            }

            return 0;
        }
#endif // __loongarch_sx

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < w; q++)
        {
            Mat top_blob_c = top_blob.channel(q);
            const float v = bottom_blob[q];
            top_blob_c.fill(v);
        }

        return 0;
    }

    if (dims == 2)
    {
        if (outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __loongarch_sx
#if __loongarch_asx
        if (elempack == 8)
        {
            if (resize_type == 1) // nearest
            {
                const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        __m256 _p = (__m256)__lasx_xvld(ptr + in_x * 8, 0);
                        __lasx_xvst(_p, outptr, 0);

                        outptr += 8;
                    }
                }
            }

            if (resize_type == 2) // bilinear
            {
                int* buf = new int[outw + outw * 2];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                linear_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 8;
                        const float* Sp = ptr + sx;

                        __m256 _a0 = __lasx_xvreplfr2vr_s(alphap[0]);
                        __m256 _a1 = __lasx_xvreplfr2vr_s(alphap[1]);

                        __m256 _S0 = (__m256)__lasx_xvld(Sp, 0);
                        __m256 _S1 = (__m256)__lasx_xvld(Sp + 8, 0);
                        __m256 _p = __lasx_xvfmul_s(_S0, _a0);
                        _p = __lasx_xvfmadd_s(_a1, _S1, _p);
                        __lasx_xvst(_p, outptr, 0);

                        alphap += 2;
                        outptr += 8;
                    }
                }

                delete[] buf;
            }

            if (resize_type == 3) // bicubic
            {
                int* buf = new int[outw + outw * 4];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                cubic_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 8;
                        const float* Sp = ptr + sx;

                        __m256 _a0 = __lasx_xvreplfr2vr_s(alphap[0]);
                        __m256 _a1 = __lasx_xvreplfr2vr_s(alphap[1]);
                        __m256 _a2 = __lasx_xvreplfr2vr_s(alphap[2]);
                        __m256 _a3 = __lasx_xvreplfr2vr_s(alphap[3]);

                        __m256 _S0 = (__m256)__lasx_xvld(Sp - 8, 0);
                        __m256 _S1 = (__m256)__lasx_xvld(Sp + 0, 0);
                        __m256 _S2 = (__m256)__lasx_xvld(Sp + 8, 0);
                        __m256 _S3 = (__m256)__lasx_xvld(Sp + 16, 0);
                        __m256 _p = __lasx_xvfmul_s(_S0, _a0);
                        _p = __lasx_xvfmadd_s(_a1, _S1, _p);
                        _p = __lasx_xvfmadd_s(_a2, _S2, _p);
                        _p = __lasx_xvfmadd_s(_a3, _S3, _p);
                        __lasx_xvst(_p, outptr, 0);

                        alphap += 4;
                        outptr += 8;
                    }
                }

                delete[] buf;
            }

            return 0;
        }
#endif // __loongarch_asx
        if (elempack == 4)
        {
            if (resize_type == 1) // nearest
            {
                const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        __m128 _p = (__m128)__lsx_vld(ptr + in_x * 4, 0);
                        __lsx_vst(_p, outptr, 0);

                        outptr += 4;
                    }
                }
            }

            if (resize_type == 2) // bilinear
            {
                int* buf = new int[outw + outw * 2];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                linear_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const float* Sp = ptr + sx;

                        __m128 _a0 = __lsx_vreplfr2vr_s(alphap[0]);
                        __m128 _a1 = __lsx_vreplfr2vr_s(alphap[1]);

                        __m128 _S0 = (__m128)__lsx_vld(Sp, 0);
                        __m128 _S1 = (__m128)__lsx_vld(Sp + 4, 0);
                        __m128 _p = __lsx_vfmul_s(_S0, _a0);
                        _p = __lsx_vfmadd_s(_a1, _S1, _p);
                        __lsx_vst(_p, outptr, 0);

                        alphap += 2;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            if (resize_type == 3) // bicubic
            {
                int* buf = new int[outw + outw * 4];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                cubic_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const float* Sp = ptr + sx;

                        __m128 _a0 = __lsx_vreplfr2vr_s(alphap[0]);
                        __m128 _a1 = __lsx_vreplfr2vr_s(alphap[1]);
                        __m128 _a2 = __lsx_vreplfr2vr_s(alphap[2]);
                        __m128 _a3 = __lsx_vreplfr2vr_s(alphap[3]);

                        __m128 _S0 = (__m128)__lsx_vld(Sp - 4, 0);
                        __m128 _S1 = (__m128)__lsx_vld(Sp + 0, 0);
                        __m128 _S2 = (__m128)__lsx_vld(Sp + 4, 0);
                        __m128 _S3 = (__m128)__lsx_vld(Sp + 8, 0);
                        __m128 _p = __lsx_vfmul_s(_S0, _a0);
                        _p = __lsx_vfmadd_s(_a1, _S1, _p);
                        _p = __lsx_vfmadd_s(_a2, _S2, _p);
                        _p = __lsx_vfmadd_s(_a3, _S3, _p);
                        __lsx_vst(_p, outptr, 0);

                        alphap += 4;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            return 0;
        }
#endif // __loongarch_sx

        if (resize_type == 1) // nearest
        {
            const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            linear_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const float* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    *outptr++ = Sp[0] * a0 + Sp[1] * a1;
                    alphap += 2;
                }
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outw * 4];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            cubic_coeffs(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const float* ptr = bottom_blob.row(y);
                float* outptr = top_blob.row(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const float* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    float a2 = alphap[2];
                    float a3 = alphap[3];
                    *outptr++ = Sp[-1] * a0 + Sp[0] * a1 + Sp[1] * a2 + Sp[2] * a3;
                    alphap += 4;
                }
            }

            delete[] buf;
        }

        return 0;
    }

    if (outw == w && outh == h)
    {
        top_blob = bottom_blob;
        return 0;
    }

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        if (resize_type == 1) // nearest
        {
            const float hs = (output_height || !size_expr.empty()) ? h / (float)outh : 1.f / height_scale;
            const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int in_y = std::min((int)(y * hs), (h - 1));

                    const float* ptr = src.row(in_y);
                    float* outptr = dst.row(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        __m256 _p = (__m256)__lasx_xvld(ptr + in_x * 8, 0);
                        __lasx_xvst(_p, outptr, 0);

                        outptr += 8;
                    }
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
            float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

            linear_coeffs(w, outw, xofs, alpha, align_corner);
            linear_coeffs(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bilinear_image_pack8(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outh + outw * 4 + outh * 4];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
            float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

            cubic_coeffs(w, outw, xofs, alpha, align_corner);
            cubic_coeffs(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bicubic_image_pack8(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return 0;
    }
#endif // __loongarch_asx
    if (elempack == 4)
    {
        if (resize_type == 1) // nearest
        {
            const float hs = (output_height || !size_expr.empty()) ? h / (float)outh : 1.f / height_scale;
            const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int in_y = std::min((int)(y * hs), (h - 1));

                    const float* ptr = src.row(in_y);
                    float* outptr = dst.row(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        __m128 _p = (__m128)__lsx_vld(ptr + in_x * 4, 0);
                        __lsx_vst(_p, outptr, 0);

                        outptr += 4;
                    }
                }
            }
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
            float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

            linear_coeffs(w, outw, xofs, alpha, align_corner);
            linear_coeffs(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bilinear_image_pack4(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outh + outw * 4 + outh * 4];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
            float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

            cubic_coeffs(w, outw, xofs, alpha, align_corner);
            cubic_coeffs(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bicubic_image_pack4(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return 0;
    }
#endif // __loongarch_sx

    if (resize_type == 1) // nearest
    {
        const float hs = (output_height || !size_expr.empty()) ? h / (float)outh : 1.f / height_scale;
        const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            for (int y = 0; y < outh; y++)
            {
                int in_y = std::min((int)(y * hs), (h - 1));

                const float* ptr = src.row(in_y);
                float* outptr = dst.row(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    *outptr++ = ptr[in_x];
                }
            }
        }
    }

    if (resize_type == 2) // bilinear
    {
        int* buf = new int[outw + outh + outw * 2 + outh * 2];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 2];
        float* beta = (float*)(buf + outw + outh + outw * 2); //new float[outh * 2];

        linear_coeffs(w, outw, xofs, alpha, align_corner);
        linear_coeffs(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    if (resize_type == 3) // bicubic
    {
        int* buf = new int[outw + outh + outw * 4 + outh * 4];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        float* alpha = (float*)(buf + outw + outh);           //new float[outw * 4];
        float* beta = (float*)(buf + outw + outh + outw * 4); //new float[outh * 4];

        cubic_coeffs(w, outw, xofs, alpha, align_corner);
        cubic_coeffs(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bicubic_image(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    return 0;
}

#if NCNN_BF16
static void interp_copy_bf16s(const unsigned short* ptr, unsigned short* outptr, int elempack)
{
    for (int i = 0; i < elempack; i++)
    {
        outptr[i] = ptr[i];
    }
}

static void resize_bilinear_image_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    Mat rowsbuf0(w);
    Mat rowsbuf1(w);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned short* S1 = src.row<const unsigned short>(sy + 1);

            const float* alphap = alpha;
            float* rows1p = rows1;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const unsigned short* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = bfloat16_to_float32(S1p[0]) * a0 + bfloat16_to_float32(S1p[1]) * a1;

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const unsigned short* S0 = src.row<const unsigned short>(sy);
            const unsigned short* S1 = src.row<const unsigned short>(sy + 1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = bfloat16_to_float32(S0p[0]) * a0 + bfloat16_to_float32(S0p[1]) * a1;
                rows1p[dx] = bfloat16_to_float32(S1p[0]) * a0 + bfloat16_to_float32(S1p[1]) * a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        unsigned short* Dp = dst.row<unsigned short>(dy);

        int dx = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _b0_lasx = __lasx_xvreplfr2vr_s(beta[0]);
        __m256 _b1_lasx = __lasx_xvreplfr2vr_s(beta[1]);
        for (; dx + 7 < w; dx += 8)
        {
            __m256 _rows0 = (__m256)__lasx_xvld(rows0 + dx, 0);
            __m256 _rows1 = (__m256)__lasx_xvld(rows1 + dx, 0);
            __m256 _Dp = __lasx_xvfmul_s(_rows0, _b0_lasx);
            _Dp = __lasx_xvfmadd_s(_b1_lasx, _rows1, _Dp);
            __lsx_vst(float2bfloat_lasx(_Dp), Dp + dx, 0);
        }
#endif // __loongarch_asx
        __m128 _b0_lsx = (__m128)__lsx_vreplfr2vr_s(beta[0]);
        __m128 _b1_lsx = (__m128)__lsx_vreplfr2vr_s(beta[1]);
        for (; dx + 3 < w; dx += 4)
        {
            __m128 _rows0 = (__m128)__lsx_vld(rows0 + dx, 0);
            __m128 _rows1 = (__m128)__lsx_vld(rows1 + dx, 0);
            __m128 _Dp = __lsx_vfmul_s(_rows0, _b0_lsx);
            _Dp = __lsx_vfmadd_s(_b1_lsx, _rows1, _Dp);
            __lsx_vstelm_d(float2bfloat_lsx(_Dp), Dp + dx, 0, 0);
        }
#endif // __loongarch_sx
        for (; dx < w; dx++)
        {
            Dp[dx] = float32_to_bfloat16(rows0[dx] * beta[0] + rows1[dx] * beta[1]);
        }

        beta += 2;
    }
}

static void resize_bicubic_image_bf16s(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    Mat rowsbuf0(w);
    Mat rowsbuf1(w);
    Mat rowsbuf2(w);
    Mat rowsbuf3(w);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    float* rows2 = rowsbuf2;
    float* rows3 = rowsbuf3;

    int prev_sy1 = -3;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows2;
            rows2 = rows3;
            rows3 = rows0_old;
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows3p = rows3;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const unsigned short* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows3p[dx] = bfloat16_to_float32(S3p[-1]) * a0 + bfloat16_to_float32(S3p[0]) * a1 + bfloat16_to_float32(S3p[1]) * a2 + bfloat16_to_float32(S3p[2]) * a3;

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize two rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            rows0 = rows2;
            rows1 = rows3;
            rows2 = rows0_old;
            rows3 = rows1_old;
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows2p = rows2;
            float* rows3p = rows3;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows2p[dx] = bfloat16_to_float32(S2p[-1]) * a0 + bfloat16_to_float32(S2p[0]) * a1 + bfloat16_to_float32(S2p[1]) * a2 + bfloat16_to_float32(S2p[2]) * a3;
                rows3p[dx] = bfloat16_to_float32(S3p[-1]) * a0 + bfloat16_to_float32(S3p[0]) * a1 + bfloat16_to_float32(S3p[1]) * a2 + bfloat16_to_float32(S3p[2]) * a3;

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 3)
        {
            // hresize three rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            float* rows2_old = rows2;
            rows0 = rows3;
            rows1 = rows0_old;
            rows2 = rows1_old;
            rows3 = rows2_old;
            const unsigned short* S1 = src.row<const unsigned short>(sy);
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows1p[dx] = bfloat16_to_float32(S1p[-1]) * a0 + bfloat16_to_float32(S1p[0]) * a1 + bfloat16_to_float32(S1p[1]) * a2 + bfloat16_to_float32(S1p[2]) * a3;
                rows2p[dx] = bfloat16_to_float32(S2p[-1]) * a0 + bfloat16_to_float32(S2p[0]) * a1 + bfloat16_to_float32(S2p[1]) * a2 + bfloat16_to_float32(S2p[2]) * a3;
                rows3p[dx] = bfloat16_to_float32(S3p[-1]) * a0 + bfloat16_to_float32(S3p[0]) * a1 + bfloat16_to_float32(S3p[1]) * a2 + bfloat16_to_float32(S3p[2]) * a3;

                alphap += 4;
            }
        }
        else
        {
            // hresize four rows
            const unsigned short* S0 = src.row<const unsigned short>(sy - 1);
            const unsigned short* S1 = src.row<const unsigned short>(sy);
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                float a2 = alphap[2];
                float a3 = alphap[3];
                rows0p[dx] = bfloat16_to_float32(S0p[-1]) * a0 + bfloat16_to_float32(S0p[0]) * a1 + bfloat16_to_float32(S0p[1]) * a2 + bfloat16_to_float32(S0p[2]) * a3;
                rows1p[dx] = bfloat16_to_float32(S1p[-1]) * a0 + bfloat16_to_float32(S1p[0]) * a1 + bfloat16_to_float32(S1p[1]) * a2 + bfloat16_to_float32(S1p[2]) * a3;
                rows2p[dx] = bfloat16_to_float32(S2p[-1]) * a0 + bfloat16_to_float32(S2p[0]) * a1 + bfloat16_to_float32(S2p[1]) * a2 + bfloat16_to_float32(S2p[2]) * a3;
                rows3p[dx] = bfloat16_to_float32(S3p[-1]) * a0 + bfloat16_to_float32(S3p[0]) * a1 + bfloat16_to_float32(S3p[1]) * a2 + bfloat16_to_float32(S3p[2]) * a3;

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        unsigned short* Dp = dst.row<unsigned short>(dy);

        int dx = 0;
#if __loongarch_sx
#if __loongarch_asx
        __m256 _b0_lasx = __lasx_xvreplfr2vr_s(beta[0]);
        __m256 _b1_lasx = __lasx_xvreplfr2vr_s(beta[1]);
        __m256 _b2_lasx = __lasx_xvreplfr2vr_s(beta[2]);
        __m256 _b3_lasx = __lasx_xvreplfr2vr_s(beta[3]);
        for (; dx + 7 < w; dx += 8)
        {
            __m256 _rows0 = (__m256)__lasx_xvld(rows0 + dx, 0);
            __m256 _rows1 = (__m256)__lasx_xvld(rows1 + dx, 0);
            __m256 _rows2 = (__m256)__lasx_xvld(rows2 + dx, 0);
            __m256 _rows3 = (__m256)__lasx_xvld(rows3 + dx, 0);
            __m256 _Dp = __lasx_xvfmul_s(_rows0, _b0_lasx);
            _Dp = __lasx_xvfmadd_s(_b1_lasx, _rows1, _Dp);
            _Dp = __lasx_xvfmadd_s(_b2_lasx, _rows2, _Dp);
            _Dp = __lasx_xvfmadd_s(_b3_lasx, _rows3, _Dp);
            __lsx_vst(float2bfloat_lasx(_Dp), Dp + dx, 0);
        }
#endif // __loongarch_asx
        __m128 _b0_lsx = (__m128)__lsx_vreplfr2vr_s(beta[0]);
        __m128 _b1_lsx = (__m128)__lsx_vreplfr2vr_s(beta[1]);
        __m128 _b2_lsx = (__m128)__lsx_vreplfr2vr_s(beta[2]);
        __m128 _b3_lsx = (__m128)__lsx_vreplfr2vr_s(beta[3]);
        for (; dx + 3 < w; dx += 4)
        {
            __m128 _rows0 = (__m128)__lsx_vld(rows0 + dx, 0);
            __m128 _rows1 = (__m128)__lsx_vld(rows1 + dx, 0);
            __m128 _rows2 = (__m128)__lsx_vld(rows2 + dx, 0);
            __m128 _rows3 = (__m128)__lsx_vld(rows3 + dx, 0);
            __m128 _Dp = __lsx_vfmul_s(_rows0, _b0_lsx);
            _Dp = __lsx_vfmadd_s(_b1_lsx, _rows1, _Dp);
            _Dp = __lsx_vfmadd_s(_b2_lsx, _rows2, _Dp);
            _Dp = __lsx_vfmadd_s(_b3_lsx, _rows3, _Dp);
            __lsx_vstelm_d(float2bfloat_lsx(_Dp), Dp + dx, 0, 0);
        }
#endif // __loongarch_sx
        for (; dx < w; dx++)
        {
            Dp[dx] = float32_to_bfloat16(rows0[dx] * beta[0] + rows1[dx] * beta[1] + rows2[dx] * beta[2] + rows3[dx] * beta[3]);
        }

        beta += 4;
    }
}

#if __loongarch_sx
static void resize_bilinear_image_pack4_bf16s_lsx(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    Mat rowsbuf0(w, (size_t)4 * 4u, 4);
    Mat rowsbuf1(w, (size_t)4 * 4u, 4);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned short* S1 = src.row<const unsigned short>(sy + 1);

            const float* alphap = alpha;
            float* rows1p = rows1;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S1p = S1 + sx;

                __m128 _a0 = (__m128)__lsx_vreplfr2vr_s(alphap[0]);
                __m128 _a1 = (__m128)__lsx_vreplfr2vr_s(alphap[1]);

                __m128 _S10 = bfloat2float_lsx(S1p);
                __m128 _S11 = bfloat2float_lsx(S1p + 4);
                __m128 _rows1 = __lsx_vfmul_s(_S10, _a0);
                _rows1 = __lsx_vfmadd_s(_a1, _S11, _rows1);
                __lsx_vst(_rows1, rows1p + dx * 4, 0);

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const unsigned short* S0 = src.row<const unsigned short>(sy);
            const unsigned short* S1 = src.row<const unsigned short>(sy + 1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;

                __m128 _a0 = (__m128)__lsx_vreplfr2vr_s(alphap[0]);
                __m128 _a1 = (__m128)__lsx_vreplfr2vr_s(alphap[1]);

                __m128 _S00 = bfloat2float_lsx(S0p);
                __m128 _S01 = bfloat2float_lsx(S0p + 4);
                __m128 _S10 = bfloat2float_lsx(S1p);
                __m128 _S11 = bfloat2float_lsx(S1p + 4);
                __m128 _rows0 = __lsx_vfmul_s(_S00, _a0);
                __m128 _rows1 = __lsx_vfmul_s(_S10, _a0);
                _rows0 = __lsx_vfmadd_s(_a1, _S01, _rows0);
                _rows1 = __lsx_vfmadd_s(_a1, _S11, _rows1);
                __lsx_vst(_rows0, rows0p + dx * 4, 0);
                __lsx_vst(_rows1, rows1p + dx * 4, 0);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        __m128 _b0 = (__m128)__lsx_vreplfr2vr_s(beta[0]);
        __m128 _b1 = (__m128)__lsx_vreplfr2vr_s(beta[1]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        unsigned short* Dp = dst.row<unsigned short>(dy);

        for (int dx = 0; dx < w; dx++)
        {
            __m128 _rows0 = (__m128)__lsx_vld(rows0p, 0);
            __m128 _rows1 = (__m128)__lsx_vld(rows1p, 0);
            __m128 _Dp = __lsx_vfmul_s(_rows0, _b0);
            _Dp = __lsx_vfmadd_s(_b1, _rows1, _Dp);
            __lsx_vstelm_d(float2bfloat_lsx(_Dp), Dp, 0, 0);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }

        beta += 2;
    }
}

static void resize_bicubic_image_pack4_bf16s_lsx(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    Mat rowsbuf0(w, (size_t)4 * 4u, 4);
    Mat rowsbuf1(w, (size_t)4 * 4u, 4);
    Mat rowsbuf2(w, (size_t)4 * 4u, 4);
    Mat rowsbuf3(w, (size_t)4 * 4u, 4);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    float* rows2 = rowsbuf2;
    float* rows3 = rowsbuf3;

    int prev_sy1 = -3;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows2;
            rows2 = rows3;
            rows3 = rows0_old;
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows3p = rows3;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S3p = S3 + sx;

                __m128 _a0 = (__m128)__lsx_vreplfr2vr_s(alphap[0]);
                __m128 _a1 = (__m128)__lsx_vreplfr2vr_s(alphap[1]);
                __m128 _a2 = (__m128)__lsx_vreplfr2vr_s(alphap[2]);
                __m128 _a3 = (__m128)__lsx_vreplfr2vr_s(alphap[3]);

                __m128 _rows3 = __lsx_vfmul_s(bfloat2float_lsx(S3p - 4), _a0);
                _rows3 = __lsx_vfmadd_s(_a1, bfloat2float_lsx(S3p), _rows3);
                _rows3 = __lsx_vfmadd_s(_a2, bfloat2float_lsx(S3p + 4), _rows3);
                _rows3 = __lsx_vfmadd_s(_a3, bfloat2float_lsx(S3p + 8), _rows3);
                __lsx_vst(_rows3, rows3p + dx * 4, 0);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize two rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            rows0 = rows2;
            rows1 = rows3;
            rows2 = rows0_old;
            rows3 = rows1_old;
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows2p = rows2;
            float* rows3p = rows3;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m128 _a0 = (__m128)__lsx_vreplfr2vr_s(alphap[0]);
                __m128 _a1 = (__m128)__lsx_vreplfr2vr_s(alphap[1]);
                __m128 _a2 = (__m128)__lsx_vreplfr2vr_s(alphap[2]);
                __m128 _a3 = (__m128)__lsx_vreplfr2vr_s(alphap[3]);

                __m128 _rows2 = __lsx_vfmul_s(bfloat2float_lsx(S2p - 4), _a0);
                __m128 _rows3 = __lsx_vfmul_s(bfloat2float_lsx(S3p - 4), _a0);
                _rows2 = __lsx_vfmadd_s(_a1, bfloat2float_lsx(S2p), _rows2);
                _rows3 = __lsx_vfmadd_s(_a1, bfloat2float_lsx(S3p), _rows3);
                _rows2 = __lsx_vfmadd_s(_a2, bfloat2float_lsx(S2p + 4), _rows2);
                _rows3 = __lsx_vfmadd_s(_a2, bfloat2float_lsx(S3p + 4), _rows3);
                _rows2 = __lsx_vfmadd_s(_a3, bfloat2float_lsx(S2p + 8), _rows2);
                _rows3 = __lsx_vfmadd_s(_a3, bfloat2float_lsx(S3p + 8), _rows3);
                __lsx_vst(_rows2, rows2p + dx * 4, 0);
                __lsx_vst(_rows3, rows3p + dx * 4, 0);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 3)
        {
            // hresize three rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            float* rows2_old = rows2;
            rows0 = rows3;
            rows1 = rows0_old;
            rows2 = rows1_old;
            rows3 = rows2_old;
            const unsigned short* S1 = src.row<const unsigned short>(sy);
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m128 _a0 = (__m128)__lsx_vreplfr2vr_s(alphap[0]);
                __m128 _a1 = (__m128)__lsx_vreplfr2vr_s(alphap[1]);
                __m128 _a2 = (__m128)__lsx_vreplfr2vr_s(alphap[2]);
                __m128 _a3 = (__m128)__lsx_vreplfr2vr_s(alphap[3]);

                __m128 _rows1 = __lsx_vfmul_s(bfloat2float_lsx(S1p - 4), _a0);
                __m128 _rows2 = __lsx_vfmul_s(bfloat2float_lsx(S2p - 4), _a0);
                __m128 _rows3 = __lsx_vfmul_s(bfloat2float_lsx(S3p - 4), _a0);
                _rows1 = __lsx_vfmadd_s(_a1, bfloat2float_lsx(S1p), _rows1);
                _rows2 = __lsx_vfmadd_s(_a1, bfloat2float_lsx(S2p), _rows2);
                _rows3 = __lsx_vfmadd_s(_a1, bfloat2float_lsx(S3p), _rows3);
                _rows1 = __lsx_vfmadd_s(_a2, bfloat2float_lsx(S1p + 4), _rows1);
                _rows2 = __lsx_vfmadd_s(_a2, bfloat2float_lsx(S2p + 4), _rows2);
                _rows3 = __lsx_vfmadd_s(_a2, bfloat2float_lsx(S3p + 4), _rows3);
                _rows1 = __lsx_vfmadd_s(_a3, bfloat2float_lsx(S1p + 8), _rows1);
                _rows2 = __lsx_vfmadd_s(_a3, bfloat2float_lsx(S2p + 8), _rows2);
                _rows3 = __lsx_vfmadd_s(_a3, bfloat2float_lsx(S3p + 8), _rows3);
                __lsx_vst(_rows1, rows1p + dx * 4, 0);
                __lsx_vst(_rows2, rows2p + dx * 4, 0);
                __lsx_vst(_rows3, rows3p + dx * 4, 0);

                alphap += 4;
            }
        }
        else
        {
            // hresize four rows
            const unsigned short* S0 = src.row<const unsigned short>(sy - 1);
            const unsigned short* S1 = src.row<const unsigned short>(sy);
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 4;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m128 _a0 = (__m128)__lsx_vreplfr2vr_s(alphap[0]);
                __m128 _a1 = (__m128)__lsx_vreplfr2vr_s(alphap[1]);
                __m128 _a2 = (__m128)__lsx_vreplfr2vr_s(alphap[2]);
                __m128 _a3 = (__m128)__lsx_vreplfr2vr_s(alphap[3]);

                __m128 _rows0 = __lsx_vfmul_s(bfloat2float_lsx(S0p - 4), _a0);
                __m128 _rows1 = __lsx_vfmul_s(bfloat2float_lsx(S1p - 4), _a0);
                __m128 _rows2 = __lsx_vfmul_s(bfloat2float_lsx(S2p - 4), _a0);
                __m128 _rows3 = __lsx_vfmul_s(bfloat2float_lsx(S3p - 4), _a0);
                _rows0 = __lsx_vfmadd_s(_a1, bfloat2float_lsx(S0p), _rows0);
                _rows1 = __lsx_vfmadd_s(_a1, bfloat2float_lsx(S1p), _rows1);
                _rows2 = __lsx_vfmadd_s(_a1, bfloat2float_lsx(S2p), _rows2);
                _rows3 = __lsx_vfmadd_s(_a1, bfloat2float_lsx(S3p), _rows3);
                _rows0 = __lsx_vfmadd_s(_a2, bfloat2float_lsx(S0p + 4), _rows0);
                _rows1 = __lsx_vfmadd_s(_a2, bfloat2float_lsx(S1p + 4), _rows1);
                _rows2 = __lsx_vfmadd_s(_a2, bfloat2float_lsx(S2p + 4), _rows2);
                _rows3 = __lsx_vfmadd_s(_a2, bfloat2float_lsx(S3p + 4), _rows3);
                _rows0 = __lsx_vfmadd_s(_a3, bfloat2float_lsx(S0p + 8), _rows0);
                _rows1 = __lsx_vfmadd_s(_a3, bfloat2float_lsx(S1p + 8), _rows1);
                _rows2 = __lsx_vfmadd_s(_a3, bfloat2float_lsx(S2p + 8), _rows2);
                _rows3 = __lsx_vfmadd_s(_a3, bfloat2float_lsx(S3p + 8), _rows3);
                __lsx_vst(_rows0, rows0p + dx * 4, 0);
                __lsx_vst(_rows1, rows1p + dx * 4, 0);
                __lsx_vst(_rows2, rows2p + dx * 4, 0);
                __lsx_vst(_rows3, rows3p + dx * 4, 0);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        __m128 _b0 = (__m128)__lsx_vreplfr2vr_s(beta[0]);
        __m128 _b1 = (__m128)__lsx_vreplfr2vr_s(beta[1]);
        __m128 _b2 = (__m128)__lsx_vreplfr2vr_s(beta[2]);
        __m128 _b3 = (__m128)__lsx_vreplfr2vr_s(beta[3]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        unsigned short* Dp = dst.row<unsigned short>(dy);

        for (int dx = 0; dx < w; dx++)
        {
            __m128 _rows0 = (__m128)__lsx_vld(rows0p, 0);
            __m128 _rows1 = (__m128)__lsx_vld(rows1p, 0);
            __m128 _rows2 = (__m128)__lsx_vld(rows2p, 0);
            __m128 _rows3 = (__m128)__lsx_vld(rows3p, 0);
            __m128 _Dp = __lsx_vfmul_s(_rows0, _b0);
            _Dp = __lsx_vfmadd_s(_b1, _rows1, _Dp);
            _Dp = __lsx_vfmadd_s(_b2, _rows2, _Dp);
            _Dp = __lsx_vfmadd_s(_b3, _rows3, _Dp);
            __lsx_vstelm_d(float2bfloat_lsx(_Dp), Dp, 0, 0);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
            rows2p += 4;
            rows3p += 4;
        }

        beta += 4;
    }
}

#if __loongarch_asx
static void resize_bilinear_image_pack8_bf16s_lasx(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    Mat rowsbuf0(w, (size_t)8 * 4u, 8);
    Mat rowsbuf1(w, (size_t)8 * 4u, 8);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned short* S1 = src.row<const unsigned short>(sy + 1);

            const float* alphap = alpha;
            float* rows1p = rows1;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const unsigned short* S1p = S1 + sx;

                __m256 _a0 = __lasx_xvreplfr2vr_s(alphap[0]);
                __m256 _a1 = __lasx_xvreplfr2vr_s(alphap[1]);

                __m256 _S10 = bfloat2float_lasx((__m128i*)S1p);
                __m256 _S11 = bfloat2float_lasx((__m128i*)(S1p + 8));
                __m256 _rows1 = __lasx_xvfmul_s(_S10, _a0);
                _rows1 = __lasx_xvfmadd_s(_a1, _S11, _rows1);
                __lasx_xvst(_rows1, rows1p + dx * 8, 0);

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const unsigned short* S0 = src.row<const unsigned short>(sy);
            const unsigned short* S1 = src.row<const unsigned short>(sy + 1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;

                __m256 _a0 = __lasx_xvreplfr2vr_s(alphap[0]);
                __m256 _a1 = __lasx_xvreplfr2vr_s(alphap[1]);

                __m256 _S00 = bfloat2float_lasx((__m128i*)S0p);
                __m256 _S01 = bfloat2float_lasx((__m128i*)(S0p + 8));
                __m256 _S10 = bfloat2float_lasx((__m128i*)S1p);
                __m256 _S11 = bfloat2float_lasx((__m128i*)(S1p + 8));
                __m256 _rows0 = __lasx_xvfmul_s(_S00, _a0);
                __m256 _rows1 = __lasx_xvfmul_s(_S10, _a0);
                _rows0 = __lasx_xvfmadd_s(_a1, _S01, _rows0);
                _rows1 = __lasx_xvfmadd_s(_a1, _S11, _rows1);
                __lasx_xvst(_rows0, rows0p + dx * 8, 0);
                __lasx_xvst(_rows1, rows1p + dx * 8, 0);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        __m256 _b0 = __lasx_xvreplfr2vr_s(beta[0]);
        __m256 _b1 = __lasx_xvreplfr2vr_s(beta[1]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        unsigned short* Dp = dst.row<unsigned short>(dy);

        for (int dx = 0; dx < w; dx++)
        {
            __m256 _rows0 = (__m256)__lasx_xvld(rows0p, 0);
            __m256 _rows1 = (__m256)__lasx_xvld(rows1p, 0);
            __m256 _Dp = __lasx_xvfmul_s(_rows0, _b0);
            _Dp = __lasx_xvfmadd_s(_b1, _rows1, _Dp);
            __lsx_vst(float2bfloat_lasx(_Dp), Dp, 0);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
        }

        beta += 2;
    }
}

static void resize_bicubic_image_pack8_bf16s_lasx(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    int w = dst.w;
    int h = dst.h;

    Mat rowsbuf0(w, (size_t)8 * 4u, 8);
    Mat rowsbuf1(w, (size_t)8 * 4u, 8);
    Mat rowsbuf2(w, (size_t)8 * 4u, 8);
    Mat rowsbuf3(w, (size_t)8 * 4u, 8);
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;
    float* rows2 = rowsbuf2;
    float* rows3 = rowsbuf3;

    int prev_sy1 = -3;

    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows2;
            rows2 = rows3;
            rows3 = rows0_old;
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows3p = rows3;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const unsigned short* S3p = S3 + sx;

                __m256 _a0 = __lasx_xvreplfr2vr_s(alphap[0]);
                __m256 _a1 = __lasx_xvreplfr2vr_s(alphap[1]);
                __m256 _a2 = __lasx_xvreplfr2vr_s(alphap[2]);
                __m256 _a3 = __lasx_xvreplfr2vr_s(alphap[3]);

                __m256 _rows3 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i*)(S3p - 8)), _a0);
                _rows3 = __lasx_xvfmadd_s(_a1, bfloat2float_lasx((__m128i*)S3p), _rows3);
                _rows3 = __lasx_xvfmadd_s(_a2, bfloat2float_lasx((__m128i*)(S3p + 8)), _rows3);
                _rows3 = __lasx_xvfmadd_s(_a3, bfloat2float_lasx((__m128i*)(S3p + 16)), _rows3);
                __lasx_xvst(_rows3, rows3p + dx * 8, 0);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 2)
        {
            // hresize two rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            rows0 = rows2;
            rows1 = rows3;
            rows2 = rows0_old;
            rows3 = rows1_old;
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows2p = rows2;
            float* rows3p = rows3;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m256 _a0 = __lasx_xvreplfr2vr_s(alphap[0]);
                __m256 _a1 = __lasx_xvreplfr2vr_s(alphap[1]);
                __m256 _a2 = __lasx_xvreplfr2vr_s(alphap[2]);
                __m256 _a3 = __lasx_xvreplfr2vr_s(alphap[3]);

                __m256 _rows2 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i*)(S2p - 8)), _a0);
                __m256 _rows3 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i*)(S3p - 8)), _a0);
                _rows2 = __lasx_xvfmadd_s(_a1, bfloat2float_lasx((__m128i*)S2p), _rows2);
                _rows3 = __lasx_xvfmadd_s(_a1, bfloat2float_lasx((__m128i*)S3p), _rows3);
                _rows2 = __lasx_xvfmadd_s(_a2, bfloat2float_lasx((__m128i*)(S2p + 8)), _rows2);
                _rows3 = __lasx_xvfmadd_s(_a2, bfloat2float_lasx((__m128i*)(S3p + 8)), _rows3);
                _rows2 = __lasx_xvfmadd_s(_a3, bfloat2float_lasx((__m128i*)(S2p + 16)), _rows2);
                _rows3 = __lasx_xvfmadd_s(_a3, bfloat2float_lasx((__m128i*)(S3p + 16)), _rows3);
                __lasx_xvst(_rows2, rows2p + dx * 8, 0);
                __lasx_xvst(_rows3, rows3p + dx * 8, 0);

                alphap += 4;
            }
        }
        else if (sy == prev_sy1 + 3)
        {
            // hresize three rows
            float* rows0_old = rows0;
            float* rows1_old = rows1;
            float* rows2_old = rows2;
            rows0 = rows3;
            rows1 = rows0_old;
            rows2 = rows1_old;
            rows3 = rows2_old;
            const unsigned short* S1 = src.row<const unsigned short>(sy);
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m256 _a0 = __lasx_xvreplfr2vr_s(alphap[0]);
                __m256 _a1 = __lasx_xvreplfr2vr_s(alphap[1]);
                __m256 _a2 = __lasx_xvreplfr2vr_s(alphap[2]);
                __m256 _a3 = __lasx_xvreplfr2vr_s(alphap[3]);

                __m256 _rows1 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i*)(S1p - 8)), _a0);
                __m256 _rows2 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i*)(S2p - 8)), _a0);
                __m256 _rows3 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i*)(S3p - 8)), _a0);
                _rows1 = __lasx_xvfmadd_s(_a1, bfloat2float_lasx((__m128i*)S1p), _rows1);
                _rows2 = __lasx_xvfmadd_s(_a1, bfloat2float_lasx((__m128i*)S2p), _rows2);
                _rows3 = __lasx_xvfmadd_s(_a1, bfloat2float_lasx((__m128i*)S3p), _rows3);
                _rows1 = __lasx_xvfmadd_s(_a2, bfloat2float_lasx((__m128i*)(S1p + 8)), _rows1);
                _rows2 = __lasx_xvfmadd_s(_a2, bfloat2float_lasx((__m128i*)(S2p + 8)), _rows2);
                _rows3 = __lasx_xvfmadd_s(_a2, bfloat2float_lasx((__m128i*)(S3p + 8)), _rows3);
                _rows1 = __lasx_xvfmadd_s(_a3, bfloat2float_lasx((__m128i*)(S1p + 16)), _rows1);
                _rows2 = __lasx_xvfmadd_s(_a3, bfloat2float_lasx((__m128i*)(S2p + 16)), _rows2);
                _rows3 = __lasx_xvfmadd_s(_a3, bfloat2float_lasx((__m128i*)(S3p + 16)), _rows3);
                __lasx_xvst(_rows1, rows1p + dx * 8, 0);
                __lasx_xvst(_rows2, rows2p + dx * 8, 0);
                __lasx_xvst(_rows3, rows3p + dx * 8, 0);

                alphap += 4;
            }
        }
        else
        {
            // hresize four rows
            const unsigned short* S0 = src.row<const unsigned short>(sy - 1);
            const unsigned short* S1 = src.row<const unsigned short>(sy);
            const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
            const unsigned short* S3 = src.row<const unsigned short>(sy + 2);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;
            float* rows2p = rows2;
            float* rows3p = rows3;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx] * 8;
                const unsigned short* S0p = S0 + sx;
                const unsigned short* S1p = S1 + sx;
                const unsigned short* S2p = S2 + sx;
                const unsigned short* S3p = S3 + sx;

                __m256 _a0 = __lasx_xvreplfr2vr_s(alphap[0]);
                __m256 _a1 = __lasx_xvreplfr2vr_s(alphap[1]);
                __m256 _a2 = __lasx_xvreplfr2vr_s(alphap[2]);
                __m256 _a3 = __lasx_xvreplfr2vr_s(alphap[3]);

                __m256 _rows0 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i*)(S0p - 8)), _a0);
                __m256 _rows1 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i*)(S1p - 8)), _a0);
                __m256 _rows2 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i*)(S2p - 8)), _a0);
                __m256 _rows3 = __lasx_xvfmul_s(bfloat2float_lasx((__m128i*)(S3p - 8)), _a0);
                _rows0 = __lasx_xvfmadd_s(_a1, bfloat2float_lasx((__m128i*)S0p), _rows0);
                _rows1 = __lasx_xvfmadd_s(_a1, bfloat2float_lasx((__m128i*)S1p), _rows1);
                _rows2 = __lasx_xvfmadd_s(_a1, bfloat2float_lasx((__m128i*)S2p), _rows2);
                _rows3 = __lasx_xvfmadd_s(_a1, bfloat2float_lasx((__m128i*)S3p), _rows3);
                _rows0 = __lasx_xvfmadd_s(_a2, bfloat2float_lasx((__m128i*)(S0p + 8)), _rows0);
                _rows1 = __lasx_xvfmadd_s(_a2, bfloat2float_lasx((__m128i*)(S1p + 8)), _rows1);
                _rows2 = __lasx_xvfmadd_s(_a2, bfloat2float_lasx((__m128i*)(S2p + 8)), _rows2);
                _rows3 = __lasx_xvfmadd_s(_a2, bfloat2float_lasx((__m128i*)(S3p + 8)), _rows3);
                _rows0 = __lasx_xvfmadd_s(_a3, bfloat2float_lasx((__m128i*)(S0p + 16)), _rows0);
                _rows1 = __lasx_xvfmadd_s(_a3, bfloat2float_lasx((__m128i*)(S1p + 16)), _rows1);
                _rows2 = __lasx_xvfmadd_s(_a3, bfloat2float_lasx((__m128i*)(S2p + 16)), _rows2);
                _rows3 = __lasx_xvfmadd_s(_a3, bfloat2float_lasx((__m128i*)(S3p + 16)), _rows3);
                __lasx_xvst(_rows0, rows0p + dx * 8, 0);
                __lasx_xvst(_rows1, rows1p + dx * 8, 0);
                __lasx_xvst(_rows2, rows2p + dx * 8, 0);
                __lasx_xvst(_rows3, rows3p + dx * 8, 0);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        __m256 _b0 = __lasx_xvreplfr2vr_s(beta[0]);
        __m256 _b1 = __lasx_xvreplfr2vr_s(beta[1]);
        __m256 _b2 = __lasx_xvreplfr2vr_s(beta[2]);
        __m256 _b3 = __lasx_xvreplfr2vr_s(beta[3]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        unsigned short* Dp = dst.row<unsigned short>(dy);

        for (int dx = 0; dx < w; dx++)
        {
            __m256 _rows0 = (__m256)__lasx_xvld(rows0p, 0);
            __m256 _rows1 = (__m256)__lasx_xvld(rows1p, 0);
            __m256 _rows2 = (__m256)__lasx_xvld(rows2p, 0);
            __m256 _rows3 = (__m256)__lasx_xvld(rows3p, 0);
            __m256 _Dp = __lasx_xvfmul_s(_rows0, _b0);
            _Dp = __lasx_xvfmadd_s(_b1, _rows1, _Dp);
            _Dp = __lasx_xvfmadd_s(_b2, _rows2, _Dp);
            _Dp = __lasx_xvfmadd_s(_b3, _rows3, _Dp);
            __lsx_vst(float2bfloat_lasx(_Dp), Dp, 0);

            Dp += 8;
            rows0p += 8;
            rows1p += 8;
            rows2p += 8;
            rows3p += 8;
        }

        beta += 4;
    }
}
#endif // __loongarch_asx
#endif // __loongarch_sx

int Interp_loongarch::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if (!size_expr.empty())
    {
        std::vector<Mat> bottom_blob_shapes(bottom_blobs.size());
        for (size_t i = 0; i < bottom_blobs.size(); i++)
        {
            bottom_blob_shapes[i] = bottom_blobs[i].shape();
        }
        eval_size_expr(bottom_blob_shapes, outw, outh);
    }

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const unsigned short* ptr = bottom_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < w; q++)
        {
            Mat top_blob_c = top_blob.channel(q);
            unsigned short* outptr = top_blob_c;
            const unsigned short* v = ptr + q * elempack;

            for (int i = 0; i < outw * outh; i++)
            {
                interp_copy_bf16s(v, outptr, elempack);
                outptr += elempack;
            }
        }

        return 0;
    }

    if (dims == 2)
    {
        if (outw == w)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, h, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (resize_type == 1) // nearest
        {
            const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                unsigned short* outptr = top_blob.row<unsigned short>(y);
                for (int x = 0; x < outw; x++)
                {
                    int in_x = std::min((int)(x * ws), (w - 1));
                    interp_copy_bf16s(ptr + in_x * elempack, outptr, elempack);
                    outptr += elempack;
                }
            }

            return 0;
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            linear_coeffs(w, outw, xofs, alpha, align_corner);

#if __loongarch_sx
#if __loongarch_asx
            if (elempack == 8)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                    unsigned short* outptr = top_blob.row<unsigned short>(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 8;
                        const unsigned short* Sp = ptr + sx;

                        __m256 _a0 = __lasx_xvreplfr2vr_s(alphap[0]);
                        __m256 _a1 = __lasx_xvreplfr2vr_s(alphap[1]);

                        __m256 _S0 = bfloat2float_lasx((__m128i*)Sp);
                        __m256 _S1 = bfloat2float_lasx((__m128i*)(Sp + 8));
                        __m256 _p = __lasx_xvfmul_s(_S0, _a0);
                        _p = __lasx_xvfmadd_s(_a1, _S1, _p);
                        __lsx_vst(float2bfloat_lasx(_p), outptr, 0);

                        alphap += 2;
                        outptr += 8;
                    }
                }

                delete[] buf;
                return 0;
            }
#endif // __loongarch_asx
            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                    unsigned short* outptr = top_blob.row<unsigned short>(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const unsigned short* Sp = ptr + sx;

                        __m128 _a0 = (__m128)__lsx_vreplfr2vr_s(alphap[0]);
                        __m128 _a1 = (__m128)__lsx_vreplfr2vr_s(alphap[1]);

                        __m128 _S0 = bfloat2float_lsx(Sp);
                        __m128 _S1 = bfloat2float_lsx(Sp + 4);
                        __m128 _p = __lsx_vfmul_s(_S0, _a0);
                        _p = __lsx_vfmadd_s(_a1, _S1, _p);
                        __lsx_vstelm_d(float2bfloat_lsx(_p), outptr, 0, 0);

                        alphap += 2;
                        outptr += 4;
                    }
                }

                delete[] buf;
                return 0;
            }
#endif // __loongarch_sx

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                unsigned short* outptr = top_blob.row<unsigned short>(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x] * elempack;
                    const unsigned short* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];

                    for (int ep = 0; ep < elempack; ep++)
                    {
                        outptr[ep] = float32_to_bfloat16(bfloat16_to_float32(Sp[ep]) * a0 + bfloat16_to_float32(Sp[elempack + ep]) * a1);
                    }

                    alphap += 2;
                    outptr += elempack;
                }
            }

            delete[] buf;
            return 0;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outw * 4];

            int* xofs = buf;
            float* alpha = (float*)(buf + outw);

            cubic_coeffs(w, outw, xofs, alpha, align_corner);

#if __loongarch_sx
#if __loongarch_asx
            if (elempack == 8)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                    unsigned short* outptr = top_blob.row<unsigned short>(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 8;
                        const unsigned short* Sp = ptr + sx;

                        __m256 _a0 = __lasx_xvreplfr2vr_s(alphap[0]);
                        __m256 _a1 = __lasx_xvreplfr2vr_s(alphap[1]);
                        __m256 _a2 = __lasx_xvreplfr2vr_s(alphap[2]);
                        __m256 _a3 = __lasx_xvreplfr2vr_s(alphap[3]);

                        __m256 _S0 = bfloat2float_lasx((__m128i*)(Sp - 8));
                        __m256 _S1 = bfloat2float_lasx((__m128i*)Sp);
                        __m256 _S2 = bfloat2float_lasx((__m128i*)(Sp + 8));
                        __m256 _S3 = bfloat2float_lasx((__m128i*)(Sp + 16));
                        __m256 _p = __lasx_xvfmul_s(_S0, _a0);
                        _p = __lasx_xvfmadd_s(_a1, _S1, _p);
                        _p = __lasx_xvfmadd_s(_a2, _S2, _p);
                        _p = __lasx_xvfmadd_s(_a3, _S3, _p);
                        __lsx_vst(float2bfloat_lasx(_p), outptr, 0);

                        alphap += 4;
                        outptr += 8;
                    }
                }

                delete[] buf;
                return 0;
            }
#endif // __loongarch_asx
            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                    unsigned short* outptr = top_blob.row<unsigned short>(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * 4;
                        const unsigned short* Sp = ptr + sx;

                        __m128 _a0 = (__m128)__lsx_vreplfr2vr_s(alphap[0]);
                        __m128 _a1 = (__m128)__lsx_vreplfr2vr_s(alphap[1]);
                        __m128 _a2 = (__m128)__lsx_vreplfr2vr_s(alphap[2]);
                        __m128 _a3 = (__m128)__lsx_vreplfr2vr_s(alphap[3]);

                        __m128 _S0 = bfloat2float_lsx(Sp - 4);
                        __m128 _S1 = bfloat2float_lsx(Sp);
                        __m128 _S2 = bfloat2float_lsx(Sp + 4);
                        __m128 _S3 = bfloat2float_lsx(Sp + 8);
                        __m128 _p = __lsx_vfmul_s(_S0, _a0);
                        _p = __lsx_vfmadd_s(_a1, _S1, _p);
                        _p = __lsx_vfmadd_s(_a2, _S2, _p);
                        _p = __lsx_vfmadd_s(_a3, _S3, _p);
                        __lsx_vstelm_d(float2bfloat_lsx(_p), outptr, 0, 0);

                        alphap += 4;
                        outptr += 4;
                    }
                }

                delete[] buf;
                return 0;
            }
#endif // __loongarch_sx

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const unsigned short* ptr = bottom_blob.row<const unsigned short>(y);
                unsigned short* outptr = top_blob.row<unsigned short>(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x] * elempack;
                    const unsigned short* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    float a2 = alphap[2];
                    float a3 = alphap[3];

                    for (int ep = 0; ep < elempack; ep++)
                    {
                        outptr[ep] = float32_to_bfloat16(bfloat16_to_float32(Sp[-elempack + ep]) * a0 + bfloat16_to_float32(Sp[ep]) * a1 + bfloat16_to_float32(Sp[elempack + ep]) * a2 + bfloat16_to_float32(Sp[elempack * 2 + ep]) * a3);
                    }

                    alphap += 4;
                    outptr += elempack;
                }
            }

            delete[] buf;
            return 0;
        }

        return 0;
    }

    if (dims == 3)
    {
        if (outw == w && outh == h)
        {
            top_blob = bottom_blob;
            return 0;
        }

        top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (resize_type == 1) // nearest
        {
            const float hs = (output_height || !size_expr.empty()) ? h / (float)outh : 1.f / height_scale;
            const float ws = (output_width || !size_expr.empty()) ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int in_y = std::min((int)(y * hs), (h - 1));

                    const unsigned short* ptr = src.row<const unsigned short>(in_y);
                    unsigned short* outptr = dst.row<unsigned short>(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));
                        interp_copy_bf16s(ptr + in_x * elempack, outptr, elempack);
                        outptr += elempack;
                    }
                }
            }

            return 0;
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;
            int* yofs = buf + outw;

            float* alpha = (float*)(buf + outw + outh);
            float* beta = (float*)(buf + outw + outh + outw * 2);

            linear_coeffs(w, outw, xofs, alpha, align_corner);
            linear_coeffs(h, outh, yofs, beta, align_corner);

#if __loongarch_sx
#if __loongarch_asx
            if (elempack == 8)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat src = bottom_blob.channel(q);
                    Mat dst = top_blob.channel(q);

                    resize_bilinear_image_pack8_bf16s_lasx(src, dst, alpha, xofs, beta, yofs);
                }

                delete[] buf;
                return 0;
            }
#endif // __loongarch_asx
            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat src = bottom_blob.channel(q);
                    Mat dst = top_blob.channel(q);

                    resize_bilinear_image_pack4_bf16s_lsx(src, dst, alpha, xofs, beta, yofs);
                }

                delete[] buf;
                return 0;
            }
#endif // __loongarch_sx

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat src = bottom_blob.channel(q);
                    Mat dst = top_blob.channel(q);

                    resize_bilinear_image_bf16s(src, dst, alpha, xofs, beta, yofs);
                }

                delete[] buf;
                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int sy = yofs[y];
                    const unsigned short* S0 = src.row<const unsigned short>(sy);
                    const unsigned short* S1 = src.row<const unsigned short>(sy + 1);
                    unsigned short* outptr = dst.row<unsigned short>(y);

                    float b0 = beta[y * 2];
                    float b1 = beta[y * 2 + 1];
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * elempack;
                        const unsigned short* S0p = S0 + sx;
                        const unsigned short* S1p = S1 + sx;
                        float a0 = alphap[0];
                        float a1 = alphap[1];

                        for (int ep = 0; ep < elempack; ep++)
                        {
                            float rows0 = bfloat16_to_float32(S0p[ep]) * a0 + bfloat16_to_float32(S0p[elempack + ep]) * a1;
                            float rows1 = bfloat16_to_float32(S1p[ep]) * a0 + bfloat16_to_float32(S1p[elempack + ep]) * a1;
                            outptr[ep] = float32_to_bfloat16(rows0 * b0 + rows1 * b1);
                        }

                        alphap += 2;
                        outptr += elempack;
                    }
                }
            }

            delete[] buf;
            return 0;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outh + outw * 4 + outh * 4];

            int* xofs = buf;
            int* yofs = buf + outw;

            float* alpha = (float*)(buf + outw + outh);
            float* beta = (float*)(buf + outw + outh + outw * 4);

            cubic_coeffs(w, outw, xofs, alpha, align_corner);
            cubic_coeffs(h, outh, yofs, beta, align_corner);

#if __loongarch_sx
#if __loongarch_asx
            if (elempack == 8)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat src = bottom_blob.channel(q);
                    Mat dst = top_blob.channel(q);

                    resize_bicubic_image_pack8_bf16s_lasx(src, dst, alpha, xofs, beta, yofs);
                }

                delete[] buf;
                return 0;
            }
#endif // __loongarch_asx
            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat src = bottom_blob.channel(q);
                    Mat dst = top_blob.channel(q);

                    resize_bicubic_image_pack4_bf16s_lsx(src, dst, alpha, xofs, beta, yofs);
                }

                delete[] buf;
                return 0;
            }
#endif // __loongarch_sx

            if (elempack == 1)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat src = bottom_blob.channel(q);
                    Mat dst = top_blob.channel(q);

                    resize_bicubic_image_bf16s(src, dst, alpha, xofs, beta, yofs);
                }

                delete[] buf;
                return 0;
            }

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int sy = yofs[y];
                    const unsigned short* S0 = src.row<const unsigned short>(sy - 1);
                    const unsigned short* S1 = src.row<const unsigned short>(sy);
                    const unsigned short* S2 = src.row<const unsigned short>(sy + 1);
                    const unsigned short* S3 = src.row<const unsigned short>(sy + 2);
                    unsigned short* outptr = dst.row<unsigned short>(y);

                    float b0 = beta[y * 4];
                    float b1 = beta[y * 4 + 1];
                    float b2 = beta[y * 4 + 2];
                    float b3 = beta[y * 4 + 3];
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * elempack;
                        const unsigned short* S0p = S0 + sx;
                        const unsigned short* S1p = S1 + sx;
                        const unsigned short* S2p = S2 + sx;
                        const unsigned short* S3p = S3 + sx;
                        float a0 = alphap[0];
                        float a1 = alphap[1];
                        float a2 = alphap[2];
                        float a3 = alphap[3];

                        for (int ep = 0; ep < elempack; ep++)
                        {
                            float rows0 = bfloat16_to_float32(S0p[-elempack + ep]) * a0 + bfloat16_to_float32(S0p[ep]) * a1 + bfloat16_to_float32(S0p[elempack + ep]) * a2 + bfloat16_to_float32(S0p[elempack * 2 + ep]) * a3;
                            float rows1 = bfloat16_to_float32(S1p[-elempack + ep]) * a0 + bfloat16_to_float32(S1p[ep]) * a1 + bfloat16_to_float32(S1p[elempack + ep]) * a2 + bfloat16_to_float32(S1p[elempack * 2 + ep]) * a3;
                            float rows2 = bfloat16_to_float32(S2p[-elempack + ep]) * a0 + bfloat16_to_float32(S2p[ep]) * a1 + bfloat16_to_float32(S2p[elempack + ep]) * a2 + bfloat16_to_float32(S2p[elempack * 2 + ep]) * a3;
                            float rows3 = bfloat16_to_float32(S3p[-elempack + ep]) * a0 + bfloat16_to_float32(S3p[ep]) * a1 + bfloat16_to_float32(S3p[elempack + ep]) * a2 + bfloat16_to_float32(S3p[elempack * 2 + ep]) * a3;
                            outptr[ep] = float32_to_bfloat16(rows0 * b0 + rows1 * b1 + rows2 * b2 + rows3 * b3);
                        }

                        alphap += 4;
                        outptr += elempack;
                    }
                }
            }

            delete[] buf;
            return 0;
        }

        return 0;
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
