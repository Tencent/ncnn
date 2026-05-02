// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "interp_mips.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_usability.h"

namespace ncnn {

#include "interp_bicubic.h"
#include "interp_bilinear.h"

#if __mips_msa
#include "interp_bicubic_pack4.h"
#include "interp_bilinear_pack4.h"
#endif

Interp_mips::Interp_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Interp_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

#if __mips_msa
        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                v4f32 _v = (v4f32)__msa_ld_w((const float*)bottom_blob + q * 4, 0);
                top_blob_c.fill(_v);
            }

            return 0;
        }
#endif // __mips_msa

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

#if __mips_msa
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

                        v4f32 _p = (v4f32)__msa_ld_w(ptr + in_x * 4, 0);
                        __msa_st_w((v4i32)_p, outptr, 0);

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

                        v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                        v4f32 _a1 = __msa_fill_w_f32(alphap[1]);

                        v4f32 _S0 = (v4f32)__msa_ld_w(Sp, 0);
                        v4f32 _S1 = (v4f32)__msa_ld_w(Sp + 4, 0);
                        v4f32 _p = __msa_fmul_w(_S0, _a0);
                        _p = __ncnn_msa_fmadd_w(_p, _S1, _a1);
                        __msa_st_w((v4i32)_p, outptr, 0);

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

                        v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                        v4f32 _a1 = __msa_fill_w_f32(alphap[1]);
                        v4f32 _a2 = __msa_fill_w_f32(alphap[2]);
                        v4f32 _a3 = __msa_fill_w_f32(alphap[3]);

                        v4f32 _S0 = (v4f32)__msa_ld_w(Sp - 4, 0);
                        v4f32 _S1 = (v4f32)__msa_ld_w(Sp + 0, 0);
                        v4f32 _S2 = (v4f32)__msa_ld_w(Sp + 4, 0);
                        v4f32 _S3 = (v4f32)__msa_ld_w(Sp + 8, 0);
                        v4f32 _p = __msa_fmul_w(_S0, _a0);
                        _p = __ncnn_msa_fmadd_w(_p, _S1, _a1);
                        _p = __ncnn_msa_fmadd_w(_p, _S2, _a2);
                        _p = __ncnn_msa_fmadd_w(_p, _S3, _a3);
                        __msa_st_w((v4i32)_p, outptr, 0);

                        alphap += 4;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            return 0;
        }
#endif // __mips_msa

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

#if __mips_msa
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

                        v4f32 _p = (v4f32)__msa_ld_w(ptr + in_x * 4, 0);
                        __msa_st_w((v4i32)_p, outptr, 0);

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
#endif // __mips_msa

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
#if __mips_msa
        v4f32 _b0 = __msa_fill_w_f32(beta[0]);
        v4f32 _b1 = __msa_fill_w_f32(beta[1]);
        for (; dx + 3 < w; dx += 4)
        {
            v4f32 _rows0 = (v4f32)__msa_ld_w(rows0 + dx, 0);
            v4f32 _rows1 = (v4f32)__msa_ld_w(rows1 + dx, 0);
            v4f32 _Dp = __msa_fmul_w(_rows0, _b0);
            _Dp = __ncnn_msa_fmadd_w(_Dp, _rows1, _b1);
            __msa_storel_d(float2bfloat_msa(_Dp), Dp + dx);
        }
#endif // __mips_msa
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
#if __mips_msa
        v4f32 _b0 = __msa_fill_w_f32(beta[0]);
        v4f32 _b1 = __msa_fill_w_f32(beta[1]);
        v4f32 _b2 = __msa_fill_w_f32(beta[2]);
        v4f32 _b3 = __msa_fill_w_f32(beta[3]);
        for (; dx + 3 < w; dx += 4)
        {
            v4f32 _rows0 = (v4f32)__msa_ld_w(rows0 + dx, 0);
            v4f32 _rows1 = (v4f32)__msa_ld_w(rows1 + dx, 0);
            v4f32 _rows2 = (v4f32)__msa_ld_w(rows2 + dx, 0);
            v4f32 _rows3 = (v4f32)__msa_ld_w(rows3 + dx, 0);
            v4f32 _Dp = __msa_fmul_w(_rows0, _b0);
            _Dp = __ncnn_msa_fmadd_w(_Dp, _rows1, _b1);
            _Dp = __ncnn_msa_fmadd_w(_Dp, _rows2, _b2);
            _Dp = __ncnn_msa_fmadd_w(_Dp, _rows3, _b3);
            __msa_storel_d(float2bfloat_msa(_Dp), Dp + dx);
        }
#endif // __mips_msa
        for (; dx < w; dx++)
        {
            Dp[dx] = float32_to_bfloat16(rows0[dx] * beta[0] + rows1[dx] * beta[1] + rows2[dx] * beta[2] + rows3[dx] * beta[3]);
        }

        beta += 4;
    }
}

#if __mips_msa
static void resize_bilinear_image_pack4_bf16s_msa(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
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

                v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                v4f32 _a1 = __msa_fill_w_f32(alphap[1]);

                v4f32 _S10 = bfloat2float_msa(S1p);
                v4f32 _S11 = bfloat2float_msa(S1p + 4);
                v4f32 _rows1 = __msa_fmul_w(_S10, _a0);
                _rows1 = __ncnn_msa_fmadd_w(_rows1, _S11, _a1);
                __msa_st_w((v4i32)_rows1, rows1p + dx * 4, 0);

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

                v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                v4f32 _a1 = __msa_fill_w_f32(alphap[1]);

                v4f32 _S00 = bfloat2float_msa(S0p);
                v4f32 _S01 = bfloat2float_msa(S0p + 4);
                v4f32 _S10 = bfloat2float_msa(S1p);
                v4f32 _S11 = bfloat2float_msa(S1p + 4);
                v4f32 _rows0 = __msa_fmul_w(_S00, _a0);
                v4f32 _rows1 = __msa_fmul_w(_S10, _a0);
                _rows0 = __ncnn_msa_fmadd_w(_rows0, _S01, _a1);
                _rows1 = __ncnn_msa_fmadd_w(_rows1, _S11, _a1);
                __msa_st_w((v4i32)_rows0, rows0p + dx * 4, 0);
                __msa_st_w((v4i32)_rows1, rows1p + dx * 4, 0);

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        v4f32 _b0 = __msa_fill_w_f32(beta[0]);
        v4f32 _b1 = __msa_fill_w_f32(beta[1]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        unsigned short* Dp = dst.row<unsigned short>(dy);

        for (int dx = 0; dx < w; dx++)
        {
            v4f32 _rows0 = (v4f32)__msa_ld_w(rows0p, 0);
            v4f32 _rows1 = (v4f32)__msa_ld_w(rows1p, 0);
            v4f32 _Dp = __msa_fmul_w(_rows0, _b0);
            _Dp = __ncnn_msa_fmadd_w(_Dp, _rows1, _b1);
            __msa_storel_d(float2bfloat_msa(_Dp), Dp);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
        }

        beta += 2;
    }
}

static void resize_bicubic_image_pack4_bf16s_msa(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
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

                v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                v4f32 _a1 = __msa_fill_w_f32(alphap[1]);
                v4f32 _a2 = __msa_fill_w_f32(alphap[2]);
                v4f32 _a3 = __msa_fill_w_f32(alphap[3]);

                v4f32 _rows3 = __msa_fmul_w(bfloat2float_msa(S3p - 4), _a0);
                _rows3 = __ncnn_msa_fmadd_w(_rows3, bfloat2float_msa(S3p), _a1);
                _rows3 = __ncnn_msa_fmadd_w(_rows3, bfloat2float_msa(S3p + 4), _a2);
                _rows3 = __ncnn_msa_fmadd_w(_rows3, bfloat2float_msa(S3p + 8), _a3);
                __msa_st_w((v4i32)_rows3, rows3p + dx * 4, 0);

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

                v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                v4f32 _a1 = __msa_fill_w_f32(alphap[1]);
                v4f32 _a2 = __msa_fill_w_f32(alphap[2]);
                v4f32 _a3 = __msa_fill_w_f32(alphap[3]);

                v4f32 _rows2 = __msa_fmul_w(bfloat2float_msa(S2p - 4), _a0);
                v4f32 _rows3 = __msa_fmul_w(bfloat2float_msa(S3p - 4), _a0);
                _rows2 = __ncnn_msa_fmadd_w(_rows2, bfloat2float_msa(S2p), _a1);
                _rows3 = __ncnn_msa_fmadd_w(_rows3, bfloat2float_msa(S3p), _a1);
                _rows2 = __ncnn_msa_fmadd_w(_rows2, bfloat2float_msa(S2p + 4), _a2);
                _rows3 = __ncnn_msa_fmadd_w(_rows3, bfloat2float_msa(S3p + 4), _a2);
                _rows2 = __ncnn_msa_fmadd_w(_rows2, bfloat2float_msa(S2p + 8), _a3);
                _rows3 = __ncnn_msa_fmadd_w(_rows3, bfloat2float_msa(S3p + 8), _a3);
                __msa_st_w((v4i32)_rows2, rows2p + dx * 4, 0);
                __msa_st_w((v4i32)_rows3, rows3p + dx * 4, 0);

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

                v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                v4f32 _a1 = __msa_fill_w_f32(alphap[1]);
                v4f32 _a2 = __msa_fill_w_f32(alphap[2]);
                v4f32 _a3 = __msa_fill_w_f32(alphap[3]);

                v4f32 _rows1 = __msa_fmul_w(bfloat2float_msa(S1p - 4), _a0);
                v4f32 _rows2 = __msa_fmul_w(bfloat2float_msa(S2p - 4), _a0);
                v4f32 _rows3 = __msa_fmul_w(bfloat2float_msa(S3p - 4), _a0);
                _rows1 = __ncnn_msa_fmadd_w(_rows1, bfloat2float_msa(S1p), _a1);
                _rows2 = __ncnn_msa_fmadd_w(_rows2, bfloat2float_msa(S2p), _a1);
                _rows3 = __ncnn_msa_fmadd_w(_rows3, bfloat2float_msa(S3p), _a1);
                _rows1 = __ncnn_msa_fmadd_w(_rows1, bfloat2float_msa(S1p + 4), _a2);
                _rows2 = __ncnn_msa_fmadd_w(_rows2, bfloat2float_msa(S2p + 4), _a2);
                _rows3 = __ncnn_msa_fmadd_w(_rows3, bfloat2float_msa(S3p + 4), _a2);
                _rows1 = __ncnn_msa_fmadd_w(_rows1, bfloat2float_msa(S1p + 8), _a3);
                _rows2 = __ncnn_msa_fmadd_w(_rows2, bfloat2float_msa(S2p + 8), _a3);
                _rows3 = __ncnn_msa_fmadd_w(_rows3, bfloat2float_msa(S3p + 8), _a3);
                __msa_st_w((v4i32)_rows1, rows1p + dx * 4, 0);
                __msa_st_w((v4i32)_rows2, rows2p + dx * 4, 0);
                __msa_st_w((v4i32)_rows3, rows3p + dx * 4, 0);

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

                v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                v4f32 _a1 = __msa_fill_w_f32(alphap[1]);
                v4f32 _a2 = __msa_fill_w_f32(alphap[2]);
                v4f32 _a3 = __msa_fill_w_f32(alphap[3]);

                v4f32 _rows0 = __msa_fmul_w(bfloat2float_msa(S0p - 4), _a0);
                v4f32 _rows1 = __msa_fmul_w(bfloat2float_msa(S1p - 4), _a0);
                v4f32 _rows2 = __msa_fmul_w(bfloat2float_msa(S2p - 4), _a0);
                v4f32 _rows3 = __msa_fmul_w(bfloat2float_msa(S3p - 4), _a0);
                _rows0 = __ncnn_msa_fmadd_w(_rows0, bfloat2float_msa(S0p), _a1);
                _rows1 = __ncnn_msa_fmadd_w(_rows1, bfloat2float_msa(S1p), _a1);
                _rows2 = __ncnn_msa_fmadd_w(_rows2, bfloat2float_msa(S2p), _a1);
                _rows3 = __ncnn_msa_fmadd_w(_rows3, bfloat2float_msa(S3p), _a1);
                _rows0 = __ncnn_msa_fmadd_w(_rows0, bfloat2float_msa(S0p + 4), _a2);
                _rows1 = __ncnn_msa_fmadd_w(_rows1, bfloat2float_msa(S1p + 4), _a2);
                _rows2 = __ncnn_msa_fmadd_w(_rows2, bfloat2float_msa(S2p + 4), _a2);
                _rows3 = __ncnn_msa_fmadd_w(_rows3, bfloat2float_msa(S3p + 4), _a2);
                _rows0 = __ncnn_msa_fmadd_w(_rows0, bfloat2float_msa(S0p + 8), _a3);
                _rows1 = __ncnn_msa_fmadd_w(_rows1, bfloat2float_msa(S1p + 8), _a3);
                _rows2 = __ncnn_msa_fmadd_w(_rows2, bfloat2float_msa(S2p + 8), _a3);
                _rows3 = __ncnn_msa_fmadd_w(_rows3, bfloat2float_msa(S3p + 8), _a3);
                __msa_st_w((v4i32)_rows0, rows0p + dx * 4, 0);
                __msa_st_w((v4i32)_rows1, rows1p + dx * 4, 0);
                __msa_st_w((v4i32)_rows2, rows2p + dx * 4, 0);
                __msa_st_w((v4i32)_rows3, rows3p + dx * 4, 0);

                alphap += 4;
            }
        }

        prev_sy1 = sy;

        v4f32 _b0 = __msa_fill_w_f32(beta[0]);
        v4f32 _b1 = __msa_fill_w_f32(beta[1]);
        v4f32 _b2 = __msa_fill_w_f32(beta[2]);
        v4f32 _b3 = __msa_fill_w_f32(beta[3]);

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* rows2p = rows2;
        float* rows3p = rows3;
        unsigned short* Dp = dst.row<unsigned short>(dy);

        for (int dx = 0; dx < w; dx++)
        {
            v4f32 _rows0 = (v4f32)__msa_ld_w(rows0p, 0);
            v4f32 _rows1 = (v4f32)__msa_ld_w(rows1p, 0);
            v4f32 _rows2 = (v4f32)__msa_ld_w(rows2p, 0);
            v4f32 _rows3 = (v4f32)__msa_ld_w(rows3p, 0);
            v4f32 _Dp = __msa_fmul_w(_rows0, _b0);
            _Dp = __ncnn_msa_fmadd_w(_Dp, _rows1, _b1);
            _Dp = __ncnn_msa_fmadd_w(_Dp, _rows2, _b2);
            _Dp = __ncnn_msa_fmadd_w(_Dp, _rows3, _b3);
            __msa_storel_d(float2bfloat_msa(_Dp), Dp);

            Dp += 4;
            rows0p += 4;
            rows1p += 4;
            rows2p += 4;
            rows3p += 4;
        }

        beta += 4;
    }
}
#endif // __mips_msa

int Interp_mips::forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

#if __mips_msa
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

                        v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                        v4f32 _a1 = __msa_fill_w_f32(alphap[1]);

                        v4f32 _S0 = bfloat2float_msa(Sp);
                        v4f32 _S1 = bfloat2float_msa(Sp + 4);
                        v4f32 _p = __msa_fmul_w(_S0, _a0);
                        _p = __ncnn_msa_fmadd_w(_p, _S1, _a1);
                        __msa_storel_d(float2bfloat_msa(_p), outptr);

                        alphap += 2;
                        outptr += 4;
                    }
                }

                delete[] buf;
                return 0;
            }
#endif // __mips_msa

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

#if __mips_msa
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

                        v4f32 _a0 = __msa_fill_w_f32(alphap[0]);
                        v4f32 _a1 = __msa_fill_w_f32(alphap[1]);
                        v4f32 _a2 = __msa_fill_w_f32(alphap[2]);
                        v4f32 _a3 = __msa_fill_w_f32(alphap[3]);

                        v4f32 _S0 = bfloat2float_msa(Sp - 4);
                        v4f32 _S1 = bfloat2float_msa(Sp);
                        v4f32 _S2 = bfloat2float_msa(Sp + 4);
                        v4f32 _S3 = bfloat2float_msa(Sp + 8);
                        v4f32 _p = __msa_fmul_w(_S0, _a0);
                        _p = __ncnn_msa_fmadd_w(_p, _S1, _a1);
                        _p = __ncnn_msa_fmadd_w(_p, _S2, _a2);
                        _p = __ncnn_msa_fmadd_w(_p, _S3, _a3);
                        __msa_storel_d(float2bfloat_msa(_p), outptr);

                        alphap += 4;
                        outptr += 4;
                    }
                }

                delete[] buf;
                return 0;
            }
#endif // __mips_msa

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

#if __mips_msa
            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat src = bottom_blob.channel(q);
                    Mat dst = top_blob.channel(q);
                    resize_bilinear_image_pack4_bf16s_msa(src, dst, alpha, xofs, beta, yofs);
                }

                delete[] buf;
                return 0;
            }
#endif // __mips_msa

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

#if __mips_msa
            if (elempack == 4)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const Mat src = bottom_blob.channel(q);
                    Mat dst = top_blob.channel(q);
                    resize_bicubic_image_pack4_bf16s_msa(src, dst, alpha, xofs, beta, yofs);
                }

                delete[] buf;
                return 0;
            }
#endif // __mips_msa

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
