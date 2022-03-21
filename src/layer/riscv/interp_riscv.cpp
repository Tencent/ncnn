// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "interp_riscv.h"

#include <math.h>

#if __riscv_vector
#ifdef RVV_SPEC_0_7
#include "riscv_v_071_fix.h"
#else
#include <riscv_vector.h>
#endif
#include "riscv_usability.h"
#endif // __riscv_vector

namespace ncnn {

#include "interp_bicubic.h"
#include "interp_bilinear.h"

#if __riscv_vector
#include "interp_bicubic_packn.h"
#include "interp_bilinear_packn.h"
#if __riscv_zfh
#include "interp_bicubic_fp16s.h"
#include "interp_bicubic_packn_fp16s.h"
#include "interp_bilinear_fp16s.h"
#include "interp_bilinear_packn_fp16s.h"
#endif
#endif

Interp_riscv::Interp_riscv()
{
#if __riscv_vector
    support_packing = true;
#if __riscv_zfh
    support_fp16_storage = true;
#endif
#endif // __riscv_vector
}

int Interp_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& reference_blob = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    int elembits = bottom_blob.elembits();

#if __riscv_vector && __riscv_zfh
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blobs, top_blobs, opt);
        else
            return forward_fp16s(bottom_blobs, top_blobs, opt);
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#endif

    int h = bottom_blob.h;
    int w = bottom_blob.w;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = reference_blob.w;
    int outh = reference_blob.h;

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __riscv_vector
        if (elempack == packn)
        {
            const word_type vl = vsetvl_e32m1(packn);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                vfloat32m1_t _v = vle32_v_f32m1((const float*)bottom_blob + q * packn, vl);
                top_blob_c.fill(_v);
            }

            return 0;
        }
#endif // __riscv_vector

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

#if __riscv_vector
        if (elempack == packn)
        {
            if (resize_type == 1) // nearest
            {
                const word_type vl = vsetvl_e32m1(packn);

                const float ws = output_width ? w / (float)outw : 1.f / width_scale;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const float* ptr = bottom_blob.row(y);
                    float* outptr = top_blob.row(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        vfloat32m1_t _p = vle32_v_f32m1(ptr + in_x * packn, vl);
                        vse32_v_f32m1(outptr, _p, vl);

                        outptr += packn;
                    }
                }
            }

            if (resize_type == 2) // bilinear
            {
                const word_type vl = vsetvl_e32m1(packn);

                int* buf = new int[outw + outw * packn];

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
                        int sx = xofs[x] * packn;
                        const float* Sp = ptr + sx;

                        vfloat32m1_t _S0 = vle32_v_f32m1(Sp, vl);
                        vfloat32m1_t _S1 = vle32_v_f32m1(Sp + packn, vl);
                        vfloat32m1_t _p = vfmacc_vf_f32m1(vfmul_vf_f32m1(_S0, alphap[0], vl), alphap[1], _S1, vl);

                        vse32_v_f32m1(outptr, _p, vl);

                        alphap += 2;
                        outptr += packn;
                    }
                }

                delete[] buf;
            }

            if (resize_type == 3) // bicubic
            {
                const word_type vl = vsetvl_e32m1(packn);

                int* buf = new int[outw + outw * packn];

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
                        int sx = xofs[x] * packn;
                        const float* Sp = ptr + sx;

                        vfloat32m1_t _S0 = vle32_v_f32m1(Sp - packn, vl);
                        vfloat32m1_t _S1 = vle32_v_f32m1(Sp, vl);
                        vfloat32m1_t _S2 = vle32_v_f32m1(Sp + packn, vl);
                        vfloat32m1_t _S3 = vle32_v_f32m1(Sp + packn * 2, vl);
                        vfloat32m1_t _p = vfmacc_vf_f32m1(vfmacc_vf_f32m1(vfmacc_vf_f32m1(vfmul_vf_f32m1(_S0, alphap[0], vl), alphap[1], _S1, vl), alphap[2], _S2, vl), alphap[3], _S3, vl);

                        vse32_v_f32m1(outptr, _p, vl);

                        alphap += 4;
                        outptr += packn;
                    }
                }

                delete[] buf;
            }

            return 0;
        }
#endif // __riscv_vector

        if (resize_type == 1) // nearest
        {
            const float ws = output_width ? w / (float)outw : 1.f / width_scale;

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

#if __riscv_vector
    if (elempack == packn)
    {
        if (resize_type == 1) // nearest
        {
            const word_type vl = vsetvl_e32m1(packn);

            const float hs = output_height ? h / (float)outh : 1.f / height_scale;
            const float ws = output_width ? w / (float)outw : 1.f / width_scale;

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

                        vfloat32m1_t _p = vle32_v_f32m1(ptr + in_x * packn, vl);
                        vse32_v_f32m1(outptr, _p, vl);

                        outptr += packn;
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

                resize_bilinear_image_packn(src, dst, alpha, xofs, beta, yofs);
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

                resize_bicubic_image_packn(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return 0;
    }
#endif // __riscv_vector

    if (resize_type == 1) // nearest
    {
        const float hs = output_height ? h / (float)outh : 1.f / height_scale;
        const float ws = output_width ? w / (float)outw : 1.f / width_scale;

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

#if __riscv_vector && __riscv_zfh
int Interp_riscv::forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const int packn = csrr_vlenb() / 2;

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

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (elempack == packn)
        {
            const word_type vl = vsetvl_e16m1(packn);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                vfloat16m1_t _v = vle16_v_f16m1((const __fp16*)bottom_blob + q * packn, vl);
                top_blob_c.fill(_v);
            }

            return 0;
        }

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < w; q++)
        {
            Mat top_blob_c = top_blob.channel(q);
            const __fp16* ptr = bottom_blob;
            top_blob_c.fill(ptr[q]);
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

        if (elempack == packn)
        {
            if (resize_type == 1) // nearest
            {
                const word_type vl = vsetvl_e16m1(packn);

                const float ws = output_width ? w / (float)outw : 1.f / width_scale;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                    __fp16* outptr = top_blob.row<__fp16>(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        vfloat16m1_t _p = vle16_v_f16m1(ptr + in_x * packn, vl);
                        vse16_v_f16m1(outptr, _p, vl);

                        outptr += packn;
                    }
                }
            }

            if (resize_type == 2) // bilinear
            {
                const word_type vl = vsetvl_e16m1(packn);

                int* buf = new int[outw + outw * packn];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                linear_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                    __fp16* outptr = top_blob.row<__fp16>(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * packn;
                        const __fp16* Sp = ptr + sx;

                        vfloat16m1_t _S0 = vle16_v_f16m1(Sp, vl);
                        vfloat16m1_t _S1 = vle16_v_f16m1(Sp + packn, vl);
                        vfloat32m2_t _p = vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S0, alphap[0], vl), alphap[1], _S1, vl);

                        vse16_v_f16m1(outptr, vfncvt_f_f_w_f16m1(_p, vl), vl);

                        alphap += 2;
                        outptr += packn;
                    }
                }

                delete[] buf;
            }

            if (resize_type == 3) // bicubic
            {
                const word_type vl = vsetvl_e16m1(packn);

                int* buf = new int[outw + outw * packn];

                int* xofs = buf;
                float* alpha = (float*)(buf + outw);

                cubic_coeffs(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                    __fp16* outptr = top_blob.row<__fp16>(y);
                    const float* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * packn;
                        const __fp16* Sp = ptr + sx;

                        vfloat16m1_t _S0 = vle16_v_f16m1(Sp - packn, vl);
                        vfloat16m1_t _S1 = vle16_v_f16m1(Sp, vl);
                        vfloat16m1_t _S2 = vle16_v_f16m1(Sp + packn, vl);
                        vfloat16m1_t _S3 = vle16_v_f16m1(Sp + packn * 2, vl);
                        vfloat32m2_t _p = vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmacc_vf_f32m2(vfwmul_vf_f32m2(_S0, alphap[0], vl), alphap[1], _S1, vl), alphap[2], _S2, vl), alphap[3], _S3, vl);

                        vse16_v_f16m1(outptr, vfncvt_f_f_w_f16m1(_p, vl), vl);

                        alphap += 4;
                        outptr += packn;
                    }
                }

                delete[] buf;
            }

            return 0;
        }

        if (resize_type == 1) // nearest
        {
            const float ws = output_width ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                __fp16* outptr = top_blob.row<__fp16>(y);
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
                const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                __fp16* outptr = top_blob.row<__fp16>(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const __fp16* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    *outptr++ = (__fp16)((float)Sp[0] * a0 + (float)Sp[1] * a1);
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
                const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                __fp16* outptr = top_blob.row<__fp16>(y);
                const float* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const __fp16* Sp = ptr + sx;
                    float a0 = alphap[0];
                    float a1 = alphap[1];
                    float a2 = alphap[2];
                    float a3 = alphap[3];
                    *outptr++ = (__fp16)((float)Sp[-1] * a0 + (float)Sp[0] * a1 + (float)Sp[1] * a2 + (float)Sp[2] * a3);
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

    if (elempack == packn)
    {
        if (resize_type == 1) // nearest
        {
            const word_type vl = vsetvl_e16m1(packn);

            const float hs = output_height ? h / (float)outh : 1.f / height_scale;
            const float ws = output_width ? w / (float)outw : 1.f / width_scale;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                for (int y = 0; y < outh; y++)
                {
                    int in_y = std::min((int)(y * hs), (h - 1));

                    const __fp16* ptr = src.row<const __fp16>(in_y);
                    __fp16* outptr = dst.row<__fp16>(y);
                    for (int x = 0; x < outw; x++)
                    {
                        int in_x = std::min((int)(x * ws), (w - 1));

                        vfloat16m1_t _p = vle16_v_f16m1(ptr + in_x * packn, vl);
                        vse16_v_f16m1(outptr, _p, vl);

                        outptr += packn;
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

                resize_bilinear_image_packn_fp16s(src, dst, alpha, xofs, beta, yofs);
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

                resize_bicubic_image_packn_fp16s(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return 0;
    }

    if (resize_type == 1) // nearest
    {
        const float hs = output_height ? h / (float)outh : 1.f / height_scale;
        const float ws = output_width ? w / (float)outw : 1.f / width_scale;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            for (int y = 0; y < outh; y++)
            {
                int in_y = std::min((int)(y * hs), (h - 1));

                const __fp16* ptr = src.row<const __fp16>(in_y);
                __fp16* outptr = dst.row<__fp16>(y);
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

            resize_bilinear_image_fp16s(src, dst, alpha, xofs, beta, yofs);
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

            resize_bicubic_image_fp16s(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    return 0;
}

int Interp_riscv::forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const int packn = csrr_vlenb() / 2;

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

    if (dims == 1 || resize_type == 1) // nearest
    {
        return forward_fp16s(bottom_blobs, top_blobs, opt);
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

        if (elempack == packn)
        {
            if (resize_type == 2) // bilinear
            {
                const word_type vl = vsetvl_e16m1(packn);

                int* buf = new int[outw + outw * packn];

                int* xofs = buf;
                __fp16* alpha = (__fp16*)(buf + outw);

                linear_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                    __fp16* outptr = top_blob.row<__fp16>(y);
                    const __fp16* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * packn;
                        const __fp16* Sp = ptr + sx;

                        vfloat16m1_t _S0 = vle16_v_f16m1(Sp, vl);
                        vfloat16m1_t _S1 = vle16_v_f16m1(Sp + packn, vl);
                        vfloat16m1_t _p = vfmacc_vf_f16m1(vfmul_vf_f16m1(_S0, alphap[0], vl), alphap[1], _S1, vl);

                        vse16_v_f16m1(outptr, _p, vl);

                        alphap += 2;
                        outptr += packn;
                    }
                }

                delete[] buf;
            }

            if (resize_type == 3) // bicubic
            {
                const word_type vl = vsetvl_e16m1(packn);

                int* buf = new int[outw + outw * packn];

                int* xofs = buf;
                __fp16* alpha = (__fp16*)(buf + outw);

                cubic_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int y = 0; y < h; y++)
                {
                    const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                    __fp16* outptr = top_blob.row<__fp16>(y);
                    const __fp16* alphap = alpha;

                    for (int x = 0; x < outw; x++)
                    {
                        int sx = xofs[x] * packn;
                        const __fp16* Sp = ptr + sx;

                        vfloat16m1_t _S0 = vle16_v_f16m1(Sp - packn, vl);
                        vfloat16m1_t _S1 = vle16_v_f16m1(Sp, vl);
                        vfloat16m1_t _S2 = vle16_v_f16m1(Sp + packn, vl);
                        vfloat16m1_t _S3 = vle16_v_f16m1(Sp + packn * 2, vl);
                        vfloat16m1_t _p = vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmacc_vf_f16m1(vfmul_vf_f16m1(_S0, alphap[0], vl), alphap[1], _S1, vl), alphap[2], _S2, vl), alphap[3], _S3, vl);

                        vse16_v_f16m1(outptr, _p, vl);

                        alphap += 4;
                        outptr += packn;
                    }
                }

                delete[] buf;
            }

            return 0;
        }

        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outw * 2];

            int* xofs = buf;
            __fp16* alpha = (__fp16*)(buf + outw);

            linear_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                __fp16* outptr = top_blob.row<__fp16>(y);
                const __fp16* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const __fp16* Sp = ptr + sx;
                    __fp16 a0 = alphap[0];
                    __fp16 a1 = alphap[1];
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
            __fp16* alpha = (__fp16*)(buf + outw);

            cubic_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int y = 0; y < h; y++)
            {
                const __fp16* ptr = bottom_blob.row<const __fp16>(y);
                __fp16* outptr = top_blob.row<__fp16>(y);
                const __fp16* alphap = alpha;

                for (int x = 0; x < outw; x++)
                {
                    int sx = xofs[x];
                    const __fp16* Sp = ptr + sx;
                    __fp16 a0 = alphap[0];
                    __fp16 a1 = alphap[1];
                    __fp16 a2 = alphap[2];
                    __fp16 a3 = alphap[3];
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

    if (elempack == packn)
    {
        if (resize_type == 2) // bilinear
        {
            int* buf = new int[outw + outh + outw * 2 + outh * 2];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            __fp16* alpha = (__fp16*)(buf + outw + outh);           //new __fp16[outw * 2];
            __fp16* beta = (__fp16*)(buf + outw + outh + outw * 2); //new __fp16[outh * 2];

            linear_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);
            linear_coeffs_fp16sa(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bilinear_image_packn_fp16sa(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        if (resize_type == 3) // bicubic
        {
            int* buf = new int[outw + outh + outw * 4 + outh * 4];

            int* xofs = buf;        //new int[outw];
            int* yofs = buf + outw; //new int[outh];

            __fp16* alpha = (__fp16*)(buf + outw + outh);           //new __fp16[outw * 4];
            __fp16* beta = (__fp16*)(buf + outw + outh + outw * 4); //new __fp16[outh * 4];

            cubic_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);
            cubic_coeffs_fp16sa(h, outh, yofs, beta, align_corner);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const Mat src = bottom_blob.channel(q);
                Mat dst = top_blob.channel(q);

                resize_bicubic_image_packn_fp16sa(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return 0;
    }

    if (resize_type == 2) // bilinear
    {
        int* buf = new int[outw + outh + outw * 2 + outh * 2];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        __fp16* alpha = (__fp16*)(buf + outw + outh);           //new __fp16[outw * 2];
        __fp16* beta = (__fp16*)(buf + outw + outh + outw * 2); //new __fp16[outh * 2];

        linear_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);
        linear_coeffs_fp16sa(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bilinear_image_fp16sa(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    if (resize_type == 3) // bicubic
    {
        int* buf = new int[outw + outh + outw * 4 + outh * 4];

        int* xofs = buf;        //new int[outw];
        int* yofs = buf + outw; //new int[outh];

        __fp16* alpha = (__fp16*)(buf + outw + outh);           //new __fp16[outw * 4];
        __fp16* beta = (__fp16*)(buf + outw + outh + outw * 4); //new __fp16[outh * 4];

        cubic_coeffs_fp16sa(w, outw, xofs, alpha, align_corner);
        cubic_coeffs_fp16sa(h, outh, yofs, beta, align_corner);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const Mat src = bottom_blob.channel(q);
            Mat dst = top_blob.channel(q);

            resize_bicubic_image_fp16sa(src, dst, alpha, xofs, beta, yofs);
        }

        delete[] buf;
    }

    return 0;
}
#endif // __riscv_vector && __riscv_zfh

} // namespace ncnn
