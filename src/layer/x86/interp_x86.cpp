// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "interp_x86.h"

#include <math.h>

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

#include "interp_bicubic.h"
#include "interp_bilinear.h"

#if __SSE2__
#include "interp_bicubic_pack4.h"
#include "interp_bilinear_pack4.h"

#if __AVX__
#include "interp_bicubic_pack8.h"
#include "interp_bilinear_pack8.h"

#if __AVX512F__
#include "interp_bicubic_pack16.h"
#include "interp_bilinear_pack16.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

Interp_x86::Interp_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Interp_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

    if (dims == 1)
    {
        top_blob.create(outw, outh, w, elemsize, elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                __m512 _v = _mm512_loadu_ps((const float*)bottom_blob + q * 16);
                top_blob_c.fill(_v);
            }

            return 0;
        }
#endif // __AVX512F__

        if (elempack == 8)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                __m256 _v = _mm256_load_ps((const float*)bottom_blob + q * 8);
                top_blob_c.fill(_v);
            }

            return 0;
        }
#endif // __AVX__

        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < w; q++)
            {
                Mat top_blob_c = top_blob.channel(q);
                __m128 _v = _mm_load_ps((const float*)bottom_blob + q * 4);
                top_blob_c.fill(_v);
            }

            return 0;
        }
#endif // __SSE2__

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

#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
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

                        __m512 _p = _mm512_load_ps(ptr + in_x * 16);
                        _mm512_store_ps(outptr, _p);

                        outptr += 16;
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
                        int sx = xofs[x] * 16;
                        const float* Sp = ptr + sx;

                        __m512 _a0 = _mm512_set1_ps(alphap[0]);
                        __m512 _a1 = _mm512_set1_ps(alphap[1]);

                        __m512 _S0 = _mm512_load_ps(Sp);
                        __m512 _S1 = _mm512_load_ps(Sp + 16);
                        __m512 _p = _mm512_mul_ps(_S0, _a0);
                        _p = _mm512_fmadd_ps(_S1, _a1, _p);
                        _mm512_store_ps(outptr, _p);

                        alphap += 2;
                        outptr += 16;
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
                        int sx = xofs[x] * 16;
                        const float* Sp = ptr + sx;

                        __m512 _a0 = _mm512_set1_ps(alphap[0]);
                        __m512 _a1 = _mm512_set1_ps(alphap[1]);
                        __m512 _a2 = _mm512_set1_ps(alphap[2]);
                        __m512 _a3 = _mm512_set1_ps(alphap[3]);

                        __m512 _S0 = _mm512_load_ps(Sp - 16);
                        __m512 _S1 = _mm512_load_ps(Sp + 0);
                        __m512 _S2 = _mm512_load_ps(Sp + 16);
                        __m512 _S3 = _mm512_load_ps(Sp + 32);
                        __m512 _p = _mm512_mul_ps(_S0, _a0);
                        _p = _mm512_fmadd_ps(_S1, _a1, _p);
                        _p = _mm512_fmadd_ps(_S2, _a2, _p);
                        _p = _mm512_fmadd_ps(_S3, _a3, _p);
                        _mm512_store_ps(outptr, _p);

                        alphap += 4;
                        outptr += 16;
                    }
                }

                delete[] buf;
            }

            return 0;
        }
#endif // __AVX512F__

        if (elempack == 8)
        {
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

                        __m256 _p = _mm256_load_ps(ptr + in_x * 8);
                        _mm256_store_ps(outptr, _p);

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

                        __m256 _a0 = _mm256_set1_ps(alphap[0]);
                        __m256 _a1 = _mm256_set1_ps(alphap[1]);

                        __m256 _S0 = _mm256_load_ps(Sp);
                        __m256 _S1 = _mm256_load_ps(Sp + 8);
                        __m256 _p = _mm256_mul_ps(_S0, _a0);
                        _p = _mm256_comp_fmadd_ps(_S1, _a1, _p);
                        _mm256_store_ps(outptr, _p);

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

                        __m256 _a0 = _mm256_set1_ps(alphap[0]);
                        __m256 _a1 = _mm256_set1_ps(alphap[1]);
                        __m256 _a2 = _mm256_set1_ps(alphap[2]);
                        __m256 _a3 = _mm256_set1_ps(alphap[3]);

                        __m256 _S0 = _mm256_load_ps(Sp - 8);
                        __m256 _S1 = _mm256_load_ps(Sp + 0);
                        __m256 _S2 = _mm256_load_ps(Sp + 8);
                        __m256 _S3 = _mm256_load_ps(Sp + 16);
                        __m256 _p = _mm256_mul_ps(_S0, _a0);
                        _p = _mm256_comp_fmadd_ps(_S1, _a1, _p);
                        _p = _mm256_comp_fmadd_ps(_S2, _a2, _p);
                        _p = _mm256_comp_fmadd_ps(_S3, _a3, _p);
                        _mm256_store_ps(outptr, _p);

                        alphap += 4;
                        outptr += 8;
                    }
                }

                delete[] buf;
            }

            return 0;
        }
#endif // __AVX__

        if (elempack == 4)
        {
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

                        __m128 _p = _mm_load_ps(ptr + in_x * 4);
                        _mm_store_ps(outptr, _p);

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

                        __m128 _a0 = _mm_set1_ps(alphap[0]);
                        __m128 _a1 = _mm_set1_ps(alphap[1]);

                        __m128 _S0 = _mm_load_ps(Sp);
                        __m128 _S1 = _mm_load_ps(Sp + 4);
                        __m128 _p = _mm_mul_ps(_S0, _a0);
                        _p = _mm_comp_fmadd_ps(_S1, _a1, _p);
                        _mm_store_ps(outptr, _p);

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

                        __m128 _a0 = _mm_set1_ps(alphap[0]);
                        __m128 _a1 = _mm_set1_ps(alphap[1]);
                        __m128 _a2 = _mm_set1_ps(alphap[2]);
                        __m128 _a3 = _mm_set1_ps(alphap[3]);

                        __m128 _S0 = _mm_load_ps(Sp - 4);
                        __m128 _S1 = _mm_load_ps(Sp + 0);
                        __m128 _S2 = _mm_load_ps(Sp + 4);
                        __m128 _S3 = _mm_load_ps(Sp + 8);
                        __m128 _p = _mm_mul_ps(_S0, _a0);
                        _p = _mm_comp_fmadd_ps(_S1, _a1, _p);
                        _p = _mm_comp_fmadd_ps(_S2, _a2, _p);
                        _p = _mm_comp_fmadd_ps(_S3, _a3, _p);
                        _mm_store_ps(outptr, _p);

                        alphap += 4;
                        outptr += 4;
                    }
                }

                delete[] buf;
            }

            return 0;
        }
#endif // __SSE2__

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

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
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

                        __m512 _p = _mm512_load_ps(ptr + in_x * 16);
                        _mm512_store_ps(outptr, _p);

                        outptr += 16;
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

                resize_bilinear_image_pack16(src, dst, alpha, xofs, beta, yofs);
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

                resize_bicubic_image_pack16(src, dst, alpha, xofs, beta, yofs);
            }

            delete[] buf;
        }

        return 0;
    }
#endif // __AVX512F__

    if (elempack == 8)
    {
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

                        __m256 _p = _mm256_load_ps(ptr + in_x * 8);
                        _mm256_store_ps(outptr, _p);

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
#endif // __AVX__

    if (elempack == 4)
    {
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

                        __m128 _p = _mm_load_ps(ptr + in_x * 4);
                        _mm_store_ps(outptr, _p);

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
#endif // __SSE2__

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

} // namespace ncnn
