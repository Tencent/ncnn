// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "eltwise_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

Eltwise_x86::Eltwise_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Eltwise_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;
    int size = w * h * elempack;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __m512 _p1 = _mm512_loadu_ps(ptr1);
                _p = _mm512_mul_ps(_p, _p1);
                _mm512_storeu_ps(outptr, _p);

                ptr += 16;
                ptr1 += 16;
                outptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _p1 = _mm256_loadu_ps(ptr1);
                _p = _mm256_mul_ps(_p, _p1);
                _mm256_storeu_ps(outptr, _p);

                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr);
                __m128 _p1 = _mm_load_ps(ptr1);
                _p = _mm_mul_ps(_p, _p1);
                _mm_store_ps(outptr, _p);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *outptr = *ptr * *ptr1;

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob2 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob2.channel(q);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p = _mm512_loadu_ps(outptr);
                    __m512 _p1 = _mm512_loadu_ps(ptr);
                    _p = _mm512_mul_ps(_p, _p1);
                    _mm512_storeu_ps(outptr, _p);

                    ptr += 16;
                    outptr += 16;
                }
#endif // __AVX512F__
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(outptr);
                    __m256 _p1 = _mm256_loadu_ps(ptr);
                    _p = _mm256_mul_ps(_p, _p1);
                    _mm256_storeu_ps(outptr, _p);

                    ptr += 8;
                    outptr += 8;
                }
#endif // __AVX__
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = _mm_load_ps(outptr);
                    __m128 _p1 = _mm_load_ps(ptr);
                    _p = _mm_mul_ps(_p, _p1);
                    _mm_store_ps(outptr, _p);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *outptr *= *ptr;

                    ptr++;
                    outptr++;
                }
            }
        }
    }
    if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr);
                    __m512 _p1 = _mm512_loadu_ps(ptr1);
                    _p = _mm512_add_ps(_p, _p1);
                    _mm512_storeu_ps(outptr, _p);

                    ptr += 16;
                    ptr1 += 16;
                    outptr += 16;
                }
#endif // __AVX512F__
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _p1 = _mm256_loadu_ps(ptr1);
                    _p = _mm256_add_ps(_p, _p1);
                    _mm256_storeu_ps(outptr, _p);

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
#endif // __AVX__
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = _mm_load_ps(ptr);
                    __m128 _p1 = _mm_load_ps(ptr1);
                    _p = _mm_add_ps(_p, _p1);
                    _mm_store_ps(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *outptr = *ptr + *ptr1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob2 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob2.channel(q);
                    float* outptr = top_blob.channel(q);

                    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    for (; i + 15 < size; i += 16)
                    {
                        __m512 _p = _mm512_loadu_ps(outptr);
                        __m512 _p1 = _mm512_loadu_ps(ptr);
                        _p = _mm512_add_ps(_p, _p1);
                        _mm512_storeu_ps(outptr, _p);

                        ptr += 16;
                        outptr += 16;
                    }
#endif // __AVX512F__
                    for (; i + 7 < size; i += 8)
                    {
                        __m256 _p = _mm256_loadu_ps(outptr);
                        __m256 _p1 = _mm256_loadu_ps(ptr);
                        _p = _mm256_add_ps(_p, _p1);
                        _mm256_storeu_ps(outptr, _p);

                        ptr += 8;
                        outptr += 8;
                    }
#endif // __AVX__
                    for (; i + 3 < size; i += 4)
                    {
                        __m128 _p = _mm_load_ps(outptr);
                        __m128 _p1 = _mm_load_ps(ptr);
                        _p = _mm_add_ps(_p, _p1);
                        _mm_store_ps(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __SSE2__
                    for (; i < size; i++)
                    {
                        *outptr += *ptr;

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                const float coeff0 = coeffs[0];
                const float coeff1 = coeffs[1];

                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                __m512 _coeff0_avx512 = _mm512_set1_ps(coeff0);
                __m512 _coeff1_avx512 = _mm512_set1_ps(coeff1);
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p = _mm512_loadu_ps(ptr);
                    __m512 _p1 = _mm512_loadu_ps(ptr1);
                    _p = _mm512_mul_ps(_p, _coeff0_avx512);
                    _p = _mm512_fmadd_ps(_p1, _coeff1_avx512, _p);
                    _mm512_storeu_ps(outptr, _p);

                    ptr += 16;
                    ptr1 += 16;
                    outptr += 16;
                }
#endif // __AVX512F__
                __m256 _coeff0_avx = _mm256_set1_ps(coeff0);
                __m256 _coeff1_avx = _mm256_set1_ps(coeff1);
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _p1 = _mm256_loadu_ps(ptr1);
                    _p = _mm256_mul_ps(_p, _coeff0_avx);
                    _p = _mm256_comp_fmadd_ps(_p1, _coeff1_avx, _p);
                    _mm256_storeu_ps(outptr, _p);

                    ptr += 8;
                    ptr1 += 8;
                    outptr += 8;
                }
#endif // __AVX__
                __m128 _coeff0 = _mm_set1_ps(coeff0);
                __m128 _coeff1 = _mm_set1_ps(coeff1);
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = _mm_load_ps(ptr);
                    __m128 _p1 = _mm_load_ps(ptr1);
                    _p = _mm_mul_ps(_p, _coeff0);
                    _p1 = _mm_mul_ps(_p1, _coeff1);
                    _p = _mm_add_ps(_p1, _p);
                    _mm_store_ps(outptr, _p);

                    ptr += 4;
                    ptr1 += 4;
                    outptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *outptr = *ptr * coeff0 + *ptr1 * coeff1;

                    ptr++;
                    ptr1++;
                    outptr++;
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob2 = bottom_blobs[b];
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob2.channel(q);
                    float* outptr = top_blob.channel(q);

                    const float coeff = coeffs[b];

                    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                    __m512 _coeff_avx512 = _mm512_set1_ps(coeff);
                    for (; i + 15 < size; i += 16)
                    {
                        __m512 _p = _mm512_loadu_ps(outptr);
                        __m512 _p1 = _mm512_loadu_ps(ptr);
                        _p = _mm512_fmadd_ps(_p1, _coeff_avx512, _p);
                        _mm512_storeu_ps(outptr, _p);

                        ptr += 16;
                        outptr += 16;
                    }
#endif // __AVX512F__
                    __m256 _coeff_avx = _mm256_set1_ps(coeff);
                    for (; i + 7 < size; i += 8)
                    {
                        __m256 _p = _mm256_loadu_ps(outptr);
                        __m256 _p1 = _mm256_loadu_ps(ptr);
                        _p = _mm256_comp_fmadd_ps(_p1, _coeff_avx, _p);
                        _mm256_storeu_ps(outptr, _p);

                        ptr += 8;
                        outptr += 8;
                    }
#endif // __AVX__
                    __m128 _coeff = _mm_set1_ps(coeff);
                    for (; i + 3 < size; i += 4)
                    {
                        __m128 _p1 = _mm_load_ps(ptr);
                        __m128 _p = _mm_load_ps(outptr);
                        _p1 = _mm_mul_ps(_p1, _coeff);
                        _p = _mm_add_ps(_p1, _p);
                        _mm_store_ps(outptr, _p);

                        ptr += 4;
                        outptr += 4;
                    }
#endif // __SSE2__
                    for (; i < size; i++)
                    {
                        *outptr += *ptr * coeff;

                        ptr++;
                        outptr++;
                    }
                }
            }
        }
    }
    if (op_type == Operation_MAX)
    {
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __m512 _p1 = _mm512_loadu_ps(ptr1);
                _p = _mm512_max_ps(_p, _p1);
                _mm512_storeu_ps(outptr, _p);

                ptr += 16;
                ptr1 += 16;
                outptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _p1 = _mm256_loadu_ps(ptr1);
                _p = _mm256_max_ps(_p, _p1);
                _mm256_storeu_ps(outptr, _p);

                ptr += 8;
                ptr1 += 8;
                outptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr);
                __m128 _p1 = _mm_load_ps(ptr1);
                _p = _mm_max_ps(_p, _p1);
                _mm_store_ps(outptr, _p);

                ptr += 4;
                ptr1 += 4;
                outptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *outptr = std::max(*ptr, *ptr1);

                ptr++;
                ptr1++;
                outptr++;
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob2 = bottom_blobs[b];
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob2.channel(q);
                float* outptr = top_blob.channel(q);

                int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
                for (; i + 15 < size; i += 16)
                {
                    __m512 _p = _mm512_loadu_ps(outptr);
                    __m512 _p1 = _mm512_loadu_ps(ptr);
                    _p = _mm512_max_ps(_p, _p1);
                    _mm512_storeu_ps(outptr, _p);

                    ptr += 16;
                    outptr += 16;
                }
#endif // __AVX512F__
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = _mm256_loadu_ps(outptr);
                    __m256 _p1 = _mm256_loadu_ps(ptr);
                    _p = _mm256_max_ps(_p, _p1);
                    _mm256_storeu_ps(outptr, _p);

                    ptr += 8;
                    outptr += 8;
                }
#endif // __AVX__
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = _mm_load_ps(outptr);
                    __m128 _p1 = _mm_load_ps(ptr);
                    _p = _mm_max_ps(_p, _p1);
                    _mm_store_ps(outptr, _p);

                    ptr += 4;
                    outptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *outptr = std::max(*ptr, *outptr);

                    ptr++;
                    outptr++;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
