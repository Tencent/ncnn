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

#include "relu_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

namespace ncnn {

ReLU_x86::ReLU_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int ReLU_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

    if (elembits == 8)
        return forward_inplace_int8(bottom_top_blob, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _zero_avx512 = _mm512_setzero_ps();
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                _mm512_storeu_ps(ptr, _mm512_max_ps(_zero_avx512, _p));
                ptr += 16;
            }
#endif // __AVX512F__
            __m256 _zero_avx = _mm256_setzero_ps();
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                _mm256_storeu_ps(ptr, _mm256_max_ps(_zero_avx, _p));
                ptr += 8;
            }
#endif // __AVX__
            __m128 _zero = _mm_setzero_ps();
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr);
                _mm_store_ps(ptr, _mm_max_ps(_zero, _p));
                ptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *ptr = std::max(*ptr, 0.f);
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _zero_avx512 = _mm512_setzero_ps();
            __m512 _slope_avx512 = _mm512_set1_ps(slope);
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __mmask16 _is_negative = _mm512_cmp_ps_mask(_p, _zero_avx512, _CMP_LT_OQ);
                _p = _mm512_mask_mul_ps(_p, _is_negative, _p, _slope_avx512);
                _mm512_storeu_ps(ptr, _p);
                ptr += 16;
            }
#endif // __AVX512F__
            __m256 _zero_avx = _mm256_setzero_ps();
            __m256 _slope_avx = _mm256_set1_ps(slope);
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _pos = _mm256_max_ps(_zero_avx, _p);
                __m256 _neg = _mm256_min_ps(_zero_avx, _p);
                _p = _mm256_add_ps(_pos, _mm256_mul_ps(_slope_avx, _neg));
                _mm256_storeu_ps(ptr, _p);
                ptr += 8;
            }
#endif // __AVX__
            __m128 _zero = _mm_setzero_ps();
            __m128 _slope = _mm_set1_ps(slope);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr);
                __m128 _pos = _mm_max_ps(_zero, _p);
                __m128 _neg = _mm_min_ps(_zero, _p);
                _p = _mm_add_ps(_pos, _mm_mul_ps(_slope, _neg));
                _mm_store_ps(ptr, _p);
                ptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
        }
    }

    return 0;
}

int ReLU_x86::forward_inplace_int8(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;
    int elempack = bottom_top_blob.elempack;

#if __SSE2__
    if (elempack == 8)
    {
        if (slope == 0.f)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                signed char* ptr = bottom_top_blob.channel(q);

                int i = 0;
                for (; i < size; i++)
                {
                    if (ptr[0] < 0)
                        ptr[0] = 0;
                    if (ptr[1] < 0)
                        ptr[1] = 0;
                    if (ptr[2] < 0)
                        ptr[2] = 0;
                    if (ptr[3] < 0)
                        ptr[3] = 0;
                    if (ptr[4] < 0)
                        ptr[4] = 0;
                    if (ptr[5] < 0)
                        ptr[5] = 0;
                    if (ptr[6] < 0)
                        ptr[6] = 0;
                    if (ptr[7] < 0)
                        ptr[7] = 0;

                    ptr += 8;
                }
            }
        }
        else
        {
            // TODO leakyrelu
        }

        return 0;
    }
#endif // __SSE2__

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            signed char* ptr = bottom_top_blob.channel(q);

            int i = 0;
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr = 0;

                ptr++;
            }
        }
    }
    else
    {
        // TODO leakyrelu
    }

    return 0;
}

} //namespace ncnn
