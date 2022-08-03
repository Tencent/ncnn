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

#include "batchnorm_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

BatchNorm_x86::BatchNorm_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int BatchNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int c = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        const float* aptr = a_data;
        const float* bptr = b_data;

        const int size = w * elempack;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p512 = _mm512_loadu_ps(ptr);
            __m512 _a512 = _mm512_loadu_ps(aptr);
            __m512 _b512 = _mm512_loadu_ps(bptr);
            _p512 = _mm512_fmadd_ps(_p512, _b512, _a512);
            _mm512_storeu_ps(ptr, _p512);
            ptr += 16;
            aptr += 16;
            bptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p256 = _mm256_loadu_ps(ptr);
            __m256 _a256 = _mm256_loadu_ps(aptr);
            __m256 _b256 = _mm256_loadu_ps(bptr);
            _p256 = _mm256_comp_fmadd_ps(_p256, _b256, _a256);
            _mm256_storeu_ps(ptr, _p256);
            ptr += 8;
            aptr += 8;
            bptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p128 = _mm_loadu_ps(ptr);
            __m128 _a128 = _mm_loadu_ps(aptr);
            __m128 _b128 = _mm_loadu_ps(bptr);
            _p128 = _mm_comp_fmadd_ps(_p128, _b128, _a128);
            _mm_storeu_ps(ptr, _p128);
            ptr += 4;
            aptr += 4;
            bptr += 4;
        }
#endif // __SSE__
        for (; i < size; i++)
        {
            *ptr = *bptr * *ptr + *aptr;
            ptr++;
            aptr++;
            bptr++;
        }
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            float a = a_data[i];
            float b = b_data[i];

#if __SSE2__
            __m128 _a128 = (elempack == 4) ? _mm_loadu_ps((const float*)a_data + i * 4) : _mm_set1_ps(a);
            __m128 _b128 = (elempack == 4) ? _mm_loadu_ps((const float*)b_data + i * 4) : _mm_set1_ps(b);
#if __AVX__
            __m256 _a256 = (elempack == 8) ? _mm256_loadu_ps((const float*)a_data + i * 8) : _mm256_insertf128_ps(_mm256_castps128_ps256(_a128), _a128, 1);
            __m256 _b256 = (elempack == 8) ? _mm256_loadu_ps((const float*)b_data + i * 8) : _mm256_insertf128_ps(_mm256_castps128_ps256(_b128), _b128, 1);
#if __AVX512F__
            __m512 _a512 = (elempack == 16) ? _mm512_loadu_ps((const float*)a_data + i * 16) : _mm512_insertf32x8(_mm512_castps256_ps512(_a256), _a256, 1);
            __m512 _b512 = (elempack == 16) ? _mm512_loadu_ps((const float*)b_data + i * 16) : _mm512_insertf32x8(_mm512_castps256_ps512(_b256), _b256, 1);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

            int j = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; j + 15 < size; j += 16)
            {
                __m512 _p512 = _mm512_loadu_ps(ptr);
                _p512 = _mm512_fmadd_ps(_p512, _b512, _a512);
                _mm512_storeu_ps(ptr, _p512);
                ptr += 16;
            }
#endif // __AVX512F__
            for (; j + 7 < size; j += 8)
            {
                __m256 _p256 = _mm256_loadu_ps(ptr);
                _p256 = _mm256_comp_fmadd_ps(_p256, _b256, _a256);
                _mm256_storeu_ps(ptr, _p256);
                ptr += 8;
            }
#endif // __AVX__
            for (; j + 3 < size; j += 4)
            {
                __m128 _p128 = _mm_loadu_ps(ptr);
                _p128 = _mm_comp_fmadd_ps(_p128, _b128, _a128);
                _mm_storeu_ps(ptr, _p128);
                ptr += 4;
            }
#endif // __SSE__
            for (; j < size; j++)
            {
                *ptr = b * *ptr + a;
                ptr++;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float a = a_data[q];
            float b = b_data[q];

#if __SSE2__
            __m128 _a128 = (elempack == 4) ? _mm_loadu_ps((const float*)a_data + q * 4) : _mm_set1_ps(a);
            __m128 _b128 = (elempack == 4) ? _mm_loadu_ps((const float*)b_data + q * 4) : _mm_set1_ps(b);
#if __AVX__
            __m256 _a256 = (elempack == 8) ? _mm256_loadu_ps((const float*)a_data + q * 8) : _mm256_insertf128_ps(_mm256_castps128_ps256(_a128), _a128, 1);
            __m256 _b256 = (elempack == 8) ? _mm256_loadu_ps((const float*)b_data + q * 8) : _mm256_insertf128_ps(_mm256_castps128_ps256(_b128), _b128, 1);
#if __AVX512F__
            __m512 _a512 = (elempack == 16) ? _mm512_loadu_ps((const float*)a_data + q * 16) : _mm512_insertf32x8(_mm512_castps256_ps512(_a256), _a256, 1);
            __m512 _b512 = (elempack == 16) ? _mm512_loadu_ps((const float*)b_data + q * 16) : _mm512_insertf32x8(_mm512_castps256_ps512(_b256), _b256, 1);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p512 = _mm512_loadu_ps(ptr);
                _p512 = _mm512_fmadd_ps(_p512, _b512, _a512);
                _mm512_storeu_ps(ptr, _p512);
                ptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p256 = _mm256_loadu_ps(ptr);
                _p256 = _mm256_comp_fmadd_ps(_p256, _b256, _a256);
                _mm256_storeu_ps(ptr, _p256);
                ptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p128 = _mm_loadu_ps(ptr);
                _p128 = _mm_comp_fmadd_ps(_p128, _b128, _a128);
                _mm_storeu_ps(ptr, _p128);
                ptr += 4;
            }
#endif // __SSE__
            for (; i < size; i++)
            {
                *ptr = b * *ptr + a;
                ptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn
