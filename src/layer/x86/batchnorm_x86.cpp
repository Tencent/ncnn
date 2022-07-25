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
    int c = bottom_top_blob.c;
    int d = bottom_top_blob.d;
    int h = bottom_top_blob.h;
    int w = bottom_top_blob.w;
    int elempack = bottom_top_blob.elempack;

    int loops;
    int size;

    // determine the loops and the size
    if(dims == 2)
    {
        loops = h;
        size = w * elempack;
    }
    else // dims = 1, 3, 4
    {
        loops = c;
        size = d * h * w * elempack;
    }


    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < loops; q++)
    {
        float* ptr;
        if (dims == 1)
            ptr = bottom_top_blob;
        else if (dims == 2)
            ptr = bottom_top_blob.row(q);
        else // dims == 3 or dims == 4
            ptr = bottom_top_blob.channel(q);

        int i = 0;

        float a, b;
        a = a_data[q];
        b = b_data[q];
#if __SSE2__
        __m128 _a128, _b128, _p128;
        if (elempack == 1)
        {
            // dims != 1
            _a128 = _mm_set1_ps(a);
            _b128 = _mm_set1_ps(b);
        }
        else if (elempack == 4)
        {
            _a128 = _mm_load_ps((const float*)a_data + q * 4);
            _b128 = _mm_load_ps((const float*)b_data + q * 4);
        }
#endif // __SSE2__

#if __AVX__
        __m256 _a256, _b256, _p256;
        if (elempack == 1)
        {
            // dims != 1
            _a256 = _mm256_set1_ps(a);
            _b256 = _mm256_set1_ps(b);
        }
        else if (elempack == 4)
        {
            _a256 = _mm256_castps128_ps256(_a128);
            _a256 = _mm256_insertf128_ps(_a256, _a128, 1);
            _b256 = _mm256_castps128_ps256(_b128);
            _b256 = _mm256_insertf128_ps(_b256, _b128, 1);
        }
        else if (elempack == 8)
        {
            _a256 = _mm256_loadu_ps((const float*)a_data + q * 8);
            _b256 = _mm256_loadu_ps((const float*)b_data + q * 8);
        }
#endif // __AVX__

#if __AVX512F__
        __m512 _a512, _b512, _p512;
        if (elempack == 1)
        {
            // dims != 1
            _a512 = _mm512_set1_ps(a);
            _b512 = _mm512_set1_ps(b);
        }
        else if (elempack == 4 || elempack == 8)
        {
            _a512 = _mm512_castps256_ps512(_a256);
            _a512 = _mm512_insertf32x8(_a512, _a256, 1);
            _b512 = _mm512_castps256_ps512(_b256);
            _b512 = _mm512_insertf32x8(_b512, _b256, 1);
        }
        else // elempack == 16
        {
            _a512 = _mm512_loadu_ps((const float*)a_data + q * 16);
            _b512 = _mm512_loadu_ps((const float*)b_data + q * 16);
        }
#endif // __AVX512F__

#if __SSE2__
#if __AVX__
#if __AVX512F__

        for (; i + 15 < size; i += 16)
        {
            if (dims == 1)
            {
                _a512 = _mm512_loadu_ps((const float*)a_data + i);
                _b512 = _mm512_loadu_ps((const float*)b_data + i);
            }

            _p512 = _mm512_loadu_ps(ptr);
            _p512 = _mm512_fmadd_ps(_p512, _b512, _a512);
            _mm512_storeu_ps(ptr, _p512);

            ptr += 16;
        }
#endif // __AVX512F__

        for (; i + 7 < size; i += 8)
        {
            if (dims == 1)
            {
                _a256 = _mm256_loadu_ps((const float*)a_data + i);
                _b256 = _mm256_loadu_ps((const float*)b_data + i);
            }

            _p256 = _mm256_loadu_ps(ptr);
            _p256 = _mm256_comp_fmadd_ps(_p256, _b256, _a256);
            _mm256_storeu_ps(ptr, _p256);

            ptr += 8;
        }

#endif // __AVX__

        for (; i + 3 < size; i += 4)
        {
            if (dims == 1)
            {
                _a128 = _mm_loadu_ps((const float*)a_data + i);
                _b128 = _mm_loadu_ps((const float*)b_data + i);
            }

            _p128 = _mm_loadu_ps(ptr);
            _p128 = _mm_comp_fmadd_ps(_p128, _b128, _a128);
            _mm_storeu_ps(ptr, _p128);

            ptr += 4;
        }
#endif // __SSE__
        for (; i < size; i++)
        {
            if (dims == 1)
                *ptr = b_data[i] * *ptr + a_data[i];
            else
                *ptr = b * *ptr + a;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn

