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

#include "sigmoid_x86.h"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include <math.h>

namespace ncnn {

Sigmoid_x86::Sigmoid_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Sigmoid_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _one_avx512 = _mm512_set1_ps(1.f);
        __m512 _zero_avx512 = _mm512_setzero_ps();
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = _mm512_div_ps(_one_avx512, _mm512_add_ps(_one_avx512, exp512_ps(_mm512_sub_ps(_zero_avx512, _p))));
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        __m256 _one_avx = _mm256_set1_ps(1.f);
        __m256 _zero_avx = _mm256_setzero_ps();
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = _mm256_div_ps(_one_avx, _mm256_add_ps(_one_avx, exp256_ps(_mm256_sub_ps(_zero_avx, _p))));
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        __m128 _one = _mm_set1_ps(1.f);
        __m128 _zero = _mm_setzero_ps();
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            _p = _mm_div_ps(_one, _mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _p))));
            _mm_store_ps(ptr, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = 1.f / (1.f + exp(-*ptr));
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
