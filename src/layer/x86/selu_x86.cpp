// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "selu_x86.h"

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

namespace ncnn {

SELU_x86::SELU_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int SELU_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int elempack = bottom_top_blob.elempack;
    int channels = bottom_top_blob.c;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _zero512 = _mm512_setzero_ps();
        __m512 _one512 = _mm512_set1_ps(1.f);
        __m512 _alpha512 = _mm512_set1_ps(alpha);
        __m512 _lambda512 = _mm512_set1_ps(lambda);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);

            __m512 _pos = _mm512_max_ps(_zero512, _p);
            __m512 _neg = _mm512_min_ps(_zero512, _p);

            __m512 _blob = exp512_ps(_neg);
            _blob = _mm512_sub_ps(_blob, _one512);
            _blob = _mm512_mul_ps(_alpha512, _blob);
            _blob = _mm512_mul_ps(_lambda512, _mm512_add_ps(_pos, _blob));

            _mm512_storeu_ps(ptr, _blob);

            ptr += 16;
        }
#endif // __AVX512F__
        __m256 _zero256 = _mm256_setzero_ps();
        __m256 _one256 = _mm256_set1_ps(1.f);
        __m256 _alpha256 = _mm256_set1_ps(alpha);
        __m256 _lambda256 = _mm256_set1_ps(lambda);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);

            __m256 _pos = _mm256_max_ps(_zero256, _p);
            __m256 _neg = _mm256_min_ps(_zero256, _p);

            __m256 _blob = exp256_ps(_neg);
            _blob = _mm256_sub_ps(_blob, _one256);
            _blob = _mm256_mul_ps(_alpha256, _blob);
            _blob = _mm256_mul_ps(_lambda256, _mm256_add_ps(_pos, _blob));

            _mm256_storeu_ps(ptr, _blob);

            ptr += 8;
        }
#endif // __AVX__
        __m128 _zero128 = _mm_setzero_ps();
        __m128 _one128 = _mm_set1_ps(1.f);
        __m128 _alpha128 = _mm_set1_ps(alpha);
        __m128 _lambda128 = _mm_set1_ps(lambda);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);

            __m128 _pos = _mm_max_ps(_zero128, _p);
            __m128 _neg = _mm_min_ps(_zero128, _p);

            __m128 _blob = exp_ps(_neg);
            _blob = _mm_sub_ps(_blob, _one128);
            _blob = _mm_mul_ps(_alpha128, _blob);
            _blob = _mm_mul_ps(_lambda128, _mm_add_ps(_pos, _blob));

            _mm_storeu_ps(ptr, _blob);

            ptr += 4;
        }
#endif // __SSE2__
        float alphaxlambda = alpha * lambda;
        for (; i < size; i++)
        {
            // y = lambda * ( max(0, x) + min(0, alpha * (exp(x) - 1)) )
            if (*ptr < 0)
                *ptr = (expf(*ptr) - 1.f) * alphaxlambda;
            else
                *ptr = *ptr * lambda;
            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
