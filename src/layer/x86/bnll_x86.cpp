// JasonZhang892 is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 JasonZhang892 <zqhy_0929@163.com>. All rights reserved.
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

#include "bnll_x86.h"

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

BNLL_x86::BNLL_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int BNLL_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
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
            __mmask16 mask = _mm512_cmp_ps_mask(_p, _zero_avx512, _CMP_GT_OQ);
            __m512 _abs_p = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(_p), _mm512_set1_epi32(0x7fffffff)));
            __m512 _tmp = log512_ps(_mm512_add_ps(_one_avx512, exp512_ps(_mm512_sub_ps(_zero_avx512, _abs_p))));
            _p = _mm512_mask_add_ps(_tmp, mask, _tmp, _p);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        __m256 _one_avx = _mm256_set1_ps(1.f);
        __m256 _zero_avx = _mm256_setzero_ps();
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 mask = _mm256_cmp_ps(_p, _mm256_setzero_ps(), _CMP_GT_OQ);
            __m256 _abs_p = _mm256_and_ps(_p, *(__m256*)_ps256_inv_sign_mask);
            __m256 _tmp = log256_ps(_mm256_add_ps(_one_avx, exp256_ps(_mm256_sub_ps(_zero_avx, _abs_p))));
            __m256 _x = _mm256_and_ps(_p, mask);
            _p = _mm256_add_ps(_x, _tmp);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        __m128 _one = _mm_set1_ps(1.f);
        __m128 _zero = _mm_setzero_ps();
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            __m128 mask = _mm_cmpgt_ps(_p, _zero);
            __m128 _abs_p = _mm_and_ps(_p, *(__m128*)_ps_inv_sign_mask);
            __m128 _tmp = log_ps(_mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _abs_p))));
            __m128 _x = _mm_and_ps(_p, mask);
            _p = _mm_add_ps(_x, _tmp);
            _mm_store_ps(ptr, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            if (*ptr > 0)
                *ptr = *ptr + logf(1.f + expf(-(*ptr)));
            else
                *ptr = logf(1.f + expf(*ptr));
            ptr++;
        }
    }
    return 0;
}

} // namespace ncnn
