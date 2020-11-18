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

#include "hardswish_x86.h"

#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__

namespace ncnn {

HardSwish_x86::HardSwish_x86()
{
    support_packing = true;
}

int HardSwish_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

#if __AVX__
    if (elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            __m256 _zero = _mm256_set1_ps(0.f);
            __m256 _one = _mm256_set1_ps(1.f);
            for (int i = 0; i < size; i++)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _ans = _mm256_set1_ps(beta);
                _ans = _mm256_fmadd_ps(_p, _mm256_set1_ps(alpha), _ans);
                _ans = _mm256_max_ps(_ans, _zero);
                _ans = _mm256_min_ps(_ans, _one);
                _ans = _mm256_mul_ps(_ans, _p);
                _mm256_storeu_ps(ptr, _ans);

                ptr += 8;
            }
        }

        return 0;
    }
#endif // __AVX__

    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            __m128 _zero = _mm_set1_ps(0.f);
            __m128 _one = _mm_set1_ps(1.f);
            for (int i = 0; i < size; i++)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _ans = _mm_set1_ps(beta);
                _ans = _mm_add_ps(_mm_mul_ps(_p, _mm_set1_ps(alpha)), _ans);
                _ans = _mm_max_ps(_ans, _zero);
                _ans = _mm_min_ps(_ans, _one);
                _ans = _mm_mul_ps(_ans, _p);
                _mm_storeu_ps(ptr, _ans);

                ptr += 4;
            }
        }

        return 0;
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __AVX__
        __m256 _zero = _mm256_set1_ps(0.f);
        __m256 _one = _mm256_set1_ps(1.f);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _ans = _mm256_set1_ps(beta);
            _ans = _mm256_fmadd_ps(_p, _mm256_set1_ps(alpha), _ans);
            _ans = _mm256_max_ps(_ans, _zero);
            _ans = _mm256_min_ps(_ans, _one);
            _ans = _mm256_mul_ps(_ans, _p);
            _mm256_storeu_ps(ptr, _ans);

            ptr += 8;
        }
#endif
        for (; i < size; i++)
        {
            if (*ptr < lower)
                *ptr = 0.f;
            else if (*ptr > upper)
                ;
            else
                *ptr = *ptr * (*ptr * alpha + beta);
            ++ptr;
        }
    }

    return 0;
}

} // namespace ncnn
