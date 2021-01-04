// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "dropout_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

namespace ncnn {

Dropout_x86::Dropout_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Dropout_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (scale == 1.f)
    {
        return 0;
    }

#if __SSE2__
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

#if __AVX__
    if (elempack == 8)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        __m256 _scale = _mm256_set1_ps(scale);

        if (dims == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float* ptr = (float*)bottom_top_blob + i * 8;
                __m256 _p = _mm256_loadu_ps(ptr);
                _p = _mm256_mul_ps(_p, _scale);
                _mm256_storeu_ps(ptr, _p);
            }
        }

        if (dims == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);

                for (int j = 0; j < w; j++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _p = _mm256_mul_ps(_p, _scale);
                    _mm256_storeu_ps(ptr, _p);
                    ptr += 8;
                }
            }
        }

        if (dims == 3)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _p = _mm256_mul_ps(_p, _scale);
                    _mm256_storeu_ps(ptr, _p);
                    ptr += 8;
                }
            }
        }

        return 0;
    }
#endif // __AVX__

    if (elempack == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        __m128 _scale = _mm_set1_ps(scale);

        if (dims == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float* ptr = (float*)bottom_top_blob + i * 4;
                __m128 _p = _mm_loadu_ps(ptr);
                _p = _mm_mul_ps(_p, _scale);
                _mm_storeu_ps(ptr, _p);
            }
        }

        if (dims == 2)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);

                for (int j = 0; j < w; j++)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    _p = _mm_mul_ps(_p, _scale);
                    _mm_storeu_ps(ptr, _p);
                    ptr += 4;
                }
            }
        }

        if (dims == 3)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    _p = _mm_mul_ps(_p, _scale);
                    _mm_storeu_ps(ptr, _p);
                    ptr += 4;
                }
            }
        }

        return 0;
    }
#endif // __SSE2__

    return Dropout::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
