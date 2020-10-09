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
#if __AVX__
#include <immintrin.h>
#endif // __AVX__

#include "batchnorm_x86.h"

namespace ncnn {

BatchNorm_x86::BatchNorm_x86()
{
#if __AVX__
    support_packing = true;
#endif // __AVX__
}

int BatchNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;

#if __AVX__
    int elempack = bottom_top_blob.elempack;

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < w; i++)
            {
                float* ptr = (float*)bottom_top_blob + i * 8;

                __m256 _a = _mm256_loadu_ps((const float*)a_data + i * 8);
                __m256 _b = _mm256_loadu_ps((const float*)b_data + i * 8);

                __m256 _p = _mm256_loadu_ps(ptr);
                _p = _mm256_fmadd_ps(_p, _b, _a);
                _mm256_storeu_ps(ptr, _p);
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                __m256 _a = _mm256_loadu_ps((const float*)a_data + i * 8);
                __m256 _b = _mm256_loadu_ps((const float*)b_data + i * 8);

                float* ptr = bottom_top_blob.row(i);

                for (int j = 0; j < w; j++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _p = _mm256_fmadd_ps(_p, _b, _a);
                    _mm256_storeu_ps(ptr, _p);

                    ptr += 8;
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int c = bottom_top_blob.c;
            int size = w * h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < c; q++)
            {
                __m256 _a = _mm256_loadu_ps((const float*)a_data + q * 8);
                __m256 _b = _mm256_loadu_ps((const float*)b_data + q * 8);

                float* ptr = bottom_top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _p = _mm256_fmadd_ps(_p, _b, _a);
                    _mm256_storeu_ps(ptr, _p);

                    ptr += 8;
                }
            }
        }

        return 0;
    }
#endif // __AVX__

    if (dims != 3)
        return BatchNorm::forward_inplace(bottom_top_blob, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    // int c = bottom_top_blob.c;
    int size = w * h;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float a = a_data[q];
        float b = b_data[q];

#if __AVX__
        int nn = size >> 3;
        int remain = size & 7;
#else
        int remain = size;
#endif // __AVX__

#if __AVX__
        __m256 _a = _mm256_set1_ps(a);
        __m256 _b = _mm256_set1_ps(b);
        for (; nn > 0; nn--)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = _mm256_fmadd_ps(_p, _b, _a);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        for (; remain > 0; remain--)
        {
            *ptr = b * *ptr + a;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
