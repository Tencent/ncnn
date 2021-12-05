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

#include "x86_activation.h"

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
    int size = w * h * d;
#if __SSE2__
    int elempack = bottom_top_blob.elempack;

#if __AVX__
    if (elempack == 8)
    {
        if (slope == 0.f)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                __m256 _zero = _mm256_set1_ps(0.f);
                for (int i = 0; i < size; i++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(ptr, _mm256_max_ps(_zero, _p));
                    ptr += 8;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                for (int i = 0; i < size; i++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(ptr, lrelu_avx(_p, slope));
                    ptr += 8;
                }
            }
        }

        return 0;
    }
#endif // __AVX__

    if (elempack == 4)
    {
        if (slope == 0.f)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                __m128 _zero = _mm_set1_ps(0.f);
                for (int i = 0; i < size; i++)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    _mm_storeu_ps(ptr, _mm_max_ps(_zero, _p));
                    ptr += 4;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                for (int i = 0; i < size; i++)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    _mm_storeu_ps(ptr, lrelu_sse(_p, slope));
                    ptr += 4;
                }
            }
        }

        return 0;
    }
#endif // __SSE2__

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int remain = size;
            for (; remain > 0; remain--)
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
            int remain = size;
            for (; remain > 0; remain--)
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
