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
#include "avx_activation.h"
#endif // __AVX__

#include "prelu_x86.h"

namespace ncnn {

PReLU_x86::PReLU_x86()
{
#if __AVX__
    support_packing = true;
#endif // __AVX__
}

int PReLU_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

#if __AVX__
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            if (num_slope > 1)
            {
                const float* slope = slope_data;

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 8;
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _slope = _mm256_loadu_ps(slope + i * 8);
                    _mm256_storeu_ps(ptr, prelu_avx(_p, _slope));
                }
            }
            else
            {
                __m256 _slope = _mm256_set1_ps(slope_data[0]);

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 8;
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(ptr, prelu_avx(_p, _slope));
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                __m256 _slope = num_slope > 1 ? _mm256_loadu_ps((const float*)slope_data + i * 8) : _mm256_set1_ps(slope_data[0]);

                for (int j = 0; j < w; j++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(ptr, prelu_avx(_p, _slope));
                    ptr += 8;
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                __m256 _slope = num_slope > 1 ? _mm256_loadu_ps((const float*)slope_data + q * 8) : _mm256_set1_ps(slope_data[0]);

                for (int i = 0; i < size; i++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    _mm256_storeu_ps(ptr, prelu_avx(_p, _slope));
                    ptr += 8;
                }
            }
        }

        return 0;
    }
#endif // __AVX__

    if (dims != 3)
        return PReLU::forward_inplace(bottom_top_blob, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    const float* slope_data_ptr = slope_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);
        float slope = num_slope > 1 ? slope_data_ptr[q] : slope_data_ptr[0];

#if __AVX__
        int nn = size >> 3;
        int remain = size - (nn << 3);
#else
        int remain = size;
#endif // __AVX__

#if __AVX__
        for (; nn > 0; nn--)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _mm256_storeu_ps(ptr, prelu_avx(_p, _mm256_set1_ps(slope)));
            ptr += 8;
        }
#endif // __AVX__
        for (; remain > 0; remain--)
        {
            if (*ptr < 0)
                *ptr *= slope;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
