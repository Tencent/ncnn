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

#include "scale_x86.h"

namespace ncnn {

Scale_x86::Scale_x86()
{
#if __AVX__
    support_packing = true;
#endif // __AVX__
}

int Scale_x86::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    int dims = bottom_top_blob.dims;

#if __AVX__
    int elempack = bottom_top_blob.elempack;

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            const float* scale = scale_blob;
            if (bias_term)
            {
                const float* bias = bias_data;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 8;

                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _s = _mm256_loadu_ps(scale + i * 8);
                    __m256 _bias = _mm256_loadu_ps(bias + i * 8);
                    _p = _mm256_fmadd_ps(_p, _s, _bias);
                    _mm256_storeu_ps(ptr, _p);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 8;

                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _s = _mm256_loadu_ps(scale + i * 8);
                    _p = _mm256_mul_ps(_p, _s);
                    _mm256_storeu_ps(ptr, _p);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    __m256 _s = _mm256_loadu_ps((const float*)scale_blob + i * 8);
                    __m256 _bias = _mm256_loadu_ps((const float*)bias_data + i * 8);

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_fmadd_ps(_p, _s, _bias);
                        _mm256_storeu_ps(ptr, _p);

                        ptr += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    __m256 _s = _mm256_loadu_ps((const float*)scale_blob + i * 8);

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_mul_ps(_p, _s);
                        _mm256_storeu_ps(ptr, _p);

                        ptr += 8;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            int channels = bottom_top_blob.c;
            int size = w * h;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    __m256 _s = _mm256_loadu_ps((const float*)scale_blob + q * 8);
                    __m256 _bias = _mm256_loadu_ps((const float*)bias_data + q * 8);

                    for (int i = 0; i < size; i++)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_fmadd_ps(_p, _s, _bias);
                        _mm256_storeu_ps(ptr, _p);

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
                    __m256 _s = _mm256_loadu_ps((const float*)scale_blob + q * 8);

                    for (int i = 0; i < size; i++)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_mul_ps(_p, _s);
                        _mm256_storeu_ps(ptr, _p);

                        ptr += 8;
                    }
                }
            }
        }

        return 0;
    }
#endif // __AVX__

    if (dims != 3)
        return Scale::forward_inplace(bottom_top_blobs, opt);

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (bias_term)
    {
        const float* scale_ptr = scale_blob;
        const float* bias_ptr = bias_data;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_ptr[q];
            float bias = bias_ptr[q];

#if __AVX__
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __AVX__

#if __AVX__
            __m256 _s = _mm256_set1_ps(s);
            __m256 _bias = _mm256_set1_ps(bias);
            for (; nn > 0; nn--)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                _p = _mm256_fmadd_ps(_p, _s, _bias);
                _mm256_storeu_ps(ptr, _p);

                ptr += 8;
            }
#endif // __AVX__

            for (; remain > 0; remain--)
            {
                *ptr = *ptr * s + bias;

                ptr++;
            }
        }
    }
    else
    {
        const float* scale_ptr = scale_blob;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale_ptr[q];

#if __AVX__
            int nn = size >> 3;
            int remain = size & 7;
#else
            int remain = size;
#endif // __AVX__

#if __AVX__
            __m256 _s = _mm256_set1_ps(s);
            for (; nn > 0; nn--)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                _p = _mm256_mul_ps(_p, _s);
                _mm256_storeu_ps(ptr, _p);

                ptr += 8;
            }
#endif // __AVX__

            for (; remain > 0; remain--)
            {
                *ptr *= s;

                ptr++;
            }
        }
    }

    return 0;
}

} // namespace ncnn
