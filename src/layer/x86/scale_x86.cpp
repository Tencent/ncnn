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

#include "scale_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"
namespace ncnn {

Scale_x86::Scale_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Scale_x86::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const int dims = bottom_top_blob.dims;

    const int elempack = bottom_top_blob.elempack;

    const float* scale = scale_blob;
    const float* bias = bias_data;

    if (dims == 1)
    {
        float* ptr = (float*)bottom_top_blob;
        int size = w * elempack;

        int remain = size;
#if __SSE2__
#if __AVX__
        int nn = size >> 3;
        remain = size & 7;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < nn; i++)
        {
            __m256 _p = _mm256_loadu_ps(ptr + i * 8);
            __m256 _s = _mm256_loadu_ps(scale + i * 8);
            if (bias_term)
            {
                __m256 _bias = _mm256_loadu_ps(bias + i * 8);
                _p = _mm256_comp_fmadd_ps(_p, _s, _bias);
            }
            else
            {
                _p = _mm256_mul_ps(_p, _s);
            }
            _mm256_storeu_ps(ptr + i * 8, _p);
        }
#else
        int nn = size >> 2;
        remain = size & 3;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < nn; i++)
        {
            __m128 _p = _mm_loadu_ps(ptr + i * 4);
            __m128 _s = _mm_loadu_ps(scale + i * 4);
            if (bias_term)
            {
                __m128 _bias = _mm_loadu_ps(bias + i * 4);
                _p = _mm_comp_fmadd_ps(_p, _s, _bias);
            }
            else
            {
                _p = _mm_mul_ps(_p, _s);
            }
            _mm_storeu_ps(ptr + i * 4, _p);
        }
#endif // __AVX__
#endif // __SSE2__
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = size - remain; i < size; i++)
        {
            if (bias_term)
            {
                ptr[i] = ptr[i] * scale[i] + bias[i];
            }
            else
            {
                ptr[i] = ptr[i] * scale[i];
            }
        }

        return 0;
    }

#if __SSE2__
#if __AVX__
    if (elempack == 8)
    {
        if (dims == 2)
        {
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
                        _p = _mm256_comp_fmadd_ps(_p, _s, _bias);
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
                        _p = _mm256_comp_fmadd_ps(_p, _s, _bias);
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

    if (elempack == 4)
    {
        if (dims == 2)
        {
            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    __m128 _s = _mm_loadu_ps((const float*)scale_blob + i * 4);
                    __m128 _bias = _mm_loadu_ps((const float*)bias_data + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        _p = _mm_add_ps(_mm_mul_ps(_p, _s), _bias);
                        _mm_storeu_ps(ptr, _p);

                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    __m128 _s = _mm_loadu_ps((const float*)scale_blob + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        _p = _mm_mul_ps(_p, _s);
                        _mm_storeu_ps(ptr, _p);

                        ptr += 4;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int size = w * h;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    __m128 _s = _mm_loadu_ps((const float*)scale_blob + q * 4);
                    __m128 _bias = _mm_loadu_ps((const float*)bias_data + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        _p = _mm_add_ps(_mm_mul_ps(_p, _s), _bias);
                        _mm_storeu_ps(ptr, _p);

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
                    __m128 _s = _mm_loadu_ps((const float*)scale_blob + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        _p = _mm_mul_ps(_p, _s);
                        _mm_storeu_ps(ptr, _p);

                        ptr += 4;
                    }
                }
            }
        }
    }
#endif // __SSE2__

    if (elempack == 1)
    {
        if (dims == 2)
        {
            int size = w;
            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);

                    float s = scale_blob[i];
                    float bias = bias_data[i];

                    int j = 0;
#if __SSE2__
#if __AVX__
                    __m256 _s = _mm256_set1_ps(s);
                    __m256 _bias = _mm256_set1_ps(bias);

                    for (; j + 7 < size; j += 8)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_comp_fmadd_ps(_p, _s, _bias);
                        _mm256_storeu_ps(ptr, _p);

                        ptr += 8;
                    }
#else
                    __m128 _s = _mm_set1_ps(s);
                    __m128 _bias = _mm_set1_ps(bias);

                    for (; j + 3 < size; j += 4)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        _p = _mm_comp_fmadd_ps(_p, _s, _bias);
                        _mm_storeu_ps(ptr, _p);

                        ptr += 4;
                    }
#endif // __AVX__
#endif // __SSE2__

                    for (; j < size; j++)
                    {
                        *ptr = *ptr * s + bias;

                        ptr++;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);

                    float s = scale_blob[i];

                    int j = 0;
#if __SSE2__
#if __AVX__
                    __m256 _s = _mm256_set1_ps(s);

                    for (; j + 7 < size; j += 8)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_mul_ps(_p, _s);
                        _mm256_storeu_ps(ptr, _p);

                        ptr += 8;
                    }
#else
                    __m128 _s = _mm_set1_ps(s);

                    for (; j + 3 < size; j += 4)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        _p = _mm_mul_ps(_p, _s);
                        _mm_storeu_ps(ptr, _p);

                        ptr += 4;
                    }
#endif // __AVX__
#endif // __SSE2__

                    for (; j < size; j++)
                    {
                        *ptr *= s;

                        ptr++;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int size = w * h;
            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < channels; i++)
                {
                    float* ptr = bottom_top_blob.channel(i);

                    float s = scale_blob[i];

                    int j = 0;
#if __SSE2__
#if __AVX__
                    __m256 _s256 = _mm256_set1_ps(s);
                    __m256 _bias256 = _mm256_set1_ps(bias_data[i]);
                    for (; j + 7 < size; j += 8)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_comp_fmadd_ps(_p, _s256, _bias256);
                        _mm256_storeu_ps(ptr, _p);

                        ptr += 8;
                    }
#endif // __AVX__
                    __m128 _s128 = _mm_set1_ps(s);
                    __m128 _bias128;
                    if (bias_term)
                        _bias128 = _mm_set1_ps(bias_data[i]);
                    for (; j < size; j += 4)
                    {
                        __m128 _p = _mm_load_ps(ptr);
                        _p = _mm_comp_fmadd_ps(_p, _s128, _bias128);
                        _mm_storeu_ps(ptr, _p);

                        ptr += 4;
                    }
#endif // __SSE2__

                    for (; j < size; j++)
                    {
                        *ptr = *ptr * s + bias_data[i];
                        ptr++;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < channels; i++)
                {
                    float* ptr = bottom_top_blob.channel(i);

                    float s = scale_blob[i];

                    int j = 0;
#if __SSE2__
#if __AVX__
                    __m256 _s256 = _mm256_set1_ps(s);
                    for (; j + 7 < size; j += 8)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_mul_ps(_p, _s256);
                        _mm256_storeu_ps(ptr, _p);

                        ptr += 8;
                    }
#endif // __AVX__

                    __m128 _s128 = _mm_set1_ps(s);
                    for (; j < size; j += 4)
                    {
                        __m128 _p = _mm_load_ps(ptr);
                        _p = _mm_mul_ps(_p, _s128);
                        _mm_storeu_ps(ptr, _p);

                        ptr += 4;
                    }
#endif // __SSE2__

                    for (; j < size; j++)
                    {
                        *ptr *= s;
                        ptr++;
                    }
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
