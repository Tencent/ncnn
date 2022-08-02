// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "dequantize_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

Dequantize_x86::Dequantize_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int Dequantize_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int elempack = bottom_blob.elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        Mat tmp;
        convert_packing(bottom_blob, tmp, 8, opt);

        Mat tmpout;
        forward(tmp, tmpout, opt);

        convert_packing(tmpout, top_blob, 16, opt);

        return 0;
    }
#endif // __AVX512F__

    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)32u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                __m256 _scale = _mm256_set1_ps(scale_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        float* ptr = (float*)top_blob + i * 8;

                        __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
                        _v = _mm256_mul_ps(_v, _scale);
                        _mm256_storeu_ps(ptr, _v);
                    }
                }
                else if (bias_data_size == 1)
                {
                    __m256 _bias = _mm256_set1_ps(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        float* ptr = (float*)top_blob + i * 8;

                        __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
                        _v = _mm256_comp_fmadd_ps(_v, _scale, _bias);
                        _mm256_storeu_ps(ptr, _v);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        float* ptr = (float*)top_blob + i * 8;

                        __m256 _bias = _mm256_loadu_ps((const float*)bias_data + i * 8);
                        __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
                        _v = _mm256_comp_fmadd_ps(_v, _scale, _bias);
                        _mm256_storeu_ps(ptr, _v);
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        float* ptr = (float*)top_blob + i * 8;

                        __m256 _scale = _mm256_loadu_ps((const float*)scale_data + i * 8);
                        __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
                        _v = _mm256_mul_ps(_v, _scale);
                        _mm256_storeu_ps(ptr, _v);
                    }
                }
                else if (bias_data_size == 1)
                {
                    __m256 _bias = _mm256_set1_ps(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        float* ptr = (float*)top_blob + i * 8;

                        __m256 _scale = _mm256_loadu_ps((const float*)scale_data + i * 8);
                        __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
                        _v = _mm256_comp_fmadd_ps(_v, _scale, _bias);
                        _mm256_storeu_ps(ptr, _v);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 8;
                        float* ptr = (float*)top_blob + i * 8;

                        __m256 _scale = _mm256_loadu_ps((const float*)scale_data + i * 8);
                        __m256 _bias = _mm256_loadu_ps((const float*)bias_data + i * 8);
                        __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
                        _v = _mm256_comp_fmadd_ps(_v, _scale, _bias);
                        _mm256_storeu_ps(ptr, _v);
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)32u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr = top_blob.row(i);

                    __m256 _scale = scale_data_size == 1 ? _mm256_set1_ps(scale_data[0]) : _mm256_loadu_ps((const float*)scale_data + i * 8);

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
                        _v = _mm256_mul_ps(_v, _scale);
                        _mm256_storeu_ps(ptr, _v);

                        intptr += 8;
                        ptr += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr = top_blob.row(i);

                    __m256 _scale = scale_data_size == 1 ? _mm256_set1_ps(scale_data[0]) : _mm256_loadu_ps((const float*)scale_data + i * 8);
                    __m256 _bias = bias_data_size == 1 ? _mm256_set1_ps(bias_data[0]) : _mm256_loadu_ps((const float*)bias_data + i * 8);

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
                        _v = _mm256_comp_fmadd_ps(_v, _scale, _bias);
                        _mm256_storeu_ps(ptr, _v);

                        intptr += 8;
                        ptr += 8;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;

            top_blob.create(w, h, channels, (size_t)32u, 8, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);

                    __m256 _scale = scale_data_size == 1 ? _mm256_set1_ps(scale_data[0]) : _mm256_loadu_ps((const float*)scale_data + q * 8);

                    for (int i = 0; i < size; i++)
                    {
                        __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
                        _v = _mm256_mul_ps(_v, _scale);
                        _mm256_storeu_ps(ptr, _v);

                        intptr += 8;
                        ptr += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);

                    __m256 _scale = scale_data_size == 1 ? _mm256_set1_ps(scale_data[0]) : _mm256_loadu_ps((const float*)scale_data + q * 8);
                    __m256 _bias = bias_data_size == 1 ? _mm256_set1_ps(bias_data[0]) : _mm256_loadu_ps((const float*)bias_data + q * 8);

                    for (int i = 0; i < size; i++)
                    {
                        __m256 _v = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)intptr));
                        _v = _mm256_comp_fmadd_ps(_v, _scale, _bias);
                        _mm256_storeu_ps(ptr, _v);

                        intptr += 8;
                        ptr += 8;
                    }
                }
            }
        }

        return 0;
    }
#else // __AVX__
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;
            int outw = w * 2;

            top_blob.create(outw, (size_t)16u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                __m128 _scale = _mm_set1_ps(scale_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_mul_ps(_v, _scale);
                        _mm_storeu_ps(ptr, _v);
                    }
                }
                else if (bias_data_size == 1)
                {
                    __m128 _bias = _mm_set1_ps(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                        _mm_storeu_ps(ptr, _v);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        __m128 _bias = _mm_loadu_ps((const float*)bias_data + i * 4);
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                        _mm_storeu_ps(ptr, _v);
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        __m128 _scale = _mm_loadu_ps((const float*)scale_data + i * 4);
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_mul_ps(_v, _scale);
                        _mm_storeu_ps(ptr, _v);
                    }
                }
                else if (bias_data_size == 1)
                {
                    __m128 _bias = _mm_set1_ps(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        __m128 _scale = _mm_loadu_ps((const float*)scale_data + i * 4);
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                        _mm_storeu_ps(ptr, _v);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < outw; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        __m128 _scale = _mm_loadu_ps((const float*)scale_data + i * 4);
                        __m128 _bias = _mm_loadu_ps((const float*)bias_data + i * 4);
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                        _mm_storeu_ps(ptr, _v);
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int outh = h * 2;

            top_blob.create(w, outh, (size_t)16u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr0 = top_blob.row(i * 2);
                    float* ptr1 = top_blob.row(i * 2 + 1);

                    __m128 _scale0 = scale_data_size == 1 ? _mm_set1_ps(scale_data[0]) : _mm_loadu_ps((const float*)scale_data + i * 8);
                    __m128 _scale1 = scale_data_size == 1 ? _mm_set1_ps(scale_data[0]) : _mm_loadu_ps((const float*)scale_data + i * 8 + 4);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(intptr + 4)));
                        _v0 = _mm_mul_ps(_v0, _scale0);
                        _v1 = _mm_mul_ps(_v1, _scale1);
                        _mm_storeu_ps(ptr0, _v0);
                        _mm_storeu_ps(ptr1, _v1);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr0 = top_blob.row(i * 2);
                    float* ptr1 = top_blob.row(i * 2 + 1);

                    __m128 _scale0 = scale_data_size == 1 ? _mm_set1_ps(scale_data[0]) : _mm_loadu_ps((const float*)scale_data + i * 8);
                    __m128 _scale1 = scale_data_size == 1 ? _mm_set1_ps(scale_data[0]) : _mm_loadu_ps((const float*)scale_data + i * 8 + 4);
                    __m128 _bias0 = bias_data_size == 1 ? _mm_set1_ps(bias_data[0]) : _mm_loadu_ps((const float*)bias_data + i * 8);
                    __m128 _bias1 = bias_data_size == 1 ? _mm_set1_ps(bias_data[0]) : _mm_loadu_ps((const float*)bias_data + i * 8 + 4);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(intptr + 4)));
                        _v0 = _mm_add_ps(_bias0, _mm_mul_ps(_v0, _scale0));
                        _v1 = _mm_add_ps(_bias1, _mm_mul_ps(_v1, _scale1));
                        _mm_storeu_ps(ptr0, _v0);
                        _mm_storeu_ps(ptr1, _v1);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;
            int outc = channels * 2;

            top_blob.create(w, h, outc, (size_t)16u, 4, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr0 = top_blob.channel(q * 2);
                    float* ptr1 = top_blob.channel(q * 2 + 1);

                    __m128 _scale0 = scale_data_size == 1 ? _mm_set1_ps(scale_data[0]) : _mm_loadu_ps((const float*)scale_data + q * 8);
                    __m128 _scale1 = scale_data_size == 1 ? _mm_set1_ps(scale_data[0]) : _mm_loadu_ps((const float*)scale_data + q * 8 + 4);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(intptr + 4)));
                        _v0 = _mm_mul_ps(_v0, _scale0);
                        _v1 = _mm_mul_ps(_v1, _scale1);
                        _mm_storeu_ps(ptr0, _v0);
                        _mm_storeu_ps(ptr1, _v1);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr0 = top_blob.channel(q * 2);
                    float* ptr1 = top_blob.channel(q * 2 + 1);

                    __m128 _scale0 = scale_data_size == 1 ? _mm_set1_ps(scale_data[0]) : _mm_loadu_ps((const float*)scale_data + q * 8);
                    __m128 _scale1 = scale_data_size == 1 ? _mm_set1_ps(scale_data[0]) : _mm_loadu_ps((const float*)scale_data + q * 8 + 4);
                    __m128 _bias0 = bias_data_size == 1 ? _mm_set1_ps(bias_data[0]) : _mm_loadu_ps((const float*)bias_data + q * 8);
                    __m128 _bias1 = bias_data_size == 1 ? _mm_set1_ps(bias_data[0]) : _mm_loadu_ps((const float*)bias_data + q * 8 + 4);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _v0 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        __m128 _v1 = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)(intptr + 4)));
                        _v0 = _mm_add_ps(_bias0, _mm_mul_ps(_v0, _scale0));
                        _v1 = _mm_add_ps(_bias1, _mm_mul_ps(_v1, _scale1));
                        _mm_storeu_ps(ptr0, _v0);
                        _mm_storeu_ps(ptr1, _v1);

                        intptr += 8;
                        ptr0 += 4;
                        ptr1 += 4;
                    }
                }
            }
        }

        return 0;
    }
#endif // __AVX__

    if (elempack == 4)
    {
        if (dims == 1)
        {
            int w = bottom_blob.w;

            top_blob.create(w, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (scale_data_size == 1)
            {
                __m128 _scale = _mm_set1_ps(scale_data[0]);

                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_mul_ps(_v, _scale);
                        _mm_storeu_ps(ptr, _v);
                    }
                }
                else if (bias_data_size == 1)
                {
                    __m128 _bias = _mm_set1_ps(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                        _mm_storeu_ps(ptr, _v);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        __m128 _bias = _mm_loadu_ps((const float*)bias_data + i * 4);
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                        _mm_storeu_ps(ptr, _v);
                    }
                }
            }
            else
            {
                if (bias_data_size == 0)
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        __m128 _scale = _mm_loadu_ps((const float*)scale_data + i * 4);
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_mul_ps(_v, _scale);
                        _mm_storeu_ps(ptr, _v);
                    }
                }
                else if (bias_data_size == 1)
                {
                    __m128 _bias = _mm_set1_ps(bias_data[0]);

                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        __m128 _scale = _mm_loadu_ps((const float*)scale_data + i * 4);
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                        _mm_storeu_ps(ptr, _v);
                    }
                }
                else
                {
                    #pragma omp parallel for num_threads(opt.num_threads)
                    for (int i = 0; i < w; i++)
                    {
                        const int* intptr = (const int*)bottom_blob + i * 4;
                        float* ptr = (float*)top_blob + i * 4;

                        __m128 _scale = _mm_loadu_ps((const float*)scale_data + i * 4);
                        __m128 _bias = _mm_loadu_ps((const float*)bias_data + i * 4);
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                        _mm_storeu_ps(ptr, _v);
                    }
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;

            top_blob.create(w, h, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr = top_blob.row(i);

                    __m128 _scale = scale_data_size == 1 ? _mm_set1_ps(scale_data[0]) : _mm_loadu_ps((const float*)scale_data + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_mul_ps(_v, _scale);
                        _mm_storeu_ps(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    const int* intptr = bottom_blob.row<const int>(i);
                    float* ptr = top_blob.row(i);

                    __m128 _scale = scale_data_size == 1 ? _mm_set1_ps(scale_data[0]) : _mm_loadu_ps((const float*)scale_data + i * 4);
                    __m128 _bias = bias_data_size == 1 ? _mm_set1_ps(bias_data[0]) : _mm_loadu_ps((const float*)bias_data + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                        _mm_storeu_ps(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
        }

        if (dims == 3)
        {
            int w = bottom_blob.w;
            int h = bottom_blob.h;
            int channels = bottom_blob.c;
            int size = w * h;

            top_blob.create(w, h, channels, (size_t)16u, elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);

                    __m128 _scale = scale_data_size == 1 ? _mm_set1_ps(scale_data[0]) : _mm_loadu_ps((const float*)scale_data + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_mul_ps(_v, _scale);
                        _mm_storeu_ps(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    const int* intptr = bottom_blob.channel(q);
                    float* ptr = top_blob.channel(q);

                    __m128 _scale = scale_data_size == 1 ? _mm_set1_ps(scale_data[0]) : _mm_loadu_ps((const float*)scale_data + q * 4);
                    __m128 _bias = bias_data_size == 1 ? _mm_set1_ps(bias_data[0]) : _mm_loadu_ps((const float*)bias_data + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                        _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                        _mm_storeu_ps(ptr, _v);

                        intptr += 4;
                        ptr += 4;
                    }
                }
            }
        }

        return 0;
    }
#endif // __SSE2__

    if (dims == 1)
    {
        int w = bottom_blob.w;

        top_blob.create(w, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        const int* intptr = bottom_blob;
        float* ptr = top_blob;

        if (scale_data_size == 1)
        {
            const float scale = scale_data[0];

            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale;
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale + bias_data[i];
                }
            }
        }
        else
        {
            if (bias_data_size == 0)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i];
                }
            }
            else if (bias_data_size == 1)
            {
                const float bias = bias_data[0];

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i] + bias;
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    ptr[i] = intptr[i] * scale_data[i] + bias_data[i];
                }
            }
        }
    }

    if (dims == 2)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;

        top_blob.create(w, h, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];

                int j = 0;
#if __SSE2__
                __m128 _scale = _mm_set1_ps(scale);
                for (; j + 3 < w; j += 4)
                {
                    __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                    _v = _mm_mul_ps(_v, _scale);
                    _mm_storeu_ps(ptr, _v);

                    intptr += 4;
                    ptr += 4;
                }
#endif // __SSE2__
                for (; j < w; j++)
                {
                    *ptr++ = *intptr++ * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                const int* intptr = bottom_blob.row<const int>(i);
                float* ptr = top_blob.row(i);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[i];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[i];

                int j = 0;
#if __SSE2__
                __m128 _scale = _mm_set1_ps(scale);
                __m128 _bias = _mm_set1_ps(bias);
                for (; j + 3 < w; j += 4)
                {
                    __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                    _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                    _mm_storeu_ps(ptr, _v);

                    intptr += 4;
                    ptr += 4;
                }
#endif // __SSE2__
                for (; j < w; j++)
                {
                    *ptr++ = *intptr++ * scale + bias;
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;
        int size = w * h;

        top_blob.create(w, h, channels, (size_t)4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (bias_data_size == 0)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];

                int i = 0;
#if __SSE2__
                __m128 _scale = _mm_set1_ps(scale);
                for (; i + 3 < size; i += 4)
                {
                    __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                    _v = _mm_mul_ps(_v, _scale);
                    _mm_storeu_ps(ptr, _v);

                    intptr += 4;
                    ptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *ptr++ = *intptr++ * scale;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                const int* intptr = bottom_blob.channel(q);
                float* ptr = top_blob.channel(q);

                const float scale = scale_data_size == 1 ? scale_data[0] : scale_data[q];
                const float bias = bias_data_size == 1 ? bias_data[0] : bias_data[q];

                int i = 0;
#if __SSE2__
                __m128 _scale = _mm_set1_ps(scale);
                __m128 _bias = _mm_set1_ps(bias);
                for (; i + 3 < size; i += 4)
                {
                    __m128 _v = _mm_cvtepi32_ps(_mm_loadu_si128((const __m128i*)intptr));
                    _v = _mm_add_ps(_bias, _mm_mul_ps(_v, _scale));
                    _mm_storeu_ps(ptr, _v);

                    intptr += 4;
                    ptr += 4;
                }
#endif // __SSE2__
                for (; i < size; i++)
                {
                    *ptr++ = *intptr++ * scale + bias;
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
