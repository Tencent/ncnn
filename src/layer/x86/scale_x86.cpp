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

    int dims = bottom_top_blob.dims;
#if __SSE2__
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        const float* scale = scale_blob;
        if (bias_term)
        {
            const float* bias = bias_data;
            if (elempack == 1)
            {
                float* ptr = (float*)bottom_top_blob;
#if __AVX__
                int nn = w >> 3;
                int remain = w & 7;
#pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < nn; i++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _s = _mm256_loadu_ps(scale);
                    __m256 _bias = _mm256_loadu_ps(bias);
                    _p = _mm256_comp_fmadd_ps(_p, _s, _bias);
                    _mm256_storeu_ps(ptr, _p);

                    ptr += 8;
                    scale += 8;
                    bias += 8;
                }
#else
                int nn = w >> 2;
                int remain = w & 3;
#pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < nn; i++)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _s = _mm_loadu_ps(scale);
                    __m128 _bias = _mm_loadu_ps(bias);
                    _p = _mm_comp_fmadd_ps(_p, _s, _bias);
                    _mm_storeu_ps(ptr, _p);

                    ptr += 4;
                    scale += 4;
                    bias += 4;
                }
#endif // __AVX__
#pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < remain; i++)
                {
                    *ptr = *ptr * *scale + *bias;
                    ptr++, scale++, bias++;
                }
            }
#if __AVX__
            if (elempack == 8)
            {
#pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 8;

                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _s = _mm256_loadu_ps(scale + i * 8);
                    __m256 _bias = _mm256_loadu_ps(bias + i * 8);
                    _p = _mm256_comp_fmadd_ps(_p, _s, _bias);
                    _mm256_storeu_ps(ptr, _p);
                }
            }
#endif // __AVX__
            if (elempack == 4)
            {
#pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _s = _mm_loadu_ps(scale + i * 4);
                    __m128 _bias = _mm_loadu_ps(bias + i * 4);
                    _p = _mm_add_ps(_mm_mul_ps(_p, _s), _bias);
                    _mm_storeu_ps(ptr, _p);
                }
            }
        }
        else
        {
            if (elempack == 1)
            {
                float* ptr = (float*)bottom_top_blob;
#if __AVX__
                int nn = w >> 3;
                int remain = w & 7;
#pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < nn; i++)
                {
                    __m256 _p = _mm256_loadu_ps(ptr);
                    __m256 _s = _mm256_loadu_ps(scale);
                    _p = _mm256_mul_ps(_p, _s);
                    _mm256_storeu_ps(ptr, _p);

                    ptr += 8;
                    scale += 8;
                }
#else
                int nn = w >> 2;
                int remain = w & 3;
#pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < nn; i++)
                {
                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _s = _mm_loadu_ps(scale);
                    _p = _mm_mul_ps(_p, _s);
                    _mm_storeu_ps(ptr, _p);

                    ptr += 4;
                    scale += 4;
                }
#endif // __AVX__
#pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < remain; i++)
                {
                    *ptr *= *scale;
                    ptr++, scale++;
                }
            }
#if __AVX__
            if (elempack == 8)
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
#endif // __AVX__
            if (elempack == 4)
            {
#pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    __m128 _p = _mm_loadu_ps(ptr);
                    __m128 _s = _mm_loadu_ps(scale + i * 4);
                    _p = _mm_mul_ps(_p, _s);
                    _mm_storeu_ps(ptr, _p);
                }
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
#if __AVX__
                if (elempack == 8)
                {
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
#endif // __AVX__
                if (elempack == 4)
                {
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
                else if (elempack == 1)
                {
                    float s = scale_blob[i];
#if __AVX__
                    int nn = w >> 3;
                    int remain = w & 7;

                    float bias = bias_data[i];
                    __m256 _s = _mm256_set1_ps(s);
                    __m256 _bias = _mm256_set1_ps(bias);

                    while (nn--)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_comp_fmadd_ps(_p, _s, _bias);
                        _mm256_storeu_ps(ptr, _p);

                        ptr += 8;
                    }
#else
                    int nn = w >> 2;
                    int remain = w & 3;

                    float bias = bias_data[i];
                    __m128 _s = _mm_set1_ps(s);
                    __m128 _bias = _mm_set1_ps(bias);

                    while (nn--)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        _p = _mm_comp_fmadd_ps(_p, _s, _bias);
                        _mm_storeu_ps(ptr, _p);

                        ptr += 4;
                    }
#endif // __AVX__

                    while (remain--)
                    {
                        *ptr = *ptr * s + bias;

                        ptr++;
                    }
                }
            }
        }
        else
        {
#pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
#if __AVX__
                if (elempack == 8)
                {
                    __m256 _s = _mm256_loadu_ps((const float*)scale_blob + i * 8);

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_mul_ps(_p, _s);
                        _mm256_storeu_ps(ptr, _p);

                        ptr += 8;
                    }
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _s = _mm_loadu_ps((const float*)scale_blob + i * 4);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        _p = _mm_mul_ps(_p, _s);
                        _mm_storeu_ps(ptr, _p);

                        ptr += 4;
                    }
                }
                else if (elempack == 1)
                {
                    float s = scale_blob[i];
#if __AVX__
                    int nn = w >> 3;
                    int remain = w & 7;

                    __m256 _s = _mm256_set1_ps(s);

                    while (nn--)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_mul_ps(_p, _s);
                        _mm256_storeu_ps(ptr, _p);

                        ptr += 8;
                    }
#else
                    int nn = w >> 2;
                    int remain = w & 3;

                    __m128 _s = _mm_set1_ps(s);
                    while (nn--)
                    {
                        __m128 _p = _mm_load_ps(ptr);
                        _p = _mm_mul_ps(_p, _s);
                        _mm_storeu_ps(ptr, _p);

                        ptr += 4;
                    }
#endif // __AVX__
                    while (remain--)
                    {
                        *ptr = *ptr * s;

                        ptr++;
                    }
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
#if __AVX__
                if (elempack == 8)
                {
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
#endif // __AVX__
                if (elempack == 4)
                {
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
                else if (elempack == 1)
                {
                    float s = scale_blob[q];
                    float bias = bias_data[q];
#if __AVX__
                    int nn = size >> 3;
                    int remain = size & 7;

                    __m256 _s = _mm256_set1_ps(s);
                    __m256 _bias = _mm256_set1_ps(bias);
                    while (nn--)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_comp_fmadd_ps(_p, _s, _bias);
                        _mm256_storeu_ps(ptr, _p);

                        ptr += 8;
                    }
#else
                    int nn = size >> 2;
                    int remain = size & 3;

                    __m128 _s = _mm_set1_ps(s);
                    __m128 _bias = _mm_set1_ps(bias);
                    while (nn--)
                    {
                        __m128 _p = _mm_load_ps(ptr);
                        _p = _mm_comp_fmadd_ps(_p, _s, _bias);
                        _mm_storeu_ps(ptr, _p);

                        ptr += 4;
                    }
#endif // __AVX__

                    while (remain--)
                    {
                        *ptr = *ptr * s + bias;

                        ptr++;
                    }
                }
            }
        }
        else
        {
#pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
#if __AVX__
                if (elempack == 8)
                {
                    __m256 _s = _mm256_loadu_ps((const float*)scale_blob + q * 8);

                    for (int i = 0; i < size; i++)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_mul_ps(_p, _s);
                        _mm256_storeu_ps(ptr, _p);

                        ptr += 8;
                    }
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    __m128 _s = _mm_loadu_ps((const float*)scale_blob + q * 4);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p = _mm_loadu_ps(ptr);
                        _p = _mm_mul_ps(_p, _s);
                        _mm_storeu_ps(ptr, _p);

                        ptr += 4;
                    }
                }
                else if (elempack == 1)
                {
                    float s = scale_blob[q];
#if __AVX__
                    int nn = size >> 3;
                    int remain = size & 7;

                    __m256 _s = _mm256_set1_ps(s);
                    while (nn--)
                    {
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _p = _mm256_mul_ps(_p, _s);
                        _mm256_storeu_ps(ptr, _p);

                        ptr += 8;
                    }
#else
                    int nn = size >> 2;
                    int remain = size & 3;

                    __m128 _s = _mm_set1_ps(s);
                    while (nn--)
                    {
                        __m128 _p = _mm_load_ps(ptr);
                        _p = _mm_mul_ps(_p, _s);
                        _mm_storeu_ps(ptr, _p);

                        ptr += 4;
                    }
#endif // __AVX__

                    while (remain--)
                    {
                        *ptr *= s;

                        ptr++;
                    }
                }
            }
        }
    }

    return 0;

#endif // __SSE2__

    return Scale::forward_inplace(bottom_top_blobs, opt);
}

} // namespace ncnn
