// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "layernorm_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

LayerNorm_x86::LayerNorm_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int LayerNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            float* ptr = (float*)bottom_top_blob;

            __m512 _fLoad;

            // mean
            float sum = 0.f;
            float sqsum = 0.f;

            __m512 _fsum = _mm512_setzero_ps();

            for (int i = 0; i < w; i++)
            {
                _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                _fsum = _mm512_add_ps(_fsum, _fLoad);
            }

            sum = _mm512_reduce_add_ps(_fsum);

            // var
            float mean = sum / (w * 16);
            __m512 _mean = _mm512_set1_ps(mean);
            __m512 _fsqsum = _mm512_setzero_ps();

            for (int i = 0; i < w; i++)
            {
                _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                _fLoad = _mm512_sub_ps(_fLoad, _mean);
                _fLoad = _mm512_mul_ps(_fLoad, _fLoad);
                _fsqsum = _mm512_add_ps(_fsqsum, _fLoad);
            }

            sqsum = _mm512_reduce_add_ps(_fsqsum);

            float var = sqsum / (w * 16);

            float a = static_cast<float>(1.f / (sqrt(var + eps)));
            float b = -mean * a;
            __m512 _a = _mm512_set1_ps(a);
            __m512 _b = _mm512_set1_ps(b);
            __m512 _gamma, _beta;

            if (affine)
            {
                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                    _fLoad = _mm512_mul_ps(_fLoad, _a);
                    _fLoad = _mm512_add_ps(_fLoad, _b);

                    _gamma = _mm512_loadu_ps((const float*)gamma_data + (i * 16));
                    _beta = _mm512_loadu_ps((const float*)beta_data + (i * 16));
                    _fLoad = _mm512_mul_ps(_fLoad, _gamma);
                    _fLoad = _mm512_add_ps(_fLoad, _beta);

                    _mm512_storeu_ps(ptr + (i * 16), _fLoad);
                }
            }
            else
            {
                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                    _fLoad = _mm512_mul_ps(_fLoad, _a);
                    _fLoad = _mm512_add_ps(_fLoad, _b);
                    _mm512_storeu_ps(ptr + (i * 16), _fLoad);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            // assert affine_size == w

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);

                __m512 _fLoad;

                // mean
                __m512 _fsum = _mm512_setzero_ps();

                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                    _fsum = _mm512_add_ps(_fsum, _fLoad);
                }

                // var
                __m512 _size = _mm512_set1_ps((float)w);
                __m512 _mean = _mm512_div_ps(_fsum, _size);
                __m512 _fsqsum = _mm512_setzero_ps();

                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                    _fLoad = _mm512_sub_ps(_fLoad, _mean);
                    _fLoad = _mm512_mul_ps(_fLoad, _fLoad);
                    _fsqsum = _mm512_add_ps(_fsqsum, _fLoad);
                }

                __m512 _var = _mm512_div_ps(_fsqsum, _size);

                __m512 _eps = _mm512_set1_ps(eps);
                __m512 _a = _mm512_add_ps(_var, _eps);
                _a = _mm512_sqrt_ps(_a);
                _a = _mm512_rcp14_ps(_a);
                __m512 _b = _mm512_mul_ps(_mean, _a);
                __m512 _gamma, _beta;

                if (affine)
                {
                    for (int i = 0; i < w; i++)
                    {
                        _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                        _fLoad = _mm512_mul_ps(_fLoad, _a);
                        _fLoad = _mm512_sub_ps(_fLoad, _b);

                        _gamma = _mm512_set1_ps(((const float*)gamma_data)[i]);
                        _beta = _mm512_set1_ps(((const float*)beta_data)[i]);
                        _fLoad = _mm512_mul_ps(_fLoad, _gamma);
                        _fLoad = _mm512_add_ps(_fLoad, _beta);

                        _mm512_storeu_ps(ptr + (i * 16), _fLoad);
                    }
                }
                else
                {
                    for (int i = 0; i < w; i++)
                    {
                        _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                        _fLoad = _mm512_mul_ps(_fLoad, _a);
                        _fLoad = _mm512_sub_ps(_fLoad, _b);
                        _mm512_storeu_ps(ptr + (i * 16), _fLoad);
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

            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr = bottom_top_blob.channel(q).row(i);

                        __m512 _fLoad;

                        // mean
                        __m512 _fsum = _mm512_setzero_ps();

                        for (int i = 0; i < w; i++)
                        {
                            _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                            _fsum = _mm512_add_ps(_fsum, _fLoad);
                        }

                        // var
                        __m512 _size = _mm512_set1_ps((float)w);
                        __m512 _mean = _mm512_div_ps(_fsum, _size);
                        __m512 _fsqsum = _mm512_setzero_ps();

                        for (int i = 0; i < w; i++)
                        {
                            _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                            _fLoad = _mm512_sub_ps(_fLoad, _mean);
                            _fLoad = _mm512_mul_ps(_fLoad, _fLoad);
                            _fsqsum = _mm512_add_ps(_fsqsum, _fLoad);
                        }

                        __m512 _var = _mm512_div_ps(_fsqsum, _size);

                        __m512 _eps = _mm512_set1_ps(eps);
                        __m512 _a = _mm512_add_ps(_var, _eps);
                        _a = _mm512_sqrt_ps(_a);
                        _a = _mm512_rcp14_ps(_a);
                        __m512 _b = _mm512_mul_ps(_mean, _a);
                        __m512 _gamma, _beta;

                        if (affine)
                        {
                            for (int j = 0; j < w; j++)
                            {
                                _fLoad = _mm512_loadu_ps(ptr + (j * 16));
                                _fLoad = _mm512_mul_ps(_fLoad, _a);
                                _fLoad = _mm512_sub_ps(_fLoad, _b);

                                _gamma = _mm512_set1_ps(((const float*)gamma_data)[j]);
                                _beta = _mm512_set1_ps(((const float*)beta_data)[j]);
                                _fLoad = _mm512_mul_ps(_fLoad, _gamma);
                                _fLoad = _mm512_add_ps(_fLoad, _beta);

                                _mm512_storeu_ps(ptr + (j * 16), _fLoad);
                            }
                        }
                        else
                        {
                            for (int j = 0; j < w; j++)
                            {
                                _fLoad = _mm512_loadu_ps(ptr + (j * 16));
                                _fLoad = _mm512_mul_ps(_fLoad, _a);
                                _fLoad = _mm512_sub_ps(_fLoad, _b);
                                _mm512_storeu_ps(ptr + (j * 16), _fLoad);
                            }
                        }
                    }
                }
            }

            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);

                    __m512 _fLoad;

                    // mean
                    __m512 _fsum = _mm512_setzero_ps();

                    for (int i = 0; i < size; i++)
                    {
                        _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                        _fsum = _mm512_add_ps(_fsum, _fLoad);
                    }

                    // var
                    __m512 _size = _mm512_set1_ps((float)size);
                    __m512 _mean = _mm512_div_ps(_fsum, _size);
                    __m512 _fsqsum = _mm512_setzero_ps();

                    for (int i = 0; i < size; i++)
                    {
                        _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                        _fLoad = _mm512_sub_ps(_fLoad, _mean);
                        _fLoad = _mm512_mul_ps(_fLoad, _fLoad);
                        _fsqsum = _mm512_add_ps(_fsqsum, _fLoad);
                    }

                    __m512 _var = _mm512_div_ps(_fsqsum, _size);

                    __m512 _eps = _mm512_set1_ps(eps);
                    __m512 _a = _mm512_add_ps(_var, _eps);
                    _a = _mm512_sqrt_ps(_a);
                    _a = _mm512_rcp14_ps(_a);
                    __m512 _b = _mm512_mul_ps(_mean, _a);
                    __m512 _gamma, _beta;

                    if (affine)
                    {
                        for (int i = 0; i < size; i++)
                        {
                            _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                            _fLoad = _mm512_mul_ps(_fLoad, _a);
                            _fLoad = _mm512_sub_ps(_fLoad, _b);

                            _gamma = _mm512_set1_ps(((const float*)gamma_data)[i]);
                            _beta = _mm512_set1_ps(((const float*)beta_data)[i]);
                            _fLoad = _mm512_mul_ps(_fLoad, _gamma);
                            _fLoad = _mm512_add_ps(_fLoad, _beta);

                            _mm512_storeu_ps(ptr + (i * 16), _fLoad);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < size; i++)
                        {
                            _fLoad = _mm512_loadu_ps(ptr + (i * 16));
                            _fLoad = _mm512_mul_ps(_fLoad, _a);
                            _fLoad = _mm512_sub_ps(_fLoad, _b);
                            _mm512_storeu_ps(ptr + (i * 16), _fLoad);
                        }
                    }
                }
            }
        }
        return 0;
    }
#endif
    if (elempack == 8)
    {
        if (dims == 1)
        {
            int w = bottom_top_blob.w;

            float* ptr = (float*)bottom_top_blob;

            __m256 _fLoad;

            // mean
            float sum = 0.f;
            float sqsum = 0.f;

            __m256 _fsum = _mm256_setzero_ps();

            for (int i = 0; i < w; i++)
            {
                _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                _fsum = _mm256_add_ps(_fsum, _fLoad);
            }

            // const float* q = (const float*)&_fsum;

            // sum = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];
            sum = _mm256_reduce_add_ps(_fsum);

            // var
            float mean = sum / (w * 8);
            __m256 _mean = _mm256_set1_ps(mean);
            __m256 _fsqsum = _mm256_setzero_ps();

            for (int i = 0; i < w; i++)
            {
                _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                _fLoad = _mm256_sub_ps(_fLoad, _mean);
                _fLoad = _mm256_mul_ps(_fLoad, _fLoad);
                _fsqsum = _mm256_add_ps(_fsqsum, _fLoad);
            }

            // q = (const float*)&_fsqsum;
            // sqsum = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];
            sqsum = _mm256_reduce_add_ps(_fsqsum);

            float var = sqsum / (w * 8);

            float a = static_cast<float>(1.f / (sqrt(var + eps)));
            float b = -mean * a;
            __m256 _a = _mm256_set1_ps(a);
            __m256 _b = _mm256_set1_ps(b);
            __m256 _gamma, _beta;

            if (affine)
            {
                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                    _fLoad = _mm256_mul_ps(_fLoad, _a);
                    _fLoad = _mm256_add_ps(_fLoad, _b);

                    _gamma = _mm256_loadu_ps((const float*)gamma_data + (i * 8));
                    _beta = _mm256_loadu_ps((const float*)beta_data + (i * 8));
                    _fLoad = _mm256_mul_ps(_fLoad, _gamma);
                    _fLoad = _mm256_add_ps(_fLoad, _beta);

                    _mm256_storeu_ps(ptr + (i * 8), _fLoad);
                }
            }
            else
            {
                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                    _fLoad = _mm256_mul_ps(_fLoad, _a);
                    _fLoad = _mm256_add_ps(_fLoad, _b);
                    _mm256_storeu_ps(ptr + (i * 8), _fLoad);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            // assert affine_size == w

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);

                __m256 _fLoad;

                // mean
                __m256 _fsum = _mm256_setzero_ps();

                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                    _fsum = _mm256_add_ps(_fsum, _fLoad);
                }

                // var
                __m256 _size = _mm256_set1_ps((float)w);
                __m256 _mean = _mm256_div_ps(_fsum, _size);
                __m256 _fsqsum = _mm256_setzero_ps();

                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                    _fLoad = _mm256_sub_ps(_fLoad, _mean);
                    _fLoad = _mm256_mul_ps(_fLoad, _fLoad);
                    _fsqsum = _mm256_add_ps(_fsqsum, _fLoad);
                }

                __m256 _var = _mm256_div_ps(_fsqsum, _size);

                __m256 _eps = _mm256_set1_ps(eps);
                __m256 _a = _mm256_add_ps(_var, _eps);
                _a = _mm256_rsqrt_ps(_a);
                __m256 _b = _mm256_mul_ps(_mean, _a);
                __m256 _gamma, _beta;

                if (affine)
                {
                    for (int i = 0; i < w; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                        _fLoad = _mm256_mul_ps(_fLoad, _a);
                        _fLoad = _mm256_sub_ps(_fLoad, _b);

                        _gamma = _mm256_set1_ps(((const float*)gamma_data)[i]);
                        _beta = _mm256_set1_ps(((const float*)beta_data)[i]);
                        _fLoad = _mm256_mul_ps(_fLoad, _gamma);
                        _fLoad = _mm256_add_ps(_fLoad, _beta);

                        _mm256_storeu_ps(ptr + (i * 8), _fLoad);
                    }
                }
                else
                {
                    for (int i = 0; i < w; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                        _fLoad = _mm256_mul_ps(_fLoad, _a);
                        _fLoad = _mm256_sub_ps(_fLoad, _b);
                        _mm256_storeu_ps(ptr + (i * 8), _fLoad);
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

            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr = bottom_top_blob.channel(q).row(i);

                        __m256 _fLoad;

                        // mean
                        __m256 _fsum = _mm256_setzero_ps();

                        for (int i = 0; i < w; i++)
                        {
                            _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                            _fsum = _mm256_add_ps(_fsum, _fLoad);
                        }

                        // var
                        __m256 _size = _mm256_set1_ps((float)w);
                        __m256 _mean = _mm256_div_ps(_fsum, _size);
                        __m256 _fsqsum = _mm256_setzero_ps();

                        for (int i = 0; i < w; i++)
                        {
                            _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                            _fLoad = _mm256_sub_ps(_fLoad, _mean);
                            _fLoad = _mm256_mul_ps(_fLoad, _fLoad);
                            _fsqsum = _mm256_add_ps(_fsqsum, _fLoad);
                        }

                        __m256 _var = _mm256_div_ps(_fsqsum, _size);

                        __m256 _eps = _mm256_set1_ps(eps);
                        __m256 _a = _mm256_add_ps(_var, _eps);
                        _a = _mm256_rsqrt_ps(_a);
                        __m256 _b = _mm256_mul_ps(_mean, _a);
                        __m256 _gamma, _beta;

                        if (affine)
                        {
                            for (int j = 0; j < w; j++)
                            {
                                _fLoad = _mm256_loadu_ps(ptr + (j * 8));
                                _fLoad = _mm256_mul_ps(_fLoad, _a);
                                _fLoad = _mm256_sub_ps(_fLoad, _b);

                                _gamma = _mm256_set1_ps(((const float*)gamma_data)[j]);
                                _beta = _mm256_set1_ps(((const float*)beta_data)[j]);
                                _fLoad = _mm256_mul_ps(_fLoad, _gamma);
                                _fLoad = _mm256_add_ps(_fLoad, _beta);

                                _mm256_storeu_ps(ptr + (j * 8), _fLoad);
                            }
                        }
                        else
                        {
                            for (int j = 0; j < w; j++)
                            {
                                _fLoad = _mm256_loadu_ps(ptr + (j * 8));
                                _fLoad = _mm256_mul_ps(_fLoad, _a);
                                _fLoad = _mm256_sub_ps(_fLoad, _b);
                                _mm256_storeu_ps(ptr + (j * 8), _fLoad);
                            }
                        }
                    }
                }
            }

            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    // int ssize = size;

                    __m256 _fLoad;

                    // mean
                    __m256 _fsum = _mm256_setzero_ps();

                    for (int i = 0; i < size; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                        _fsum = _mm256_add_ps(_fsum, _fLoad);
                    }

                    // const float* sum = (const float*)&_fsum;

                    // var
                    __m256 _size = _mm256_set1_ps((float)size);
                    __m256 _mean = _mm256_div_ps(_fsum, _size);
                    __m256 _fsqsum = _mm256_setzero_ps();

                    for (int i = 0; i < size; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                        _fLoad = _mm256_sub_ps(_fLoad, _mean);
                        _fLoad = _mm256_mul_ps(_fLoad, _fLoad);
                        _fsqsum = _mm256_add_ps(_fsqsum, _fLoad);
                    }

                    // const float* sqsum = (const float*)&_fsqsum;

                    __m256 _var = _mm256_div_ps(_fsqsum, _size);

                    // float a = static_cast<float>(1.f / (sqrt(var + eps)));
                    // float b = -mean * a;
                    __m256 _eps = _mm256_set1_ps(eps);
                    __m256 _a = _mm256_add_ps(_var, _eps);
                    _a = _mm256_rsqrt_ps(_a);
                    __m256 _b = _mm256_mul_ps(_mean, _a);
                    __m256 _gamma, _beta;

                    if (affine)
                    {
                        for (int i = 0; i < size; i++)
                        {
                            _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                            _fLoad = _mm256_mul_ps(_fLoad, _a);
                            _fLoad = _mm256_sub_ps(_fLoad, _b);

                            // _gamma = _mm256_loadu_ps((const float*)gamma_data + (i * 8));
                            // _beta = _mm256_loadu_ps((const float*)beta_data + (i * 8));
                            _gamma = _mm256_set1_ps(((const float*)gamma_data)[i]);
                            _beta = _mm256_set1_ps(((const float*)beta_data)[i]);
                            _fLoad = _mm256_mul_ps(_fLoad, _gamma);
                            _fLoad = _mm256_add_ps(_fLoad, _beta);

                            _mm256_storeu_ps(ptr + (i * 8), _fLoad);
                        }
                    }
                    else
                    {
                        for (int i = 0; i < size; i++)
                        {
                            _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                            _fLoad = _mm256_mul_ps(_fLoad, _a);
                            _fLoad = _mm256_sub_ps(_fLoad, _b);
                            _mm256_storeu_ps(ptr + (i * 8), _fLoad);
                        }
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
            int w = bottom_top_blob.w;

            float* ptr = bottom_top_blob;

            __m128 _fLoad;

            // mean
            float sum = 0.f;
            float sqsum = 0.f;

            __m128 _fsum = _mm_setzero_ps();

            for (int i = 0; i < w; i++)
            {
                _fLoad = _mm_load_ps(ptr + (i * 4));
                _fsum = _mm_add_ps(_fsum, _fLoad);
            }

            // const float* q = (const float*)&_fsum;

            // sum = q[0] + q[1] + q[2] + q[3];
            sum = _mm_reduce_add_ps(_fsum);

            // var
            float mean = sum / (w * 4);
            __m128 _mean = _mm_set1_ps(mean);
            __m128 _fsqsum = _mm_setzero_ps();

            for (int i = 0; i < w; i++)
            {
                _fLoad = _mm_load_ps(ptr + (i * 4));
                _fLoad = _mm_sub_ps(_fLoad, _mean);
                _fLoad = _mm_mul_ps(_fLoad, _fLoad);
                _fsqsum = _mm_add_ps(_fsqsum, _fLoad);
            }

            // q = (const float*)&_fsqsum;
            // sqsum = q[0] + q[1] + q[2] + q[3];
            sqsum = _mm_reduce_add_ps(_fsqsum);

            float var = sqsum / (w * 4);

            float a = static_cast<float>(1.f / (sqrt(var + eps)));
            float b = -mean * a;
            __m128 _a = _mm_set1_ps(a);
            __m128 _b = _mm_set1_ps(b);
            __m128 _gamma, _beta;

            if (affine)
            {
                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm_load_ps(ptr + (i * 4));
                    _fLoad = _mm_mul_ps(_fLoad, _a);
                    _fLoad = _mm_add_ps(_fLoad, _b);

                    _gamma = _mm_load_ps((const float*)gamma_data + (i * 4));
                    _beta = _mm_load_ps((const float*)beta_data + (i * 4));
                    _fLoad = _mm_mul_ps(_fLoad, _gamma);
                    _fLoad = _mm_add_ps(_fLoad, _beta);

                    _mm_store_ps(ptr + (i * 4), _fLoad);
                }
            }
            else
            {
                for (int i = 0; i < w; i++)
                {
                    _fLoad = _mm_load_ps(ptr + (i * 4));
                    _fLoad = _mm_mul_ps(_fLoad, _a);
                    _fLoad = _mm_add_ps(_fLoad, _b);
                    _mm_store_ps(ptr + (i * 4), _fLoad);
                }
            }
        }

        if (dims == 2)
        {
            int w = bottom_top_blob.w;
            int h = bottom_top_blob.h;
            // assert affine_size == w

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);

                __m128 _fLoad;

                // mean
                __m128 _fsum = _mm_setzero_ps();

                for (int j = 0; j < w; j++)
                {
                    _fLoad = _mm_load_ps(ptr + (j * 4));
                    _fsum = _mm_add_ps(_fsum, _fLoad);
                }

                // var
                __m128 _size = _mm_set1_ps((float)w);
                __m128 _mean = _mm_div_ps(_fsum, _size);
                __m128 _fsqsum = _mm_setzero_ps();

                for (int j = 0; j < w; j++)
                {
                    _fLoad = _mm_load_ps(ptr + (j * 4));
                    _fLoad = _mm_sub_ps(_fLoad, _mean);
                    _fLoad = _mm_mul_ps(_fLoad, _fLoad);
                    _fsqsum = _mm_add_ps(_fsqsum, _fLoad);
                }
                __m128 _var = _mm_div_ps(_fsqsum, _size);

                __m128 _eps = _mm_set1_ps(eps);
                __m128 _a = _mm_add_ps(_var, _eps);
                _a = _mm_rsqrt_ps(_a);
                __m128 _b = _mm_mul_ps(_mean, _a);
                __m128 _gamma, _beta;

                if (affine)
                {
                    for (int j = 0; j < w; j++)
                    {
                        _fLoad = _mm_load_ps(ptr + (j * 4));
                        _fLoad = _mm_mul_ps(_fLoad, _a);
                        _fLoad = _mm_sub_ps(_fLoad, _b);

                        _gamma = _mm_set1_ps(((const float*)gamma_data)[j]);
                        _beta = _mm_set1_ps(((const float*)beta_data)[j]);
                        _fLoad = _mm_mul_ps(_fLoad, _gamma);
                        _fLoad = _mm_add_ps(_fLoad, _beta);

                        _mm_store_ps(ptr + (j * 4), _fLoad);
                    }
                }
                else
                {
                    for (int j = 0; j < w; j++)
                    {
                        _fLoad = _mm_load_ps(ptr + (j * 4));
                        _fLoad = _mm_mul_ps(_fLoad, _a);
                        _fLoad = _mm_sub_ps(_fLoad, _b);
                        _mm_store_ps(ptr + (j * 4), _fLoad);
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

            if (affine_size == w)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    for (int i = 0; i < h; i++)
                    {
                        float* ptr = bottom_top_blob.channel(q).row(i);

                        __m128 _fLoad;

                        // mean
                        __m128 _fsum = _mm_setzero_ps();

                        for (int j = 0; j < w; j++)
                        {
                            _fLoad = _mm_load_ps(ptr + (j * 4));
                            _fsum = _mm_add_ps(_fsum, _fLoad);
                        }

                        // var
                        __m128 _size = _mm_set1_ps((float)w);
                        __m128 _mean = _mm_div_ps(_fsum, _size);
                        __m128 _fsqsum = _mm_setzero_ps();

                        for (int j = 0; j < w; j++)
                        {
                            _fLoad = _mm_load_ps(ptr + (j * 4));
                            _fLoad = _mm_sub_ps(_fLoad, _mean);
                            _fLoad = _mm_mul_ps(_fLoad, _fLoad);
                            _fsqsum = _mm_add_ps(_fsqsum, _fLoad);
                        }

                        __m128 _var = _mm_div_ps(_fsqsum, _size);
                        __m128 _eps = _mm_set1_ps(eps);
                        __m128 _a = _mm_add_ps(_var, _eps);
                        _a = _mm_rsqrt_ps(_a);
                        __m128 _b = _mm_mul_ps(_mean, _a);
                        __m128 _gamma, _beta;

                        if (affine)
                        {
                            for (int j = 0; j < w; j++)
                            {
                                _fLoad = _mm_load_ps(ptr + (j * 4));
                                _fLoad = _mm_mul_ps(_fLoad, _a);
                                _fLoad = _mm_sub_ps(_fLoad, _b);

                                _gamma = _mm_set1_ps(((const float*)gamma_data)[j]);
                                _beta = _mm_set1_ps(((const float*)beta_data)[j]);
                                _fLoad = _mm_mul_ps(_fLoad, _gamma);
                                _fLoad = _mm_add_ps(_fLoad, _beta);

                                _mm_store_ps(ptr + (j * 4), _fLoad);
                            }
                        }
                        else
                        {
                            for (int j = 0; j < w; j++)
                            {
                                _fLoad = _mm_load_ps(ptr + (j * 4));
                                _fLoad = _mm_mul_ps(_fLoad, _a);
                                _fLoad = _mm_sub_ps(_fLoad, _b);
                                _mm_store_ps(ptr + (j * 4), _fLoad);
                            }
                        }
                    }
                }
            }

            else // if (affine_size == size)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);

                    __m128 _fLoad;

                    // mean

                    __m128 _fsum = _mm_setzero_ps();

                    for (int j = 0; j < size; j++)
                    {
                        _fLoad = _mm_load_ps(ptr + (j * 4));
                        _fsum = _mm_add_ps(_fsum, _fLoad);
                    }

                    // var
                    __m128 _size = _mm_set1_ps((float)size);
                    __m128 _mean = _mm_div_ps(_fsum, _size);
                    __m128 _fsqsum = _mm_setzero_ps();

                    for (int j = 0; j < size; j++)
                    {
                        _fLoad = _mm_load_ps(ptr + (j * 4));
                        _fLoad = _mm_sub_ps(_fLoad, _mean);

                        _fLoad = _mm_mul_ps(_fLoad, _fLoad);
                        _fsqsum = _mm_add_ps(_fsqsum, _fLoad);
                    }

                    __m128 _var = _mm_div_ps(_fsqsum, _size);
                    __m128 _eps = _mm_set1_ps(eps);
                    __m128 _a = _mm_add_ps(_var, _eps);
                    _a = _mm_rsqrt_ps(_a);
                    __m128 _b = _mm_mul_ps(_mean, _a);
                    __m128 _gamma, _beta;

                    if (affine)
                    {
                        for (int j = 0; j < size; j++)
                        {
                            _fLoad = _mm_load_ps(ptr + (j * 4));
                            _fLoad = _mm_mul_ps(_fLoad, _a);
                            _fLoad = _mm_sub_ps(_fLoad, _b);

                            _gamma = _mm_set1_ps(((const float*)gamma_data)[j]);
                            _beta = _mm_set1_ps(((const float*)beta_data)[j]);
                            _fLoad = _mm_mul_ps(_fLoad, _gamma);
                            _fLoad = _mm_add_ps(_fLoad, _beta);

                            _mm_store_ps(ptr + (j * 4), _fLoad);
                        }
                    }
                    else
                    {
                        for (int j = 0; j < size; j++)
                        {
                            _fLoad = _mm_load_ps(ptr + (j * 4));
                            _fLoad = _mm_mul_ps(_fLoad, _a);
                            _fLoad = _mm_sub_ps(_fLoad, _b);
                            _mm_store_ps(ptr + (j * 4), _fLoad);
                        }
                    }
                }
            }
        }

        return 0;
    }
#endif // __SSE2__

    if (dims != 3)
        return LayerNorm::forward_inplace(bottom_top_blob, opt);

#if __SSE2__
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;
#if __AVX__

    if (affine_size == w)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.channel(q).row(i);
                int ww = w / 8;
                int remainw = ww * 8;

                __m256 _fLoad;

                // mean
                float sum = 0.f;
                float sqsum = 0.f;

                __m256 _fsum = _mm256_setzero_ps();

                for (int i = 0; i < ww; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                    _fsum = _mm256_add_ps(_fsum, _fLoad);
                }

                // const float* q = (const float*)&_fsum;

                // sum = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];
                sum = _mm256_reduce_add_ps(_fsum);

                for (int i = remainw; i < w; i++)
                    sum += ptr[i];

                // var
                float mean = sum / w;
                __m256 _mean = _mm256_set1_ps(mean);
                __m256 _fsqsum = _mm256_setzero_ps();

                for (int i = 0; i < ww; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                    _fLoad = _mm256_sub_ps(_fLoad, _mean);
                    _fLoad = _mm256_mul_ps(_fLoad, _fLoad);
                    _fsqsum = _mm256_add_ps(_fsqsum, _fLoad);
                }

                // q = (const float*)&_fsqsum;
                // sqsum = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];
                sqsum = _mm256_reduce_add_ps(_fsqsum);

                for (int i = remainw; i < w; i++)
                {
                    sqsum += (ptr[i] - mean) * (ptr[i] - mean);
                }

                float var = sqsum / w;

                float a = static_cast<float>(1.f / (sqrt(var + eps)));
                float b = -mean * a;
                __m256 _a = _mm256_set1_ps(a);
                __m256 _b = _mm256_set1_ps(b);
                __m256 _gamma, _beta;

                if (affine)
                {
                    for (int i = 0; i < ww; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                        _fLoad = _mm256_mul_ps(_fLoad, _a);
                        _fLoad = _mm256_add_ps(_fLoad, _b);

                        _gamma = _mm256_loadu_ps((const float*)gamma_data + (i * 8));
                        _beta = _mm256_loadu_ps((const float*)beta_data + (i * 8));
                        _fLoad = _mm256_mul_ps(_fLoad, _gamma);
                        _fLoad = _mm256_add_ps(_fLoad, _beta);

                        _mm256_storeu_ps(ptr + (i * 8), _fLoad);
                    }

                    for (int i = remainw; i < w; i++)
                    {
                        ptr[i] = (ptr[i] * a + b) * gamma_data[i] + beta_data[i];
                    }
                }
                else
                {
                    for (int i = 0; i < ww; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                        _fLoad = _mm256_mul_ps(_fLoad, _a);
                        _fLoad = _mm256_add_ps(_fLoad, _b);
                        _mm256_storeu_ps(ptr + (i * 8), _fLoad);
                    }
                    for (int i = remainw; i < w; i++)
                    {
                        ptr[i] = ptr[i] * a + b;
                    }
                }
            }
        }
    }
    else // if (affine_size == size)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            {
                float* ptr = bottom_top_blob.channel(q);
                int ssize = size / 8;
                int remain_size = ssize * 8;

                __m256 _fLoad;

                // mean
                float sum = 0.f;
                float sqsum = 0.f;

                __m256 _fsum = _mm256_setzero_ps();

                for (int i = 0; i < ssize; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                    _fsum = _mm256_add_ps(_fsum, _fLoad);
                }

                // const float* q = (const float*)&_fsum;

                // sum = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];
                sum = _mm256_reduce_add_ps(_fsum);

                for (int i = remain_size; i < size; i++)
                    sum += ptr[i];

                // var
                float mean = sum / size;
                __m256 _mean = _mm256_set1_ps(mean);
                __m256 _fsqsum = _mm256_setzero_ps();

                for (int i = 0; i < ssize; i++)
                {
                    _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                    _fLoad = _mm256_sub_ps(_fLoad, _mean);
                    _fLoad = _mm256_mul_ps(_fLoad, _fLoad);
                    _fsqsum = _mm256_add_ps(_fsqsum, _fLoad);
                }

                // q = (const float*)&_fsqsum;
                // sqsum = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];
                sqsum = _mm256_reduce_add_ps(_fsqsum);

                for (int i = remain_size; i < size; i++)
                {
                    sqsum += (ptr[i] - mean) * (ptr[i] - mean);
                }

                float var = sqsum / size;

                float a = static_cast<float>(1.f / (sqrt(var + eps)));
                float b = -mean * a;
                __m256 _a = _mm256_set1_ps(a);
                __m256 _b = _mm256_set1_ps(b);
                __m256 _gamma, _beta;

                if (affine)
                {
                    for (int i = 0; i < ssize; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                        _fLoad = _mm256_mul_ps(_fLoad, _a);
                        _fLoad = _mm256_add_ps(_fLoad, _b);

                        _gamma = _mm256_loadu_ps((const float*)gamma_data + (i * 8));
                        _beta = _mm256_loadu_ps((const float*)beta_data + (i * 8));
                        _fLoad = _mm256_mul_ps(_fLoad, _gamma);
                        _fLoad = _mm256_add_ps(_fLoad, _beta);

                        _mm256_storeu_ps(ptr + (i * 8), _fLoad);
                    }

                    for (int i = remain_size; i < size; i++)
                    {
                        ptr[i] = (ptr[i] * a + b) * gamma_data[i] + beta_data[i];
                    }
                }
                else
                {
                    for (int i = 0; i < ssize; i++)
                    {
                        _fLoad = _mm256_loadu_ps(ptr + (i * 8));
                        _fLoad = _mm256_mul_ps(_fLoad, _a);
                        _fLoad = _mm256_add_ps(_fLoad, _b);
                        _mm256_storeu_ps(ptr + (i * 8), _fLoad);
                    }
                    for (int i = remain_size; i < size; i++)
                    {
                        ptr[i] = ptr[i] * a + b;
                    }
                }
            }
        }
    }

    return 0;
#endif // __AVX__
    if (affine_size == w)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.channel(q).row(i);

                int ww = w / 4;
                int remainw = ww * 4;

                __m128 _fLoad;

                // mean
                float sum = 0.f;
                float sqsum = 0.f;

                __m128 _fsum = _mm_setzero_ps();

                for (int j = 0; j < ww; j++)
                {
                    _fLoad = _mm_load_ps(ptr + (j * 4));
                    _fsum = _mm_add_ps(_fsum, _fLoad);
                }

                // const float* q = (const float*)&_fsum;
                // sum = q[0] + q[1] + q[2] + q[3];
                sum = _mm_reduce_add_ps(_fsum);

                for (int j = remainw; j < w; j++)
                    sum += ptr[j];

                // var
                float mean = sum / w;
                __m128 _mean = _mm_set1_ps(mean);
                __m128 _fsqsum = _mm_setzero_ps();

                for (int j = 0; j < ww; j++)
                {
                    _fLoad = _mm_sub_ps(_fLoad, _mean);
                    _fLoad = _mm_mul_ps(_fLoad, _fLoad);
                    _fsqsum = _mm_add_ps(_fsqsum, _fLoad);
                }

                // q = (const float*)&_fsqsum;
                // sqsum = q[0] + q[1] + q[2] + q[3];
                sqsum = _mm_reduce_add_ps(_fsqsum);

                for (int j = remainw; j < w; j++)
                {
                    sqsum += (ptr[j] - mean) * (ptr[j] - mean);
                }
                float var = sqsum / w;

                float a = static_cast<float>(1.f / (sqrt(var + eps)));
                float b = -mean * a;
                __m128 _a = _mm_set1_ps(a);
                __m128 _b = _mm_set1_ps(b);
                __m128 _gamma, _beta;

                if (affine)
                {
                    for (int j = 0; j < ww; j++)
                    {
                        _fLoad = _mm_load_ps(ptr + (j * 4));
                        _fLoad = _mm_mul_ps(_fLoad, _a);
                        _fLoad = _mm_add_ps(_fLoad, _b);

                        _gamma = _mm_load_ps((const float*)gamma_data + (j * 4));
                        _beta = _mm_load_ps((const float*)beta_data + (j * 4));
                        _fLoad = _mm_mul_ps(_fLoad, _gamma);
                        _fLoad = _mm_add_ps(_fLoad, _beta);

                        _mm_store_ps(ptr + (j * 4), _fLoad);
                    }

                    for (int j = remainw; j < w; j++)
                    {
                        ptr[j] = (ptr[j] * a + b) * gamma_data[j] + beta_data[j];
                    }
                }
                else
                {
                    for (int j = 0; j < ww; j++)
                    {
                        _fLoad = _mm_load_ps(ptr + (j * 4));
                        _fLoad = _mm_mul_ps(_fLoad, _a);
                        _fLoad = _mm_add_ps(_fLoad, _b);
                        _mm_store_ps(ptr + (j * 4), _fLoad);
                    }
                    for (int j = remainw; j < w; j++)
                    {
                        ptr[j] = ptr[j] * a + b;
                    }
                }
            }
        }
    }

    else // if (affine_size == size)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            int ssize = size / 4;
            int remainsize = ssize * 4;

            __m128 _fLoad;

            // mean
            float sum = 0.f;
            float sqsum = 0.f;

            __m128 _fsum = _mm_setzero_ps();

            for (int i = 0; i < ssize; i++)
            {
                _fLoad = _mm_load_ps(ptr + (i * 4));
                _fsum = _mm_add_ps(_fsum, _fLoad);
            }

            // const float* q = (const float*)&_fsum;

            // sum = q[0] + q[1] + q[2] + q[3];
            sum = _mm_reduce_add_ps(_fsum);

            for (int i = remainsize; i < size; i++)
                sum += ptr[i];

            float mean = sum / size;
            __m128 _mean = _mm_set1_ps(mean);
            __m128 _fsqsum = _mm_setzero_ps();

            for (int i = 0; i < ssize; i++)
            {
                _fLoad = _mm_load_ps(ptr + (i * 4));
                _fLoad = _mm_sub_ps(_fLoad, _mean);
                _fLoad = _mm_mul_ps(_fLoad, _fLoad);
                _fsqsum = _mm_add_ps(_fsqsum, _fLoad);
            }

            // q = (const float*)&_fsqsum;
            // sqsum = q[0] + q[1] + q[2] + q[3];
            sqsum = _mm_reduce_add_ps(_fsqsum);

            for (int i = remainsize; i < size; i++)
            {
                sqsum += (ptr[i] - mean) * (ptr[i] - mean);
            }

            // var
            float var = sqsum / size;
            float a = static_cast<float>(1.f / (sqrt(var + eps)));
            float b = -mean * a;
            __m128 _a = _mm_set1_ps(a);
            __m128 _b = _mm_set1_ps(b);
            __m128 _gamma, _beta;

            if (affine)
            {
                for (int i = 0; i < ssize; i++)
                {
                    _fLoad = _mm_load_ps(ptr + (i * 4));
                    _fLoad = _mm_mul_ps(_fLoad, _a);
                    _fLoad = _mm_add_ps(_fLoad, _b);

                    _gamma = _mm_load_ps((const float*)gamma_data + (i * 4));
                    _beta = _mm_load_ps((const float*)beta_data + (i * 4));
                    _fLoad = _mm_mul_ps(_fLoad, _gamma);
                    _fLoad = _mm_add_ps(_fLoad, _beta);

                    _mm_store_ps(ptr + (i * 4), _fLoad);
                }
                for (int i = remainsize; i < size; i++)
                {
                    ptr[i] = (ptr[i] * a + b) * gamma_data[i] + beta_data[i];
                }
            }
            else
            {
                for (int i = 0; i < size; i++)
                {
                    _fLoad = _mm_load_ps(ptr + (i * 4));
                    _fLoad = _mm_mul_ps(_fLoad, _a);
                    _fLoad = _mm_add_ps(_fLoad, _b);
                    _mm_store_ps(ptr + (i * 4), _fLoad);
                }
                for (int i = remainsize; i < size; i++)
                {
                    ptr[i] = (ptr[i] * a + b);
                }
            }
        }
    }

    return 0;
#endif // __SSE2__
    return LayerNorm::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
