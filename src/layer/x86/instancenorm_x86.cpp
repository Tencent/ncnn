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

#include "instancenorm_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

InstanceNorm_x86::InstanceNorm_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int InstanceNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    // x = (x - mean) / (sqrt(var + eps)) * gamma + beta

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int c = bottom_top_blob.c;
    int size = w * h;
    int elempack = bottom_top_blob.elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            __m512 _fLoad;

            // mean
            __m512 _fsum = _mm512_setzero_ps();

            for (int j = 0; j < size; j++)
            {
                _fLoad = _mm512_loadu_ps(ptr + (j * 16));
                _fsum = _mm512_add_ps(_fsum, _fLoad);
            }

            // var
            __m512 _size = _mm512_set1_ps((float)size);
            __m512 _mean = _mm512_div_ps(_fsum, _size);
            __m512 _fsqsum = _mm512_setzero_ps();

            for (int j = 0; j < size; j++)
            {
                _fLoad = _mm512_loadu_ps(ptr + (j * 16));
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

            if (affine)
            {
                __m512 _gamma = _mm512_loadu_ps((const float*)gamma_data + (q * 16));
                __m512 _beta = _mm512_loadu_ps((const float*)beta_data + (q * 16));

                _a = _mm512_mul_ps(_a, _gamma);
                _b = _mm512_mul_ps(_b, _gamma);
                _b = _mm512_sub_ps(_b, _beta);
            }

            for (int j = 0; j < size; j++)
            {
                _fLoad = _mm512_loadu_ps(ptr + (j * 16));
                _fLoad = _mm512_mul_ps(_fLoad, _a);
                _fLoad = _mm512_sub_ps(_fLoad, _b);
                _mm512_storeu_ps(ptr + (j * 16), _fLoad);
            }
        }

        return 0;
    }
#endif
    if (elempack == 8)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            __m256 _fLoad;

            // mean
            __m256 _fsum = _mm256_setzero_ps();

            for (int j = 0; j < size; j++)
            {
                _fLoad = _mm256_loadu_ps(ptr + (j * 8));
                _fsum = _mm256_add_ps(_fsum, _fLoad);
            }

            // var
            __m256 _size = _mm256_set1_ps((float)size);
            __m256 _mean = _mm256_div_ps(_fsum, _size);
            __m256 _fsqsum = _mm256_setzero_ps();

            for (int j = 0; j < size; j++)
            {
                _fLoad = _mm256_loadu_ps(ptr + (j * 8));
                _fLoad = _mm256_sub_ps(_fLoad, _mean);

                _fLoad = _mm256_mul_ps(_fLoad, _fLoad);
                _fsqsum = _mm256_add_ps(_fsqsum, _fLoad);
            }
            __m256 _var = _mm256_div_ps(_fsqsum, _size);
            __m256 _eps = _mm256_set1_ps(eps);
            __m256 _a = _mm256_add_ps(_var, _eps);
            _a = _mm256_rsqrt_ps(_a);
            __m256 _b = _mm256_mul_ps(_mean, _a);

            if (affine)
            {
                __m256 _gamma = _mm256_loadu_ps((const float*)gamma_data + (q * 8));
                __m256 _beta = _mm256_loadu_ps((const float*)beta_data + (q * 8));

                _a = _mm256_mul_ps(_a, _gamma);
                _b = _mm256_mul_ps(_b, _gamma);
                _b = _mm256_sub_ps(_b, _beta);
            }

            for (int j = 0; j < size; j++)
            {
                _fLoad = _mm256_loadu_ps(ptr + (j * 8));
                _fLoad = _mm256_mul_ps(_fLoad, _a);
                _fLoad = _mm256_sub_ps(_fLoad, _b);
                _mm256_storeu_ps(ptr + (j * 8), _fLoad);
            }
        }

        return 0;
    }
#endif
    if (elempack == 4)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
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

            if (affine)
            {
                __m128 _gamma = _mm_load_ps((const float*)gamma_data + (q * 4));
                __m128 _beta = _mm_load_ps((const float*)beta_data + (q * 4));

                _a = _mm_mul_ps(_a, _gamma);
                _b = _mm_mul_ps(_b, _gamma);
                _b = _mm_sub_ps(_b, _beta);
            }

            for (int j = 0; j < size; j++)
            {
                _fLoad = _mm_load_ps(ptr + (j * 4));
                _fLoad = _mm_mul_ps(_fLoad, _a);
                _fLoad = _mm_sub_ps(_fLoad, _b);
                _mm_store_ps(ptr + (j * 4), _fLoad);
            }
        }

        return 0;
    }
#endif

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        // mean
        int i = 0;
        float sum = 0.f;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _fLoad512;
        __m512 _fsum512 = _mm512_setzero_ps();

        for (; i + 15 < size; i += 16)
        {
            _fLoad512 = _mm512_loadu_ps(ptr);
            _fsum512 = _mm512_add_ps(_fsum512, _fLoad512);
            ptr += 16;
        }
        sum += _mm512_reduce_add_ps(_fsum512);
#endif // __AVX512F__
        __m256 _fLoad256;
        __m256 _fsum256 = _mm256_setzero_ps();
        for (; i + 7 < size; i += 8)
        {
            _fLoad256 = _mm256_loadu_ps(ptr);
            _fsum256 = _mm256_add_ps(_fsum256, _fLoad256);
            ptr += 8;
        }
        sum += _mm256_reduce_add_ps(_fsum256);
#endif // __AVX__
        __m128 _fLoad128;
        __m128 _fsum128 = _mm_setzero_ps();
        for (; i + 3 < size; i += 4)
        {
            _fLoad128 = _mm_loadu_ps(ptr);
            _fsum128 = _mm_add_ps(_fsum128, _fLoad128);
            ptr += 4;
        }
        sum += _mm_reduce_add_ps(_fsum128);
#endif // __SSE2__
        for (; i < size; i++)
        {
            sum += *ptr;
            ptr++;
        }

        // var
        ptr = bottom_top_blob.channel(q);
        float sqsum = 0.f;
        float mean = sum / size;
        i = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _fsqsum512 = _mm512_setzero_ps();
        __m512 _mean512 = _mm512_set1_ps(mean);
        for (; i + 15 < size; i += 16)
        {
            _fLoad512 = _mm512_loadu_ps(ptr);
            _fLoad512 = _mm512_sub_ps(_fLoad512, _mean512);
            _fLoad512 = _mm512_mul_ps(_fLoad512, _fLoad512);
            _fsqsum512 = _mm512_add_ps(_fsqsum512, _fLoad512);
            ptr += 16;
        }

        sqsum += _mm512_reduce_add_ps(_fsqsum512);

        __m256 _mean256 = _mm512_castps512_ps256(_mean512);
#else
        __m256 _mean256 = _mm256_set1_ps(mean);
#endif // __AVX512F__
        __m256 _fsqsum256 = _mm256_setzero_ps();
        for (; i + 7 < size; i += 8)
        {
            _fLoad256 = _mm256_loadu_ps(ptr);
            _fLoad256 = _mm256_sub_ps(_fLoad256, _mean256);
            _fLoad256 = _mm256_mul_ps(_fLoad256, _fLoad256);
            _fsqsum256 = _mm256_add_ps(_fsqsum256, _fLoad256);
            ptr += 8;
        }

        sqsum += _mm256_reduce_add_ps(_fsqsum256);

        __m128 _mean128 = _mm256_castps256_ps128(_mean256);
#else
        __m128 _mean128 = _mm_set1_ps(mean);
#endif // __AVX__
        __m128 _fsqsum128 = _mm_setzero_ps();
        for (; i + 3 < size; i += 4)
        {
            _fLoad128 = _mm_loadu_ps(ptr);
            _fLoad128 = _mm_sub_ps(_fLoad128, _mean128);
            _fLoad128 = _mm_mul_ps(_fLoad128, _fLoad128);
            _fsqsum128 = _mm_add_ps(_fsqsum128, _fLoad128);
            ptr += 4;
        }

        sqsum += _mm_reduce_add_ps(_fsqsum128);
#endif // __SSE2__

        float tmp = 0.f;
        for (; i < size; i++)
        {
            tmp = (*ptr) - mean;
            sqsum += tmp * tmp;

            ptr++;
        }

        float var = sqsum / size;
        ptr = bottom_top_blob.channel(q);
        float a, b;
        i = 0;

        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = static_cast<float>(gamma / (sqrt(var + eps)));
            b = -mean * a + beta;
        }
        else
        {
            a = static_cast<float>(1.f / (sqrt(var + eps)));
            b = -mean * a;
        }

#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _a512 = _mm512_set1_ps(a);
        __m512 _b512 = _mm512_set1_ps(b);
        // __m512 _fLoad512 = _mm512_setzero_ps();

        for (; i + 15 < size; i += 16)
        {
            _fLoad512 = _mm512_loadu_ps(ptr);
            _fLoad512 = _mm512_mul_ps(_fLoad512, _a512);
            _fLoad512 = _mm512_add_ps(_fLoad512, _b512);

            _mm512_storeu_ps(ptr, _fLoad512);
            ptr += 16;
        }
        __m256 _a256 = _mm512_castps512_ps256(_a512);
        __m256 _b256 = _mm512_castps512_ps256(_b512);
#else
        __m256 _a256 = _mm256_set1_ps(a);
        __m256 _b256 = _mm256_set1_ps(b);

#endif // __AVX512F__
        // __m256 _fLoad256 = _mm256_setzero_ps();

        for (; i + 7 < size; i += 8)
        {
            _fLoad256 = _mm256_loadu_ps(ptr);
            _fLoad256 = _mm256_mul_ps(_fLoad256, _a256);
            _fLoad256 = _mm256_add_ps(_fLoad256, _b256);

            _mm256_storeu_ps(ptr, _fLoad256);
            ptr += 8;
        }
        __m128 _a128 = _mm256_castps256_ps128(_a256);
        __m128 _b128 = _mm256_castps256_ps128(_b256);
#else
        __m128 _a128 = _mm_set1_ps(a);
        __m128 _b128 = _mm_set1_ps(b);

#endif // __AVX__
        // __m128 _fLoad128 = _mm_setzero_ps();

        for (; i + 3 < size; i += 4)
        {
            _fLoad128 = _mm_loadu_ps(ptr);
            _fLoad128 = _mm_mul_ps(_fLoad128, _a128);
            _fLoad128 = _mm_add_ps(_fLoad128, _b128);

            _mm_storeu_ps(ptr, _fLoad128);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = *ptr * a + b;

            ptr++;
        }
    }

    return 0;
}

} // namespace ncnn
