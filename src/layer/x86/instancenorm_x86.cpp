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
    int elempack = bottom_top_blob.elempack;
    const int size = w * h * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < c; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
        float mean[16] = {0.f}, var[16] = {0.f};

        // get the mean
#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _fsum512 = _mm512_setzero_ps();
        __m256 _fsum256;

        for (; i + 15 < size; i += 16)
        {
            _fsum512 = _mm512_add_ps(_fsum512, _mm512_loadu_ps(ptr));

            ptr += 16;
        }

        if (elempack == 16)
        {
            __m512 _mean512 = _mm512_div_ps(_fsum512, _mm512_set1_ps((float)size / 16));

            _mm512_storeu_ps(mean, _mean512);
        }
        else // elempack = 8 or 4 or 1
            _fsum256 = _mm256_add_ps(_mm512_castps512_ps256(_fsum512), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_fsum512), 1)));
#else
        __m256 _fsum256 = _mm256_setzero_ps();
#endif // __AVX512F__
        __m128 _fsum128;

        for (; i + 7 < size; i += 8)
        {
            _fsum256 = _mm256_add_ps(_fsum256, _mm256_loadu_ps(ptr));

            ptr += 8;
        }

        if (elempack == 8)
        {
            __m256 _mean256 = _mm256_div_ps(_fsum256, _mm256_set1_ps((float)size / 8));

            _mm256_storeu_ps(mean, _mean256);
            _mm256_storeu_ps(mean + 8, _mean256);
        }
        else // elempack = 4 or 1
            _fsum128 = _mm_add_ps(_mm256_castps256_ps128(_fsum256), _mm256_extractf128_ps(_fsum256, 1));
#else
        __m128 _fsum128 = _mm_setzero_ps();
#endif // __AVX__
        float sum = 0;

        for (; i + 3 < size; i += 4)
        {
            _fsum128 = _mm_add_ps(_fsum128, _mm_loadu_ps(ptr));

            ptr += 4;
        }

        if (elempack == 4)
        {
            __m128 _mean128 = _mm_div_ps(_fsum128, _mm_set1_ps((float)size / 4));

            _mm_storeu_ps(mean, _mean128);
            _mm_storeu_ps(mean + 4, _mean128);
            _mm_storeu_ps(mean + 8, _mean128);
            _mm_storeu_ps(mean + 12, _mean128);
        }
        else // elempack = 1
            sum = _mm_reduce_add_ps(_fsum128);
#else
        float sum = 0;
#endif // __SSE2__
        for (; i < size; i++)
        {
            sum += *ptr++;
        }

        if (elempack == 1)
        {
            float tmp = sum / size;
            for (int i = 0; i < 16; i++)
                mean[i] = tmp;
        }

        // get the var
        ptr = bottom_top_blob.channel(q);
        i = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _fsqsum512 = _mm512_setzero_ps();
        __m512 _mean512 = _mm512_loadu_ps(mean);
        __m256 _fsqsum256;

        for (; i + 15 < size; i += 16)
        {
            __m512 _fLoad = _mm512_loadu_ps(ptr);
            _fLoad = _mm512_sub_ps(_fLoad, _mean512);
            _fsqsum512 = _mm512_fmadd_ps(_fLoad, _fLoad, _fsqsum512);

            ptr += 16;
        }

        if (elempack == 16)
        {
            __m512 _var512 = _mm512_div_ps(_fsqsum512, _mm512_set1_ps((float)size / 16));

            _mm512_storeu_ps(var, _var512);
        }
        else // elempack = 8 or 4 or 1
            _fsqsum256 = _mm256_add_ps(_mm512_castps512_ps256(_fsqsum512), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_fsqsum512), 1)));
#else
        __m256 _fsqsum256 = _mm256_setzero_ps();
#endif // __AVX512F__
        __m256 _mean256 = _mm256_loadu_ps(mean);
        __m128 _fsqsum128;

        for (; i + 7 < size; i += 8)
        {
            __m256 _fLoad = _mm256_loadu_ps(ptr);
            _fLoad = _mm256_sub_ps(_fLoad, _mean256);
            _fsqsum256 = _mm256_comp_fmadd_ps(_fLoad, _fLoad, _fsqsum256);

            ptr += 8;
        }

        if (elempack == 8)
        {
            __m256 _var256 = _mm256_div_ps(_fsqsum256, _mm256_set1_ps((float)size / 8));

            _mm256_storeu_ps(var, _var256);
            _mm256_storeu_ps(var + 8, _var256);
        }
        else // elempack = 4 or 1
            _fsqsum128 = _mm_add_ps(_mm256_castps256_ps128(_fsqsum256), _mm256_extractf128_ps(_fsqsum256, 1));
#else
        __m128 _fsqsum128 = _mm_setzero_ps();
#endif // __AVX__
        __m128 _mean128 = _mm_loadu_ps(mean);
        float fsqsum = 0;

        for (; i + 3 < size; i += 4)
        {
            __m128 _fLoad = _mm_loadu_ps(ptr);
            _fLoad = _mm_sub_ps(_fLoad, _mean128);
            _fsqsum128 = _mm_comp_fmadd_ps(_fLoad, _fLoad, _fsqsum128);

            ptr += 4;
        }

        if (elempack == 4)
        {
            __m128 _var128 = _mm_div_ps(_fsqsum128, _mm_set1_ps((float)size / 4));

            _mm_storeu_ps(var, _var128);
            _mm_storeu_ps(var + 4, _var128);
            _mm_storeu_ps(var + 8, _var128);
            _mm_storeu_ps(var + 12, _var128);
        }
        else //elempack = 1
            fsqsum = _mm_reduce_add_ps(_fsqsum128);
#else
        float fsqsum = 0;
#endif // __SSE2__
        for (; i < size; i++)
        {
            float tmp = (*ptr++) - mean[0];
            fsqsum += tmp * tmp;
        }

        if (elempack == 1)
        {
            float tmp = fsqsum / size;
            for (int i = 0; i < 16; i++)
                var[i] = tmp;
        }

        ptr = bottom_top_blob.channel(q);
        i = 0;

        float a;
        float b;
        if (affine)
        {
            float gamma = gamma_data[q];
            float beta = beta_data[q];

            a = static_cast<float>(gamma / (sqrt(var[0] + eps)));
            b = -mean[0] * a + beta;
        }
        else
        {
            a = static_cast<float>(1.f / (sqrt(var[0] + eps)));
            b = -mean[0] * a;
        }

#if __SSE2__
        __m128 _a128;
        __m128 _b128;
        __m128 _gamma128;
        __m128 _beta128;
        if (affine)
        {
            __m128 _mean128 = _mm_loadu_ps(mean);

            _gamma128 = (elempack == 4) ? _mm_loadu_ps((const float*)gamma_data + q * 4) : _mm_set1_ps(gamma_data[q]);
            _beta128 = (elempack == 4) ? _mm_loadu_ps((const float*)beta_data + q * 4) : _mm_set1_ps(beta_data[q]);

            __m128 _var_eps128 = _mm_add_ps(_mm_loadu_ps(var), _mm_set1_ps(eps));
            __m128 _sqrtvar128 = _mm_sqrt_ps(_var_eps128);

            _a128 = _mm_div_ps(_gamma128, _sqrtvar128);
            _b128 = _mm_sub_ps(_beta128, _mm_mul_ps(_mean128, _a128));
        }
        else
        {
            __m128 _mean128 = _mm_loadu_ps(mean);

            __m128 _var_eps128 = _mm_add_ps(_mm_loadu_ps(var), _mm_set1_ps(eps));
            __m128 _sqrtvar128 = _mm_sqrt_ps(_var_eps128);

            _a128 = _mm_div_ps(_mm_set1_ps(1.f), _sqrtvar128);
            _b128 = _mm_mul_ps(_mm_sub_ps(_mm_setzero_ps(), _mean128), _a128);
        }
#if __AVX__
        __m256 _a256;
        __m256 _b256;
        __m256 _gamma256;
        __m256 _beta256;
        if (affine)
        {
            __m256 _mean256 = _mm256_loadu_ps(mean);

            _gamma256 = (elempack == 8) ? _mm256_loadu_ps((const float*)gamma_data + q * 8) : _mm256_insertf128_ps(_mm256_castps128_ps256(_gamma128), _gamma128, 1);
            _beta256 = (elempack == 8) ? _mm256_loadu_ps((const float*)beta_data + q * 8) : _mm256_insertf128_ps(_mm256_castps128_ps256(_beta128), _beta128, 1);

            __m256 _var_eps256 = _mm256_add_ps(_mm256_loadu_ps(var), _mm256_set1_ps(eps));
            __m256 _sqrtvar256 = _mm256_sqrt_ps(_var_eps256);

            _a256 = _mm256_div_ps(_gamma256, _sqrtvar256);
            _b256 = _mm256_sub_ps(_beta256, _mm256_mul_ps(_mean256, _a256));
        }
        else
        {
            __m256 _mean256 = _mm256_loadu_ps(mean);

            __m256 _var_eps256 = _mm256_add_ps(_mm256_loadu_ps(var), _mm256_set1_ps(eps));
            __m256 _sqrtvar256 = _mm256_sqrt_ps(_var_eps256);

            _a256 = _mm256_div_ps(_mm256_set1_ps(1.f), _sqrtvar256);
            _b256 = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), _mean256), _a256);
        }
#if __AVX512F__
        __m512 _a512;
        __m512 _b512;
        __m512 _gamma512;
        __m512 _beta512;
        if (affine)
        {
            __m512 _mean512 = _mm512_loadu_ps(mean);

            _gamma512 = (elempack == 16) ? _mm512_loadu_ps((const float*)gamma_data + q * 16) : _mm512_insertf32x8(_mm512_castps256_ps512(_gamma256), _gamma256, 1);
            _beta512 = (elempack == 16) ? _mm512_loadu_ps((const float*)beta_data + q * 16) : _mm512_insertf32x8(_mm512_castps256_ps512(_beta256), _beta256, 1);

            __m512 _var_eps512 = _mm512_add_ps(_mm512_loadu_ps(var), _mm512_set1_ps(eps));
            __m512 _sqrtvar512 = _mm512_sqrt_ps(_var_eps512);

            _a512 = _mm512_div_ps(_gamma512, _sqrtvar512);
            _b512 = _mm512_sub_ps(_beta512, _mm512_mul_ps(_mean512, _a512));
        }
        else
        {
            __m512 _mean512 = _mm512_loadu_ps(mean);

            __m512 _var_eps512 = _mm512_add_ps(_mm512_loadu_ps(var), _mm512_set1_ps(eps));
            __m512 _sqrtvar512 = _mm512_sqrt_ps(_var_eps512);

            _a512 = _mm512_div_ps(_mm512_set1_ps(1.f), _sqrtvar512);
            _b512 = _mm512_mul_ps(_mm512_sub_ps(_mm512_setzero_ps(), _mean512), _a512);
        }

        for (; i + 15 < size; i += 16)
        {
            __m512 _fLoad = _mm512_loadu_ps(ptr);
            _fLoad = _mm512_fmadd_ps(_fLoad, _a512, _b512);
            _mm512_storeu_ps(ptr, _fLoad);

            ptr += 16;
        }

#endif // __AVX512F__

        for (; i + 7 < size; i += 8)
        {
            __m256 _fLoad = _mm256_loadu_ps(ptr);
            _fLoad = _mm256_comp_fmadd_ps(_fLoad, _a256, _b256);
            _mm256_storeu_ps(ptr, _fLoad);

            ptr += 8;
        }
#endif // __AVX__

        for (; i + 3 < size; i += 4)
        {
            __m128 _fLoad = _mm_loadu_ps(ptr);
            _fLoad = _mm_comp_fmadd_ps(_fLoad, _a128, _b128);
            _mm_storeu_ps(ptr, _fLoad);

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