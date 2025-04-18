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

static void layernorm(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _mean_avx512 = _mm512_set1_ps(0.f);
#endif // __AVX512F__
    __m256 _mean_avx = _mm256_set1_ps(0.f);
#endif // __AVX__
    __m128 _mean = _mm_set1_ps(0.f);
#endif // __SSE2__
    float mean = 0.f;
    {
        const float* ptr0 = ptr;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr0);
            _mean_avx512 = _mm512_add_ps(_mean_avx512, _p);
            ptr0 += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr0);
            _mean_avx = _mm256_add_ps(_mean_avx, _p);
            ptr0 += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr0);
            _mean = _mm_add_ps(_mean, _p);
            ptr0 += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            mean += ptr0[0];
            ptr0++;
        }
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        __m512 _elemcount = _mm512_set1_ps((float)elemcount);
        _mean_avx512 = _mm512_div_ps(_mean_avx512, _elemcount);
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
#if __AVX512F__
        {
            __m256 _mean0 = _mm512_castps512_ps256(_mean_avx512);
            __m256 _mean1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_mean_avx512), 1));
            _mean_avx = _mm256_add_ps(_mean_avx, _mean0);
            _mean_avx = _mm256_add_ps(_mean_avx, _mean1);
        }
#endif // __AVX512F__

        __m256 _elemcount = _mm256_set1_ps((float)elemcount);
        _mean_avx = _mm256_div_ps(_mean_avx, _elemcount);
#if __AVX512F__
        _mean_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_mean_avx), _mean_avx, 1);
#endif // __AVX512F__
    }
#endif // __AVX__
    if (elempack == 4)
    {
#if __AVX__
#if __AVX512F__
        {
            __m256 _mean0 = _mm512_castps512_ps256(_mean_avx512);
            __m256 _mean1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_mean_avx512), 1));
            _mean_avx = _mm256_add_ps(_mean_avx, _mean0);
            _mean_avx = _mm256_add_ps(_mean_avx, _mean1);
        }
#endif // __AVX512F__
        {
            __m128 _mean0 = _mm256_castps256_ps128(_mean_avx);
            __m128 _mean1 = _mm256_extractf128_ps(_mean_avx, 1);
            _mean = _mm_add_ps(_mean, _mean0);
            _mean = _mm_add_ps(_mean, _mean1);
        }
#endif // __AVX__

        __m128 _elemcount = _mm_set1_ps((float)elemcount);
        _mean = _mm_div_ps(_mean, _elemcount);
#if __AVX__
        _mean_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_mean), _mean, 1);
#if __AVX512F__
        _mean_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_mean_avx), _mean_avx, 1);
#endif // __AVX512F__
#endif // __AVX__
    }
#endif // __SSE2__
    if (elempack == 1)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        mean += _mm512_comp_reduce_add_ps(_mean_avx512);
#endif // __AVX512F__
        mean += _mm256_reduce_add_ps(_mean_avx);
#endif // __AVX__
        mean += _mm_reduce_add_ps(_mean);
#endif // __SSE2__

        mean = mean / elemcount;
#if __SSE2__
        _mean = _mm_set1_ps(mean);
#if __AVX__
        _mean_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_mean), _mean, 1);
#if __AVX512F__
        _mean_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_mean_avx), _mean_avx, 1);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _var_avx512 = _mm512_set1_ps(0.f);
#endif // __AVX512F__
    __m256 _var_avx = _mm256_set1_ps(0.f);
#endif // __AVX__
    __m128 _var = _mm_set1_ps(0.f);
#endif // __SSE2__
    float var = 0.f;
    {
        const float* ptr0 = ptr;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr0);
            _p = _mm512_sub_ps(_p, _mean_avx512);
            _var_avx512 = _mm512_fmadd_ps(_p, _p, _var_avx512);
            ptr0 += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr0);
            _p = _mm256_sub_ps(_p, _mean_avx);
            _var_avx = _mm256_comp_fmadd_ps(_p, _p, _var_avx);
            ptr0 += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr0);
            _p = _mm_sub_ps(_p, _mean);
            _var = _mm_comp_fmadd_ps(_p, _p, _var);
            ptr0 += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = ptr0[0] - mean;
            var += v * v;
            ptr0++;
        }
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        __m512 _elemcount = _mm512_set1_ps((float)elemcount);
        __m512 _eps = _mm512_set1_ps(eps);
        _var_avx512 = _mm512_div_ps(_var_avx512, _elemcount);
        _var_avx512 = _mm512_add_ps(_var_avx512, _eps);
        __m256 _var0 = _mm256_rsqrt_ps(_mm512_extractf32x8_ps(_var_avx512, 0));
        __m256 _var1 = _mm256_rsqrt_ps(_mm512_extractf32x8_ps(_var_avx512, 1));
        _var_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_var0), _var1, 1);
        _mean_avx512 = _mm512_mul_ps(_mean_avx512, _var_avx512);
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
#if __AVX512F__
        {
            __m256 _var0 = _mm512_castps512_ps256(_var_avx512);
            __m256 _var1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_var_avx512), 1));
            _var_avx = _mm256_add_ps(_var_avx, _var0);
            _var_avx = _mm256_add_ps(_var_avx, _var1);
        }
#endif // __AVX512F__

        __m256 _elemcount = _mm256_set1_ps((float)elemcount);
        __m256 _eps = _mm256_set1_ps(eps);
        _var_avx = _mm256_div_ps(_var_avx, _elemcount);
        _var_avx = _mm256_add_ps(_var_avx, _eps);
        _var_avx = _mm256_rsqrt_ps(_var_avx);
        _mean_avx = _mm256_mul_ps(_mean_avx, _var_avx);
#if __AVX512F__
        _var_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_var_avx), _var_avx, 1);
        _mean_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_mean_avx), _mean_avx, 1);
#endif // __AVX512F__
    }
#endif // __AVX__
    if (elempack == 4)
    {
#if __AVX__
#if __AVX512F__
        {
            __m256 _var0 = _mm512_castps512_ps256(_var_avx512);
            __m256 _var1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_var_avx512), 1));
            _var_avx = _mm256_add_ps(_var_avx, _var0);
            _var_avx = _mm256_add_ps(_var_avx, _var1);
        }
#endif // __AVX512F__
        {
            __m128 _var0 = _mm256_castps256_ps128(_var_avx);
            __m128 _var1 = _mm256_extractf128_ps(_var_avx, 1);
            _var = _mm_add_ps(_var, _var0);
            _var = _mm_add_ps(_var, _var1);
        }
#endif // __AVX__

        __m128 _elemcount = _mm_set1_ps((float)elemcount);
        __m128 _eps = _mm_set1_ps(eps);
        _var = _mm_div_ps(_var, _elemcount);
        _var = _mm_add_ps(_var, _eps);
        _var = _mm_rsqrt_ps(_var);
        _mean = _mm_mul_ps(_mean, _var);
#if __AVX__
        _var_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_var), _var, 1);
        _mean_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_mean), _mean, 1);
#if __AVX512F__
        _var_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_var_avx), _var_avx, 1);
        _mean_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_mean_avx), _mean_avx, 1);
#endif // __AVX512F__
#endif // __AVX__
    }
#endif // __SSE2__
    if (elempack == 1)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        var += _mm512_comp_reduce_add_ps(_var_avx512);
#endif // __AVX512F__
        var += _mm256_reduce_add_ps(_var_avx);
#endif // __AVX__
        var += _mm_reduce_add_ps(_var);
#endif // __SSE2__

        var = 1.f / sqrtf(var / elemcount + eps);
        mean = mean * var;
#if __SSE2__
        _var = _mm_set1_ps(var);
        _mean = _mm_set1_ps(mean);
#if __AVX__
        _var_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_var), _var, 1);
        _mean_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_mean), _mean, 1);
#if __AVX512F__
        _var_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_var_avx), _var_avx, 1);
        _mean_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_mean_avx), _mean_avx, 1);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
    }

    if (gamma_ptr && beta_ptr)
    {
        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __m512 _gamma = _mm512_set1_ps(gamma_ptr[0]);
                __m512 _beta = _mm512_set1_ps(beta_ptr[0]);
                _p = _mm512_fmsub_ps(_p, _var_avx512, _mean_avx512);
                _p = _mm512_fmadd_ps(_p, _gamma, _beta);
                _mm512_storeu_ps(ptr, _p);
                ptr += 16;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __m256 _gamma0 = _mm256_set1_ps(gamma_ptr[0]);
                __m256 _gamma1 = _mm256_set1_ps(gamma_ptr[1]);
                __m512 _gamma = _mm512_insertf32x8(_mm512_castps256_ps512(_gamma0), _gamma1, 1);
                __m256 _beta0 = _mm256_set1_ps(beta_ptr[0]);
                __m256 _beta1 = _mm256_set1_ps(beta_ptr[1]);
                __m512 _beta = _mm512_insertf32x8(_mm512_castps256_ps512(_beta0), _beta1, 1);
                _p = _mm512_fmsub_ps(_p, _var_avx512, _mean_avx512);
                _p = _mm512_fmadd_ps(_p, _gamma, _beta);
                _mm512_storeu_ps(ptr, _p);
                ptr += 16;
                gamma_ptr += 2;
                beta_ptr += 2;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _gamma = _mm256_set1_ps(gamma_ptr[0]);
                __m256 _beta = _mm256_set1_ps(beta_ptr[0]);
                _p = _mm256_comp_fmsub_ps(_p, _var_avx, _mean_avx);
                _p = _mm256_comp_fmadd_ps(_p, _gamma, _beta);
                _mm256_storeu_ps(ptr, _p);
                ptr += 8;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
#endif // __AVX__
        if (elempack == 4)
        {
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __m128 _gamma0 = _mm_set1_ps(gamma_ptr[0]);
                __m128 _gamma1 = _mm_set1_ps(gamma_ptr[1]);
                __m128 _gamma2 = _mm_set1_ps(gamma_ptr[2]);
                __m128 _gamma3 = _mm_set1_ps(gamma_ptr[3]);
                __m256 _gamma01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_gamma0), _gamma1, 1);
                __m256 _gamma23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_gamma2), _gamma3, 1);
                __m512 _gamma = _mm512_insertf32x8(_mm512_castps256_ps512(_gamma01), _gamma23, 1);
                __m128 _beta0 = _mm_set1_ps(beta_ptr[0]);
                __m128 _beta1 = _mm_set1_ps(beta_ptr[1]);
                __m128 _beta2 = _mm_set1_ps(beta_ptr[2]);
                __m128 _beta3 = _mm_set1_ps(beta_ptr[3]);
                __m256 _beta01 = _mm256_insertf128_ps(_mm256_castps128_ps256(_beta0), _beta1, 1);
                __m256 _beta23 = _mm256_insertf128_ps(_mm256_castps128_ps256(_beta2), _beta3, 1);
                __m512 _beta = _mm512_insertf32x8(_mm512_castps256_ps512(_beta01), _beta23, 1);
                _p = _mm512_fmsub_ps(_p, _var_avx512, _mean_avx512);
                _p = _mm512_fmadd_ps(_p, _gamma, _beta);
                _mm512_storeu_ps(ptr, _p);
                ptr += 16;
                gamma_ptr += 4;
                beta_ptr += 4;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m128 _gamma0 = _mm_set1_ps(gamma_ptr[0]);
                __m128 _gamma1 = _mm_set1_ps(gamma_ptr[1]);
                __m256 _gamma = _mm256_insertf128_ps(_mm256_castps128_ps256(_gamma0), _gamma1, 1);
                __m128 _beta0 = _mm_set1_ps(beta_ptr[0]);
                __m128 _beta1 = _mm_set1_ps(beta_ptr[1]);
                __m256 _beta = _mm256_insertf128_ps(_mm256_castps128_ps256(_beta0), _beta1, 1);
                _p = _mm256_comp_fmsub_ps(_p, _var_avx, _mean_avx);
                _p = _mm256_comp_fmadd_ps(_p, _gamma, _beta);
                _mm256_storeu_ps(ptr, _p);
                ptr += 8;
                gamma_ptr += 2;
                beta_ptr += 2;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _gamma = _mm_set1_ps(gamma_ptr[0]);
                __m128 _beta = _mm_set1_ps(beta_ptr[0]);
                _p = _mm_comp_fmsub_ps(_p, _var, _mean);
                _p = _mm_comp_fmadd_ps(_p, _gamma, _beta);
                _mm_storeu_ps(ptr, _p);
                ptr += 4;
                gamma_ptr += 1;
                beta_ptr += 1;
            }
        }
        if (elempack == 1)
        {
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __m512 _gamma = _mm512_loadu_ps(gamma_ptr);
                __m512 _beta = _mm512_loadu_ps(beta_ptr);
                _p = _mm512_fmsub_ps(_p, _var_avx512, _mean_avx512);
                _p = _mm512_fmadd_ps(_p, _gamma, _beta);
                _mm512_storeu_ps(ptr, _p);
                ptr += 16;
                gamma_ptr += 16;
                beta_ptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _gamma = _mm256_loadu_ps(gamma_ptr);
                __m256 _beta = _mm256_loadu_ps(beta_ptr);
                _p = _mm256_comp_fmsub_ps(_p, _var_avx, _mean_avx);
                _p = _mm256_comp_fmadd_ps(_p, _gamma, _beta);
                _mm256_storeu_ps(ptr, _p);
                ptr += 8;
                gamma_ptr += 8;
                beta_ptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _gamma = _mm_loadu_ps(gamma_ptr);
                __m128 _beta = _mm_loadu_ps(beta_ptr);
                _p = _mm_comp_fmsub_ps(_p, _var, _mean);
                _p = _mm_comp_fmadd_ps(_p, _gamma, _beta);
                _mm_storeu_ps(ptr, _p);
                ptr += 4;
                gamma_ptr += 4;
                beta_ptr += 4;
            }
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            ptr[0] = (ptr[0] * var - mean) * gamma_ptr[0] + beta_ptr[0];
            ptr++;
            gamma_ptr++;
            beta_ptr++;
        }
    }
    else
    {
        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = _mm512_fmsub_ps(_p, _var_avx512, _mean_avx512);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = _mm256_comp_fmsub_ps(_p, _var_avx, _mean_avx);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _p = _mm_comp_fmsub_ps(_p, _var, _mean);
            _mm_storeu_ps(ptr, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * var - mean;
            ptr++;
        }
    }
}

int LayerNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;

    if (dims == 1)
    {
        // assert affine_size == w

        float* ptr = bottom_top_blob;
        layernorm(ptr, gamma_data, beta_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        // assert affine_size == w

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            layernorm(ptr, gamma_data, beta_data, eps, w, elempack);
        }
    }

    if (dims == 3)
    {
        if (affine_size == w)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.channel(q).row(i);
                    layernorm(ptr, gamma_data, beta_data, eps, w, elempack);
                }
            }
        }
        else // if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                layernorm(ptr, gamma_data, beta_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}

} // namespace ncnn
