// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "rmsnorm_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

RMSNorm_x86::RMSNorm_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

static void rmsnorm(float* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    const int size = elemcount * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _rms_avx512 = _mm512_set1_ps(0.f);
#endif // __AVX512F__
    __m256 _rms_avx = _mm256_set1_ps(0.f);
#endif // __AVX__
    __m128 _rms = _mm_set1_ps(0.f);
#endif // __SSE2__
    float rms = 0.f;
    {
        const float* ptr0 = ptr;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr0);
            _rms_avx512 = _mm512_fmadd_ps(_p, _p, _rms_avx512);
            ptr0 += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr0);
            _rms_avx = _mm256_comp_fmadd_ps(_p, _p, _rms_avx);
            ptr0 += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr0);
            _rms = _mm_comp_fmadd_ps(_p, _p, _rms);
            ptr0 += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            rms += ptr0[0] * ptr0[0];
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

        _rms_avx512 = _mm512_div_ps(_rms_avx512, _elemcount);
        _rms_avx512 = _mm512_add_ps(_rms_avx512, _eps);

        __m256 _rms0 = _mm256_rsqrt_ps(_mm512_extractf32x8_ps(_rms_avx512, 0));
        __m256 _rms1 = _mm256_rsqrt_ps(_mm512_extractf32x8_ps(_rms_avx512, 1));
        _rms_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_rms0), _rms1, 1);
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
#if __AVX512F__
        {
            __m256 _rms0 = _mm512_castps512_ps256(_rms_avx512);
            __m256 _rms1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_rms_avx512), 1));
            _rms_avx = _mm256_add_ps(_rms_avx, _rms0);
            _rms_avx = _mm256_add_ps(_rms_avx, _rms1);
        }
#endif // __AVX512F__

        __m256 _elemcount = _mm256_set1_ps((float)elemcount);
        __m256 _eps = _mm256_set1_ps(eps);

        _rms_avx = _mm256_div_ps(_rms_avx, _elemcount);
        _rms_avx = _mm256_add_ps(_rms_avx, _eps);

        _rms_avx = _mm256_rsqrt_ps(_rms_avx);
#if __AVX512F__
        _rms_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_rms_avx), _rms_avx, 1);
#endif // __AVX512F__
    }
#endif // __AVX__
    if (elempack == 4)
    {
#if __AVX__
#if __AVX512F__
        {
            __m256 _rms0 = _mm512_castps512_ps256(_rms_avx512);
            __m256 _rms1 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(_rms_avx512), 1));
            _rms_avx = _mm256_add_ps(_rms_avx, _rms0);
            _rms_avx = _mm256_add_ps(_rms_avx, _rms1);
        }
#endif // __AVX512F__
        {
            __m128 _rms0 = _mm256_castps256_ps128(_rms_avx);
            __m128 _rms1 = _mm256_extractf128_ps(_rms_avx, 1);
            _rms = _mm_add_ps(_rms, _rms0);
            _rms = _mm_add_ps(_rms, _rms1);
        }
#endif // __AVX__

        __m128 _elemcount = _mm_set1_ps((float)elemcount);
        __m128 _eps = _mm_set1_ps(eps);

        _rms = _mm_div_ps(_rms, _elemcount);
        _rms = _mm_add_ps(_rms, _eps);

        _rms = _mm_rsqrt_ps(_rms);
#if __AVX__
        _rms_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_rms), _rms, 1);
#if __AVX512F__
        _rms_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_rms_avx), _rms_avx, 1);
#endif // __AVX512F__
#endif // __AVX__
    }
#endif // __SSE2__
    if (elempack == 1)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        rms += _mm512_comp_reduce_add_ps(_rms_avx512);
#endif // __AVX512F__
        rms += _mm256_reduce_add_ps(_rms_avx);
#endif // __AVX__
        rms += _mm_reduce_add_ps(_rms);
#endif // __SSE2__

        rms = 1.f / sqrtf(rms / elemcount + eps);
#if __SSE2__
        _rms = _mm_set1_ps(rms);
#if __AVX__
        _rms_avx = _mm256_insertf128_ps(_mm256_castps128_ps256(_rms), _rms, 1);
#if __AVX512F__
        _rms_avx512 = _mm512_insertf32x8(_mm512_castps256_ps512(_rms_avx), _rms_avx, 1);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
    }

    if (gamma_ptr)
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
                _p = _mm512_mul_ps(_p, _rms_avx512);
                _p = _mm512_mul_ps(_p, _gamma);
                _mm512_storeu_ps(ptr, _p);
                ptr += 16;
                gamma_ptr += 1;
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
                _p = _mm512_mul_ps(_p, _rms_avx512);
                _p = _mm512_mul_ps(_p, _gamma);
                _mm512_storeu_ps(ptr, _p);
                ptr += 16;
                gamma_ptr += 2;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _gamma = _mm256_set1_ps(gamma_ptr[0]);
                _p = _mm256_mul_ps(_p, _rms_avx);
                _p = _mm256_mul_ps(_p, _gamma);
                _mm256_storeu_ps(ptr, _p);
                ptr += 8;
                gamma_ptr += 1;
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
                _p = _mm512_mul_ps(_p, _rms_avx512);
                _p = _mm512_mul_ps(_p, _gamma);
                _mm512_storeu_ps(ptr, _p);
                ptr += 16;
                gamma_ptr += 4;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m128 _gamma0 = _mm_set1_ps(gamma_ptr[0]);
                __m128 _gamma1 = _mm_set1_ps(gamma_ptr[1]);
                __m256 _gamma = _mm256_insertf128_ps(_mm256_castps128_ps256(_gamma0), _gamma1, 1);
                _p = _mm256_mul_ps(_p, _rms_avx);
                _p = _mm256_mul_ps(_p, _gamma);
                _mm256_storeu_ps(ptr, _p);
                ptr += 8;
                gamma_ptr += 2;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _gamma = _mm_set1_ps(gamma_ptr[0]);
                _p = _mm_mul_ps(_p, _rms);
                _p = _mm_mul_ps(_p, _gamma);
                _mm_storeu_ps(ptr, _p);
                ptr += 4;
                gamma_ptr += 1;
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
                _p = _mm512_mul_ps(_p, _rms_avx512);
                _p = _mm512_mul_ps(_p, _gamma);
                _mm512_storeu_ps(ptr, _p);
                ptr += 16;
                gamma_ptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _gamma = _mm256_loadu_ps(gamma_ptr);
                _p = _mm256_mul_ps(_p, _rms_avx);
                _p = _mm256_mul_ps(_p, _gamma);
                _mm256_storeu_ps(ptr, _p);
                ptr += 8;
                gamma_ptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _gamma = _mm_loadu_ps(gamma_ptr);
                _p = _mm_mul_ps(_p, _rms);
                _p = _mm_mul_ps(_p, _gamma);
                _mm_storeu_ps(ptr, _p);
                ptr += 4;
                gamma_ptr += 4;
            }
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            ptr[0] = (ptr[0] * rms) * gamma_ptr[0];
            ptr++;
            gamma_ptr++;
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
            _p = _mm512_mul_ps(_p, _rms_avx512);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = _mm256_mul_ps(_p, _rms_avx);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _p = _mm_mul_ps(_p, _rms);
            _mm_storeu_ps(ptr, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            ptr[0] = ptr[0] * rms;
            ptr++;
        }
    }
}

int RMSNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        // assert affine_size == w

        float* ptr = bottom_top_blob;
        rmsnorm(ptr, gamma_data, eps, w * elempack, 1);
    }

    if (dims == 2)
    {
        // assert affine_size == w

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            rmsnorm(ptr, gamma_data, eps, w, elempack);
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
                    rmsnorm(ptr, gamma_data, eps, w, elempack);
                }
            }
        }
        else // if (affine_size == w * h)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                rmsnorm(ptr, gamma_data, eps, w * h, elempack);
            }
        }
    }

    return 0;
}

} // namespace ncnn
