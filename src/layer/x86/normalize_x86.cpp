// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_x86.h"

#include <math.h>

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

Normalize_x86::Normalize_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

static NCNN_FORCEINLINE float normalize_x86_coeff(float ssum, float eps, int eps_mode)
{
    if (eps_mode == 0)
    {
        return 1.f / sqrtf(ssum + eps);
    }

    if (eps_mode == 1)
    {
        const float v = sqrtf(ssum);
        return 1.f / (v > eps ? v : eps);
    }

    return 1.f / sqrtf(ssum > eps ? ssum : eps);
}

static float normalize_x86_sumsq(const float* ptr, int size)
{
    float sum = 0.f;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sum_avx512 = _mm512_setzero_ps();
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        _sum_avx512 = _mm512_fmadd_ps(_p, _p, _sum_avx512);
        ptr += 16;
    }
    sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
    __m256 _sum_avx = _mm256_setzero_ps();
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        _sum_avx = _mm256_comp_fmadd_ps(_p, _p, _sum_avx);
        ptr += 8;
    }
    sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
    __m128 _sum = _mm_setzero_ps();
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _sum = _mm_comp_fmadd_ps(_p, _p, _sum);
        ptr += 4;
    }
    sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
    for (; i < size; i++)
    {
        sum += ptr[0] * ptr[0];
        ptr++;
    }

    return sum;
}

static void normalize_x86_mul_scalar(float* ptr, int size, float scale)
{
    int i = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _scale_avx512 = _mm512_set1_ps(scale);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        _mm512_storeu_ps(ptr, _mm512_mul_ps(_p, _scale_avx512));
        ptr += 16;
    }
#endif // __AVX512F__
    __m256 _scale_avx = _mm256_set1_ps(scale);
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        _mm256_storeu_ps(ptr, _mm256_mul_ps(_p, _scale_avx));
        ptr += 8;
    }
#endif // __AVX__
    __m128 _scale = _mm_set1_ps(scale);
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        _mm_storeu_ps(ptr, _mm_mul_ps(_p, _scale));
        ptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        ptr[0] *= scale;
        ptr++;
    }
}

static void normalize_x86_mul_pack(float* ptr, int spatial_size, const float* factors, int elempack)
{
    if (elempack == 1)
    {
        normalize_x86_mul_scalar(ptr, spatial_size, factors[0]);
        return;
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        const __m512 _factor = _mm512_loadu_ps(factors);

        for (int i = 0; i < spatial_size; i++)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _mm512_storeu_ps(ptr, _mm512_mul_ps(_p, _factor));
            ptr += 16;
        }

        return;
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        const __m256 _factor = _mm256_loadu_ps(factors);

        for (int i = 0; i < spatial_size; i++)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _mm256_storeu_ps(ptr, _mm256_mul_ps(_p, _factor));
            ptr += 8;
        }

        return;
    }
#endif // __AVX__
    if (elempack == 4)
    {
        const __m128 _factor = _mm_loadu_ps(factors);

        for (int i = 0; i < spatial_size; i++)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _mm_storeu_ps(ptr, _mm_mul_ps(_p, _factor));
            ptr += 4;
        }

        return;
    }
#endif // __SSE2__

    for (int i = 0; i < spatial_size; i++)
    {
        for (int k = 0; k < elempack; k++)
        {
            ptr[k] *= factors[k];
        }
        ptr += elempack;
    }
}

static void normalize_x86_sum_squares_lanewise(const float* ptr, float* sums, int spatial_size, int elempack)
{
    for (int i = 0; i < elempack; i++)
    {
        sums[i] = 0.f;
    }

    if (elempack == 1)
    {
        sums[0] = normalize_x86_sumsq(ptr, spatial_size);
        return;
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        __m512 _sum = _mm512_setzero_ps();

        for (int i = 0; i < spatial_size; i++)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _sum = _mm512_fmadd_ps(_p, _p, _sum);
            ptr += 16;
        }

        _mm512_storeu_ps(sums, _sum);
        return;
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        __m256 _sum = _mm256_setzero_ps();

        for (int i = 0; i < spatial_size; i++)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _sum = _mm256_comp_fmadd_ps(_p, _p, _sum);
            ptr += 8;
        }

        _mm256_storeu_ps(sums, _sum);
        return;
    }
#endif // __AVX__
    if (elempack == 4)
    {
        __m128 _sum = _mm_setzero_ps();

        for (int i = 0; i < spatial_size; i++)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _sum = _mm_comp_fmadd_ps(_p, _p, _sum);
            ptr += 4;
        }

        _mm_storeu_ps(sums, _sum);
        return;
    }
#endif // __SSE2__

    for (int i = 0; i < spatial_size; i++)
    {
        for (int k = 0; k < elempack; k++)
        {
            sums[k] += ptr[k] * ptr[k];
        }
        ptr += elempack;
    }
}

static float normalize_x86_reduce_sumsq_pack(const float* ptr, int elempack)
{
#if __SSE2__
#if __AVX__
#if __AVX512F__
    if (elempack == 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        return _mm512_comp_reduce_add_ps(_mm512_mul_ps(_p, _p));
    }
#endif // __AVX512F__
    if (elempack == 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        return _mm256_reduce_add_ps(_mm256_mul_ps(_p, _p));
    }
#endif // __AVX__
    if (elempack == 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        return _mm_reduce_add_ps(_mm_mul_ps(_p, _p));
    }
#endif // __SSE2__

    float sum = 0.f;
    for (int i = 0; i < elempack; i++)
    {
        sum += ptr[i] * ptr[i];
    }
    return sum;
}

static void normalize_x86_inplace_per_element(float* ptr, int size, float scale, float eps, int eps_mode)
{
    int i = 0;

#if __SSE2__
    const __m128 _scale = _mm_set1_ps(scale);
    const __m128 _eps = _mm_set1_ps(eps);
    const __m128 _one = _mm_set1_ps(1.f);
#if __AVX__
    const __m256 _scale_avx = _mm256_set1_ps(scale);
    const __m256 _eps_avx = _mm256_set1_ps(eps);
    const __m256 _one_avx = _mm256_set1_ps(1.f);
#if __AVX512F__
    const __m512 _scale_avx512 = _mm512_set1_ps(scale);
    const __m512 _eps_avx512 = _mm512_set1_ps(eps);
    const __m512 _one_avx512 = _mm512_set1_ps(1.f);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr);
        __m512 _sq = _mm512_mul_ps(_p, _p);
        __m512 _a;

        if (eps_mode == 0)
            _a = _mm512_div_ps(_one_avx512, _mm512_sqrt_ps(_mm512_add_ps(_sq, _eps_avx512)));
        else if (eps_mode == 1)
            _a = _mm512_div_ps(_one_avx512, _mm512_max_ps(_mm512_sqrt_ps(_sq), _eps_avx512));
        else
            _a = _mm512_div_ps(_one_avx512, _mm512_sqrt_ps(_mm512_max_ps(_sq, _eps_avx512)));

        _p = _mm512_mul_ps(_p, _a);
        _p = _mm512_mul_ps(_p, _scale_avx512);
        _mm512_storeu_ps(ptr, _p);
        ptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr);
        __m256 _sq = _mm256_mul_ps(_p, _p);
        __m256 _a;

        if (eps_mode == 0)
            _a = _mm256_div_ps(_one_avx, _mm256_sqrt_ps(_mm256_add_ps(_sq, _eps_avx)));
        else if (eps_mode == 1)
            _a = _mm256_div_ps(_one_avx, _mm256_max_ps(_mm256_sqrt_ps(_sq), _eps_avx));
        else
            _a = _mm256_div_ps(_one_avx, _mm256_sqrt_ps(_mm256_max_ps(_sq, _eps_avx)));

        _p = _mm256_mul_ps(_p, _a);
        _p = _mm256_mul_ps(_p, _scale_avx);
        _mm256_storeu_ps(ptr, _p);
        ptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr);
        __m128 _sq = _mm_mul_ps(_p, _p);
        __m128 _a;

        if (eps_mode == 0)
            _a = _mm_div_ps(_one, _mm_sqrt_ps(_mm_add_ps(_sq, _eps)));
        else if (eps_mode == 1)
            _a = _mm_div_ps(_one, _mm_max_ps(_mm_sqrt_ps(_sq), _eps));
        else
            _a = _mm_div_ps(_one, _mm_sqrt_ps(_mm_max_ps(_sq, _eps)));

        _p = _mm_mul_ps(_p, _a);
        _p = _mm_mul_ps(_p, _scale);
        _mm_storeu_ps(ptr, _p);
        ptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        float a = normalize_x86_coeff(ptr[0] * ptr[0], eps, eps_mode);
        ptr[0] = ptr[0] * a * scale;
        ptr++;
    }
}

int Normalize_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int spatial_size = w * h * d;

    if (dims == 1 || dims == 2)
    {
        float* ptr = bottom_top_blob;
        const int size = spatial_size * elempack;
        const float scale = scale_data[0];

        if (across_spatial)
        {
            const float a = normalize_x86_coeff(normalize_x86_sumsq(ptr, size), eps, eps_mode);
            normalize_x86_mul_scalar(ptr, size, a * scale);
            return 0;
        }

        if (across_channel)
        {
            normalize_x86_inplace_per_element(ptr, size, scale, eps, eps_mode);
            return 0;
        }

        return 0;
    }

    if (across_spatial && across_channel)
    {
        Mat square_sum_blob;
        square_sum_blob.create(channels, 4u, opt.workspace_allocator);
        if (square_sum_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            square_sum_blob[q] = normalize_x86_sumsq(bottom_top_blob.channel(q), spatial_size * elempack);
        }

        float ssum = 0.f;
        for (int q = 0; q < channels; q++)
        {
            ssum += square_sum_blob[q];
        }

        const float a = normalize_x86_coeff(ssum, eps, eps_mode);

        if (channel_shared)
        {
            const float scale = a * scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                normalize_x86_mul_scalar(bottom_top_blob.channel(q), spatial_size * elempack, scale);
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float factors[16];
                const float* scale_ptr = (const float*)scale_data + q * elempack;

                for (int i = 0; i < elempack; i++)
                {
                    factors[i] = a * scale_ptr[i];
                }

                normalize_x86_mul_pack(bottom_top_blob.channel(q), spatial_size, factors, elempack);
            }
        }

        return 0;
    }

    if (across_spatial && !across_channel)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float factors[16];
            float sums[16];
            float* ptr = bottom_top_blob.channel(q);

            normalize_x86_sum_squares_lanewise(ptr, sums, spatial_size, elempack);

            if (channel_shared)
            {
                for (int i = 0; i < elempack; i++)
                {
                    factors[i] = normalize_x86_coeff(sums[i], eps, eps_mode) * scale_data[0];
                }
            }
            else
            {
                const float* scale_ptr = (const float*)scale_data + q * elempack;

                for (int i = 0; i < elempack; i++)
                {
                    factors[i] = normalize_x86_coeff(sums[i], eps, eps_mode) * scale_ptr[i];
                }
            }

            normalize_x86_mul_pack(ptr, spatial_size, factors, elempack);
        }

        return 0;
    }

    if (!across_spatial && across_channel)
    {
        Mat coeffs_blob;
        coeffs_blob.create(spatial_size, 4u, opt.workspace_allocator);
        if (coeffs_blob.empty())
            return -100;

        float* coeffs = (float*)coeffs_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < spatial_size; i++)
        {
            float ssum = 0.f;
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = (const float*)bottom_top_blob.channel(q) + i * elempack;
                ssum += normalize_x86_reduce_sumsq_pack(ptr, elempack);
            }

            coeffs[i] = normalize_x86_coeff(ssum, eps, eps_mode);
        }

        if (channel_shared)
        {
            const float shared_scale = scale_data[0];

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                if (elempack == 1)
                {
                    for (int i = 0; i < spatial_size; i++)
                    {
                        ptr[i] *= coeffs[i] * shared_scale;
                    }
                    continue;
                }

#if __SSE2__
#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    for (int i = 0; i < spatial_size; i++)
                    {
                        __m512 _coeff = _mm512_set1_ps(coeffs[i] * shared_scale);
                        __m512 _p = _mm512_loadu_ps(ptr);
                        _mm512_storeu_ps(ptr, _mm512_mul_ps(_p, _coeff));
                        ptr += 16;
                    }
                    continue;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    for (int i = 0; i < spatial_size; i++)
                    {
                        __m256 _coeff = _mm256_set1_ps(coeffs[i] * shared_scale);
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _mm256_storeu_ps(ptr, _mm256_mul_ps(_p, _coeff));
                        ptr += 8;
                    }
                    continue;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    for (int i = 0; i < spatial_size; i++)
                    {
                        __m128 _coeff = _mm_set1_ps(coeffs[i] * shared_scale);
                        __m128 _p = _mm_loadu_ps(ptr);
                        _mm_storeu_ps(ptr, _mm_mul_ps(_p, _coeff));
                        ptr += 4;
                    }
                    continue;
                }
#endif // __SSE2__

                for (int i = 0; i < spatial_size; i++)
                {
                    const float factor = coeffs[i] * shared_scale;
                    for (int k = 0; k < elempack; k++)
                    {
                        ptr[k] *= factor;
                    }
                    ptr += elempack;
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                const float* scale_ptr = (const float*)scale_data + q * elempack;

                if (elempack == 1)
                {
                    const float scale = scale_ptr[0];
                    for (int i = 0; i < spatial_size; i++)
                    {
                        ptr[i] *= coeffs[i] * scale;
                    }
                    continue;
                }

#if __SSE2__
#if __AVX__
#if __AVX512F__
                if (elempack == 16)
                {
                    const __m512 _scale = _mm512_loadu_ps(scale_ptr);
                    for (int i = 0; i < spatial_size; i++)
                    {
                        __m512 _coeff = _mm512_set1_ps(coeffs[i]);
                        __m512 _factor = _mm512_mul_ps(_coeff, _scale);
                        __m512 _p = _mm512_loadu_ps(ptr);
                        _mm512_storeu_ps(ptr, _mm512_mul_ps(_p, _factor));
                        ptr += 16;
                    }
                    continue;
                }
#endif // __AVX512F__
                if (elempack == 8)
                {
                    const __m256 _scale = _mm256_loadu_ps(scale_ptr);
                    for (int i = 0; i < spatial_size; i++)
                    {
                        __m256 _coeff = _mm256_set1_ps(coeffs[i]);
                        __m256 _factor = _mm256_mul_ps(_coeff, _scale);
                        __m256 _p = _mm256_loadu_ps(ptr);
                        _mm256_storeu_ps(ptr, _mm256_mul_ps(_p, _factor));
                        ptr += 8;
                    }
                    continue;
                }
#endif // __AVX__
                if (elempack == 4)
                {
                    const __m128 _scale = _mm_loadu_ps(scale_ptr);
                    for (int i = 0; i < spatial_size; i++)
                    {
                        __m128 _coeff = _mm_set1_ps(coeffs[i]);
                        __m128 _factor = _mm_mul_ps(_coeff, _scale);
                        __m128 _p = _mm_loadu_ps(ptr);
                        _mm_storeu_ps(ptr, _mm_mul_ps(_p, _factor));
                        ptr += 4;
                    }
                    continue;
                }
#endif // __SSE2__

                for (int i = 0; i < spatial_size; i++)
                {
                    const float coeff = coeffs[i];
                    for (int k = 0; k < elempack; k++)
                    {
                        ptr[k] *= coeff * scale_ptr[k];
                    }
                    ptr += elempack;
                }
            }
        }

        return 0;
    }

    return 0;
}

} // namespace ncnn
