// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "groupnorm_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

namespace ncnn {

GroupNorm_x86::GroupNorm_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

static void groupnorm(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, int elempack, size_t cstep)
{
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
    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;

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

        mean = mean / (channels * size);
#if __SSE2__
        _mean = _mm_set1_ps(mean);
#if __AVX__
        _mean_avx = combine4x2_ps(_mean, _mean);
#if __AVX512F__
        _mean_avx512 = combine8x2_ps(_mean_avx, _mean_avx);
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
    for (int q = 0; q < channels; q++)
    {
        const float* ptr0 = ptr + cstep * q * elempack;

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

        var = 1.f / sqrtf(var / (channels * size) + eps);
        mean = mean * var;
#if __SSE2__
        _var = _mm_set1_ps(var);
        _mean = _mm_set1_ps(mean);
#if __AVX__
        _var_avx = combine4x2_ps(_var, _var);
        _mean_avx = combine4x2_ps(_mean, _mean);
#if __AVX512F__
        _var_avx512 = combine8x2_ps(_var_avx, _var_avx);
        _mean_avx512 = combine8x2_ps(_mean_avx, _mean_avx);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
    }

    // v = v * var - mean;
    // v = (v * var - mean) * gamma + beta
    //   = v * var * gamma - mean * gamma + beta
    //   = v * (var * gamma) - (mean * gamma - beta)

    if (gamma_ptr && beta_ptr)
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q * elempack;

#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _a_avx512 = _mm512_set1_ps(0.f);
            __m512 _b_avx512 = _mm512_set1_ps(0.f);
#endif // __AVX512F__
            __m256 _a_avx = _mm256_set1_ps(0.f);
            __m256 _b_avx = _mm256_set1_ps(0.f);
#endif // __AVX__
            __m128 _a = _mm_set1_ps(0.f);
            __m128 _b = _mm_set1_ps(0.f);
#endif // __SSE2__
            float a = 0.f;
            float b = 0.f;

#if __SSE2__
#if __AVX__
#if __AVX512F__
            if (elempack == 16)
            {
                __m512 _gamma = _mm512_loadu_ps(gamma_ptr + q * elempack);
                __m512 _beta = _mm512_loadu_ps(beta_ptr + q * elempack);

                _a_avx512 = _mm512_mul_ps(_var_avx512, _gamma);
                _b_avx512 = _mm512_fmsub_ps(_mean_avx512, _gamma, _beta);
            }
#endif // __AVX512F__
            if (elempack == 8)
            {
                __m256 _gamma = _mm256_loadu_ps(gamma_ptr + q * elempack);
                __m256 _beta = _mm256_loadu_ps(beta_ptr + q * elempack);

                _a_avx = _mm256_mul_ps(_var_avx, _gamma);
                _b_avx = _mm256_comp_fmsub_ps(_mean_avx, _gamma, _beta);
#if __AVX512F__
                _a_avx512 = combine8x2_ps(_a_avx, _a_avx);
                _b_avx512 = combine8x2_ps(_b_avx, _b_avx);
#endif // __AVX512F__
            }
#endif // __AVX__
            if (elempack == 4)
            {
                __m128 _gamma = _mm_loadu_ps(gamma_ptr + q * elempack);
                __m128 _beta = _mm_loadu_ps(beta_ptr + q * elempack);

                _a = _mm_mul_ps(_var, _gamma);
                _b = _mm_comp_fmsub_ps(_mean, _gamma, _beta);
#if __AVX__
                _a_avx = combine4x2_ps(_a, _a);
                _b_avx = combine4x2_ps(_b, _b);
#if __AVX512F__
                _a_avx512 = combine8x2_ps(_a_avx, _a_avx);
                _b_avx512 = combine8x2_ps(_b_avx, _b_avx);
#endif // __AVX512F__
#endif // __AVX__
            }
#endif // __SSE2__
            if (elempack == 1)
            {
                const float gamma = gamma_ptr[q];
                const float beta = beta_ptr[q];

                a = var * gamma;
                b = mean * gamma - beta;
#if __SSE2__
                _a = _mm_set1_ps(a);
                _b = _mm_set1_ps(b);
#if __AVX__
                _a_avx = combine4x2_ps(_a, _a);
                _b_avx = combine4x2_ps(_b, _b);
#if __AVX512F__
                _a_avx512 = combine8x2_ps(_a_avx, _a_avx);
                _b_avx512 = combine8x2_ps(_b_avx, _b_avx);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
            }

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr0);
                _p = _mm512_fmsub_ps(_p, _a_avx512, _b_avx512);
                _mm512_storeu_ps(ptr0, _p);
                ptr0 += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr0);
                _p = _mm256_comp_fmsub_ps(_p, _a_avx, _b_avx);
                _mm256_storeu_ps(ptr0, _p);
                ptr0 += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr0);
                _p = _mm_comp_fmsub_ps(_p, _a, _b);
                _mm_storeu_ps(ptr0, _p);
                ptr0 += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *ptr0 = *ptr0 * a - b;
                ptr0++;
            }
        }
    }
    else
    {
        for (int q = 0; q < channels; q++)
        {
            float* ptr0 = ptr + cstep * q * elempack;

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr0);
                _p = _mm512_fmsub_ps(_p, _var_avx512, _mean_avx512);
                _mm512_storeu_ps(ptr0, _p);
                ptr0 += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr0);
                _p = _mm256_comp_fmsub_ps(_p, _var_avx, _mean_avx);
                _mm256_storeu_ps(ptr0, _p);
                ptr0 += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr0);
                _p = _mm_comp_fmsub_ps(_p, _var, _mean);
                _mm_storeu_ps(ptr0, _p);
                ptr0 += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *ptr0 = *ptr0 * var - mean;
                ptr0++;
            }
        }
    }
}

int GroupNorm_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    const int channels_g = channels / group;

    int g_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        g_elempack = channels_g % 16 == 0 ? 16 : channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
#elif __AVX__
        g_elempack = channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
#else
        g_elempack = channels_g % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    Mat bottom_top_blob_unpacked = bottom_top_blob;
    if (elempack > g_elempack)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_top_blob, bottom_top_blob_unpacked, g_elempack, opt_p);
        if (bottom_top_blob_unpacked.empty())
            return -100;
    }

    if (dims == 1)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob_unpacked.range(g * channels_g / g_elempack, channels_g / g_elempack);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, 1 * g_elempack, g_elempack, 1);
        }
    }

    if (dims == 2)
    {
        const int w = bottom_top_blob_unpacked.w;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob_unpacked.row_range(g * channels_g / g_elempack, channels_g / g_elempack);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, w * g_elempack, g_elempack, w);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = bottom_top_blob_unpacked.w * bottom_top_blob_unpacked.h * bottom_top_blob_unpacked.d;
        const size_t cstep = bottom_top_blob_unpacked.cstep;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Mat bottom_top_blob_g = bottom_top_blob_unpacked.channel_range(g * channels_g / g_elempack, channels_g / g_elempack);
            const float* gamma_ptr = affine ? (const float*)gamma_data + g * channels_g : 0;
            const float* beta_ptr = affine ? (const float*)beta_data + g * channels_g : 0;
            groupnorm(bottom_top_blob_g, gamma_ptr, beta_ptr, eps, channels_g / g_elempack, size * g_elempack, g_elempack, cstep);
        }
    }

    if (g_elempack != elempack)
    {
        convert_packing(bottom_top_blob_unpacked, bottom_top_blob, elempack, opt);
    }

    return 0;
}

} // namespace ncnn
