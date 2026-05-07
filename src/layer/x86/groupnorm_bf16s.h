// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void groupnorm_bf16s_sse_avx512bf16(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, int elempack, size_t cstep);
#endif

static void groupnorm_bf16s_sse(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, int elempack, size_t cstep)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        groupnorm_bf16s_sse_avx512bf16(ptr, gamma_ptr, beta_ptr, eps, channels, size, elempack, cstep);
        return;
    }
#endif

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
        const unsigned short* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr0));
            _mean_avx512 = _mm512_add_ps(_mean_avx512, _p);
            ptr0 += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr0));
            _mean_avx = _mm256_add_ps(_mean_avx, _p);
            ptr0 += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr0));
            _mean = _mm_add_ps(_mean, _p);
            ptr0 += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            mean += bfloat16_to_float32(*ptr0);
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
        const unsigned short* ptr0 = ptr + cstep * q * elempack;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr0));
            _p = _mm512_sub_ps(_p, _mean_avx512);
            _var_avx512 = _mm512_fmadd_ps(_p, _p, _var_avx512);
            ptr0 += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr0));
            _p = _mm256_sub_ps(_p, _mean_avx);
            _var_avx = _mm256_comp_fmadd_ps(_p, _p, _var_avx);
            ptr0 += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr0));
            _p = _mm_sub_ps(_p, _mean);
            _var = _mm_comp_fmadd_ps(_p, _p, _var);
            ptr0 += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr0) - mean;
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
            unsigned short* ptr0 = ptr + cstep * q * elempack;

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
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr0));
                _p = _mm512_fmsub_ps(_p, _a_avx512, _b_avx512);
                _mm256_storeu_si256((__m256i*)ptr0, float2bfloat_avx512(_p));
                ptr0 += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr0));
                _p = _mm256_comp_fmsub_ps(_p, _a_avx, _b_avx);
                _mm_storeu_si128((__m128i*)ptr0, float2bfloat_avx(_p));
                ptr0 += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr0));
                _p = _mm_comp_fmsub_ps(_p, _a, _b);
                _mm_storel_epi64((__m128i*)ptr0, float2bfloat_sse(_p, _p));
                ptr0 += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *ptr0 = float32_to_bfloat16(bfloat16_to_float32(*ptr0) * a - b);
                ptr0++;
            }
        }
    }
    else
    {
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr0 = ptr + cstep * q * elempack;

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr0));
                _p = _mm512_fmsub_ps(_p, _var_avx512, _mean_avx512);
                _mm256_storeu_si256((__m256i*)ptr0, float2bfloat_avx512(_p));
                ptr0 += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr0));
                _p = _mm256_comp_fmsub_ps(_p, _var_avx, _mean_avx);
                _mm_storeu_si128((__m128i*)ptr0, float2bfloat_avx(_p));
                ptr0 += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr0));
                _p = _mm_comp_fmsub_ps(_p, _var, _mean);
                _mm_storel_epi64((__m128i*)ptr0, float2bfloat_sse(_p, _p));
                ptr0 += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *ptr0 = float32_to_bfloat16(bfloat16_to_float32(*ptr0) * var - mean);
                ptr0++;
            }
        }
    }
}
