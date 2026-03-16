// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void layernorm_bf16s_sse_avx512bf16(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack);
#endif

static void layernorm_bf16s_sse(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        layernorm_bf16s_sse_avx512bf16(ptr, gamma_ptr, beta_ptr, eps, elemcount, elempack);
        return;
    }
#endif

    const int size = elemcount * elempack;

    // convert bf16 -> fp32, accumulate mean
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _mean_avx512 = _mm512_setzero_ps();
#endif // __AVX512F__
    __m256 _mean_avx = _mm256_setzero_ps();
#endif // __AVX__
    __m128 _mean = _mm_setzero_ps();
#endif // __SSE2__
    float mean = 0.f;
    {
        const unsigned short* ptr0 = ptr;

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
            mean += bfloat16_to_float32(*ptr0++);
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
        _mean_avx512 = combine8x2_ps(_mean_avx, _mean_avx);
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
        _mean_avx = combine4x2_ps(_mean, _mean);
#if __AVX512F__
        _mean_avx512 = combine8x2_ps(_mean_avx, _mean_avx);
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
        _mean_avx = combine4x2_ps(_mean, _mean);
#if __AVX512F__
        _mean_avx512 = combine8x2_ps(_mean_avx, _mean_avx);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
    }

    // accumulate var
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _var_avx512 = _mm512_setzero_ps();
#endif // __AVX512F__
    __m256 _var_avx = _mm256_setzero_ps();
#endif // __AVX__
    __m128 _var = _mm_setzero_ps();
#endif // __SSE2__
    float var = 0.f;
    {
        const unsigned short* ptr0 = ptr;

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
            float v = bfloat16_to_float32(*ptr0++) - mean;
            var += v * v;
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
        _var_avx512 = combine8x2_ps(_var0, _var1);
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
        _var_avx512 = combine8x2_ps(_var_avx, _var_avx);
        _mean_avx512 = combine8x2_ps(_mean_avx, _mean_avx);
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
        _var_avx = combine4x2_ps(_var, _var);
        _mean_avx = combine4x2_ps(_mean, _mean);
#if __AVX512F__
        _var_avx512 = combine8x2_ps(_var_avx, _var_avx);
        _mean_avx512 = combine8x2_ps(_mean_avx, _mean_avx);
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
        _var_avx = combine4x2_ps(_var, _var);
        _mean_avx = combine4x2_ps(_mean, _mean);
#if __AVX512F__
        _var_avx512 = combine8x2_ps(_var_avx, _var_avx);
        _mean_avx512 = combine8x2_ps(_mean_avx, _mean_avx);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
    }

    // norm and store bf16
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
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                __m512 _gamma = _mm512_set1_ps(gamma_ptr[0]);
                __m512 _beta = _mm512_set1_ps(beta_ptr[0]);
                _p = _mm512_fmsub_ps(_p, _var_avx512, _mean_avx512);
                _p = _mm512_fmadd_ps(_p, _gamma, _beta);
                _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
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
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                __m256 _gamma0 = _mm256_set1_ps(gamma_ptr[0]);
                __m256 _gamma1 = _mm256_set1_ps(gamma_ptr[1]);
                __m512 _gamma = combine8x2_ps(_gamma0, _gamma1);
                __m256 _beta0 = _mm256_set1_ps(beta_ptr[0]);
                __m256 _beta1 = _mm256_set1_ps(beta_ptr[1]);
                __m512 _beta = combine8x2_ps(_beta0, _beta1);
                _p = _mm512_fmsub_ps(_p, _var_avx512, _mean_avx512);
                _p = _mm512_fmadd_ps(_p, _gamma, _beta);
                _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
                ptr += 16;
                gamma_ptr += 2;
                beta_ptr += 2;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                __m256 _gamma = _mm256_set1_ps(gamma_ptr[0]);
                __m256 _beta = _mm256_set1_ps(beta_ptr[0]);
                _p = _mm256_comp_fmsub_ps(_p, _var_avx, _mean_avx);
                _p = _mm256_comp_fmadd_ps(_p, _gamma, _beta);
                _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
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
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                __m128 _gamma0 = _mm_set1_ps(gamma_ptr[0]);
                __m128 _gamma1 = _mm_set1_ps(gamma_ptr[1]);
                __m128 _gamma2 = _mm_set1_ps(gamma_ptr[2]);
                __m128 _gamma3 = _mm_set1_ps(gamma_ptr[3]);
                __m512 _gamma = combine4x4_ps(_gamma0, _gamma1, _gamma2, _gamma3);
                __m128 _beta0 = _mm_set1_ps(beta_ptr[0]);
                __m128 _beta1 = _mm_set1_ps(beta_ptr[1]);
                __m128 _beta2 = _mm_set1_ps(beta_ptr[2]);
                __m128 _beta3 = _mm_set1_ps(beta_ptr[3]);
                __m512 _beta = combine4x4_ps(_beta0, _beta1, _beta2, _beta3);
                _p = _mm512_fmsub_ps(_p, _var_avx512, _mean_avx512);
                _p = _mm512_fmadd_ps(_p, _gamma, _beta);
                _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
                ptr += 16;
                gamma_ptr += 4;
                beta_ptr += 4;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                __m128 _gamma0 = _mm_set1_ps(gamma_ptr[0]);
                __m128 _gamma1 = _mm_set1_ps(gamma_ptr[1]);
                __m256 _gamma = combine4x2_ps(_gamma0, _gamma1);
                __m128 _beta0 = _mm_set1_ps(beta_ptr[0]);
                __m128 _beta1 = _mm_set1_ps(beta_ptr[1]);
                __m256 _beta = combine4x2_ps(_beta0, _beta1);
                _p = _mm256_comp_fmsub_ps(_p, _var_avx, _mean_avx);
                _p = _mm256_comp_fmadd_ps(_p, _gamma, _beta);
                _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
                ptr += 8;
                gamma_ptr += 2;
                beta_ptr += 2;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                __m128 _gamma = _mm_set1_ps(gamma_ptr[0]);
                __m128 _beta = _mm_set1_ps(beta_ptr[0]);
                _p = _mm_comp_fmsub_ps(_p, _var, _mean);
                _p = _mm_comp_fmadd_ps(_p, _gamma, _beta);
                _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
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
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                __m512 _gamma = _mm512_loadu_ps(gamma_ptr);
                __m512 _beta = _mm512_loadu_ps(beta_ptr);
                _p = _mm512_fmsub_ps(_p, _var_avx512, _mean_avx512);
                _p = _mm512_fmadd_ps(_p, _gamma, _beta);
                _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
                ptr += 16;
                gamma_ptr += 16;
                beta_ptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                __m256 _gamma = _mm256_loadu_ps(gamma_ptr);
                __m256 _beta = _mm256_loadu_ps(beta_ptr);
                _p = _mm256_comp_fmsub_ps(_p, _var_avx, _mean_avx);
                _p = _mm256_comp_fmadd_ps(_p, _gamma, _beta);
                _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
                ptr += 8;
                gamma_ptr += 8;
                beta_ptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                __m128 _gamma = _mm_loadu_ps(gamma_ptr);
                __m128 _beta = _mm_loadu_ps(beta_ptr);
                _p = _mm_comp_fmsub_ps(_p, _var, _mean);
                _p = _mm_comp_fmadd_ps(_p, _gamma, _beta);
                _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
                ptr += 4;
                gamma_ptr += 4;
                beta_ptr += 4;
            }
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16((bfloat16_to_float32(*ptr) * var - mean) * *gamma_ptr + *beta_ptr);
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
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            _p = _mm512_fmsub_ps(_p, _var_avx512, _mean_avx512);
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            _p = _mm256_comp_fmsub_ps(_p, _var_avx, _mean_avx);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            _p = _mm_comp_fmsub_ps(_p, _var, _mean);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * var - mean);
            ptr++;
        }
    }
}
