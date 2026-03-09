// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void rmsnorm_bf16s_sse_avx512bf16(unsigned short* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack);
#endif

static void rmsnorm_bf16s_sse(unsigned short* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        rmsnorm_bf16s_sse_avx512bf16(ptr, gamma_ptr, eps, elemcount, elempack);
        return;
    }
#endif

    const int size = elemcount * elempack;

    // accumulate rms
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _rms_avx512 = _mm512_setzero_ps();
#endif // __AVX512F__
    __m256 _rms_avx = _mm256_setzero_ps();
#endif // __AVX__
    __m128 _rms = _mm_setzero_ps();
#endif // __SSE2__
    float rms = 0.f;
    {
        const unsigned short* ptr0 = ptr;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr0));
            _rms_avx512 = _mm512_fmadd_ps(_p, _p, _rms_avx512);
            ptr0 += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr0));
            _rms_avx = _mm256_comp_fmadd_ps(_p, _p, _rms_avx);
            ptr0 += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr0));
            _rms = _mm_comp_fmadd_ps(_p, _p, _rms);
            ptr0 += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr0++);
            rms += v * v;
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
        _rms_avx512 = combine8x2_ps(_rms0, _rms1);
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
        _rms_avx512 = combine8x2_ps(_rms_avx, _rms_avx);
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
        _rms_avx = combine4x2_ps(_rms, _rms);
#if __AVX512F__
        _rms_avx512 = combine8x2_ps(_rms_avx, _rms_avx);
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
        _rms_avx = combine4x2_ps(_rms, _rms);
#if __AVX512F__
        _rms_avx512 = combine8x2_ps(_rms_avx, _rms_avx);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
    }

    // normalize and store bf16
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
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                __m512 _gamma = _mm512_set1_ps(gamma_ptr[0]);
                _p = _mm512_mul_ps(_p, _rms_avx512);
                _p = _mm512_mul_ps(_p, _gamma);
                _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
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
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                __m256 _gamma0 = _mm256_set1_ps(gamma_ptr[0]);
                __m256 _gamma1 = _mm256_set1_ps(gamma_ptr[1]);
                __m512 _gamma = combine8x2_ps(_gamma0, _gamma1);
                _p = _mm512_mul_ps(_p, _rms_avx512);
                _p = _mm512_mul_ps(_p, _gamma);
                _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
                ptr += 16;
                gamma_ptr += 2;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                __m256 _gamma = _mm256_set1_ps(gamma_ptr[0]);
                _p = _mm256_mul_ps(_p, _rms_avx);
                _p = _mm256_mul_ps(_p, _gamma);
                _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
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
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                __m128 _gamma0 = _mm_set1_ps(gamma_ptr[0]);
                __m128 _gamma1 = _mm_set1_ps(gamma_ptr[1]);
                __m128 _gamma2 = _mm_set1_ps(gamma_ptr[2]);
                __m128 _gamma3 = _mm_set1_ps(gamma_ptr[3]);
                __m512 _gamma = combine4x4_ps(_gamma0, _gamma1, _gamma2, _gamma3);
                _p = _mm512_mul_ps(_p, _rms_avx512);
                _p = _mm512_mul_ps(_p, _gamma);
                _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
                ptr += 16;
                gamma_ptr += 4;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                __m128 _gamma0 = _mm_set1_ps(gamma_ptr[0]);
                __m128 _gamma1 = _mm_set1_ps(gamma_ptr[1]);
                __m256 _gamma = combine4x2_ps(_gamma0, _gamma1);
                _p = _mm256_mul_ps(_p, _rms_avx);
                _p = _mm256_mul_ps(_p, _gamma);
                _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
                ptr += 8;
                gamma_ptr += 2;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                __m128 _gamma = _mm_set1_ps(gamma_ptr[0]);
                _p = _mm_mul_ps(_p, _rms);
                _p = _mm_mul_ps(_p, _gamma);
                _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
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
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                __m512 _gamma = _mm512_loadu_ps(gamma_ptr);
                _p = _mm512_mul_ps(_p, _rms_avx512);
                _p = _mm512_mul_ps(_p, _gamma);
                _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
                ptr += 16;
                gamma_ptr += 16;
            }
#endif // __AVX512F__
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                __m256 _gamma = _mm256_loadu_ps(gamma_ptr);
                _p = _mm256_mul_ps(_p, _rms_avx);
                _p = _mm256_mul_ps(_p, _gamma);
                _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
                ptr += 8;
                gamma_ptr += 8;
            }
#endif // __AVX__
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                __m128 _gamma = _mm_loadu_ps(gamma_ptr);
                _p = _mm_mul_ps(_p, _rms);
                _p = _mm_mul_ps(_p, _gamma);
                _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
                ptr += 4;
                gamma_ptr += 4;
            }
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * rms * *gamma_ptr);
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
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            _p = _mm512_mul_ps(_p, _rms_avx512);
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
            ptr += 16;
        }
#endif // __AVX512F__
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            _p = _mm256_mul_ps(_p, _rms_avx);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
            ptr += 8;
        }
#endif // __AVX__
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            _p = _mm_mul_ps(_p, _rms);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * rms);
            ptr++;
        }
    }
}
