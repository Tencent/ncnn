// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void batchnorm_bf16s_sse_avx512bf16(unsigned short* ptr, const float* a, const float* b, int size, int elempack);
void batchnorm_bf16s_per_element_sse_avx512bf16(unsigned short* ptr, const float* a, const float* b, int size);
#endif

static void batchnorm_bf16s_sse(unsigned short* ptr, const float* a, const float* b, int size, int elempack)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        batchnorm_bf16s_sse_avx512bf16(ptr, a, b, size, elempack);
        return;
    }
#endif

    // Load a/b into SIMD registers with correct elempack broadcasting
#if __SSE2__
    __m128 _a128 = (elempack == 4) ? _mm_loadu_ps(a) : _mm_set1_ps(a[0]);
    __m128 _b128 = (elempack == 4) ? _mm_loadu_ps(b) : _mm_set1_ps(b[0]);
#if __AVX__
    __m256 _a256 = (elempack == 8) ? _mm256_loadu_ps(a) : combine4x2_ps(_a128, _a128);
    __m256 _b256 = (elempack == 8) ? _mm256_loadu_ps(b) : combine4x2_ps(_b128, _b128);
#if __AVX512F__
    __m512 _a512 = (elempack == 16) ? _mm512_loadu_ps(a) : combine8x2_ps(_a256, _a256);
    __m512 _b512 = (elempack == 16) ? _mm512_loadu_ps(b) : combine8x2_ps(_b256, _b256);
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
    float sa = a[0];
    float sb = b[0];

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
        _p = _mm512_fmadd_ps(_p, _b512, _a512);
        _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
        ptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
        _p = _mm256_comp_fmadd_ps(_p, _b256, _a256);
        _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
        ptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
        _p = _mm_comp_fmadd_ps(_p, _b128, _a128);
        _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
        ptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(sb * bfloat16_to_float32(*ptr) + sa);
        ptr++;
    }
}

static void batchnorm_bf16s_per_element_sse(unsigned short* ptr, const float* a, const float* b, int size)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        batchnorm_bf16s_per_element_sse_avx512bf16(ptr, a, b, size);
        return;
    }
#endif

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(ptr + i)));
        __m512 _a = _mm512_loadu_ps(a + i);
        __m512 _b = _mm512_loadu_ps(b + i);
        _p = _mm512_fmadd_ps(_p, _b, _a);
        _mm256_storeu_si256((__m256i*)(ptr + i), float2bfloat_avx512(_p));
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(ptr + i)));
        __m256 _a = _mm256_loadu_ps(a + i);
        __m256 _b = _mm256_loadu_ps(b + i);
        _p = _mm256_comp_fmadd_ps(_p, _b, _a);
        _mm_storeu_si128((__m128i*)(ptr + i), float2bfloat_avx(_p));
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(ptr + i)));
        __m128 _a = _mm_loadu_ps(a + i);
        __m128 _b = _mm_loadu_ps(b + i);
        _p = _mm_comp_fmadd_ps(_p, _b, _a);
        _mm_storel_epi64((__m128i*)(ptr + i), float2bfloat_sse(_p, _p));
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(b[i] * bfloat16_to_float32(ptr[i]) + a[i]);
    }
}
