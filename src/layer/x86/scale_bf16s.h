// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void scale_bf16s_sse_avx512bf16(unsigned short* ptr, const float* scale, const float* bias, int size, int elempack);
void scale_bf16s_no_bias_sse_avx512bf16(unsigned short* ptr, const float* scale, int size, int elempack);
#endif

static void scale_bf16s_sse(unsigned short* ptr, const float* scale, const float* bias, int size, int elempack)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        scale_bf16s_sse_avx512bf16(ptr, scale, bias, size, elempack);
        return;
    }
#endif

#if __SSE2__
    __m128 _s128 = (elempack == 4) ? _mm_loadu_ps(scale) : _mm_set1_ps(scale[0]);
    __m128 _b128 = (elempack == 4) ? _mm_loadu_ps(bias) : _mm_set1_ps(bias[0]);
#if __AVX__
    __m256 _s256 = (elempack == 8) ? _mm256_loadu_ps(scale) : combine4x2_ps(_s128, _s128);
    __m256 _b256 = (elempack == 8) ? _mm256_loadu_ps(bias) : combine4x2_ps(_b128, _b128);
#if __AVX512F__
    __m512 _s512 = (elempack == 16) ? _mm512_loadu_ps(scale) : combine8x2_ps(_s256, _s256);
    __m512 _b512 = (elempack == 16) ? _mm512_loadu_ps(bias) : combine8x2_ps(_b256, _b256);
#endif
#endif
#endif
    float s = scale[0];
    float b = bias[0];

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
        _p = _mm512_fmadd_ps(_p, _s512, _b512);
        _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
        ptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
        _p = _mm256_comp_fmadd_ps(_p, _s256, _b256);
        _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
        ptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
        _p = _mm_comp_fmadd_ps(_p, _s128, _b128);
        _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
        ptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * s + b);
        ptr++;
    }
}

static void scale_bf16s_no_bias_sse(unsigned short* ptr, const float* scale, int size, int elempack)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        scale_bf16s_no_bias_sse_avx512bf16(ptr, scale, size, elempack);
        return;
    }
#endif

#if __SSE2__
    __m128 _s128 = (elempack == 4) ? _mm_loadu_ps(scale) : _mm_set1_ps(scale[0]);
#if __AVX__
    __m256 _s256 = (elempack == 8) ? _mm256_loadu_ps(scale) : combine4x2_ps(_s128, _s128);
#if __AVX512F__
    __m512 _s512 = (elempack == 16) ? _mm512_loadu_ps(scale) : combine8x2_ps(_s256, _s256);
#endif
#endif
#endif
    float s = scale[0];

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
        _p = _mm512_mul_ps(_p, _s512);
        _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
        ptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
        _p = _mm256_mul_ps(_p, _s256);
        _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
        ptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
        _p = _mm_mul_ps(_p, _s128);
        _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
        ptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * s);
        ptr++;
    }
}
