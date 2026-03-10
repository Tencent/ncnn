// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void softmax_bf16s_to_fp32_avx512bf16(const unsigned short* src, float* dst, int size);
void softmax_fp32_to_bf16s_avx512bf16(const float* src, unsigned short* dst, int size);
#endif

static void softmax_bf16s_to_fp32(const unsigned short* src, float* dst, int size)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        softmax_bf16s_to_fp32_avx512bf16(src, dst, size);
        return;
    }
#endif

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(src + i)));
        _mm512_storeu_ps(dst + i, _p);
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(src + i)));
        _mm256_storeu_ps(dst + i, _p);
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(src + i)));
        _mm_storeu_ps(dst + i, _p);
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        dst[i] = bfloat16_to_float32(src[i]);
    }
}

static void softmax_fp32_to_bf16s(const float* src, unsigned short* dst, int size)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        softmax_fp32_to_bf16s_avx512bf16(src, dst, size);
        return;
    }
#endif

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(src + i);
        _mm256_storeu_si256((__m256i*)(dst + i), float2bfloat_avx512(_p));
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(src + i);
        _mm_storeu_si128((__m128i*)(dst + i), float2bfloat_avx(_p));
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(src + i);
        _mm_storel_epi64((__m128i*)(dst + i), float2bfloat_sse(_p, _p));
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        dst[i] = float32_to_bfloat16(src[i]);
    }
}
