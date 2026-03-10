// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void instancenorm_bf16s_sse_avx512bf16(unsigned short* ptr, int size, float a, float b);
void instancenorm_bf16s_compute_mean_var_avx512bf16(const unsigned short* ptr, int size, float& mean, float& var);
#endif

static void instancenorm_bf16s_sse(unsigned short* ptr, int size, float a, float b)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        instancenorm_bf16s_sse_avx512bf16(ptr, size, a, b);
        return;
    }
#endif

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _a_avx512 = _mm512_set1_ps(a);
    __m512 _b_avx512 = _mm512_set1_ps(b);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
        _p = _mm512_fmadd_ps(_p, _a_avx512, _b_avx512);
        _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
        ptr += 16;
    }
#endif // __AVX512F__
    __m256 _a_avx = _mm256_set1_ps(a);
    __m256 _b_avx = _mm256_set1_ps(b);
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
        _p = _mm256_comp_fmadd_ps(_p, _a_avx, _b_avx);
        _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
        ptr += 8;
    }
#endif // __AVX__
    __m128 _a = _mm_set1_ps(a);
    __m128 _b = _mm_set1_ps(b);
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
        _p = _mm_comp_fmadd_ps(_p, _a, _b);
        _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
        ptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * a + b);
        ptr++;
    }
}

static void instancenorm_bf16s_compute_mean_var(const unsigned short* ptr, int size, float& mean, float& var)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        instancenorm_bf16s_compute_mean_var_avx512bf16(ptr, size, mean, var);
        return;
    }
#endif

    float sum = 0.f;
    float sqsum = 0.f;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sum_avx512 = _mm512_setzero_ps();
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
        ptr += 16;
    }
    sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
    __m256 _sum_avx = _mm256_setzero_ps();
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
        _sum_avx = _mm256_add_ps(_sum_avx, _p);
        ptr += 8;
    }
    sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
    __m128 _sum = _mm_setzero_ps();
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
        _sum = _mm_add_ps(_sum, _p);
        ptr += 4;
    }
    sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__
    for (; i < size; i++)
    {
        sum += bfloat16_to_float32(*ptr);
        ptr++;
    }

    mean = sum / size;

    ptr -= size;
    i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sqsum_avx512 = _mm512_setzero_ps();
    __m512 _mean_avx512 = _mm512_set1_ps(mean);
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
        __m512 _diff = _mm512_sub_ps(_p, _mean_avx512);
        _sqsum_avx512 = _mm512_fmadd_ps(_diff, _diff, _sqsum_avx512);
        ptr += 16;
    }
    sqsum += _mm512_comp_reduce_add_ps(_sqsum_avx512);
#endif // __AVX512F__
    __m256 _sqsum_avx = _mm256_setzero_ps();
    __m256 _mean_avx = _mm256_set1_ps(mean);
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
        __m256 _diff = _mm256_sub_ps(_p, _mean_avx);
        _sqsum_avx = _mm256_comp_fmadd_ps(_diff, _diff, _sqsum_avx);
        ptr += 8;
    }
    sqsum += _mm256_reduce_add_ps(_sqsum_avx);
#endif // __AVX__
    __m128 _sqsum = _mm_setzero_ps();
    __m128 _mean_sse = _mm_set1_ps(mean);
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
        __m128 _diff = _mm_sub_ps(_p, _mean_sse);
        _sqsum = _mm_comp_fmadd_ps(_diff, _diff, _sqsum);
        ptr += 4;
    }
    sqsum += _mm_reduce_add_ps(_sqsum);
#endif // __SSE2__
    for (; i < size; i++)
    {
        float v = bfloat16_to_float32(*ptr) - mean;
        sqsum += v * v;
        ptr++;
    }

    var = sqsum / size;
}
