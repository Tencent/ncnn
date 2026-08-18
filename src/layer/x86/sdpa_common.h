// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static float sdpa_exp_submax_fp32(float* ptr, int size, float max)
{
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sum_avx512 = _mm512_setzero_ps();
    __m512 _max_avx512 = _mm512_set1_ps(max);
#endif // __AVX512F__
    __m256 _sum_avx = _mm256_setzero_ps();
    __m256 _max_avx = _mm256_set1_ps(max);
#endif // __AVX__
    __m128 _sum = _mm_setzero_ps();
    __m128 _max = _mm_set1_ps(max);
#endif // __SSE2__
    float sum = 0.f;

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr + i);
        _p = exp512_ps(_mm512_sub_ps(_p, _max_avx512));
        _mm512_storeu_ps(ptr + i, _p);
        _sum_avx512 = _mm512_add_ps(_sum_avx512, _p);
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr + i);
        _p = exp256_ps(_mm256_sub_ps(_p, _max_avx));
        _mm256_storeu_ps(ptr + i, _p);
        _sum_avx = _mm256_add_ps(_sum_avx, _p);
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr + i);
        _p = exp_ps(_mm_sub_ps(_p, _max));
        _mm_storeu_ps(ptr + i, _p);
        _sum = _mm_add_ps(_sum, _p);
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        ptr[i] = expf(ptr[i] - max);
        sum += ptr[i];
    }

#if __SSE2__
#if __AVX__
#if __AVX512F__
    sum += _mm512_comp_reduce_add_ps(_sum_avx512);
#endif // __AVX512F__
    sum += _mm256_reduce_add_ps(_sum_avx);
#endif // __AVX__
    sum += _mm_reduce_add_ps(_sum);
#endif // __SSE2__

    return sum;
}

static void sdpa_normalize_fp32(float* out, float sum, int size)
{
    float inv_sum = 1.f / sum;
    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _inv_sum_avx512 = _mm512_set1_ps(inv_sum);
    for (; i + 15 < size; i += 16)
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(out + i), _inv_sum_avx512));
#endif // __AVX512F__
    __m256 _inv_sum_avx = _mm256_set1_ps(inv_sum);
    for (; i + 7 < size; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), _inv_sum_avx));
#endif // __AVX__
    __m128 _inv_sum = _mm_set1_ps(inv_sum);
    for (; i + 3 < size; i += 4)
        _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(out + i), _inv_sum));
#endif // __SSE2__
    for (; i < size; i++)
        out[i] *= inv_sum;
}
