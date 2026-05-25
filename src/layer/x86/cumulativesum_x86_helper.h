// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
void cumulative_sum_prefix_sum_row_avx2(float* ptr, int w);
void cumulative_sum_add_avx2(const float* ptr, float* outptr, int size);
#endif

#if __SSE2__
static inline __m128 cumulative_sum_prefix_sum4_ps(__m128 v)
{
    __m128 t = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), 4));
    v = _mm_add_ps(v, t);
    t = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), 8));
    v = _mm_add_ps(v, t);
    return v;
}
#endif

#if __AVX2__
static void cumulative_sum_prefix_sum_row_avx2_impl(float* ptr, int w)
{
    int j = 0;
    float sum = 0.f;

    __m256 base = _mm256_setzero_ps();
    for (; j + 8 <= w; j += 8)
    {
        __m256 v = _mm256_loadu_ps(ptr + j);
        __m256 t = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(v), 4));
        v = _mm256_add_ps(v, t);
        t = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(v), 8));
        v = _mm256_add_ps(v, t);
        __m256 lo = _mm256_permute2f128_ps(v, v, 0x08);
        lo = _mm256_shuffle_ps(lo, lo, _MM_SHUFFLE(3, 3, 3, 3));
        v = _mm256_add_ps(v, lo);
        v = _mm256_add_ps(v, base);
        _mm256_storeu_ps(ptr + j, v);
        __m256 last = _mm256_permute2f128_ps(v, v, 0x11);
        base = _mm256_shuffle_ps(last, last, _MM_SHUFFLE(3, 3, 3, 3));
    }
    if (j > 0)
        sum = ptr[j - 1];

    for (; j < w; j++)
    {
        sum += ptr[j];
        ptr[j] = sum;
    }
}

static void cumulative_sum_add_avx2_impl(const float* ptr, float* outptr, int size)
{
    int i = 0;

    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr + i);
        __m256 _outp = _mm256_loadu_ps(outptr + i);
        _outp = _mm256_add_ps(_outp, _p);
        _mm256_storeu_ps(outptr + i, _outp);
    }

    for (; i < size; i++)
    {
        outptr[i] += ptr[i];
    }
}
#endif // __AVX2__

static void cumulative_sum_prefix_sum_row(float* ptr, int w)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        cumulative_sum_prefix_sum_row_avx2(ptr, w);
        return;
    }
#endif

    int j = 0;
    float sum = 0.f;

#if __AVX__
    for (; j + 8 <= w; j += 8)
    {
        __m256 v = _mm256_loadu_ps(ptr + j);
        __m128 v0 = cumulative_sum_prefix_sum4_ps(_mm256_castps256_ps128(v));
        __m128 v1 = cumulative_sum_prefix_sum4_ps(_mm256_extractf128_ps(v, 1));
        v1 = _mm_add_ps(v1, _mm_shuffle_ps(v0, v0, _MM_SHUFFLE(3, 3, 3, 3)));

        __m256 out = _mm256_castps128_ps256(v0);
        out = _mm256_insertf128_ps(out, v1, 1);
        out = _mm256_add_ps(out, _mm256_set1_ps(sum));
        _mm256_storeu_ps(ptr + j, out);
        sum = ptr[j + 7];
    }
#elif __SSE2__
    for (; j + 4 <= w; j += 4)
    {
        __m128 v = cumulative_sum_prefix_sum4_ps(_mm_loadu_ps(ptr + j));
        v = _mm_add_ps(v, _mm_set1_ps(sum));
        _mm_storeu_ps(ptr + j, v);
        sum = ptr[j + 3];
    }
#endif

    for (; j < w; j++)
    {
        sum += ptr[j];
        ptr[j] = sum;
    }
}

static void cumulative_sum_add(const float* ptr, float* outptr, int size)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        cumulative_sum_add_avx2(ptr, outptr, size);
        return;
    }
#endif

    int i = 0;

#if __AVX__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr + i);
        __m256 _outp = _mm256_loadu_ps(outptr + i);
        _outp = _mm256_add_ps(_outp, _p);
        _mm256_storeu_ps(outptr + i, _outp);
    }
#elif __SSE2__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr + i);
        __m128 _outp = _mm_loadu_ps(outptr + i);
        _outp = _mm_add_ps(_outp, _p);
        _mm_storeu_ps(outptr + i, _outp);
    }
#endif

    for (; i < size; i++)
    {
        outptr[i] += ptr[i];
    }
}
