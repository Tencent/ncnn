// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
void cumulative_sum_avx2(float* ptr, int w);
#endif

#if __SSE2__
static inline __m128 cumulative_sum4_ps(__m128 _p)
{
    __m128 _t = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(_p), 4));
    _p = _mm_add_ps(_p, _t);
    _t = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(_p), 8));
    _p = _mm_add_ps(_p, _t);
    return _p;
}
#endif

static void cumulative_sum(float* ptr, int w)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        cumulative_sum_avx2(ptr, w);
        return;
    }
#endif

    int j = 0;
    float sum = 0.f;

#if __AVX2__
    __m256 _sum = _mm256_setzero_ps();
    for (; j + 8 <= w; j += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr + j);
        __m256 _t = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(_p), 4));
        _p = _mm256_add_ps(_p, _t);
        _t = _mm256_castsi256_ps(_mm256_slli_si256(_mm256_castps_si256(_p), 8));
        _p = _mm256_add_ps(_p, _t);
        __m256 _lo = _mm256_permute2f128_ps(_p, _p, 0x08);
        _lo = _mm256_shuffle_ps(_lo, _lo, _MM_SHUFFLE(3, 3, 3, 3));
        _p = _mm256_add_ps(_p, _lo);
        _p = _mm256_add_ps(_p, _sum);
        _mm256_storeu_ps(ptr + j, _p);
        __m256 _last = _mm256_permute2f128_ps(_p, _p, 0x11);
        _sum = _mm256_shuffle_ps(_last, _last, _MM_SHUFFLE(3, 3, 3, 3));
    }
    if (j > 0)
        sum = ptr[j - 1];
#elif __AVX__
    for (; j + 8 <= w; j += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr + j);
        __m128 _p0 = cumulative_sum4_ps(_mm256_castps256_ps128(_p));
        __m128 _p1 = cumulative_sum4_ps(_mm256_extractf128_ps(_p, 1));
        _p1 = _mm_add_ps(_p1, _mm_shuffle_ps(_p0, _p0, _MM_SHUFFLE(3, 3, 3, 3)));

        __m256 _out = _mm256_castps128_ps256(_p0);
        _out = _mm256_insertf128_ps(_out, _p1, 1);
        _out = _mm256_add_ps(_out, _mm256_set1_ps(sum));
        _mm256_storeu_ps(ptr + j, _out);
        sum = ptr[j + 7];
    }
#elif __SSE2__
    for (; j + 4 <= w; j += 4)
    {
        __m128 _p = cumulative_sum4_ps(_mm_loadu_ps(ptr + j));
        _p = _mm_add_ps(_p, _mm_set1_ps(sum));
        _mm_storeu_ps(ptr + j, _p);
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
