// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
void cumulative_sum_avx2(float* ptr, int w);
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

#if __SSE2__
#if __AVX__
#if __AVX512F__
    __m512 _sum_avx512 = _mm512_setzero_ps();

    for (; j + 15 < w; j += 16)
    {
        __m512 _p = _mm512_loadu_ps(ptr + j);

        // prefix within each 128-bit quarter
        __m512 _t = _mm512_maskz_shuffle_ps(
                        (__mmask16)0xeeee, _p, _p, _MM_SHUFFLE(2, 1, 0, 0));
        _p = _mm512_add_ps(_p, _t);

        _t = _mm512_maskz_shuffle_ps(
                 (__mmask16)0xcccc, _p, _p, _MM_SHUFFLE(1, 0, 0, 0));
        _p = _mm512_add_ps(_p, _t);

        // propagate quarter prefix
        __m512 _qsum = _mm512_shuffle_ps(_p, _p, _MM_SHUFFLE(3, 3, 3, 3));

        _t = _mm512_maskz_shuffle_f32x4(
                 (__mmask16)0xfff0, _qsum, _qsum, _MM_SHUFFLE(2, 1, 0, 0));
        _p = _mm512_add_ps(_p, _t);

        _qsum = _mm512_shuffle_ps(_p, _p, _MM_SHUFFLE(3, 3, 3, 3));

        _t = _mm512_maskz_shuffle_f32x4(
                 (__mmask16)0xff00, _qsum, _qsum, _MM_SHUFFLE(1, 0, 0, 0));
        _p = _mm512_add_ps(_p, _t);

        _p = _mm512_add_ps(_p, _sum_avx512);
        _mm512_storeu_ps(ptr + j, _p);

        __m512 _last = _mm512_shuffle_f32x4(_p, _p, _MM_SHUFFLE(3, 3, 3, 3));
        _sum_avx512 = _mm512_shuffle_ps(_last, _last, _MM_SHUFFLE(3, 3, 3, 3));
    }

    if (j > 0)
        sum = ptr[j - 1];
#endif // __AVX512F__

    __m256 _sum_avx = _mm256_set1_ps(sum);

    for (; j + 7 < w; j += 8)
    {
        __m256 _p = _mm256_loadu_ps(ptr + j);

#if __AVX2__
        __m256 _t = _mm256_castsi256_ps(
                        _mm256_slli_si256(_mm256_castps_si256(_p), 4));
        _p = _mm256_add_ps(_p, _t);

        _t = _mm256_castsi256_ps(
                 _mm256_slli_si256(_mm256_castps_si256(_p), 8));
        _p = _mm256_add_ps(_p, _t);

        __m256 _lo = _mm256_permute2f128_ps(_p, _p, 0x08);
        _lo = _mm256_shuffle_ps(_lo, _lo, _MM_SHUFFLE(3, 3, 3, 3));
        _p = _mm256_add_ps(_p, _lo);
#else
        __m128 _p0 = _mm256_castps256_ps128(_p);
        __m128 _p1 = _mm256_extractf128_ps(_p, 1);

        __m128 _t0 = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(_p0), 4));
        _p0 = _mm_add_ps(_p0, _t0);
        _t0 = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(_p0), 8));
        _p0 = _mm_add_ps(_p0, _t0);

        __m128 _t1 = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(_p1), 4));
        _p1 = _mm_add_ps(_p1, _t1);
        _t1 = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(_p1), 8));
        _p1 = _mm_add_ps(_p1, _t1);

        _p1 = _mm_add_ps(_p1, _mm_shuffle_ps(_p0, _p0, _MM_SHUFFLE(3, 3, 3, 3)));

        _p = _mm256_insertf128_ps(_mm256_castps128_ps256(_p0), _p1, 1);
#endif

        _p = _mm256_add_ps(_p, _sum_avx);
        _mm256_storeu_ps(ptr + j, _p);

        __m256 _last = _mm256_permute2f128_ps(_p, _p, 0x11);
        _sum_avx = _mm256_shuffle_ps(_last, _last, _MM_SHUFFLE(3, 3, 3, 3));
    }

    if (j > 0)
        sum = ptr[j - 1];
#endif // __AVX__

    __m128 _sum_sse = _mm_set1_ps(sum);

    for (; j + 3 < w; j += 4)
    {
        __m128 _p = _mm_loadu_ps(ptr + j);

        __m128 _t = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(_p), 4));
        _p = _mm_add_ps(_p, _t);

        _t = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(_p), 8));
        _p = _mm_add_ps(_p, _t);

        _p = _mm_add_ps(_p, _sum_sse);
        _mm_storeu_ps(ptr + j, _p);

        _sum_sse = _mm_shuffle_ps(_p, _p, _MM_SHUFFLE(3, 3, 3, 3));
    }

    if (j > 0)
        sum = ptr[j - 1];
#endif // __SSE2__

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
