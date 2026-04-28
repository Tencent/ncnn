// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cumulativesum_x86.h"

#include "x86_usability.h"

namespace ncnn {

void prefix_sum_row_avx2(float* ptr, int w)
{
    int j = 0;

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

    float sum = (j > 0) ? ptr[j - 1] : 0.0f;
    for (; j < w; j++)
    {
        sum += ptr[j];
        ptr[j] = sum;
    }
}

void cumulative_sum_add_avx2(const float* ptr, float* outptr, int size)
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

} // namespace ncnn
