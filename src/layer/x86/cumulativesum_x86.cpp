// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cumulativesum_x86.h"

#include "x86_usability.h"

namespace ncnn {

CumulativeSum_x86::CumulativeSum_x86()
{
}

// in-place Kogge-Stone prefix sum over a contiguous row of w floats.
// AVX2 8-lane when available, SSE2 4-lane fallback, scalar tail.
static inline void prefix_sum_row(float* ptr, int w)
{
    int j = 0;

#if __AVX2__
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
#elif __SSE2__
    __m128 base = _mm_setzero_ps();
    for (; j + 4 <= w; j += 4)
    {
        __m128 v = _mm_loadu_ps(ptr + j);
        __m128 t = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), 4));
        v = _mm_add_ps(v, t);
        t = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), 8));
        v = _mm_add_ps(v, t);
        v = _mm_add_ps(v, base);
        _mm_storeu_ps(ptr + j, v);
        base = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));
    }
#endif

    float sum = (j > 0) ? ptr[j - 1] : 0.0f;
    for (; j < w; j++)
    {
        sum += ptr[j];
        ptr[j] = sum;
    }
}

int CumulativeSum_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    if (bottom_top_blob.elembits() != 32 || bottom_top_blob.elempack != 1)
        return CumulativeSum::forward_inplace(bottom_top_blob, opt);

    const int dims = bottom_top_blob.dims;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    // only specialize inner-dim scans that carry a x[k] = x[k] + x[k-1] dependency;
    // outer-dim scans are already auto-vectorized by the compiler.
    if (dims == 1)
    {
        prefix_sum_row(bottom_top_blob, bottom_top_blob.w);
        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            prefix_sum_row(bottom_top_blob.row(i), w);
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 2)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int c = bottom_top_blob.c;
        #pragma omp parallel for collapse(2) num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            for (int i = 0; i < h; i++)
            {
                prefix_sum_row(bottom_top_blob.channel(q).row(i), w);
            }
        }
        return 0;
    }

    return CumulativeSum::forward_inplace(bottom_top_blob, opt);
}

} // namespace ncnn
