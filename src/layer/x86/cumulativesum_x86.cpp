// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cumulativesum_x86.h"

#include "x86_usability.h"

#include "cpu.h"

namespace ncnn {

#if __SSE2__
static inline __m128 prefix_sum4_ps(__m128 v)
{
    __m128 t = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), 4));
    v = _mm_add_ps(v, t);
    t = _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(v), 8));
    v = _mm_add_ps(v, t);
    return v;
}
#endif

CumulativeSum_x86::CumulativeSum_x86()
{
}

static inline void prefix_sum_row(float* ptr, int w)
{
    int j = 0;
    float sum = 0.f;

#if __AVX__
    for (; j + 8 <= w; j += 8)
    {
        __m256 v = _mm256_loadu_ps(ptr + j);
        __m128 v0 = prefix_sum4_ps(_mm256_castps256_ps128(v));
        __m128 v1 = prefix_sum4_ps(_mm256_extractf128_ps(v, 1));
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
        __m128 v = prefix_sum4_ps(_mm_loadu_ps(ptr + j));
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

static inline void cumulative_sum_add(const float* ptr, float* outptr, int size)
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

int CumulativeSum_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int dims = bottom_top_blob.dims;
    const int positive_axis = axis < 0 ? dims + axis : axis;

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__
    void (*prefix_sum_row_impl)(float*, int) = prefix_sum_row;
    void (*cumulative_sum_add_impl)(const float*, float*, int) = cumulative_sum_add;
    if (ncnn::cpu_support_x86_avx2())
    {
        prefix_sum_row_impl = prefix_sum_row_avx2;
        cumulative_sum_add_impl = cumulative_sum_add_avx2;
    }
#else
    void (*prefix_sum_row_impl)(float*, int) = prefix_sum_row;
    void (*cumulative_sum_add_impl)(const float*, float*, int) = cumulative_sum_add;
#endif

    if (dims == 1)
    {
        prefix_sum_row_impl(bottom_top_blob, bottom_top_blob.w);
        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        for (int i = 1; i < h; i++)
        {
            const float* prev_row = bottom_top_blob.row(i - 1);
            float* this_row = bottom_top_blob.row(i);
            cumulative_sum_add_impl(prev_row, this_row, w);
        }
        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            prefix_sum_row_impl(bottom_top_blob.row(i), w);
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 0)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int c = bottom_top_blob.c;
        const int size = w * h;

        for (int q = 1; q < c; q++)
        {
            const float* prev = bottom_top_blob.channel(q - 1);
            float* cur = bottom_top_blob.channel(q);
            cumulative_sum_add_impl(prev, cur, size);
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 1)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int c = bottom_top_blob.c;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            Mat this_channel = bottom_top_blob.channel(q);
            for (int i = 1; i < h; i++)
            {
                const float* prev_row = this_channel.row(i - 1);
                float* this_row = this_channel.row(i);
                cumulative_sum_add_impl(prev_row, this_row, w);
            }
        }
        return 0;
    }

    if (dims == 3 && positive_axis == 2)
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int c = bottom_top_blob.c;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int idx = 0; idx < c * h; idx++)
        {
            const int q = idx / h;
            const int i = idx - q * h;
            prefix_sum_row_impl(bottom_top_blob.channel(q).row(i), w);
        }
        return 0;
    }

    return -100;
}

} // namespace ncnn
