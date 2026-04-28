// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
int cumulative_sum_forward_inplace_avx2(Mat& bottom_top_blob, int axis, const Option& opt);
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

static void cumulative_sum_prefix_sum_row(float* ptr, int w)
{
    int j = 0;
    float sum = 0.f;

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
    if (j > 0)
        sum = ptr[j - 1];
#elif __AVX__
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

static int cumulative_sum_forward_inplace(Mat& bottom_top_blob, int axis, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__
    if (ncnn::cpu_support_x86_avx2())
    {
        return cumulative_sum_forward_inplace_avx2(bottom_top_blob, axis, opt);
    }
#endif

    const int dims = bottom_top_blob.dims;
    const int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1)
    {
        cumulative_sum_prefix_sum_row(bottom_top_blob, bottom_top_blob.w);
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
            cumulative_sum_add(prev_row, this_row, w);
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
            cumulative_sum_prefix_sum_row(bottom_top_blob.row(i), w);
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
            cumulative_sum_add(prev, cur, size);
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
                cumulative_sum_add(prev_row, this_row, w);
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
            cumulative_sum_prefix_sum_row(bottom_top_blob.channel(q).row(i), w);
        }
        return 0;
    }

    return -100;
}
