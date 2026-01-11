// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "clip_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

namespace ncnn {

Clip_x86::Clip_x86()
{
#if __SSE2__
    support_packing = true;
    support_any_packing = true;
#endif // __SSE2__
}

int Clip_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __AVX512F__
        __m512 _min_avx512 = _mm512_set1_ps(min);
        __m512 _max_avx512 = _mm512_set1_ps(max);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _p = _mm512_max_ps(_p, _min_avx512);
            _p = _mm512_min_ps(_p, _max_avx512);
            _mm512_storeu_ps(ptr, _p);
            ptr += 16;
        }
        if (i < size)
        {
            const unsigned int remain = size - i;
            __mmask16 _mask = (__mmask16)((1u << remain) - 1);
            __m512 _p = _mm512_maskz_loadu_ps(_mask, ptr);
            _p = _mm512_max_ps(_p, _min_avx512);
            _p = _mm512_min_ps(_p, _max_avx512);
            _mm512_mask_storeu_ps(ptr, _mask, _p);
        }
#else // __AVX512F__
#if __SSE2__
#if __AVX__
        __m256 _min_avx = _mm256_set1_ps(min);
        __m256 _max_avx = _mm256_set1_ps(max);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _p = _mm256_max_ps(_p, _min_avx);
            _p = _mm256_min_ps(_p, _max_avx);
            _mm256_storeu_ps(ptr, _p);
            ptr += 8;
        }
#endif // __AVX__
        __m128 _min = _mm_set1_ps(min);
        __m128 _max = _mm_set1_ps(max);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            _p = _mm_max_ps(_p, _min);
            _p = _mm_min_ps(_p, _max);
            _mm_store_ps(ptr, _p);
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            if (*ptr < min)
                *ptr = min;
            if (*ptr > max)
                *ptr = max;
            ptr++;
        }
#endif // __AVX512F__
    }

    return 0;
}

} //namespace ncnn
