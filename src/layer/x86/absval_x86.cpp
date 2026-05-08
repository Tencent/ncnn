// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "absval_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "cpu.h"

namespace ncnn {

AbsVal_x86::AbsVal_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
    support_fp16_storage = cpu_support_x86_f16c();
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int AbsVal_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    int elembits = bottom_top_blob.elembits();

    if (elembits == 16)
        return forward_inplace_bf16s_fp16s(bottom_top_blob, opt);

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
        __m512 _sign_mask_avx512 = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _mm512_storeu_ps(ptr, _mm512_and_ps(_p, _sign_mask_avx512));
            ptr += 16;
        }
        if (i < size)
        {
            const __mmask16 _mask = (__mmask16)((1u << (size - i)) - 1);
            __m512 _p = _mm512_maskz_loadu_ps(_mask, ptr);
            _mm512_mask_storeu_ps(ptr, _mask, _mm512_and_ps(_p, _sign_mask_avx512));
        }
#else // __AVX512F__
#if __SSE2__
#if __AVX__
        __m256 _sign_mask_avx = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _mm256_storeu_ps(ptr, _mm256_and_ps(_p, _sign_mask_avx));
            ptr += 8;
        }
#endif // __AVX__
        __m128 _sign_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            _mm_store_ps(ptr, _mm_and_ps(_p, _sign_mask));
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = *ptr > 0.f ? *ptr : -*ptr;
            ptr++;
        }
#endif // __AVX512F__
    }

    return 0;
}

int AbsVal_x86::forward_inplace_bf16s_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int d = bottom_top_blob.d;
    const int channels = bottom_top_blob.c;
    const int elempack = bottom_top_blob.elempack;
    const int size = w * h * d * elempack;

    // fp16/bf16 abs: sign bit is bit 15 for both formats.
    // Reinterpret pairs of 16-bit values as float and apply AND with
    // 0x7fff7fff to clear both sign bits in one 32-bit operation.
    // No fp32 round-trip required, no F16C instructions needed.

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __AVX512F__
        __m512i _sign_mask_avx512 = _mm512_set1_epi32(0x7fff7fff);
        for (; i + 31 < size; i += 32)
        {
            __m512i _p = _mm512_loadu_si512((const __m512i*)ptr);
            _mm512_storeu_si512((__m512i*)ptr, _mm512_and_si512(_p, _sign_mask_avx512));
            ptr += 32;
        }
        if (i < size)
        {
            const unsigned int remain = size - i;
            const __mmask16 _mask = (__mmask16)((1u << ((remain + 1) / 2)) - 1);
            __m512i _p = _mm512_maskz_loadu_epi32(_mask, (const __m512i*)ptr);
            _mm512_mask_storeu_epi32((__m512i*)ptr, _mask, _mm512_and_si512(_p, _sign_mask_avx512));
        }
#else // __AVX512F__
#if __SSE2__
#if __AVX__
        __m256 _sign_mask_avx = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff7fff));
        for (; i + 15 < size; i += 16)
        {
            __m256 _p = _mm256_castsi256_ps(_mm256_loadu_si256((const __m256i*)ptr));
            _mm256_storeu_si256((__m256i*)ptr, _mm256_castps_si256(_mm256_and_ps(_p, _sign_mask_avx)));
            ptr += 16;
        }
#endif // __AVX__
        __m128i _sign_mask = _mm_set1_epi32(0x7fff7fff);
        for (; i + 7 < size; i += 8)
        {
            __m128i _p = _mm_load_si128((const __m128i*)ptr);
            _mm_store_si128((__m128i*)ptr, _mm_and_si128(_p, _sign_mask));
            ptr += 8;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            *ptr = *ptr & 0x7fffu;
            ptr++;
        }
#endif // __AVX512F__
    }

    return 0;
}

} // namespace ncnn
