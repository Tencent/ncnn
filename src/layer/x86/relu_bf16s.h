// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

static int relu_bf16s(Mat& a, float slope, const Option& opt)
{
    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = a.channel(q);

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _zero_avx512 = _mm512_setzero_ps();
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                _p = _mm512_max_ps(_p, _zero_avx512);
                _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
                ptr += 16;
            }
            if (i < size)
            {
                const unsigned int remain = size - i;
                __mmask16 _mask = (__mmask16)((1u << remain) - 1);
                __m512 _p = bfloat2float_avx512(_mm256_maskz_loadu_epi16(_mask, ptr));
                _p = _mm512_max_ps(_p, _zero_avx512);
                _mm256_mask_storeu_epi16(ptr, _mask, float2bfloat_avx512(_p));
                i += remain;
            }
#else  // __AVX512F__
            __m256 _zero_avx = _mm256_setzero_ps();
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                _p = _mm256_max_ps(_p, _zero_avx);
                _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
                ptr += 8;
            }
            __m128 _zero = _mm_setzero_ps();
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                _p = _mm_max_ps(_p, _zero);
                _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
                ptr += 4;
            }
#endif // __AVX512F__
#else  // __AVX__
            __m128 _zero = _mm_setzero_ps();
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                _p = _mm_max_ps(_p, _zero);
                _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
                ptr += 4;
            }
#endif // __AVX__
#endif // __SSE2__
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0.f) v = 0.f;
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = a.channel(q);

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _zero_avx512 = _mm512_setzero_ps();
            __m512 _slope_avx512 = _mm512_set1_ps(slope);
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                __mmask16 _is_negative = _mm512_cmp_ps_mask(_p, _zero_avx512, _CMP_LT_OQ);
                _p = _mm512_mask_mul_ps(_p, _is_negative, _p, _slope_avx512);
                _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
                ptr += 16;
            }
            if (i < size)
            {
                const unsigned int remain = size - i;
                __mmask16 _mask = (__mmask16)((1u << remain) - 1);
                __m512 _p = bfloat2float_avx512(_mm256_maskz_loadu_epi16(_mask, ptr));
                __mmask16 _is_negative = _mm512_cmp_ps_mask(_p, _zero_avx512, _CMP_LT_OQ);
                _p = _mm512_mask_mul_ps(_p, _is_negative, _p, _slope_avx512);
                _mm256_mask_storeu_epi16(ptr, _mask, float2bfloat_avx512(_p));
                i += remain;
            }
#else  // __AVX512F__
            __m256 _zero_avx = _mm256_setzero_ps();
            __m256 _slope_avx = _mm256_set1_ps(slope);
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
                __m256 _pos = _mm256_max_ps(_zero_avx, _p);
                __m256 _neg = _mm256_min_ps(_zero_avx, _p);
                _p = _mm256_add_ps(_pos, _mm256_mul_ps(_slope_avx, _neg));
                _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
                ptr += 8;
            }
            __m128 _zero = _mm_setzero_ps();
            __m128 _slope = _mm_set1_ps(slope);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                __m128 _pos = _mm_max_ps(_zero, _p);
                __m128 _neg = _mm_min_ps(_zero, _p);
                _p = _mm_add_ps(_pos, _mm_mul_ps(_slope, _neg));
                _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
                ptr += 4;
            }
#endif // __AVX512F__
#else  // __AVX__
            __m128 _zero = _mm_setzero_ps();
            __m128 _slope = _mm_set1_ps(slope);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
                __m128 _pos = _mm_max_ps(_zero, _p);
                __m128 _neg = _mm_min_ps(_zero, _p);
                _p = _mm_add_ps(_pos, _mm_mul_ps(_slope, _neg));
                _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
                ptr += 4;
            }
#endif // __AVX__
#endif // __SSE2__
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0.f) v *= slope;
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }

    return 0;
}
