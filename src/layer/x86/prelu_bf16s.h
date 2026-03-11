// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void prelu_bf16s_sse_avx512bf16(unsigned short* ptr, const float* slope, int size, int elempack);
void prelu_bf16s_per_element_sse_avx512bf16(unsigned short* ptr, const float* slope, int size, int num_threads);
void prelu_bf16s_single_slope_sse_avx512bf16(unsigned short* ptr, float slope, int size, int num_threads);
#endif

static void prelu_bf16s_sse(unsigned short* ptr, const float* slope, int size, int elempack)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        prelu_bf16s_sse_avx512bf16(ptr, slope, size, elempack);
        return;
    }
#endif

#if __SSE2__
    __m128 _slope128 = (elempack == 4) ? _mm_loadu_ps(slope) : _mm_set1_ps(slope[0]);
    __m128 _zero = _mm_setzero_ps();
#if __AVX__
    __m256 _slope256 = (elempack == 8) ? _mm256_loadu_ps(slope) : combine4x2_ps(_slope128, _slope128);
    __m256 _zero_avx = _mm256_setzero_ps();
#if __AVX512F__
    __m512 _slope512 = (elempack == 16) ? _mm512_loadu_ps(slope) : combine8x2_ps(_slope256, _slope256);
    __m512 _zero_avx512 = _mm512_setzero_ps();
#endif
#endif
#endif
    float s = slope[0];

    int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    for (; i + 15 < size; i += 16)
    {
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
        __mmask16 _mask = _mm512_cmp_ps_mask(_p, _zero_avx512, _CMP_LT_OQ);
        __m512 _ps = _mm512_mul_ps(_p, _slope512);
        _p = _mm512_mask_mov_ps(_p, _mask, _ps);
        _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
        ptr += 16;
    }
#endif // __AVX512F__
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
        __m256 _ps = _mm256_mul_ps(_p, _slope256);
        _p = _mm256_blendv_ps(_p, _ps, _mm256_cmp_ps(_p, _zero_avx, _CMP_LT_OQ));
        _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
        ptr += 8;
    }
#endif // __AVX__
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
        __m128 _ps = _mm_mul_ps(_p, _slope128);
        __m128 _mask = _mm_cmplt_ps(_p, _zero);
        _p = _mm_or_ps(_mm_andnot_ps(_mask, _p), _mm_and_ps(_mask, _ps));
        _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
        ptr += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        float v = bfloat16_to_float32(*ptr);
        if (v < 0.f)
            v *= s;
        *ptr = float32_to_bfloat16(v);
        ptr++;
    }
}

static void prelu_bf16s_per_element_sse(unsigned short* ptr, const float* slope, int size, int num_threads)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        prelu_bf16s_per_element_sse_avx512bf16(ptr, slope, size, num_threads);
        return;
    }
#endif

    int nn_size = 0;
    int remain_size_start = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    nn_size = (size - remain_size_start) / 16;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 16;
        __m512 _zero_avx512 = _mm512_setzero_ps();
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(ptr + i)));
        __m512 _slope = _mm512_loadu_ps(slope + i);
        __mmask16 _mask = _mm512_cmp_ps_mask(_p, _zero_avx512, _CMP_LT_OQ);
        __m512 _ps = _mm512_mul_ps(_p, _slope);
        _p = _mm512_mask_mov_ps(_p, _mask, _ps);
        _mm256_storeu_si256((__m256i*)(ptr + i), float2bfloat_avx512(_p));
    }
    remain_size_start += nn_size * 16;
#endif // __AVX512F__
    nn_size = (size - remain_size_start) / 8;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 8;
        __m256 _zero_avx = _mm256_setzero_ps();
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(ptr + i)));
        __m256 _slope = _mm256_loadu_ps(slope + i);
        __m256 _ps = _mm256_mul_ps(_p, _slope);
        _p = _mm256_blendv_ps(_p, _ps, _mm256_cmp_ps(_p, _zero_avx, _CMP_LT_OQ));
        _mm_storeu_si128((__m128i*)(ptr + i), float2bfloat_avx(_p));
    }
    remain_size_start += nn_size * 8;
#endif // __AVX__
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        __m128 _zero = _mm_setzero_ps();
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(ptr + i)));
        __m128 _slope = _mm_loadu_ps(slope + i);
        __m128 _ps = _mm_mul_ps(_p, _slope);
        __m128 _mask = _mm_cmplt_ps(_p, _zero);
        _p = _mm_or_ps(_mm_andnot_ps(_mask, _p), _mm_and_ps(_mask, _ps));
        _mm_storel_epi64((__m128i*)(ptr + i), float2bfloat_sse(_p, _p));
    }
    remain_size_start += nn_size * 4;
#endif // __SSE2__
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        float v = bfloat16_to_float32(ptr[i]);
        if (v < 0.f)
            v *= slope[i];
        ptr[i] = float32_to_bfloat16(v);
    }
}

static void prelu_bf16s_single_slope_sse(unsigned short* ptr, float slope, int size, int num_threads)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        prelu_bf16s_single_slope_sse_avx512bf16(ptr, slope, size, num_threads);
        return;
    }
#endif

    int nn_size = 0;
    int remain_size_start = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
    nn_size = (size - remain_size_start) / 16;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 16;
        __m512 _zero_avx512 = _mm512_setzero_ps();
        __m512 _slope512 = _mm512_set1_ps(slope);
        __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(ptr + i)));
        __mmask16 _mask = _mm512_cmp_ps_mask(_p, _zero_avx512, _CMP_LT_OQ);
        __m512 _ps = _mm512_mul_ps(_p, _slope512);
        _p = _mm512_mask_mov_ps(_p, _mask, _ps);
        _mm256_storeu_si256((__m256i*)(ptr + i), float2bfloat_avx512(_p));
    }
    remain_size_start += nn_size * 16;
#endif // __AVX512F__
    nn_size = (size - remain_size_start) / 8;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 8;
        __m256 _zero_avx = _mm256_setzero_ps();
        __m256 _slope256 = _mm256_set1_ps(slope);
        __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(ptr + i)));
        __m256 _ps = _mm256_mul_ps(_p, _slope256);
        _p = _mm256_blendv_ps(_p, _ps, _mm256_cmp_ps(_p, _zero_avx, _CMP_LT_OQ));
        _mm_storeu_si128((__m128i*)(ptr + i), float2bfloat_avx(_p));
    }
    remain_size_start += nn_size * 8;
#endif // __AVX__
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        __m128 _zero = _mm_setzero_ps();
        __m128 _slope128 = _mm_set1_ps(slope);
        __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(ptr + i)));
        __m128 _ps = _mm_mul_ps(_p, _slope128);
        __m128 _mask = _mm_cmplt_ps(_p, _zero);
        _p = _mm_or_ps(_mm_andnot_ps(_mask, _p), _mm_and_ps(_mask, _ps));
        _mm_storel_epi64((__m128i*)(ptr + i), float2bfloat_sse(_p, _p));
    }
    remain_size_start += nn_size * 4;
#endif // __SSE2__
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        float v = bfloat16_to_float32(ptr[i]);
        if (v < 0.f)
            v *= slope;
        ptr[i] = float32_to_bfloat16(v);
    }
}
