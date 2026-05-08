// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void gelu_bf16s_avx512bf16(Mat& a, int fast_gelu, const Option& opt);
#endif

static void gelu_bf16s(Mat& a, int fast_gelu, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        gelu_bf16s_avx512bf16(a, fast_gelu, opt);
        return;
    }
#endif

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = a.channel(q);

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (fast_gelu)
        {
            __m512 _half512 = _mm512_set1_ps(0.5f);
            __m512 _one512 = _mm512_set1_ps(1.f);
            __m512 _fast1c512 = _mm512_set1_ps(0.79788452f);
            __m512 _fast2c512 = _mm512_set1_ps(0.044715f);
            for (; i + 15 < size; i += 16)
            {
                __m512 _pLoad = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));

                __m512 _cube = _mm512_mul_ps(_pLoad, _pLoad);
                _cube = _mm512_mul_ps(_pLoad, _cube);

                __m512 _blob = _mm512_mul_ps(_fast2c512, _cube);
                _blob = _mm512_add_ps(_pLoad, _blob);
                _blob = _mm512_mul_ps(_fast1c512, _blob);
                _blob = tanh512_ps(_blob);
                _blob = _mm512_add_ps(_one512, _blob);

                _blob = _mm512_mul_ps(_half512, _mm512_mul_ps(_blob, _pLoad));

                _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_blob));

                ptr += 16;
            }
            if (i < size)
            {
                const unsigned int remain = size - i;
                __mmask16 _mask = (__mmask16)((1u << remain) - 1);
                __m512 _pLoad = bfloat2float_avx512(_mm256_maskz_loadu_epi16(_mask, ptr));

                __m512 _cube = _mm512_mul_ps(_pLoad, _pLoad);
                _cube = _mm512_mul_ps(_pLoad, _cube);

                __m512 _blob = _mm512_mul_ps(_fast2c512, _cube);
                _blob = _mm512_add_ps(_pLoad, _blob);
                _blob = _mm512_mul_ps(_fast1c512, _blob);
                _blob = tanh512_ps(_blob);
                _blob = _mm512_add_ps(_one512, _blob);

                _blob = _mm512_mul_ps(_half512, _mm512_mul_ps(_blob, _pLoad));

                _mm256_mask_storeu_epi16(ptr, _mask, float2bfloat_avx512(_blob));
                i += remain;
            }
        }
        else
        {
            __m512 _half512 = _mm512_set1_ps(0.5f);
            __m512 _one512 = _mm512_set1_ps(1.f);
            __m512 _inv_sqrt2_512 = _mm512_set1_ps(0.70710678f);
            for (; i + 15 < size; i += 16)
            {
                __m512 _pLoad = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));

                __m512 _erf = erf512_ps(_mm512_mul_ps(_pLoad, _inv_sqrt2_512));
                __m512 _blob = _mm512_add_ps(_one512, _erf);
                _blob = _mm512_mul_ps(_half512, _mm512_mul_ps(_blob, _pLoad));

                _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_blob));

                ptr += 16;
            }
            if (i < size)
            {
                const unsigned int remain = size - i;
                __mmask16 _mask = (__mmask16)((1u << remain) - 1);
                __m512 _pLoad = bfloat2float_avx512(_mm256_maskz_loadu_epi16(_mask, ptr));

                __m512 _erf = erf512_ps(_mm512_mul_ps(_pLoad, _inv_sqrt2_512));
                __m512 _blob = _mm512_add_ps(_one512, _erf);
                _blob = _mm512_mul_ps(_half512, _mm512_mul_ps(_blob, _pLoad));

                _mm256_mask_storeu_epi16(ptr, _mask, float2bfloat_avx512(_blob));
                i += remain;
            }
        }
#else  // __AVX512F__
        if (fast_gelu)
        {
            __m256 _half256 = _mm256_set1_ps(0.5f);
            __m256 _one256 = _mm256_set1_ps(1.f);
            __m256 _fast1c256 = _mm256_set1_ps(0.79788452f);
            __m256 _fast2c256 = _mm256_set1_ps(0.044715f);
            for (; i + 7 < size; i += 8)
            {
                __m256 _pLoad = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));

                __m256 _cube = _mm256_mul_ps(_pLoad, _pLoad);
                _cube = _mm256_mul_ps(_pLoad, _cube);

                __m256 _blob = _mm256_mul_ps(_fast2c256, _cube);
                _blob = _mm256_add_ps(_pLoad, _blob);
                _blob = _mm256_mul_ps(_fast1c256, _blob);
                _blob = tanh256_ps(_blob);
                _blob = _mm256_add_ps(_one256, _blob);

                _blob = _mm256_mul_ps(_half256, _mm256_mul_ps(_blob, _pLoad));

                _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_blob));

                ptr += 8;
            }
        }
        else
        {
            __m256 _half256 = _mm256_set1_ps(0.5f);
            __m256 _one256 = _mm256_set1_ps(1.f);
            __m256 _inv_sqrt2_256 = _mm256_set1_ps(0.70710678f);
            for (; i + 7 < size; i += 8)
            {
                __m256 _pLoad = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));

                __m256 _erf = erf256_ps(_mm256_mul_ps(_pLoad, _inv_sqrt2_256));
                __m256 _blob = _mm256_add_ps(_one256, _erf);
                _blob = _mm256_mul_ps(_half256, _mm256_mul_ps(_blob, _pLoad));

                _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_blob));

                ptr += 8;
            }
        }
        if (fast_gelu)
        {
            __m128 _half128 = _mm_set1_ps(0.5f);
            __m128 _one128 = _mm_set1_ps(1.f);
            __m128 _fast1c128 = _mm_set1_ps(0.79788452f);
            __m128 _fast2c128 = _mm_set1_ps(0.044715f);
            for (; i + 3 < size; i += 4)
            {
                __m128 _pLoad = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));

                __m128 _cube = _mm_mul_ps(_pLoad, _pLoad);
                _cube = _mm_mul_ps(_pLoad, _cube);

                __m128 _blob = _mm_mul_ps(_fast2c128, _cube);
                _blob = _mm_add_ps(_pLoad, _blob);
                _blob = _mm_mul_ps(_fast1c128, _blob);
                _blob = tanh_ps(_blob);
                _blob = _mm_add_ps(_one128, _blob);

                _blob = _mm_mul_ps(_half128, _mm_mul_ps(_blob, _pLoad));

                _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_blob, _blob));

                ptr += 4;
            }
        }
        else
        {
            __m128 _half128 = _mm_set1_ps(0.5f);
            __m128 _one128 = _mm_set1_ps(1.f);
            __m128 _inv_sqrt2_128 = _mm_set1_ps(0.70710678f);
            for (; i + 3 < size; i += 4)
            {
                __m128 _pLoad = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));

                __m128 _erf = erf_ps(_mm_mul_ps(_pLoad, _inv_sqrt2_128));
                __m128 _blob = _mm_add_ps(_one128, _erf);
                _blob = _mm_mul_ps(_half128, _mm_mul_ps(_blob, _pLoad));

                _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_blob, _blob));

                ptr += 4;
            }
        }
#endif // __AVX512F__
#else  // __AVX__
        if (fast_gelu)
        {
            __m128 _half128 = _mm_set1_ps(0.5f);
            __m128 _one128 = _mm_set1_ps(1.f);
            __m128 _fast1c128 = _mm_set1_ps(0.79788452f);
            __m128 _fast2c128 = _mm_set1_ps(0.044715f);
            for (; i + 3 < size; i += 4)
            {
                __m128 _pLoad = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));

                __m128 _cube = _mm_mul_ps(_pLoad, _pLoad);
                _cube = _mm_mul_ps(_pLoad, _cube);

                __m128 _blob = _mm_mul_ps(_fast2c128, _cube);
                _blob = _mm_add_ps(_pLoad, _blob);
                _blob = _mm_mul_ps(_fast1c128, _blob);
                _blob = tanh_ps(_blob);
                _blob = _mm_add_ps(_one128, _blob);

                _blob = _mm_mul_ps(_half128, _mm_mul_ps(_blob, _pLoad));

                _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_blob, _blob));

                ptr += 4;
            }
        }
        else
        {
            __m128 _half128 = _mm_set1_ps(0.5f);
            __m128 _one128 = _mm_set1_ps(1.f);
            __m128 _inv_sqrt2_128 = _mm_set1_ps(0.70710678f);
            for (; i + 3 < size; i += 4)
            {
                __m128 _pLoad = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));

                __m128 _erf = erf_ps(_mm_mul_ps(_pLoad, _inv_sqrt2_128));
                __m128 _blob = _mm_add_ps(_one128, _erf);
                _blob = _mm_mul_ps(_half128, _mm_mul_ps(_blob, _pLoad));

                _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_blob, _blob));

                ptr += 4;
            }
        }
#endif // __AVX__
#endif // __SSE2__
        if (fast_gelu)
        {
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                v = 0.5f * v * (1.0f + tanhf(0.79788452f * (v + 0.044715f * v * v * v)));
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
        else
        {
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                v = 0.5f * v * (1.0f + erff(0.70710678f * v));
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }
}
