// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void hardswish_bf16s_avx512bf16(Mat& a, float alpha, float beta, float lower, float upper, const Option& opt);
#endif

static void hardswish_bf16s(Mat& a, float alpha, float beta, float lower, float upper, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        hardswish_bf16s_avx512bf16(a, alpha, beta, lower, upper, opt);
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
        __m512 _alpha_avx512 = _mm512_set1_ps(alpha);
        __m512 _beta_avx512 = _mm512_set1_ps(beta);
        __m512 _zero_avx512 = _mm512_setzero_ps();
        __m512 _one_avx512 = _mm512_set1_ps(1.f);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            __m512 _ans = _mm512_fmadd_ps(_p, _alpha_avx512, _beta_avx512);
            _ans = _mm512_max_ps(_ans, _zero_avx512);
            _ans = _mm512_min_ps(_ans, _one_avx512);
            _ans = _mm512_mul_ps(_ans, _p);
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_ans));
            ptr += 16;
        }
        if (i < size)
        {
            const unsigned int remain = size - i;
            __mmask16 _mask = (__mmask16)((1u << remain) - 1);
            __m512 _p = bfloat2float_avx512(_mm256_maskz_loadu_epi16(_mask, ptr));
            __m512 _ans = _mm512_fmadd_ps(_p, _alpha_avx512, _beta_avx512);
            _ans = _mm512_max_ps(_ans, _zero_avx512);
            _ans = _mm512_min_ps(_ans, _one_avx512);
            _ans = _mm512_mul_ps(_ans, _p);
            _mm256_mask_storeu_epi16(ptr, _mask, float2bfloat_avx512(_ans));
            i += remain;
        }
#else  // __AVX512F__
        __m256 _alpha_avx = _mm256_set1_ps(alpha);
        __m256 _beta_avx = _mm256_set1_ps(beta);
        __m256 _zero_avx = _mm256_setzero_ps();
        __m256 _one_avx = _mm256_set1_ps(1.f);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            __m256 _ans = _mm256_comp_fmadd_ps(_p, _alpha_avx, _beta_avx);
            _ans = _mm256_max_ps(_ans, _zero_avx);
            _ans = _mm256_min_ps(_ans, _one_avx);
            _ans = _mm256_mul_ps(_ans, _p);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_ans));
            ptr += 8;
        }
        __m128 _alpha_sse = _mm_set1_ps(alpha);
        __m128 _beta_sse = _mm_set1_ps(beta);
        __m128 _zero = _mm_setzero_ps();
        __m128 _one = _mm_set1_ps(1.f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            __m128 _ans = _mm_comp_fmadd_ps(_p, _alpha_sse, _beta_sse);
            _ans = _mm_max_ps(_ans, _zero);
            _ans = _mm_min_ps(_ans, _one);
            _ans = _mm_mul_ps(_ans, _p);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_ans, _ans));
            ptr += 4;
        }
#endif // __AVX512F__
#else  // __AVX__
        __m128 _alpha_sse = _mm_set1_ps(alpha);
        __m128 _beta_sse = _mm_set1_ps(beta);
        __m128 _zero = _mm_setzero_ps();
        __m128 _one = _mm_set1_ps(1.f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            __m128 _ans = _mm_comp_fmadd_ps(_p, _alpha_sse, _beta_sse);
            _ans = _mm_max_ps(_ans, _zero);
            _ans = _mm_min_ps(_ans, _one);
            _ans = _mm_mul_ps(_ans, _p);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_ans, _ans));
            ptr += 4;
        }
#endif // __AVX__
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < lower)
                v = 0.f;
            else if (v > upper)
                ;
            else
                v = v * (v * alpha + beta);
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
}
