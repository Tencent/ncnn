// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void bnll_bf16s_avx512bf16(Mat& a, const Option& opt);
#endif

static void bnll_bf16s(Mat& a, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        bnll_bf16s_avx512bf16(a, opt);
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
        __m512 _one_avx512 = _mm512_set1_ps(1.f);
        __m512 _zero_avx512 = _mm512_setzero_ps();
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            __mmask16 mask = _mm512_cmp_ps_mask(_p, _zero_avx512, _CMP_GT_OQ);
            __m512 _abs_p = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(_p), _mm512_set1_epi32(0x7fffffff)));
            __m512 _tmp = log512_ps(_mm512_add_ps(_one_avx512, exp512_ps(_mm512_sub_ps(_zero_avx512, _abs_p))));
            _p = _mm512_mask_add_ps(_tmp, mask, _tmp, _p);
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
            ptr += 16;
        }
        if (i < size)
        {
            const unsigned int remain = size - i;
            __mmask16 _mask = (__mmask16)((1u << remain) - 1);
            __m512 _p = bfloat2float_avx512(_mm256_maskz_loadu_epi16(_mask, ptr));
            __mmask16 mask = _mm512_cmp_ps_mask(_p, _zero_avx512, _CMP_GT_OQ);
            __m512 _abs_p = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(_p), _mm512_set1_epi32(0x7fffffff)));
            __m512 _tmp = log512_ps(_mm512_add_ps(_one_avx512, exp512_ps(_mm512_sub_ps(_zero_avx512, _abs_p))));
            _p = _mm512_mask_add_ps(_tmp, mask, _tmp, _p);
            _mm256_mask_storeu_epi16(ptr, _mask, float2bfloat_avx512(_p));
            i += remain;
        }
#else  // __AVX512F__
        __m256 _one_avx = _mm256_set1_ps(1.f);
        __m256 _zero_avx = _mm256_setzero_ps();
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            __m256 mask = _mm256_cmp_ps(_p, _mm256_setzero_ps(), _CMP_GT_OQ);
            __m256 _abs_p = _mm256_and_ps(_p, *(__m256*)_ps256_inv_sign_mask);
            __m256 _tmp = log256_ps(_mm256_add_ps(_one_avx, exp256_ps(_mm256_sub_ps(_zero_avx, _abs_p))));
            __m256 _x = _mm256_and_ps(_p, mask);
            _p = _mm256_add_ps(_x, _tmp);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
            ptr += 8;
        }
        __m128 _one = _mm_set1_ps(1.f);
        __m128 _zero = _mm_setzero_ps();
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            __m128 mask = _mm_cmpgt_ps(_p, _zero);
            __m128 _abs_p = _mm_and_ps(_p, *(__m128*)_ps_inv_sign_mask);
            __m128 _tmp = log_ps(_mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _abs_p))));
            __m128 _x = _mm_and_ps(_p, mask);
            _p = _mm_add_ps(_x, _tmp);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
            ptr += 4;
        }
#endif // __AVX512F__
#else  // __AVX__
        __m128 _one = _mm_set1_ps(1.f);
        __m128 _zero = _mm_setzero_ps();
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            __m128 mask = _mm_cmpgt_ps(_p, _zero);
            __m128 _abs_p = _mm_and_ps(_p, *(__m128*)_ps_inv_sign_mask);
            __m128 _tmp = log_ps(_mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _abs_p))));
            __m128 _x = _mm_and_ps(_p, mask);
            _p = _mm_add_ps(_x, _tmp);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
            ptr += 4;
        }
#endif // __AVX__
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v > 0)
                v = v + logf(1.f + expf(-v));
            else
                v = logf(1.f + expf(v));
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
}
