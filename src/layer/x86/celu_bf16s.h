// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void celu_bf16s_avx512bf16(Mat& a, float alpha, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXNECONVERT && __AVX__ && !__AVX512F__ && !__AVXNECONVERT__
void celu_bf16s_avxneconvert(Mat& a, float alpha, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
void celu_bf16s_avx2(Mat& a, float alpha, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__ && !__AVX512BF16__
void celu_bf16s_fma(Mat& a, float alpha, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__ && !__AVX512BF16__
void celu_bf16s_fma4(Mat& a, float alpha, const Option& opt);
#endif

static void celu_bf16s(Mat& a, float alpha, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        celu_bf16s_avx512bf16(a, alpha, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXNECONVERT && __AVX__ && !__AVX512F__ && !__AVXNECONVERT__
    if (ncnn::cpu_support_x86_avx_ne_convert())
    {
        celu_bf16s_avxneconvert(a, alpha, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx2())
    {
        celu_bf16s_avx2(a, alpha, opt);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_fma())
    {
        celu_bf16s_fma(a, alpha, opt);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_fma4())
    {
        celu_bf16s_fma4(a, alpha, opt);
        return;
    }
#endif

#if __SSE2__
    // For negative inputs, keep the exponent nonpositive and limit the error amplified by alpha.
    const float inv_alpha = alpha >= FLT_MIN && alpha <= 1024.f ? 1.f / alpha : 0.f;
#endif // __SSE2__

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads) if (channels > 1)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = a.channel(q);

        int i = 0;
#if __SSE2__
        if (inv_alpha > 0.f)
        {
#if __AVX512F__
            const __m512 _alpha = _mm512_set1_ps(alpha);
            const __m512 _inv_alpha = _mm512_set1_ps(inv_alpha);
            const __m512 _one = _mm512_set1_ps(1.f);
            const __m512 _zero = _mm512_setzero_ps();
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
                __m512 _y = _mm512_mul_ps(_mm512_sub_ps(exp512_ps(_mm512_mul_ps(_p, _inv_alpha)), _one), _alpha);
                __mmask16 _neg = _mm512_cmp_ps_mask(_p, _zero, _CMP_LT_OQ);
                _mm256_mask_storeu_epi16(ptr, _neg, float2bfloat_avx512(_y));
                ptr += 16;
            }
            if (i < size)
            {
                const unsigned int remain = size - i;
                __mmask16 _mask = (__mmask16)((1u << remain) - 1);
                __m512 _p = bfloat2float_avx512(_mm256_maskz_loadu_epi16(_mask, ptr));
                __m512 _y = _mm512_mul_ps(_mm512_sub_ps(exp512_ps(_mm512_mul_ps(_p, _inv_alpha)), _one), _alpha);
                __mmask16 _neg = _mm512_cmp_ps_mask(_p, _zero, _CMP_LT_OQ);
                _mm256_mask_storeu_epi16(ptr, _mask & _neg, float2bfloat_avx512(_y));
                i += remain;
                ptr += remain;
            }
#else // __AVX512F__
#if __AVX__
            const __m256 _alpha_avx = _mm256_set1_ps(alpha);
            const __m256 _inv_alpha_avx = _mm256_set1_ps(inv_alpha);
            const __m256 _one_avx = _mm256_set1_ps(1.f);
            const __m256 _zero_avx = _mm256_setzero_ps();
            for (; i + 7 < size; i += 8)
            {
                __m128i _v = _mm_loadu_si128((const __m128i*)ptr);
                __m256 _p = bfloat2float_avx(_v);
                __m256 _y = _mm256_mul_ps(_mm256_sub_ps(exp256_ps(_mm256_mul_ps(_p, _inv_alpha_avx)), _one_avx), _alpha_avx);
                __m256i _neg = _mm256_castps_si256(_mm256_cmp_ps(_p, _zero_avx, _CMP_LT_OQ));
                __m128i _mask = _mm_packs_epi32(_mm256_castsi256_si128(_neg), _mm256_extractf128_si256(_neg, 1));
                _mm_storeu_si128((__m128i*)ptr, _mm_blendv_epi8(_v, float2bfloat_avx(_y), _mask));
                ptr += 8;
            }
#endif // __AVX__
            const __m128 _alpha = _mm_set1_ps(alpha);
            const __m128 _inv_alpha = _mm_set1_ps(inv_alpha);
            const __m128 _one = _mm_set1_ps(1.f);
            const __m128 _zero = _mm_setzero_ps();
            for (; i + 3 < size; i += 4)
            {
                __m128i _v = _mm_loadl_epi64((const __m128i*)ptr);
                __m128 _p = bfloat2float_sse(_v);
                __m128 _y = _mm_mul_ps(_mm_sub_ps(exp_ps(_mm_mul_ps(_p, _inv_alpha)), _one), _alpha);
                __m128i _neg = _mm_castps_si128(_mm_cmplt_ps(_p, _zero));
                __m128i _mask = _mm_packs_epi32(_neg, _neg);
                _mm_storel_epi64((__m128i*)ptr, _mm_or_si128(_mm_and_si128(_mask, float2bfloat_sse(_y)), _mm_andnot_si128(_mask, _v)));
                ptr += 4;
            }
#endif // __AVX512F__
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
#if __SSE2__
            // Keep the scalar comparison ordered under fast math.
            if (_mm_movemask_ps(_mm_cmplt_ss(_mm_set_ss(v), _mm_setzero_ps())) & 1)
#else
            if (v < 0.f)
#endif
                *ptr = float32_to_bfloat16((expf(v / alpha) - 1.f) * alpha);
            ptr++;
        }
    }
}
