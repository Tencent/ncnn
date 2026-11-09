// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void softplus_bf16s_avx512bf16(Mat& a, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXNECONVERT && __AVX__ && !__AVX512F__ && !__AVXNECONVERT__
void softplus_bf16s_avxneconvert(Mat& a, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
void softplus_bf16s_avx2(Mat& a, const Option& opt);
#endif

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__ && !__AVX512BF16__
void softplus_bf16s_fma(Mat& a, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__ && !__AVX512BF16__
void softplus_bf16s_fma4(Mat& a, const Option& opt);
#endif

static void softplus_bf16s(Mat& a, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        softplus_bf16s_avx512bf16(a, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVXNECONVERT && __AVX__ && !__AVX512F__ && !__AVXNECONVERT__
    if (ncnn::cpu_support_x86_avx_ne_convert())
    {
        softplus_bf16s_avxneconvert(a, opt);
        return;
    }
#endif

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx2())
    {
        softplus_bf16s_avx2(a, opt);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_fma())
    {
        softplus_bf16s_fma(a, opt);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_fma4())
    {
        softplus_bf16s_fma4(a, opt);
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
        __m512 _exp_hi_avx512 = _mm512_set1_ps(88.3762626647949f);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            __m512 _y = log512_ps(_mm512_add_ps(exp512_ps(_p), _one_avx512));
            // exp clamps its argument, so keep nan and values beyond the clamped range unchanged
            __mmask16 _ovf = _mm512_cmp_ps_mask(_p, _exp_hi_avx512, _CMP_NLE_UQ);
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_mm512_mask_mov_ps(_y, _ovf, _p)));
            ptr += 16;
        }
        if (i < size)
        {
            const unsigned int remain = size - i;
            __mmask16 _mask = (__mmask16)((1u << remain) - 1);
            __m512 _p = bfloat2float_avx512(_mm256_maskz_loadu_epi16(_mask, ptr));
            __m512 _y = log512_ps(_mm512_add_ps(exp512_ps(_p), _one_avx512));
            __mmask16 _ovf = _mm512_cmp_ps_mask(_p, _exp_hi_avx512, _CMP_NLE_UQ);
            _mm256_mask_storeu_epi16(ptr, _mask, float2bfloat_avx512(_mm512_mask_mov_ps(_y, _ovf, _p)));
            i += remain;
        }
#else  // __AVX512F__
        __m256 _one_avx = _mm256_set1_ps(1.f);
        __m256 _exp_hi_avx = _mm256_set1_ps(88.3762626647949f);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            __m256 _y = log256_ps(_mm256_add_ps(exp256_ps(_p), _one_avx));
            __m256 _ovf = _mm256_cmp_ps(_p, _exp_hi_avx, _CMP_NLE_UQ);
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_mm256_blendv_ps(_y, _p, _ovf)));
            ptr += 8;
        }
        __m128 _one = _mm_set1_ps(1.f);
        __m128 _exp_hi = _mm_set1_ps(88.3762626647949f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            __m128 _y = log_ps(_mm_add_ps(exp_ps(_p), _one));
            __m128 _ovf = _mm_cmpnle_ps(_p, _exp_hi);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_mm_or_ps(_mm_and_ps(_ovf, _p), _mm_andnot_ps(_ovf, _y))));
            ptr += 4;
        }
#endif // __AVX512F__
#else  // __AVX__
        __m128 _one = _mm_set1_ps(1.f);
        __m128 _exp_hi = _mm_set1_ps(88.3762626647949f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            __m128 _y = log_ps(_mm_add_ps(exp_ps(_p), _one));
            __m128 _ovf = _mm_cmpnle_ps(_p, _exp_hi);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_mm_or_ps(_mm_and_ps(_ovf, _p), _mm_andnot_ps(_ovf, _y))));
            ptr += 4;
        }
#endif // __AVX__
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v <= 88.3762626647949f)
                *ptr = float32_to_bfloat16(logf(expf(v) + 1.f));
            ptr++;
        }
    }
}
