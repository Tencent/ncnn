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

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    const float inv_alpha = 1.f / alpha;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = a.channel(q);

        int i = 0;
#if __SSE2__
#if __AVX512F__
        __m512 _alpha_avx512 = _mm512_set1_ps(alpha);
        __m512 _inv_alpha_avx512 = _mm512_set1_ps(inv_alpha);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(celu_avx512(_p, _alpha_avx512, _inv_alpha_avx512)));
            ptr += 16;
        }
        if (i < size)
        {
            const unsigned int remain = size - i;
            __mmask16 _mask = (__mmask16)((1u << remain) - 1);
            __m512 _p = bfloat2float_avx512(_mm256_maskz_loadu_epi16(_mask, ptr));
            _mm256_mask_storeu_epi16(ptr, _mask, float2bfloat_avx512(celu_avx512(_p, _alpha_avx512, _inv_alpha_avx512)));
            i += remain;
        }
#else // __AVX512F__
#if __AVX__
        __m256 _alpha_avx = _mm256_set1_ps(alpha);
        __m256 _inv_alpha_avx = _mm256_set1_ps(inv_alpha);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_load_si128((const __m128i*)ptr));
            _mm_store_si128((__m128i*)ptr, float2bfloat_avx(celu_avx(_p, _alpha_avx, _inv_alpha_avx)));
            ptr += 8;
        }
#endif // __AVX__
        __m128 _alpha_sse = _mm_set1_ps(alpha);
        __m128 _inv_alpha_sse = _mm_set1_ps(inv_alpha);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            _p = celu_sse(_p, _alpha_sse, _inv_alpha_sse);
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p));
            ptr += 4;
        }
#endif // __AVX512F__
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < 0.f)
                v = (expf(v * inv_alpha) - 1.f) * alpha;
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
}
