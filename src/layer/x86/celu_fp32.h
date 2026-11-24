// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void celu_fp32_fma(Mat& bottom_top_blob, float alpha, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void celu_fp32_fma4(Mat& bottom_top_blob, float alpha, const Option& opt);
#endif

static void celu_fp32(Mat& bottom_top_blob, float alpha, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
    {
        celu_fp32_fma(bottom_top_blob, alpha, opt);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
    {
        celu_fp32_fma4(bottom_top_blob, alpha, opt);
        return;
    }
#endif

#if __SSE2__
    // For negative inputs, keep the exponent nonpositive and limit the error amplified by alpha.
    const float inv_alpha = alpha >= FLT_MIN && alpha <= 1024.f ? 1.f / alpha : 0.f;
#endif // __SSE2__

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads) if (channels > 1)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

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
                __m512 _p = _mm512_loadu_ps(ptr);
                __m512 _y = _mm512_mul_ps(_mm512_sub_ps(exp512_ps(_mm512_mul_ps(_p, _inv_alpha)), _one), _alpha);
                __mmask16 _neg = _mm512_cmp_ps_mask(_p, _zero, _CMP_LT_OQ);
                _mm512_storeu_ps(ptr, _mm512_mask_mov_ps(_p, _neg, _y));
                ptr += 16;
            }
            if (i < size)
            {
                const unsigned int remain = size - i;
                __mmask16 _mask = (__mmask16)((1u << remain) - 1);
                __m512 _p = _mm512_maskz_loadu_ps(_mask, ptr);
                __m512 _y = _mm512_mul_ps(_mm512_sub_ps(exp512_ps(_mm512_mul_ps(_p, _inv_alpha)), _one), _alpha);
                __mmask16 _neg = _mm512_cmp_ps_mask(_p, _zero, _CMP_LT_OQ);
                _mm512_mask_storeu_ps(ptr, _mask, _mm512_mask_mov_ps(_p, _neg, _y));
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
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _y = _mm256_mul_ps(_mm256_sub_ps(exp256_ps(_mm256_mul_ps(_p, _inv_alpha_avx)), _one_avx), _alpha_avx);
                __m256 _neg = _mm256_cmp_ps(_p, _zero_avx, _CMP_LT_OQ);
                _mm256_storeu_ps(ptr, _mm256_blendv_ps(_p, _y, _neg));
                ptr += 8;
            }
#endif // __AVX__
            const __m128 _alpha = _mm_set1_ps(alpha);
            const __m128 _inv_alpha = _mm_set1_ps(inv_alpha);
            const __m128 _one = _mm_set1_ps(1.f);
            const __m128 _zero = _mm_setzero_ps();
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_loadu_ps(ptr);
                __m128 _y = _mm_mul_ps(_mm_sub_ps(exp_ps(_mm_mul_ps(_p, _inv_alpha)), _one), _alpha);
                __m128 _neg = _mm_cmplt_ps(_p, _zero);
#if defined(__clang__) && defined(__FAST_MATH__)
                // Clang can invert the comparison under fast math, selecting NaN lanes.
                __m128i _abs = _mm_and_si128(_mm_castps_si128(_p), _mm_set1_epi32(0x7fffffff));
                __m128i _nan = _mm_cmpgt_epi32(_abs, _mm_set1_epi32(0x7f800000));
                _neg = _mm_andnot_ps(_mm_castsi128_ps(_nan), _neg);
#endif
                _mm_storeu_ps(ptr, _mm_or_ps(_mm_and_ps(_neg, _y), _mm_andnot_ps(_neg, _p)));
                ptr += 4;
            }
#endif // __AVX512F__
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
#if __SSE2__
            // Keep the scalar comparison ordered under fast math.
            if (_mm_movemask_ps(_mm_cmplt_ss(_mm_set_ss(*ptr), _mm_setzero_ps())) & 1)
#else
            if (*ptr < 0.f)
#endif
                *ptr = (expf(*ptr / alpha) - 1.f) * alpha;
            ptr++;
        }
    }
}
