// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void softplus_fp32_fma(Mat& bottom_top_blob, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void softplus_fp32_fma4(Mat& bottom_top_blob, const Option& opt);
#endif

static void softplus_fp32(Mat& bottom_top_blob, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
    {
        softplus_fp32_fma(bottom_top_blob, opt);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
    {
        softplus_fp32_fma4(bottom_top_blob, opt);
        return;
    }
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;
#if __AVX512F__
        __m512 _one_avx512 = _mm512_set1_ps(1.f);
        __m512 _exp_hi_avx512 = _mm512_set1_ps(88.3762626647949f);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            __m512 _y = log512_ps(_mm512_add_ps(exp512_ps(_p), _one_avx512));
            // exp clamps its argument, so keep nan and values beyond the clamped range unchanged
            __mmask16 _ovf = _mm512_cmp_ps_mask(_p, _exp_hi_avx512, _CMP_NLE_UQ);
            _mm512_storeu_ps(ptr, _mm512_mask_mov_ps(_y, _ovf, _p));
            ptr += 16;
        }
        if (i < size)
        {
            const unsigned int remain = size - i;
            __mmask16 _mask = (__mmask16)((1u << remain) - 1);
            __m512 _p = _mm512_maskz_loadu_ps(_mask, ptr);
            __m512 _y = log512_ps(_mm512_add_ps(exp512_ps(_p), _one_avx512));
            __mmask16 _ovf = _mm512_cmp_ps_mask(_p, _exp_hi_avx512, _CMP_NLE_UQ);
            _mm512_mask_storeu_ps(ptr, _mask, _mm512_mask_mov_ps(_y, _ovf, _p));
        }
#else // __AVX512F__
#if __SSE2__
#if __AVX__
        __m256 _one_avx = _mm256_set1_ps(1.f);
        __m256 _exp_hi_avx = _mm256_set1_ps(88.3762626647949f);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            __m256 _y = log256_ps(_mm256_add_ps(exp256_ps(_p), _one_avx));
            __m256 _ovf = _mm256_cmp_ps(_p, _exp_hi_avx, _CMP_NLE_UQ);
            _mm256_storeu_ps(ptr, _mm256_blendv_ps(_y, _p, _ovf));
            ptr += 8;
        }
#endif // __AVX__
        __m128 _one = _mm_set1_ps(1.f);
        __m128 _exp_hi = _mm_set1_ps(88.3762626647949f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            __m128 _y = log_ps(_mm_add_ps(exp_ps(_p), _one));
            __m128 _ovf = _mm_cmpnle_ps(_p, _exp_hi);
            _mm_store_ps(ptr, _mm_or_ps(_mm_and_ps(_ovf, _p), _mm_andnot_ps(_ovf, _y)));
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            if (*ptr <= 88.3762626647949f)
                *ptr = logf(expf(*ptr) + 1.f);
            ptr++;
        }
#endif // __AVX512F__
    }
}
