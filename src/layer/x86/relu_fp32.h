// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void relu_fp32_fma(Mat& bottom_top_blob, float slope, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void relu_fp32_fma4(Mat& bottom_top_blob, float slope, const Option& opt);
#endif

static void relu_fp32(Mat& bottom_top_blob, float slope, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (slope != 0.f && ncnn::cpu_support_x86_fma())
    {
        relu_fp32_fma(bottom_top_blob, slope, opt);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (slope != 0.f && ncnn::cpu_support_x86_fma4())
    {
        relu_fp32_fma4(bottom_top_blob, slope, opt);
        return;
    }
#endif

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _zero_avx512 = _mm512_setzero_ps();
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                _mm512_storeu_ps(ptr, _mm512_max_ps(_zero_avx512, _p));
                ptr += 16;
            }
#endif // __AVX512F__
            __m256 _zero_avx = _mm256_setzero_ps();
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                _mm256_storeu_ps(ptr, _mm256_max_ps(_zero_avx, _p));
                ptr += 8;
            }
#endif // __AVX__
            __m128 _zero = _mm_setzero_ps();
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr);
                _mm_store_ps(ptr, _mm_max_ps(_zero, _p));
                ptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                *ptr = std::max(*ptr, 0.f);
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
            __m512 _zero_avx512 = _mm512_setzero_ps();
            __m512 _slope_avx512 = _mm512_set1_ps(slope);
            for (; i + 15 < size; i += 16)
            {
                __m512 _p = _mm512_loadu_ps(ptr);
                __mmask16 _is_negative = _mm512_cmp_ps_mask(_p, _zero_avx512, _CMP_LT_OQ);
                _p = _mm512_mask_mul_ps(_p, _is_negative, _p, _slope_avx512);
                _mm512_storeu_ps(ptr, _p);
                ptr += 16;
            }
#endif // __AVX512F__
            __m256 _zero_avx = _mm256_setzero_ps();
            __m256 _slope_avx = _mm256_set1_ps(slope);
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = _mm256_loadu_ps(ptr);
                __m256 _pos = _mm256_max_ps(_zero_avx, _p);
                __m256 _neg = _mm256_min_ps(_zero_avx, _p);
                _p = _mm256_add_ps(_pos, _mm256_mul_ps(_slope_avx, _neg));
                _mm256_storeu_ps(ptr, _p);
                ptr += 8;
            }
#endif // __AVX__
            __m128 _zero = _mm_setzero_ps();
            __m128 _slope = _mm_set1_ps(slope);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = _mm_load_ps(ptr);
                __m128 _pos = _mm_max_ps(_zero, _p);
                __m128 _neg = _mm_min_ps(_zero, _p);
                _p = _mm_add_ps(_pos, _mm_mul_ps(_slope, _neg));
                _mm_store_ps(ptr, _p);
                ptr += 4;
            }
#endif // __SSE2__
            for (; i < size; i++)
            {
                if (*ptr < 0)
                    *ptr *= slope;
                ptr++;
            }
        }
    }
}
