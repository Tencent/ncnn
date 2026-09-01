// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__
void elu_fp32_fma(Mat& bottom_top_blob, float alpha, const Option& opt);
#endif

static void elu_fp32(Mat& bottom_top_blob, float alpha, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__
    if (ncnn::cpu_support_x86_fma())
    {
        elu_fp32_fma(bottom_top_blob, alpha, opt);
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

#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _alpha512 = _mm512_set1_ps(alpha);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _mm512_storeu_ps(ptr, elu_avx512(_p, _alpha512));

            ptr += 16;
        }
#endif // __AVX512F__
        __m256 _alpha256 = _mm256_set1_ps(alpha);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _mm256_storeu_ps(ptr, elu_avx(_p, _alpha256));

            ptr += 8;
        }
#endif // __AVX__
        __m128 _alpha128 = _mm_set1_ps(alpha);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_load_ps(ptr);
            _mm_store_ps(ptr, elu_sse(_p, _alpha128));

            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            if (*ptr < 0.f)
                *ptr = alpha * (expf(*ptr) - 1.f);
            ptr++;
        }
    }
}
