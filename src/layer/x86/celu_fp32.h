// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void celu_fp32_fma(Mat& bottom_top_blob, float alpha, const Option& opt);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void celu_fp32_fma4(Mat& bottom_top_blob, float alpha, const Option& opt);
#endif

#if __SSE2__
static void celu_fp32_scalar(Mat& a, int size, float alpha, const Option& opt)
{
    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < a.c; q++)
    {
        float* ptr = a.channel(q);
        // Preserve division because large alpha amplifies reciprocal rounding errors.
        const volatile float divisor = alpha;
        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < 0.f)
                ptr[i] = (expf(ptr[i] / divisor) - 1.f) * alpha;
        }
    }
}
#endif // __SSE2__

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

    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;
    int size = w * h * d * elempack;

#if __SSE2__
    // Negative alpha can exceed the SIMD exponential's range. Large alpha amplifies rounding near one.
    if (alpha <= 0.f || alpha > 1024.f)
    {
        celu_fp32_scalar(bottom_top_blob, size, alpha, opt);
        return;
    }
    const float inv_alpha = 1.f / alpha;
#endif // __SSE2__

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        int i = 0;

#if __SSE2__
#if __AVX__
#if __AVX512F__
        __m512 _alpha512 = _mm512_set1_ps(alpha);
        __m512 _inv_alpha512 = _mm512_set1_ps(inv_alpha);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = _mm512_loadu_ps(ptr);
            _mm512_storeu_ps(ptr, celu_avx512(_p, _alpha512, _inv_alpha512));

            ptr += 16;
        }
#endif // __AVX512F__
        __m256 _alpha256 = _mm256_set1_ps(alpha);
        __m256 _inv_alpha256 = _mm256_set1_ps(inv_alpha);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = _mm256_loadu_ps(ptr);
            _mm256_storeu_ps(ptr, celu_avx(_p, _alpha256, _inv_alpha256));

            ptr += 8;
        }
#endif // __AVX__
        __m128 _alpha128 = _mm_set1_ps(alpha);
        __m128 _inv_alpha128 = _mm_set1_ps(inv_alpha);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = _mm_loadu_ps(ptr);
            _mm_storeu_ps(ptr, celu_sse(_p, _alpha128, _inv_alpha128));

            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            if (*ptr < 0.f)
                *ptr = (expf(*ptr / alpha) - 1.f) * alpha;
            ptr++;
        }
    }
}
