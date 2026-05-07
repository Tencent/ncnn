// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void selu_bf16s_avx512bf16(Mat& a, float alphaxlambda, float lambda, const Option& opt);
#endif

static void selu_bf16s(Mat& a, float alphaxlambda, float lambda, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        selu_bf16s_avx512bf16(a, alphaxlambda, lambda, opt);
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
        __m512 _zero512 = _mm512_setzero_ps();
        __m512 _one512 = _mm512_set1_ps(1.f);
        __m512 _alpha512 = _mm512_set1_ps(alphaxlambda / lambda);
        __m512 _lambda512 = _mm512_set1_ps(lambda);
        for (; i + 15 < size; i += 16)
        {
            __m512 _p = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)ptr));

            __m512 _pos = _mm512_max_ps(_zero512, _p);
            __m512 _neg = _mm512_min_ps(_zero512, _p);

            __m512 _blob = exp512_ps(_neg);
            _blob = _mm512_sub_ps(_blob, _one512);
            _blob = _mm512_mul_ps(_alpha512, _blob);
            _blob = _mm512_mul_ps(_lambda512, _mm512_add_ps(_pos, _blob));

            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_blob));

            ptr += 16;
        }
#endif // __AVX512F__
        __m256 _zero256 = _mm256_setzero_ps();
        __m256 _one256 = _mm256_set1_ps(1.f);
        __m256 _alpha256 = _mm256_set1_ps(alphaxlambda / lambda);
        __m256 _lambda256 = _mm256_set1_ps(lambda);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));

            __m256 _pos = _mm256_max_ps(_zero256, _p);
            __m256 _neg = _mm256_min_ps(_zero256, _p);

            __m256 _blob = exp256_ps(_neg);
            _blob = _mm256_sub_ps(_blob, _one256);
            _blob = _mm256_mul_ps(_alpha256, _blob);
            _blob = _mm256_mul_ps(_lambda256, _mm256_add_ps(_pos, _blob));

            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_blob));

            ptr += 8;
        }
#endif // __AVX__
        __m128 _zero128 = _mm_setzero_ps();
        __m128 _one128 = _mm_set1_ps(1.f);
        __m128 _alpha128 = _mm_set1_ps(alphaxlambda / lambda);
        __m128 _lambda128 = _mm_set1_ps(lambda);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));

            __m128 _pos = _mm_max_ps(_zero128, _p);
            __m128 _neg = _mm_min_ps(_zero128, _p);

            __m128 _blob = exp_ps(_neg);
            _blob = _mm_sub_ps(_blob, _one128);
            _blob = _mm_mul_ps(_alpha128, _blob);
            _blob = _mm_mul_ps(_lambda128, _mm_add_ps(_pos, _blob));

            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_blob, _blob));

            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < 0)
                v = (expf(v) - 1.f) * alphaxlambda;
            else
                v = v * lambda;
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
}
