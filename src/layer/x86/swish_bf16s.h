// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void swish_bf16s_avx512bf16(Mat& a, const Option& opt);
#endif

static void swish_bf16s(Mat& a, const Option& opt)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        swish_bf16s_avx512bf16(a, opt);
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
            _p = _mm512_div_ps(_p, _mm512_add_ps(_one_avx512, exp512_ps(_mm512_sub_ps(_zero_avx512, _p))));
            _mm256_storeu_si256((__m256i*)ptr, float2bfloat_avx512(_p));
            ptr += 16;
        }
#endif // __AVX512F__
        __m256 _one_avx = _mm256_set1_ps(1.f);
        __m256 _zero_avx = _mm256_setzero_ps();
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx(_mm_loadu_si128((const __m128i*)ptr));
            _p = _mm256_div_ps(_p, _mm256_add_ps(_one_avx, exp256_ps(_mm256_sub_ps(_zero_avx, _p))));
            _mm_storeu_si128((__m128i*)ptr, float2bfloat_avx(_p));
            ptr += 8;
        }
#endif // __AVX__
        __m128 _one = _mm_set1_ps(1.f);
        __m128 _zero = _mm_setzero_ps();
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)ptr));
            _p = _mm_div_ps(_p, _mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _p))));
            _mm_storel_epi64((__m128i*)ptr, float2bfloat_sse(_p, _p));
            ptr += 4;
        }
#endif // __SSE2__
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            v = v / (1.f + expf(-v));
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
}
