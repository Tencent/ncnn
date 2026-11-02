// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__FMA__ && !__FMA4__
void glu_avx2(const float* input, const float* gate, float* output, int size);
void glu_pack_avx2(const float* input, const float* gate, float* output, int size, int elempack, int out_elempack);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void glu_fma(const float* input, const float* gate, float* output, int size);
void glu_pack_fma(const float* input, const float* gate, float* output, int size, int elempack, int out_elempack);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void glu_fma4(const float* input, const float* gate, float* output, int size);
void glu_pack_fma4(const float* input, const float* gate, float* output, int size, int elempack, int out_elempack);
#endif

static void glu_compute(const float* input, const float* gate, float* output, int size)
{
    int i = 0;
#if __AVX512F__
    const __m512 _one_avx512 = _mm512_set1_ps(1.f);
    const __m512 _zero_avx512 = _mm512_setzero_ps();
    for (; i + 15 < size; i += 16)
    {
        const __m512 _input = _mm512_loadu_ps(input + i);
        const __m512 _gate = _mm512_loadu_ps(gate + i);
        const __m512 _sigmoid = _mm512_div_ps(_one_avx512, _mm512_add_ps(_one_avx512, exp512_ps(_mm512_sub_ps(_zero_avx512, _gate))));
        _mm512_storeu_ps(output + i, _mm512_mul_ps(_input, _sigmoid));
    }
    if (i < size)
    {
        const __mmask16 _mask = (__mmask16)((1u << (size - i)) - 1);
        const __m512 _input = _mm512_maskz_loadu_ps(_mask, input + i);
        const __m512 _gate = _mm512_maskz_loadu_ps(_mask, gate + i);
        const __m512 _sigmoid = _mm512_div_ps(_one_avx512, _mm512_add_ps(_one_avx512, exp512_ps(_mm512_sub_ps(_zero_avx512, _gate))));
        _mm512_mask_storeu_ps(output + i, _mask, _mm512_mul_ps(_input, _sigmoid));
    }
#else // __AVX512F__
#if __AVX__
    const __m256 _one_avx = _mm256_set1_ps(1.f);
    const __m256 _zero_avx = _mm256_setzero_ps();
    for (; i + 7 < size; i += 8)
    {
        const __m256 _input = _mm256_loadu_ps(input + i);
        const __m256 _gate = _mm256_loadu_ps(gate + i);
        const __m256 _sigmoid = _mm256_div_ps(_one_avx, _mm256_add_ps(_one_avx, exp256_ps(_mm256_sub_ps(_zero_avx, _gate))));
        _mm256_storeu_ps(output + i, _mm256_mul_ps(_input, _sigmoid));
    }
#endif // __AVX__
#if __SSE2__
    const __m128 _one = _mm_set1_ps(1.f);
    const __m128 _zero = _mm_setzero_ps();
    for (; i + 3 < size; i += 4)
    {
        const __m128 _input = _mm_loadu_ps(input + i);
        const __m128 _gate = _mm_loadu_ps(gate + i);
        const __m128 _sigmoid = _mm_div_ps(_one, _mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _gate))));
        _mm_storeu_ps(output + i, _mm_mul_ps(_input, _sigmoid));
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        output[i] = input[i] * (1.f / (1.f + expf(-gate[i])));
    }
#endif // __AVX512F__
}

static void glu(const float* input, const float* gate, float* output, int size)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_avx2() && ncnn::cpu_support_x86_fma())
    {
        glu_avx2(input, gate, output, size);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
    {
        glu_fma(input, gate, output, size);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
    {
        glu_fma4(input, gate, output, size);
        return;
    }
#endif
    glu_compute(input, gate, output, size);
}

static void glu_pack4to1(const float* input, const float* gate, float* output, int size)
{
    int i = 0;
#if __SSE2__
    const __m128 _one = _mm_set1_ps(1.f);
    const __m128 _zero = _mm_setzero_ps();
    for (; i + 3 < size; i += 4)
    {
        const __m128 _input = _mm_set_ps(input[12], input[8], input[4], input[0]);
        const __m128 _gate = _mm_set_ps(gate[12], gate[8], gate[4], gate[0]);
        const __m128 _sigmoid = _mm_div_ps(_one, _mm_add_ps(_one, exp_ps(_mm_sub_ps(_zero, _gate))));
        _mm_storeu_ps(output, _mm_mul_ps(_input, _sigmoid));
        input += 16;
        gate += 16;
        output += 4;
    }
#endif // __SSE2__
    for (; i < size; i++)
    {
        *output++ = *input * (1.f / (1.f + expf(-*gate)));
        input += 4;
        gate += 4;
    }
}

static void glu_pack(const float* input, const float* gate, float* output, int size, int elempack, int out_elempack)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_avx2() && ncnn::cpu_support_x86_fma())
    {
        glu_pack_avx2(input, gate, output, size, elempack, out_elempack);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
    {
        glu_pack_fma(input, gate, output, size, elempack, out_elempack);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
    {
        glu_pack_fma4(input, gate, output, size, elempack, out_elempack);
        return;
    }
#endif

    if (out_elempack == 1)
    {
        glu_pack4to1(input, gate, output, size);
        return;
    }

    // Each value/gate subvector stays within one input pack. Write the result
    // directly in the smaller output pack without an intermediate tensor.
    for (int i = 0; i < size; i++)
    {
        glu_compute(input, gate, output, out_elempack);
        input += elempack;
        gate += elempack;
        output += out_elempack;
    }
}
