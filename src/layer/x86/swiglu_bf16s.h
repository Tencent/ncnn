// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__FMA__ && !__FMA4__
void swiglu_bf16s_avx2(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt, float scale, float offset, float limit);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void swiglu_bf16s_fma(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt, float scale, float offset, float limit);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void swiglu_bf16s_fma4(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt, float scale, float offset, float limit);
#endif

static void swiglu_bf16s(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt, float scale, float offset, float limit)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_avx2())
    {
        swiglu_bf16s_avx2(gate, up, top_blob, opt, scale, offset, limit);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
    {
        swiglu_bf16s_fma(gate, up, top_blob, opt, scale, offset, limit);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
    {
        swiglu_bf16s_fma4(gate, up, top_blob, opt, scale, offset, limit);
        return;
    }
#endif

    const int channels = gate.c;
    const int size = gate.w * gate.h * gate.d * gate.elempack;
    const bool use_limit = limit > 0.f;

    const int block_size = 1024;
    const int nn_block = (size + block_size - 1) / block_size;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < channels * nn_block; ii++)
    {
        const int q = ii / nn_block;
        const int offseti = ii % nn_block * block_size;
        const int len = std::min(size - offseti, block_size);

        const unsigned short* gateptr = (const unsigned short*)gate.channel(q) + offseti;
        const unsigned short* upptr = (const unsigned short*)up.channel(q) + offseti;
        unsigned short* outptr = (unsigned short*)top_blob.channel(q) + offseti;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        const __m512 _one_avx512 = _mm512_set1_ps(1.f);
        const __m512 _nscale_avx512 = _mm512_set1_ps(-scale);
        const __m512 _offset_avx512 = _mm512_set1_ps(offset);
        const __m512 _limit_avx512 = _mm512_set1_ps(limit);
        const __m512 _mlimit_avx512 = _mm512_set1_ps(-limit);
        for (; i + 15 < len; i += 16)
        {
            __m512 _g = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)gateptr));
            __m512 _u = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)upptr));
            if (use_limit)
            {
                _g = _mm512_min_ps(_g, _limit_avx512);
                _u = _mm512_min_ps(_mm512_max_ps(_u, _mlimit_avx512), _limit_avx512);
            }
            _g = _mm512_div_ps(_g, _mm512_add_ps(_one_avx512, exp512_ps(_mm512_mul_ps(_nscale_avx512, _g))));
            _mm256_storeu_si256((__m256i*)outptr, float2bfloat_avx512(_mm512_mul_ps(_g, _mm512_add_ps(_u, _offset_avx512))));
            gateptr += 16;
            upptr += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        const __m256 _one_avx = _mm256_set1_ps(1.f);
        const __m256 _nscale_avx = _mm256_set1_ps(-scale);
        const __m256 _offset_avx = _mm256_set1_ps(offset);
        const __m256 _limit_avx = _mm256_set1_ps(limit);
        const __m256 _mlimit_avx = _mm256_set1_ps(-limit);
        for (; i + 7 < len; i += 8)
        {
            __m256 _g = bfloat2float_avx(_mm_loadu_si128((const __m128i*)gateptr));
            __m256 _u = bfloat2float_avx(_mm_loadu_si128((const __m128i*)upptr));
            if (use_limit)
            {
                _g = _mm256_min_ps(_g, _limit_avx);
                _u = _mm256_min_ps(_mm256_max_ps(_u, _mlimit_avx), _limit_avx);
            }
            _g = _mm256_div_ps(_g, _mm256_add_ps(_one_avx, exp256_ps(_mm256_mul_ps(_nscale_avx, _g))));
            _mm_storeu_si128((__m128i*)outptr, float2bfloat_avx(_mm256_mul_ps(_g, _mm256_add_ps(_u, _offset_avx))));
            gateptr += 8;
            upptr += 8;
            outptr += 8;
        }
#endif // __AVX__
        const __m128 _one_sse = _mm_set1_ps(1.f);
        const __m128 _nscale_sse = _mm_set1_ps(-scale);
        const __m128 _offset_sse = _mm_set1_ps(offset);
        const __m128 _limit_sse = _mm_set1_ps(limit);
        const __m128 _mlimit_sse = _mm_set1_ps(-limit);
        for (; i + 3 < len; i += 4)
        {
            __m128 _g = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)gateptr));
            __m128 _u = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)upptr));
            if (use_limit)
            {
                _g = _mm_min_ps(_g, _limit_sse);
                _u = _mm_min_ps(_mm_max_ps(_u, _mlimit_sse), _limit_sse);
            }
            _g = _mm_div_ps(_g, _mm_add_ps(_one_sse, exp_ps(_mm_mul_ps(_nscale_sse, _g))));
            _mm_storel_epi64((__m128i*)outptr, float2bfloat_sse(_mm_mul_ps(_g, _mm_add_ps(_u, _offset_sse))));
            gateptr += 4;
            upptr += 4;
            outptr += 4;
        }
#endif // __SSE2__
        for (; i < len; i++)
        {
            float g = bfloat16_to_float32(gateptr[0]);
            float u = bfloat16_to_float32(upptr[0]);
            if (use_limit)
            {
                g = std::min(g, limit);
                u = std::min(std::max(u, -limit), limit);
            }
            outptr[0] = float32_to_bfloat16(g / (1.f + expf(-scale * g)) * (u + offset));
            gateptr++;
            upptr++;
            outptr++;
        }
    }
}
