// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <float.h>

#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__FMA__ && !__FMA4__
void swiglu_fp32_avx2(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt, float scale, float offset, float limit);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
void swiglu_fp32_fma(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt, float scale, float offset, float limit);
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
void swiglu_fp32_fma4(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt, float scale, float offset, float limit);
#endif

static void swiglu_fp32(const Mat& gate, const Mat& up, Mat& top_blob, const Option& opt, float scale, float offset, float limit)
{
#if NCNN_RUNTIME_CPU && NCNN_AVX2 && __AVX__ && !__AVX2__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_avx2())
    {
        swiglu_fp32_avx2(gate, up, top_blob, opt, scale, offset, limit);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma())
    {
        swiglu_fp32_fma(gate, up, top_blob, opt, scale, offset, limit);
        return;
    }
#endif
#if NCNN_RUNTIME_CPU && NCNN_FMA4 && __AVX__ && !__FMA__ && !__FMA4__
    if (ncnn::cpu_support_x86_fma4())
    {
        swiglu_fp32_fma4(gate, up, top_blob, opt, scale, offset, limit);
        return;
    }
#endif

    const int channels = gate.c;
    const int size = gate.w * gate.h * gate.d * gate.elempack;
    const float climit = limit > 0.f ? limit : FLT_MAX;
    const float mlimit = -climit;

    const int block_size = 1024;
    const int nn_block = (size + block_size - 1) / block_size;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int ii = 0; ii < channels * nn_block; ii++)
    {
        const int q = ii / nn_block;
        const int offseti = ii % nn_block * block_size;
        const int len = std::min(size - offseti, block_size);

        const float* gateptr = (const float*)gate.channel(q) + offseti;
        const float* upptr = (const float*)up.channel(q) + offseti;
        float* outptr = (float*)top_blob.channel(q) + offseti;

        int i = 0;
#if __SSE2__
#if __AVX__
#if __AVX512F__
        const __m512 _one_avx512 = _mm512_set1_ps(1.f);
        const __m512 _nscale_avx512 = _mm512_set1_ps(-scale);
        const __m512 _offset_avx512 = _mm512_set1_ps(offset);
        const __m512 _limit_avx512 = _mm512_set1_ps(climit);
        const __m512 _mlimit_avx512 = _mm512_set1_ps(mlimit);
        for (; i + 15 < len; i += 16)
        {
            __m512 _g = _mm512_loadu_ps(gateptr);
            __m512 _u = _mm512_loadu_ps(upptr);
            _g = _mm512_min_ps(_g, _limit_avx512);
            _u = _mm512_min_ps(_mm512_max_ps(_u, _mlimit_avx512), _limit_avx512);
            _g = _mm512_div_ps(_g, _mm512_add_ps(_one_avx512, exp512_ps(_mm512_mul_ps(_nscale_avx512, _g))));
            _mm512_storeu_ps(outptr, _mm512_mul_ps(_g, _mm512_add_ps(_u, _offset_avx512)));
            gateptr += 16;
            upptr += 16;
            outptr += 16;
        }
#endif // __AVX512F__
        const __m256 _one_avx = _mm256_set1_ps(1.f);
        const __m256 _nscale_avx = _mm256_set1_ps(-scale);
        const __m256 _offset_avx = _mm256_set1_ps(offset);
        const __m256 _limit_avx = _mm256_set1_ps(climit);
        const __m256 _mlimit_avx = _mm256_set1_ps(mlimit);
        for (; i + 7 < len; i += 8)
        {
            __m256 _g = _mm256_loadu_ps(gateptr);
            __m256 _u = _mm256_loadu_ps(upptr);
            _g = _mm256_min_ps(_g, _limit_avx);
            _u = _mm256_min_ps(_mm256_max_ps(_u, _mlimit_avx), _limit_avx);
            _g = _mm256_div_ps(_g, _mm256_add_ps(_one_avx, exp256_ps(_mm256_mul_ps(_nscale_avx, _g))));
            _mm256_storeu_ps(outptr, _mm256_mul_ps(_g, _mm256_add_ps(_u, _offset_avx)));
            gateptr += 8;
            upptr += 8;
            outptr += 8;
        }
#endif // __AVX__
        const __m128 _one_sse = _mm_set1_ps(1.f);
        const __m128 _nscale_sse = _mm_set1_ps(-scale);
        const __m128 _offset_sse = _mm_set1_ps(offset);
        const __m128 _limit_sse = _mm_set1_ps(climit);
        const __m128 _mlimit_sse = _mm_set1_ps(mlimit);
        for (; i + 3 < len; i += 4)
        {
            __m128 _g = _mm_loadu_ps(gateptr);
            __m128 _u = _mm_loadu_ps(upptr);
            _g = _mm_min_ps(_g, _limit_sse);
            _u = _mm_min_ps(_mm_max_ps(_u, _mlimit_sse), _limit_sse);
            _g = _mm_div_ps(_g, _mm_add_ps(_one_sse, exp_ps(_mm_mul_ps(_nscale_sse, _g))));
            _mm_storeu_ps(outptr, _mm_mul_ps(_g, _mm_add_ps(_u, _offset_sse)));
            gateptr += 4;
            upptr += 4;
            outptr += 4;
        }
#endif // __SSE2__
        for (; i < len; i++)
        {
            float g = gateptr[0];
            float u = upptr[0];
            g = std::min(g, climit);
            u = std::min(std::max(u, mlimit), climit);
            outptr[0] = g / (1.f + expf(-scale * g)) * (u + offset);
            gateptr++;
            upptr++;
            outptr++;
        }
    }
}
