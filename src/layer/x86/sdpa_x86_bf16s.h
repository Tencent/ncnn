// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SDPA_X86_BF16S_H
#define SDPA_X86_BF16S_H

#include <float.h>

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && __AVX512F__ && !__AVX512BF16__
void decode_qk_dot_bf16s_avx512bf16(float* s, const float* q, const unsigned short* K, int n_start, int block_n, int d, float scale);
void decode_pv_gemv_bf16s_avx512bf16(float* out, const float* s, const unsigned short* V, int n_start, int block_n, int out_d);
void qk_gemm_bf16s_avx512bf16(float* S, const float* Q, const unsigned short* K, int m, int n, int d, float scale);
void pv_gemm_bf16s_avx512bf16(float* O, const float* P, const unsigned short* V, int m, int n, int d);
#endif

// ---------------------------------------------------------------------------
// decode_qk_dot_bf16s : Q(fp32) dot K(bf16) -> S(fp32)
// ---------------------------------------------------------------------------

#if __AVX512F__
static inline void decode_qk_dot_bf16s_avx512_kernel(float* s, const float* q, const unsigned short* K, int n_start, int block_n, int d, float scale)
{
    int j = 0;
    if (d >= 256)
    {
        for (; j + 7 < block_n; j += 8)
        {
            const unsigned short* k0 = K + (n_start + j + 0) * d;
            const unsigned short* k1 = K + (n_start + j + 1) * d;
            const unsigned short* k2 = K + (n_start + j + 2) * d;
            const unsigned short* k3 = K + (n_start + j + 3) * d;
            const unsigned short* k4 = K + (n_start + j + 4) * d;
            const unsigned short* k5 = K + (n_start + j + 5) * d;
            const unsigned short* k6 = K + (n_start + j + 6) * d;
            const unsigned short* k7 = K + (n_start + j + 7) * d;

            if (j + 15 < block_n)
            {
                _mm_prefetch((const char*)(K + (n_start + j + 8) * d), _MM_HINT_T1);
                _mm_prefetch((const char*)(K + (n_start + j + 9) * d), _MM_HINT_T1);
                _mm_prefetch((const char*)(K + (n_start + j + 10) * d), _MM_HINT_T1);
                _mm_prefetch((const char*)(K + (n_start + j + 11) * d), _MM_HINT_T1);
                _mm_prefetch((const char*)(K + (n_start + j + 12) * d), _MM_HINT_T1);
                _mm_prefetch((const char*)(K + (n_start + j + 13) * d), _MM_HINT_T1);
                _mm_prefetch((const char*)(K + (n_start + j + 14) * d), _MM_HINT_T1);
                _mm_prefetch((const char*)(K + (n_start + j + 15) * d), _MM_HINT_T1);
            }

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            __m512 acc4 = _mm512_setzero_ps();
            __m512 acc5 = _mm512_setzero_ps();
            __m512 acc6 = _mm512_setzero_ps();
            __m512 acc7 = _mm512_setzero_ps();

            int k = 0;
            for (; k + 15 < d; k += 16)
            {
                __m512 qv = _mm512_loadu_ps(q + k);
                acc0 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k0 + k))), acc0);
                acc1 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k1 + k))), acc1);
                acc2 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k2 + k))), acc2);
                acc3 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k3 + k))), acc3);
                acc4 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k4 + k))), acc4);
                acc5 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k5 + k))), acc5);
                acc6 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k6 + k))), acc6);
                acc7 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k7 + k))), acc7);
            }
            if (k < d)
            {
                __mmask16 mask_d = (__mmask16)((1u << (d - k)) - 1);
                __m512 qv = _mm512_maskz_loadu_ps(mask_d, q + k);
                __mmask16 mask16 = (__mmask16)((1u << (d - k)) - 1);
                acc0 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k0 + k)), acc0);
                acc1 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k1 + k)), acc1);
                acc2 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k2 + k)), acc2);
                acc3 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k3 + k)), acc3);
                acc4 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k4 + k)), acc4);
                acc5 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k5 + k)), acc5);
                acc6 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k6 + k)), acc6);
                acc7 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k7 + k)), acc7);
            }

            s[j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
            s[j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
            s[j + 2] = _mm512_comp_reduce_add_ps(acc2) * scale;
            s[j + 3] = _mm512_comp_reduce_add_ps(acc3) * scale;
            s[j + 4] = _mm512_comp_reduce_add_ps(acc4) * scale;
            s[j + 5] = _mm512_comp_reduce_add_ps(acc5) * scale;
            s[j + 6] = _mm512_comp_reduce_add_ps(acc6) * scale;
            s[j + 7] = _mm512_comp_reduce_add_ps(acc7) * scale;
        }
    }

    for (; j + 3 < block_n; j += 4)
    {
        const unsigned short* k0 = K + (n_start + j + 0) * d;
        const unsigned short* k1 = K + (n_start + j + 1) * d;
        const unsigned short* k2 = K + (n_start + j + 2) * d;
        const unsigned short* k3 = K + (n_start + j + 3) * d;

        if (j + 7 < block_n)
        {
            _mm_prefetch((const char*)(K + (n_start + j + 4) * d), _MM_HINT_T1);
            _mm_prefetch((const char*)(K + (n_start + j + 5) * d), _MM_HINT_T1);
            _mm_prefetch((const char*)(K + (n_start + j + 6) * d), _MM_HINT_T1);
            _mm_prefetch((const char*)(K + (n_start + j + 7) * d), _MM_HINT_T1);
        }

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        int k = 0;
        for (; k + 15 < d; k += 16)
        {
            __m512 qv = _mm512_loadu_ps(q + k);
            acc0 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k0 + k))), acc0);
            acc1 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k1 + k))), acc1);
            acc2 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k2 + k))), acc2);
            acc3 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k3 + k))), acc3);
        }
        if (k < d)
        {
            __mmask16 mask_d = (__mmask16)((1u << (d - k)) - 1);
            __m512 qv = _mm512_maskz_loadu_ps(mask_d, q + k);
            __mmask16 mask16 = (__mmask16)((1u << (d - k)) - 1);
            acc0 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k0 + k)), acc0);
            acc1 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k1 + k)), acc1);
            acc2 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k2 + k)), acc2);
            acc3 = _mm512_fmadd_ps(qv, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k3 + k)), acc3);
        }

        s[j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
        s[j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
        s[j + 2] = _mm512_comp_reduce_add_ps(acc2) * scale;
        s[j + 3] = _mm512_comp_reduce_add_ps(acc3) * scale;
    }

    for (; j < block_n; j++)
    {
        if (j + 4 < block_n)
            _mm_prefetch((const char*)(K + (n_start + j + 4) * d), _MM_HINT_T1);

        __m512 acc = _mm512_setzero_ps();
        int k = 0;
        const unsigned short* kptr = K + (n_start + j) * d;
        for (; k + 15 < d; k += 16)
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(q + k), bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + k))), acc);
        if (k < d)
        {
            __mmask16 mask_d = (__mmask16)((1u << (d - k)) - 1);
            __mmask16 mask16 = (__mmask16)((1u << (d - k)) - 1);
            acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask_d, q + k), bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, kptr + k)), acc);
        }
        s[j] = _mm512_comp_reduce_add_ps(acc) * scale;
    }
}
#endif // __AVX512F__

#if __AVX__
static inline void decode_qk_dot_bf16s_avx_kernel(float* s, const float* q, const unsigned short* K, int n_start, int block_n, int d, float scale)
{
    int j = 0;
    for (; j + 1 < block_n; j += 2)
    {
        const unsigned short* k0 = K + (n_start + j + 0) * d;
        const unsigned short* k1 = K + (n_start + j + 1) * d;

        if (j + 5 < block_n)
        {
            _mm_prefetch((const char*)(K + (n_start + j + 2) * d), _MM_HINT_T1);
            _mm_prefetch((const char*)(K + (n_start + j + 3) * d), _MM_HINT_T1);
        }

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        int k = 0;
        for (; k + 7 < d; k += 8)
        {
            __m256 qv = _mm256_loadu_ps(q + k);
            acc0 = _mm256_comp_fmadd_ps(qv, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(k0 + k))), acc0);
            acc1 = _mm256_comp_fmadd_ps(qv, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(k1 + k))), acc1);
        }

        float sum0 = _mm256_reduce_add_ps(acc0);
        float sum1 = _mm256_reduce_add_ps(acc1);

        for (; k < d; k++)
        {
            sum0 += q[k] * bfloat16_to_float32(k0[k]);
            sum1 += q[k] * bfloat16_to_float32(k1[k]);
        }

        s[j + 0] = sum0 * scale;
        s[j + 1] = sum1 * scale;
    }

    for (; j < block_n; j++)
    {
        if (j + 2 < block_n)
            _mm_prefetch((const char*)(K + (n_start + j + 2) * d), _MM_HINT_T1);

        const unsigned short* kptr = K + (n_start + j) * d;
        __m256 acc = _mm256_setzero_ps();
        int k = 0;
        for (; k + 7 < d; k += 8)
            acc = _mm256_comp_fmadd_ps(_mm256_loadu_ps(q + k), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(kptr + k))), acc);
        float sum = _mm256_reduce_add_ps(acc);
        for (; k < d; k++)
            sum += q[k] * bfloat16_to_float32(kptr[k]);
        s[j] = sum * scale;
    }
}
#endif // __AVX__

#if __SSE2__
static inline void decode_qk_dot_bf16s_sse2_kernel(float* s, const float* q, const unsigned short* K, int n_start, int block_n, int d, float scale)
{
    for (int j = 0; j < block_n; j++)
    {
        if (j + 4 < block_n)
            _mm_prefetch((const char*)(K + (n_start + j + 4) * d), _MM_HINT_T1);

        __m128 acc = _mm_setzero_ps();
        int k = 0;
        const unsigned short* kptr = K + (n_start + j) * d;
        for (; k + 3 < d; k += 4)
            acc = _mm_comp_fmadd_ps(_mm_loadu_ps(q + k), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(kptr + k))), acc);
        float sum = _mm_reduce_add_ps(acc);
        for (; k < d; k++)
            sum += q[k] * bfloat16_to_float32(kptr[k]);
        s[j] = sum * scale;
    }
}
#endif // __SSE2__

static inline void decode_qk_dot_bf16s_scalar_kernel(float* s, const float* q, const unsigned short* K, int n_start, int block_n, int d, float scale)
{
    for (int j = 0; j < block_n; j++)
    {
        float sum = 0.f;
        const unsigned short* kptr = K + (n_start + j) * d;
        for (int k = 0; k < d; k++)
            sum += q[k] * bfloat16_to_float32(kptr[k]);
        s[j] = sum * scale;
    }
}


// ---------------------------------------------------------------------------
// decode_pv_gemv_bf16s : S(fp32) gemv V(bf16) -> out(fp32)
// ---------------------------------------------------------------------------

#if __AVX512F__
static inline void decode_pv_gemv_bf16s_avx512_kernel(float* out, const float* s, const unsigned short* V, int n_start, int block_n, int out_d)
{
    int j = 0;
    for (; j + 1 < block_n; j += 2)
    {
        if (j + 6 < block_n)
            _mm_prefetch((const char*)(V + (n_start + j + 6) * out_d), _MM_HINT_T1);

        __m512 pvec0 = _mm512_set1_ps(s[j]);
        __m512 pvec1 = _mm512_set1_ps(s[j + 1]);
        int k = 0;
        for (; k + 31 < out_d; k += 32)
        {
            __m512 oval0 = _mm512_loadu_ps(out + k);
            __m512 oval1 = _mm512_loadu_ps(out + k + 16);
            __m512 v00 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + (n_start + j) * out_d + k)));
            __m512 v01 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + (n_start + j) * out_d + k + 16)));
            __m512 v10 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + (n_start + j + 1) * out_d + k)));
            __m512 v11 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + (n_start + j + 1) * out_d + k + 16)));
            oval0 = _mm512_fmadd_ps(pvec0, v00, oval0);
            oval1 = _mm512_fmadd_ps(pvec0, v01, oval1);
            oval0 = _mm512_fmadd_ps(pvec1, v10, oval0);
            oval1 = _mm512_fmadd_ps(pvec1, v11, oval1);
            _mm512_storeu_ps(out + k, oval0);
            _mm512_storeu_ps(out + k + 16, oval1);
        }
        if (k + 15 < out_d)
        {
            __m512 oval = _mm512_loadu_ps(out + k);
            __m512 v0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + (n_start + j) * out_d + k)));
            __m512 v1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + (n_start + j + 1) * out_d + k)));
            oval = _mm512_fmadd_ps(pvec0, v0, oval);
            oval = _mm512_fmadd_ps(pvec1, v1, oval);
            _mm512_storeu_ps(out + k, oval);
            k += 16;
        }
        if (k < out_d)
        {
            __mmask16 mask_d = (__mmask16)((1u << (out_d - k)) - 1);
            __mmask16 mask16 = (__mmask16)((1u << (out_d - k)) - 1);
            __m512 oval = _mm512_maskz_loadu_ps(mask_d, out + k);
            __m512 v0 = bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, V + (n_start + j) * out_d + k));
            __m512 v1 = bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, V + (n_start + j + 1) * out_d + k));
            oval = _mm512_fmadd_ps(pvec0, v0, oval);
            oval = _mm512_fmadd_ps(pvec1, v1, oval);
            _mm512_mask_storeu_ps(out + k, mask_d, oval);
        }
    }
    for (; j < block_n; j++)
    {
        __m512 pvec512 = _mm512_set1_ps(s[j]);
        int k = 0;
        for (; k + 31 < out_d; k += 32)
        {
            __m512 oval0 = _mm512_loadu_ps(out + k);
            __m512 oval1 = _mm512_loadu_ps(out + k + 16);
            __m512 v0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + (n_start + j) * out_d + k)));
            __m512 v1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + (n_start + j) * out_d + k + 16)));
            oval0 = _mm512_fmadd_ps(pvec512, v0, oval0);
            oval1 = _mm512_fmadd_ps(pvec512, v1, oval1);
            _mm512_storeu_ps(out + k, oval0);
            _mm512_storeu_ps(out + k + 16, oval1);
        }
        if (k + 15 < out_d)
        {
            __m512 oval = _mm512_loadu_ps(out + k);
            __m512 vval = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + (n_start + j) * out_d + k)));
            _mm512_storeu_ps(out + k, _mm512_fmadd_ps(pvec512, vval, oval));
            k += 16;
        }
        if (k < out_d)
        {
            __mmask16 mask_d = (__mmask16)((1u << (out_d - k)) - 1);
            __mmask16 mask16 = (__mmask16)((1u << (out_d - k)) - 1);
            __m512 oval = _mm512_maskz_loadu_ps(mask_d, out + k);
            __m512 vval = bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, V + (n_start + j) * out_d + k));
            _mm512_mask_storeu_ps(out + k, mask_d, _mm512_fmadd_ps(pvec512, vval, oval));
        }
    }
}
#endif // __AVX512F__

#if __AVX__
static inline void decode_pv_gemv_bf16s_avx_kernel(float* out, const float* s, const unsigned short* V, int n_start, int block_n, int out_d)
{
    int j = 0;
    for (; j < block_n; j++)
    {
        if (j + 4 < block_n)
            _mm_prefetch((const char*)(V + (n_start + j + 4) * out_d), _MM_HINT_T1);

        int k = 0;
        __m256 pvec256 = _mm256_set1_ps(s[j]);
        for (; k + 7 < out_d; k += 8)
        {
            __m256 oval = _mm256_loadu_ps(out + k);
            __m256 vval = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(V + (n_start + j) * out_d + k)));
            _mm256_storeu_ps(out + k, _mm256_comp_fmadd_ps(pvec256, vval, oval));
        }
        for (; k < out_d; k++)
            out[k] += s[j] * bfloat16_to_float32(V[(n_start + j) * out_d + k]);
    }
}
#endif // __AVX__

#if __SSE2__
static inline void decode_pv_gemv_bf16s_sse2_kernel(float* out, const float* s, const unsigned short* V, int n_start, int block_n, int out_d)
{
    for (int j = 0; j < block_n; j++)
    {
        if (j + 4 < block_n)
            _mm_prefetch((const char*)(V + (n_start + j + 4) * out_d), _MM_HINT_T1);

        __m128 pvec128 = _mm_set1_ps(s[j]);
        int k = 0;
        for (; k + 3 < out_d; k += 4)
        {
            __m128 oval = _mm_loadu_ps(out + k);
            __m128 vval = bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(V + (n_start + j) * out_d + k)));
            _mm_storeu_ps(out + k, _mm_comp_fmadd_ps(pvec128, vval, oval));
        }
        for (; k < out_d; k++)
            out[k] += s[j] * bfloat16_to_float32(V[(n_start + j) * out_d + k]);
    }
}
#endif // __SSE2__

static inline void decode_pv_gemv_bf16s_scalar_kernel(float* out, const float* s, const unsigned short* V, int n_start, int block_n, int out_d)
{
    for (int j = 0; j < block_n; j++)
    {
        float p = s[j];
        const unsigned short* vptr = V + (n_start + j) * out_d;
        for (int k = 0; k < out_d; k++)
            out[k] += p * bfloat16_to_float32(vptr[k]);
    }
}


// ---------------------------------------------------------------------------
// sdpa_decode_bf16s : full decode with bf16 K/V
// ---------------------------------------------------------------------------

static inline void sdpa_decode_bf16s(float* out, const float* q,
                                     const unsigned short* K, const unsigned short* V, const float* mask,
                                     int n, int d, int out_d, float scale)
{
    const int BLOCK_N = 128;
#if __AVX512F__
    __attribute__((aligned(64))) float s[BLOCK_N];
#elif __AVX__
    __attribute__((aligned(32))) float s[BLOCK_N];
#elif __SSE2__
    __attribute__((aligned(16))) float s[BLOCK_N];
#else
    float s[BLOCK_N];
#endif

    // vec_zero
    {
#if __AVX512F__
        __m512 zero512 = _mm512_setzero_ps();
        int i = 0;
        for (; i + 15 < out_d; i += 16)
            _mm512_storeu_ps(out + i, zero512);
        if (i < out_d)
        {
            __mmask16 mask = (__mmask16)((1u << (out_d - i)) - 1);
            _mm512_mask_storeu_ps(out + i, mask, zero512);
        }
#else
        int i = 0;
#if __AVX__
        __m256 zero256 = _mm256_setzero_ps();
        for (; i + 7 < out_d; i += 8)
            _mm256_storeu_ps(out + i, zero256);
#endif
#if __SSE2__
        __m128 zero128 = _mm_setzero_ps();
        for (; i + 3 < out_d; i += 4)
            _mm_storeu_ps(out + i, zero128);
#endif
        for (; i < out_d; i++)
            out[i] = 0.f;
#endif
    }

    float m = -FLT_MAX;
    float l = 0.f;

    for (int n_start = 0; n_start < n; n_start += BLOCK_N)
    {
        int block_n = std::min(BLOCK_N, n - n_start);

#if __AVX512F__
        decode_qk_dot_bf16s_avx512_kernel(s, q, K, n_start, block_n, d, scale);
#elif __AVX__
        decode_qk_dot_bf16s_avx_kernel(s, q, K, n_start, block_n, d, scale);
#elif __SSE2__
        decode_qk_dot_bf16s_sse2_kernel(s, q, K, n_start, block_n, d, scale);
#else
        decode_qk_dot_bf16s_scalar_kernel(s, q, K, n_start, block_n, d, scale);
#endif

        if (mask)
        {
#if __AVX512F__
            int j = 0;
            for (; j + 15 < block_n; j += 16)
                _mm512_storeu_ps(s + j, _mm512_add_ps(_mm512_loadu_ps(s + j), _mm512_loadu_ps(mask + n_start + j)));
            if (j < block_n)
            {
                __mmask16 mask_n = (__mmask16)((1u << (block_n - j)) - 1);
                _mm512_mask_storeu_ps(s + j, mask_n,
                                      _mm512_add_ps(_mm512_maskz_loadu_ps(mask_n, s + j), _mm512_maskz_loadu_ps(mask_n, mask + n_start + j)));
            }
#elif __AVX__
            int j = 0;
            for (; j + 7 < block_n; j += 8)
                _mm256_storeu_ps(s + j, _mm256_add_ps(_mm256_loadu_ps(s + j), _mm256_loadu_ps(mask + n_start + j)));
            for (; j < block_n; j++)
                s[j] += mask[n_start + j];
#elif __SSE2__
            int j = 0;
            for (; j + 3 < block_n; j += 4)
                _mm_storeu_ps(s + j, _mm_add_ps(_mm_loadu_ps(s + j), _mm_loadu_ps(mask + n_start + j)));
            for (; j < block_n; j++)
                s[j] += mask[n_start + j];
#else
            for (int j = 0; j < block_n; j++)
                s[j] += mask[n_start + j];
#endif
        }

        // tile max
#if __AVX512F__
        __m512 vmax = _mm512_set1_ps(-FLT_MAX);
        int j = 0;
        for (; j + 15 < block_n; j += 16)
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(s + j));
        if (j < block_n)
        {
            __mmask16 mask_n = (__mmask16)((1u << (block_n - j)) - 1);
            vmax = _mm512_max_ps(vmax, _mm512_mask_loadu_ps(_mm512_set1_ps(-FLT_MAX), mask_n, s + j));
        }
        float tile_m = _mm512_comp_reduce_max_ps(vmax);
#elif __AVX__
        __m256 vmax = _mm256_set1_ps(-FLT_MAX);
        int j = 0;
        for (; j + 7 < block_n; j += 8)
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(s + j));
        float tile_m = _mm256_reduce_max_ps(vmax);
        for (; j < block_n; j++)
            tile_m = std::max(tile_m, s[j]);
#elif __SSE2__
        __m128 vmax = _mm_set1_ps(-FLT_MAX);
        int j = 0;
        for (; j + 3 < block_n; j += 4)
            vmax = _mm_max_ps(vmax, _mm_loadu_ps(s + j));
        float tile_m = _mm_reduce_max_ps(vmax);
        for (; j < block_n; j++)
            tile_m = std::max(tile_m, s[j]);
#else
        float tile_m = -FLT_MAX;
        for (int j = 0; j < block_n; j++)
            tile_m = std::max(tile_m, s[j]);
#endif

        float new_m = std::max(m, tile_m);
        if (m != new_m)
        {
            float scale_factor = expf(m - new_m);
            l *= scale_factor;
            // vec_scale(out, scale_factor, out_d);
            {
#if __AVX512F__
                __m512 vscale512 = _mm512_set1_ps(scale_factor);
                int i = 0;
                for (; i + 15 < out_d; i += 16)
                    _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(out + i), vscale512));
                if (i < out_d)
                {
                    __mmask16 mask = (__mmask16)((1u << (out_d - i)) - 1);
                    _mm512_mask_storeu_ps(out + i, mask, _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, out + i), vscale512));
                }
#else
                int i = 0;
#if __AVX__
                __m256 vscale256 = _mm256_set1_ps(scale_factor);
                for (; i + 7 < out_d; i += 8)
                    _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), vscale256));
#endif
#if __SSE2__
                __m128 vscale128 = _mm_set1_ps(scale_factor);
                for (; i + 3 < out_d; i += 4)
                    _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(out + i), vscale128));
#endif
                for (; i < out_d; i++)
                    out[i] *= scale_factor;
#endif
            }
        }

        // exp and sum
#if __AVX512F__
        __m512 vm_new = _mm512_set1_ps(new_m);
        __m512 vsum = _mm512_setzero_ps();
        j = 0;
        for (; j + 15 < block_n; j += 16)
        {
            __m512 pvec = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(s + j), vm_new));
            _mm512_storeu_ps(s + j, pvec);
            vsum = _mm512_add_ps(vsum, pvec);
        }
        if (j < block_n)
        {
            __mmask16 mask_n = (__mmask16)((1u << (block_n - j)) - 1);
            __m512 pvec = exp512_ps(_mm512_sub_ps(_mm512_maskz_loadu_ps(mask_n, s + j), vm_new));
            _mm512_mask_storeu_ps(s + j, mask_n, pvec);
            vsum = _mm512_mask_add_ps(vsum, mask_n, vsum, pvec);
        }
        l += _mm512_comp_reduce_add_ps(vsum);
#elif __AVX__
        __m256 vm_new = _mm256_set1_ps(new_m);
        __m256 vsum = _mm256_setzero_ps();
        j = 0;
        for (; j + 7 < block_n; j += 8)
        {
            __m256 pvec = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(s + j), vm_new));
            _mm256_storeu_ps(s + j, pvec);
            vsum = _mm256_add_ps(vsum, pvec);
        }
        float l_add = _mm256_reduce_add_ps(vsum);
        for (; j < block_n; j++)
        {
            s[j] = expf(s[j] - new_m);
            l_add += s[j];
        }
        l += l_add;
#elif __SSE2__
        __m128 vm_new = _mm_set1_ps(new_m);
        __m128 vsum = _mm_setzero_ps();
        j = 0;
        for (; j + 3 < block_n; j += 4)
        {
            __m128 pvec = exp_ps(_mm_sub_ps(_mm_loadu_ps(s + j), vm_new));
            _mm_storeu_ps(s + j, pvec);
            vsum = _mm_add_ps(vsum, pvec);
        }
        float l_add = _mm_reduce_add_ps(vsum);
        for (; j < block_n; j++)
        {
            s[j] = expf(s[j] - new_m);
            l_add += s[j];
        }
        l += l_add;
#else
        float l_add = 0.f;
        for (int j = 0; j < block_n; j++)
        {
            s[j] = expf(s[j] - new_m);
            l_add += s[j];
        }
        l += l_add;
#endif

#if __AVX512F__
        decode_pv_gemv_bf16s_avx512_kernel(out, s, V, n_start, block_n, out_d);
#elif __AVX__
        decode_pv_gemv_bf16s_avx_kernel(out, s, V, n_start, block_n, out_d);
#elif __SSE2__
        decode_pv_gemv_bf16s_sse2_kernel(out, s, V, n_start, block_n, out_d);
#else
        decode_pv_gemv_bf16s_scalar_kernel(out, s, V, n_start, block_n, out_d);
#endif

        m = new_m;
    }

    float inv_l = 1.f / l;
    // vec_scale(out, inv_l, out_d);
    {
#if __AVX512F__
        __m512 vscale512 = _mm512_set1_ps(inv_l);
        int i = 0;
        for (; i + 15 < out_d; i += 16)
            _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(out + i), vscale512));
        if (i < out_d)
        {
            __mmask16 mask = (__mmask16)((1u << (out_d - i)) - 1);
            _mm512_mask_storeu_ps(out + i, mask, _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, out + i), vscale512));
        }
#else
        int i = 0;
#if __AVX__
        __m256 vscale256 = _mm256_set1_ps(inv_l);
        for (; i + 7 < out_d; i += 8)
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), vscale256));
#endif
#if __SSE2__
        __m128 vscale128 = _mm_set1_ps(inv_l);
        for (; i + 3 < out_d; i += 4)
            _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(out + i), vscale128));
#endif
        for (; i < out_d; i++)
            out[i] *= inv_l;
#endif
    }
}


// ---------------------------------------------------------------------------
// sdpa_decode_chunk_bf16s / sdpa_decode_reduce_bf16s
// ---------------------------------------------------------------------------

static inline void sdpa_decode_chunk_bf16s(
    float* out, float* m_out, float* l_out,
    const float* q, const unsigned short* K, const unsigned short* V, const float* mask,
    int n_start, int n_end, int d, int out_d, float scale)
{
    const int BLOCK_N = 128;
#if __AVX512F__
    __attribute__((aligned(64))) float s[BLOCK_N];
#elif __AVX__
    __attribute__((aligned(32))) float s[BLOCK_N];
#elif __SSE2__
    __attribute__((aligned(16))) float s[BLOCK_N];
#else
    float s[BLOCK_N];
#endif

    // vec_zero
    {
#if __AVX512F__
        __m512 zero512 = _mm512_setzero_ps();
        int i = 0;
        for (; i + 15 < out_d; i += 16)
            _mm512_storeu_ps(out + i, zero512);
        if (i < out_d)
        {
            __mmask16 mask = (__mmask16)((1u << (out_d - i)) - 1);
            _mm512_mask_storeu_ps(out + i, mask, zero512);
        }
#else
        int i = 0;
#if __AVX__
        __m256 zero256 = _mm256_setzero_ps();
        for (; i + 7 < out_d; i += 8)
            _mm256_storeu_ps(out + i, zero256);
#endif
#if __SSE2__
        __m128 zero128 = _mm_setzero_ps();
        for (; i + 3 < out_d; i += 4)
            _mm_storeu_ps(out + i, zero128);
#endif
        for (; i < out_d; i++)
            out[i] = 0.f;
#endif
    }

    float m = -FLT_MAX;
    float l = 0.f;

    for (int n = n_start; n < n_end; n += BLOCK_N)
    {
        int block_n = std::min(BLOCK_N, n_end - n);

#if __AVX512F__
        decode_qk_dot_bf16s_avx512_kernel(s, q, K, n, block_n, d, scale);
#elif __AVX__
        decode_qk_dot_bf16s_avx_kernel(s, q, K, n, block_n, d, scale);
#elif __SSE2__
        decode_qk_dot_bf16s_sse2_kernel(s, q, K, n, block_n, d, scale);
#else
        decode_qk_dot_bf16s_scalar_kernel(s, q, K, n, block_n, d, scale);
#endif

        if (mask)
        {
#if __AVX512F__
            int j = 0;
            for (; j + 15 < block_n; j += 16)
                _mm512_storeu_ps(s + j, _mm512_add_ps(_mm512_loadu_ps(s + j), _mm512_loadu_ps(mask + n + j)));
            if (j < block_n)
            {
                __mmask16 mask_n = (__mmask16)((1u << (block_n - j)) - 1);
                _mm512_mask_storeu_ps(s + j, mask_n,
                                      _mm512_add_ps(_mm512_maskz_loadu_ps(mask_n, s + j), _mm512_maskz_loadu_ps(mask_n, mask + n + j)));
            }
#elif __AVX__
            int j = 0;
            for (; j + 7 < block_n; j += 8)
                _mm256_storeu_ps(s + j, _mm256_add_ps(_mm256_loadu_ps(s + j), _mm256_loadu_ps(mask + n + j)));
            for (; j < block_n; j++)
                s[j] += mask[n + j];
#elif __SSE2__
            int j = 0;
            for (; j + 3 < block_n; j += 4)
                _mm_storeu_ps(s + j, _mm_add_ps(_mm_loadu_ps(s + j), _mm_loadu_ps(mask + n + j)));
            for (; j < block_n; j++)
                s[j] += mask[n + j];
#else
            for (int j = 0; j < block_n; j++)
                s[j] += mask[n + j];
#endif
        }

        // tile max
#if __AVX512F__
        __m512 vmax = _mm512_set1_ps(-FLT_MAX);
        int j = 0;
        for (; j + 15 < block_n; j += 16)
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(s + j));
        if (j < block_n)
        {
            __mmask16 mask_n = (__mmask16)((1u << (block_n - j)) - 1);
            vmax = _mm512_max_ps(vmax, _mm512_mask_loadu_ps(_mm512_set1_ps(-FLT_MAX), mask_n, s + j));
        }
        float tile_m = _mm512_comp_reduce_max_ps(vmax);
#elif __AVX__
        __m256 vmax = _mm256_set1_ps(-FLT_MAX);
        int j = 0;
        for (; j + 7 < block_n; j += 8)
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(s + j));
        float tile_m = _mm256_reduce_max_ps(vmax);
        for (; j < block_n; j++)
            tile_m = std::max(tile_m, s[j]);
#elif __SSE2__
        __m128 vmax = _mm_set1_ps(-FLT_MAX);
        int j = 0;
        for (; j + 3 < block_n; j += 4)
            vmax = _mm_max_ps(vmax, _mm_loadu_ps(s + j));
        float tile_m = _mm_reduce_max_ps(vmax);
        for (; j < block_n; j++)
            tile_m = std::max(tile_m, s[j]);
#else
        float tile_m = -FLT_MAX;
        for (int j = 0; j < block_n; j++)
            tile_m = std::max(tile_m, s[j]);
#endif

        float new_m = std::max(m, tile_m);
        if (m != new_m)
        {
            float scale_factor = expf(m - new_m);
            l *= scale_factor;
            // vec_scale
            {
#if __AVX512F__
                __m512 vscale512 = _mm512_set1_ps(scale_factor);
                int i = 0;
                for (; i + 15 < out_d; i += 16)
                    _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(out + i), vscale512));
                if (i < out_d)
                {
                    __mmask16 mask = (__mmask16)((1u << (out_d - i)) - 1);
                    _mm512_mask_storeu_ps(out + i, mask, _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, out + i), vscale512));
                }
#else
                int i = 0;
#if __AVX__
                __m256 vscale256 = _mm256_set1_ps(scale_factor);
                for (; i + 7 < out_d; i += 8)
                    _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), vscale256));
#endif
#if __SSE2__
                __m128 vscale128 = _mm_set1_ps(scale_factor);
                for (; i + 3 < out_d; i += 4)
                    _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(out + i), vscale128));
#endif
                for (; i < out_d; i++)
                    out[i] *= scale_factor;
#endif
            }
        }

        // exp and sum
#if __AVX512F__
        __m512 vm_new = _mm512_set1_ps(new_m);
        __m512 vsum = _mm512_setzero_ps();
        j = 0;
        for (; j + 15 < block_n; j += 16)
        {
            __m512 pvec = exp512_ps(_mm512_sub_ps(_mm512_loadu_ps(s + j), vm_new));
            _mm512_storeu_ps(s + j, pvec);
            vsum = _mm512_add_ps(vsum, pvec);
        }
        if (j < block_n)
        {
            __mmask16 mask_n = (__mmask16)((1u << (block_n - j)) - 1);
            __m512 pvec = exp512_ps(_mm512_sub_ps(_mm512_maskz_loadu_ps(mask_n, s + j), vm_new));
            _mm512_mask_storeu_ps(s + j, mask_n, pvec);
            vsum = _mm512_mask_add_ps(vsum, mask_n, vsum, pvec);
        }
        l += _mm512_comp_reduce_add_ps(vsum);
#elif __AVX__
        __m256 vm_new = _mm256_set1_ps(new_m);
        __m256 vsum = _mm256_setzero_ps();
        j = 0;
        for (; j + 7 < block_n; j += 8)
        {
            __m256 pvec = exp256_ps(_mm256_sub_ps(_mm256_loadu_ps(s + j), vm_new));
            _mm256_storeu_ps(s + j, pvec);
            vsum = _mm256_add_ps(vsum, pvec);
        }
        float l_add = _mm256_reduce_add_ps(vsum);
        for (; j < block_n; j++)
        {
            s[j] = expf(s[j] - new_m);
            l_add += s[j];
        }
        l += l_add;
#elif __SSE2__
        __m128 vm_new = _mm_set1_ps(new_m);
        __m128 vsum = _mm_setzero_ps();
        j = 0;
        for (; j + 3 < block_n; j += 4)
        {
            __m128 pvec = exp_ps(_mm_sub_ps(_mm_loadu_ps(s + j), vm_new));
            _mm_storeu_ps(s + j, pvec);
            vsum = _mm_add_ps(vsum, pvec);
        }
        float l_add = _mm_reduce_add_ps(vsum);
        for (; j < block_n; j++)
        {
            s[j] = expf(s[j] - new_m);
            l_add += s[j];
        }
        l += l_add;
#else
        float l_add = 0.f;
        for (int j = 0; j < block_n; j++)
        {
            s[j] = expf(s[j] - new_m);
            l_add += s[j];
        }
        l += l_add;
#endif

#if __AVX512F__
        decode_pv_gemv_bf16s_avx512_kernel(out, s, V, n, block_n, out_d);
#elif __AVX__
        decode_pv_gemv_bf16s_avx_kernel(out, s, V, n, block_n, out_d);
#elif __SSE2__
        decode_pv_gemv_bf16s_sse2_kernel(out, s, V, n, block_n, out_d);
#else
        decode_pv_gemv_bf16s_scalar_kernel(out, s, V, n, block_n, out_d);
#endif

        m = new_m;
    }

    *m_out = m;
    *l_out = l;
}

static inline void sdpa_decode_reduce_bf16s(
    float* out, int out_d,
    const float* partials, int num_chunks, int partial_stride)
{
    float M_final = -FLT_MAX;
    float S_final = 0.f;
    // vec_zero
    {
#if __AVX512F__
        __m512 zero512 = _mm512_setzero_ps();
        int i = 0;
        for (; i + 15 < out_d; i += 16)
            _mm512_storeu_ps(out + i, zero512);
        if (i < out_d)
        {
            __mmask16 mask = (__mmask16)((1u << (out_d - i)) - 1);
            _mm512_mask_storeu_ps(out + i, mask, zero512);
        }
#else
        int i = 0;
#if __AVX__
        __m256 zero256 = _mm256_setzero_ps();
        for (; i + 7 < out_d; i += 8)
            _mm256_storeu_ps(out + i, zero256);
#endif
#if __SSE2__
        __m128 zero128 = _mm_setzero_ps();
        for (; i + 3 < out_d; i += 4)
            _mm_storeu_ps(out + i, zero128);
#endif
        for (; i < out_d; i++)
            out[i] = 0.f;
#endif
    }

    for (int c = 0; c < num_chunks; c++)
    {
        const float* p = partials + c * partial_stride;
        float M_chunk = p[0];
        float S_chunk = p[1];
        if (S_chunk == 0.f) continue;

        const float* VKQ_chunk = p + 2;

        float M_new = std::max(M_final, M_chunk);
        float scale_final = expf(M_final - M_new);
        float scale_chunk = expf(M_chunk - M_new);

        for (int k = 0; k < out_d; k++)
        {
            out[k] = out[k] * scale_final + VKQ_chunk[k] * scale_chunk;
        }

        S_final = S_final * scale_final + S_chunk * scale_chunk;
        M_final = M_new;
    }

    if (S_final != 0.f)
    {
        float inv_s = 1.f / S_final;
        // vec_scale
        {
#if __AVX512F__
            __m512 vscale512 = _mm512_set1_ps(inv_s);
            int i = 0;
            for (; i + 15 < out_d; i += 16)
                _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(out + i), vscale512));
            if (i < out_d)
            {
                __mmask16 mask = (__mmask16)((1u << (out_d - i)) - 1);
                _mm512_mask_storeu_ps(out + i, mask, _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, out + i), vscale512));
            }
#else
            int i = 0;
#if __AVX__
            __m256 vscale256 = _mm256_set1_ps(inv_s);
            for (; i + 7 < out_d; i += 8)
                _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(out + i), vscale256));
#endif
#if __SSE2__
            __m128 vscale128 = _mm_set1_ps(inv_s);
            for (; i + 3 < out_d; i += 4)
                _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(out + i), vscale128));
#endif
            for (; i < out_d; i++)
                out[i] *= inv_s;
#endif
        }
    }
}


// ---------------------------------------------------------------------------
// qk_gemm_bf16s : Q(fp32) x K^T(bf16) -> S(fp32)   [prefill]
// ---------------------------------------------------------------------------

#if __AVX512F__
static void qk_gemm_bf16s_avx512(float* S, const float* Q, const unsigned short* K,
                                 int m, int n, int d, float scale)
{
    int i = 0;
    for (; i + 8 <= m; i += 8)
    {
        int j = 0;
        for (; j + 2 <= n; j += 2)
        {
            const unsigned short* k0 = K + (j + 0) * d;
            const unsigned short* k1 = K + (j + 1) * d;

            __m512 acc[8][2];
            for (int mi = 0; mi < 8; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
            }

            int k = 0;
            for (; k + 15 < d; k += 16)
            {
                __m512 kv0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k0 + k)));
                __m512 kv1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k1 + k)));

                for (int mi = 0; mi < 8; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * d + k);
                    acc[mi][0] = _mm512_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            if (k < d)
            {
                __mmask16 mask = (__mmask16)((1u << (d - k)) - 1);
                __mmask16 mask16 = (__mmask16)((1u << (d - k)) - 1);
                __m512 kv0 = bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k0 + k));
                __m512 kv1 = bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k1 + k));

                for (int mi = 0; mi < 8; mi++)
                {
                    __m512 qvec = _mm512_maskz_loadu_ps(mask, Q + (i + mi) * d + k);
                    acc[mi][0] = _mm512_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 8; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm512_comp_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm512_comp_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const unsigned short* kptr = K + j * d;

            __m512 acc[8];
            for (int mi = 0; mi < 8; mi++)
                acc[mi] = _mm512_setzero_ps();

            int k = 0;
            for (; k + 15 < d; k += 16)
            {
                __m512 kvec = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + k)));
                for (int mi = 0; mi < 8; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * d + k);
                    acc[mi] = _mm512_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            if (k < d)
            {
                __mmask16 mask = (__mmask16)((1u << (d - k)) - 1);
                __mmask16 mask16 = (__mmask16)((1u << (d - k)) - 1);
                __m512 kvec = bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, kptr + k));
                for (int mi = 0; mi < 8; mi++)
                {
                    __m512 qvec = _mm512_maskz_loadu_ps(mask, Q + (i + mi) * d + k);
                    acc[mi] = _mm512_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 8; mi++)
                S[(i + mi) * n + j] = _mm512_comp_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i + 4 <= m; i += 4)
    {
        int j = 0;
        for (; j + 2 <= n; j += 2)
        {
            const unsigned short* k0 = K + (j + 0) * d;
            const unsigned short* k1 = K + (j + 1) * d;

            __m512 acc[4][2];
            for (int mi = 0; mi < 4; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
            }

            int k = 0;
            for (; k + 15 < d; k += 16)
            {
                __m512 kv0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k0 + k)));
                __m512 kv1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k1 + k)));

                for (int mi = 0; mi < 4; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * d + k);
                    acc[mi][0] = _mm512_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            if (k < d)
            {
                __mmask16 mask = (__mmask16)((1u << (d - k)) - 1);
                __mmask16 mask16 = (__mmask16)((1u << (d - k)) - 1);
                __m512 kv0 = bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k0 + k));
                __m512 kv1 = bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k1 + k));

                for (int mi = 0; mi < 4; mi++)
                {
                    __m512 qvec = _mm512_maskz_loadu_ps(mask, Q + (i + mi) * d + k);
                    acc[mi][0] = _mm512_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 4; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm512_comp_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm512_comp_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const unsigned short* kptr = K + j * d;

            __m512 acc[4];
            for (int mi = 0; mi < 4; mi++)
                acc[mi] = _mm512_setzero_ps();

            int k = 0;
            for (; k + 15 < d; k += 16)
            {
                __m512 kvec = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + k)));
                for (int mi = 0; mi < 4; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * d + k);
                    acc[mi] = _mm512_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            if (k < d)
            {
                __mmask16 mask = (__mmask16)((1u << (d - k)) - 1);
                __mmask16 mask16 = (__mmask16)((1u << (d - k)) - 1);
                __m512 kvec = bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, kptr + k));
                for (int mi = 0; mi < 4; mi++)
                {
                    __m512 qvec = _mm512_maskz_loadu_ps(mask, Q + (i + mi) * d + k);
                    acc[mi] = _mm512_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 4; mi++)
                S[(i + mi) * n + j] = _mm512_comp_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i < m; i++)
    {
        int j = 0;
        for (; j + 3 < n; j += 4)
        {
            const float* qptr = Q + i * d;
            const unsigned short* k0 = K + (j + 0) * d;
            const unsigned short* k1 = K + (j + 1) * d;
            const unsigned short* k2 = K + (j + 2) * d;
            const unsigned short* k3 = K + (j + 3) * d;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            int k = 0;
            for (; k + 15 < d; k += 16)
            {
                __m512 qvec = _mm512_loadu_ps(qptr + k);
                acc0 = _mm512_fmadd_ps(qvec, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k0 + k))), acc0);
                acc1 = _mm512_fmadd_ps(qvec, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k1 + k))), acc1);
                acc2 = _mm512_fmadd_ps(qvec, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k2 + k))), acc2);
                acc3 = _mm512_fmadd_ps(qvec, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(k3 + k))), acc3);
            }

            if (k < d)
            {
                __mmask16 mask = (__mmask16)((1u << (d - k)) - 1);
                __mmask16 mask16 = (__mmask16)((1u << (d - k)) - 1);
                __m512 qvec = _mm512_maskz_loadu_ps(mask, qptr + k);
                acc0 = _mm512_fmadd_ps(qvec, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k0 + k)), acc0);
                acc1 = _mm512_fmadd_ps(qvec, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k1 + k)), acc1);
                acc2 = _mm512_fmadd_ps(qvec, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k2 + k)), acc2);
                acc3 = _mm512_fmadd_ps(qvec, bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, k3 + k)), acc3);
            }

            S[i * n + j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
            S[i * n + j + 2] = _mm512_comp_reduce_add_ps(acc2) * scale;
            S[i * n + j + 3] = _mm512_comp_reduce_add_ps(acc3) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * d;
            const unsigned short* kptr = K + j * d;
            int k = 0;
            __m512 vacc = _mm512_setzero_ps();
            for (; k + 15 < d; k += 16)
                vacc = _mm512_fmadd_ps(_mm512_loadu_ps(qptr + k), bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(kptr + k))), vacc);
            if (k < d)
            {
                __mmask16 mask = (__mmask16)((1u << (d - k)) - 1);
                __mmask16 mask16 = (__mmask16)((1u << (d - k)) - 1);
                vacc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, qptr + k), bfloat2float_avx512(_mm256_maskz_loadu_epi16(mask16, kptr + k)), vacc);
            }
            S[i * n + j] = _mm512_comp_reduce_add_ps(vacc) * scale;
        }
    }
}
#endif // __AVX512F__


#if __AVX__
static void qk_gemm_bf16s_avx(float* S, const float* Q, const unsigned short* K,
                              int m, int n, int d, float scale)
{
    int i = 0;
    for (; i + 6 <= m; i += 6)
    {
        int j = 0;
        for (; j + 2 <= n; j += 2)
        {
            const unsigned short* k0 = K + (j + 0) * d;
            const unsigned short* k1 = K + (j + 1) * d;

            __m256 acc[6][2];
            for (int mi = 0; mi < 6; mi++)
            {
                acc[mi][0] = _mm256_setzero_ps();
                acc[mi][1] = _mm256_setzero_ps();
            }

            int k = 0;
            for (; k + 7 < d; k += 8)
            {
                __m256 kv0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(k0 + k)));
                __m256 kv1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(k1 + k)));

                for (int mi = 0; mi < 6; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * d + k);
                    acc[mi][0] = _mm256_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm256_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 6; mi++)
            {
                float sum0 = _mm256_reduce_add_ps(acc[mi][0]);
                float sum1 = _mm256_reduce_add_ps(acc[mi][1]);
                for (; k < d; k++)
                {
                    float qv = Q[(i + mi) * d + k];
                    sum0 += qv * bfloat16_to_float32(k0[k]);
                    sum1 += qv * bfloat16_to_float32(k1[k]);
                }
                S[(i + mi) * n + j + 0] = sum0 * scale;
                S[(i + mi) * n + j + 1] = sum1 * scale;
            }
        }

        for (; j < n; j++)
        {
            const unsigned short* kptr = K + j * d;
            __m256 acc[6];
            for (int mi = 0; mi < 6; mi++)
                acc[mi] = _mm256_setzero_ps();

            int k = 0;
            for (; k + 7 < d; k += 8)
            {
                __m256 kvec = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(kptr + k)));
                for (int mi = 0; mi < 6; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * d + k);
                    acc[mi] = _mm256_comp_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 6; mi++)
            {
                float sum = _mm256_reduce_add_ps(acc[mi]);
                for (; k < d; k++)
                    sum += Q[(i + mi) * d + k] * bfloat16_to_float32(kptr[k]);
                S[(i + mi) * n + j] = sum * scale;
            }
        }
    }

    for (; i < m; i++)
    {
        int j = 0;
        for (; j + 1 < n; j += 2)
        {
            const float* qptr = Q + i * d;
            const unsigned short* k0 = K + (j + 0) * d;
            const unsigned short* k1 = K + (j + 1) * d;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            int k = 0;
            for (; k + 7 < d; k += 8)
            {
                __m256 qvec = _mm256_loadu_ps(qptr + k);
                acc0 = _mm256_comp_fmadd_ps(qvec, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(k0 + k))), acc0);
                acc1 = _mm256_comp_fmadd_ps(qvec, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(k1 + k))), acc1);
            }

            float sum0 = _mm256_reduce_add_ps(acc0);
            float sum1 = _mm256_reduce_add_ps(acc1);
            for (; k < d; k++)
            {
                float qv = qptr[k];
                sum0 += qv * bfloat16_to_float32(k0[k]);
                sum1 += qv * bfloat16_to_float32(k1[k]);
            }
            S[i * n + j + 0] = sum0 * scale;
            S[i * n + j + 1] = sum1 * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * d;
            const unsigned short* kptr = K + j * d;
            float sum = 0.f;
            int k = 0;
            __m256 vacc = _mm256_setzero_ps();
            for (; k + 7 < d; k += 8)
                vacc = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr + k), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(kptr + k))), vacc);
            sum = _mm256_reduce_add_ps(vacc);
            for (; k < d; k++)
                sum += qptr[k] * bfloat16_to_float32(kptr[k]);
            S[i * n + j] = sum * scale;
        }
    }
}
#endif // __AVX__

#if __SSE2__
static void qk_gemm_bf16s_sse2(float* S, const float* Q, const unsigned short* K,
                               int m, int n, int d, float scale)
{
    for (int i = 0; i < m; i++)
    {
        int j = 0;
        for (; j + 1 < n; j += 2)
        {
            const float* qptr = Q + i * d;
            const unsigned short* k0 = K + (j + 0) * d;
            const unsigned short* k1 = K + (j + 1) * d;

            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();
            int k = 0;
            for (; k + 3 < d; k += 4)
            {
                __m128 qvec = _mm_loadu_ps(qptr + k);
                acc0 = _mm_comp_fmadd_ps(qvec, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(k0 + k))), acc0);
                acc1 = _mm_comp_fmadd_ps(qvec, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(k1 + k))), acc1);
            }
            float sum0 = _mm_reduce_add_ps(acc0);
            float sum1 = _mm_reduce_add_ps(acc1);
            for (; k < d; k++)
            {
                float qv = qptr[k];
                sum0 += qv * bfloat16_to_float32(k0[k]);
                sum1 += qv * bfloat16_to_float32(k1[k]);
            }
            S[i * n + j + 0] = sum0 * scale;
            S[i * n + j + 1] = sum1 * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * d;
            const unsigned short* kptr = K + j * d;
            __m128 acc = _mm_setzero_ps();
            int k = 0;
            for (; k + 3 < d; k += 4)
                acc = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr + k), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(kptr + k))), acc);
            float sum = _mm_reduce_add_ps(acc);
            for (; k < d; k++)
                sum += qptr[k] * bfloat16_to_float32(kptr[k]);
            S[i * n + j] = sum * scale;
        }
    }
}
#endif // __SSE2__

static void qk_gemm_bf16s_scalar(float* S, const float* Q, const unsigned short* K,
                                 int m, int n, int d, float scale)
{
    for (int i = 0; i < m; i++)
    {
        const float* qptr = Q + i * d;
        for (int j = 0; j < n; j++)
        {
            const unsigned short* kptr = K + j * d;
            float sum = 0.f;
            for (int k = 0; k < d; k++)
                sum += qptr[k] * bfloat16_to_float32(kptr[k]);
            S[i * n + j] = sum * scale;
        }
    }
}


// ---------------------------------------------------------------------------
// pv_gemm_bf16s : P(fp32) x V(bf16) -> O(fp32)   [prefill]
// ---------------------------------------------------------------------------

#if __AVX512F__
static void pv_gemm_bf16s_avx512(float* O, const float* P, const unsigned short* V, int m, int n, int d)
{
    int dd = 0;
    for (; dd + 127 < d; dd += 128)
    {
        int i = 0;
        for (; i + 4 <= m; i += 4)
        {
            float* op[4];
            const float* pptr[4];
            for (int mi = 0; mi < 4; mi++)
            {
                op[mi] = O + (i + mi) * d + dd;
                pptr[mi] = P + (i + mi) * n;
            }

            __m512 acc[4][8];
            for (int mi = 0; mi < 4; mi++)
            {
                acc[mi][0] = _mm512_loadu_ps(op[mi] + 0 * 16);
                acc[mi][1] = _mm512_loadu_ps(op[mi] + 1 * 16);
                acc[mi][2] = _mm512_loadu_ps(op[mi] + 2 * 16);
                acc[mi][3] = _mm512_loadu_ps(op[mi] + 3 * 16);
                acc[mi][4] = _mm512_loadu_ps(op[mi] + 4 * 16);
                acc[mi][5] = _mm512_loadu_ps(op[mi] + 5 * 16);
                acc[mi][6] = _mm512_loadu_ps(op[mi] + 6 * 16);
                acc[mi][7] = _mm512_loadu_ps(op[mi] + 7 * 16);
            }

            for (int j = 0; j < n; j++)
            {
                __m512 v0 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 0 * 16)));
                __m512 v1 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 1 * 16)));
                __m512 v2 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 2 * 16)));
                __m512 v3 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 3 * 16)));
                __m512 v4 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 4 * 16)));
                __m512 v5 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 5 * 16)));
                __m512 v6 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 6 * 16)));
                __m512 v7 = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 7 * 16)));

                for (int mi = 0; mi < 4; mi++)
                {
                    __m512 pvec = _mm512_set1_ps(pptr[mi][j]);
                    acc[mi][0] = _mm512_fmadd_ps(pvec, v0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(pvec, v1, acc[mi][1]);
                    acc[mi][2] = _mm512_fmadd_ps(pvec, v2, acc[mi][2]);
                    acc[mi][3] = _mm512_fmadd_ps(pvec, v3, acc[mi][3]);
                    acc[mi][4] = _mm512_fmadd_ps(pvec, v4, acc[mi][4]);
                    acc[mi][5] = _mm512_fmadd_ps(pvec, v5, acc[mi][5]);
                    acc[mi][6] = _mm512_fmadd_ps(pvec, v6, acc[mi][6]);
                    acc[mi][7] = _mm512_fmadd_ps(pvec, v7, acc[mi][7]);
                }
            }

            for (int mi = 0; mi < 4; mi++)
            {
                _mm512_storeu_ps(op[mi] + 0 * 16, acc[mi][0]);
                _mm512_storeu_ps(op[mi] + 1 * 16, acc[mi][1]);
                _mm512_storeu_ps(op[mi] + 2 * 16, acc[mi][2]);
                _mm512_storeu_ps(op[mi] + 3 * 16, acc[mi][3]);
                _mm512_storeu_ps(op[mi] + 4 * 16, acc[mi][4]);
                _mm512_storeu_ps(op[mi] + 5 * 16, acc[mi][5]);
                _mm512_storeu_ps(op[mi] + 6 * 16, acc[mi][6]);
                _mm512_storeu_ps(op[mi] + 7 * 16, acc[mi][7]);
            }
        }

        for (; i < m; i++)
        {
            float* optr = O + i * d + dd;
            const float* pptr = P + i * n;
            __m512 acc0 = _mm512_loadu_ps(optr + 0 * 16);
            __m512 acc1 = _mm512_loadu_ps(optr + 1 * 16);
            __m512 acc2 = _mm512_loadu_ps(optr + 2 * 16);
            __m512 acc3 = _mm512_loadu_ps(optr + 3 * 16);
            __m512 acc4 = _mm512_loadu_ps(optr + 4 * 16);
            __m512 acc5 = _mm512_loadu_ps(optr + 5 * 16);
            __m512 acc6 = _mm512_loadu_ps(optr + 6 * 16);
            __m512 acc7 = _mm512_loadu_ps(optr + 7 * 16);

            for (int j = 0; j < n; j++)
            {
                __m512 pvec = _mm512_set1_ps(pptr[j]);
                acc0 = _mm512_fmadd_ps(pvec, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 0 * 16))), acc0);
                acc1 = _mm512_fmadd_ps(pvec, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 1 * 16))), acc1);
                acc2 = _mm512_fmadd_ps(pvec, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 2 * 16))), acc2);
                acc3 = _mm512_fmadd_ps(pvec, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 3 * 16))), acc3);
                acc4 = _mm512_fmadd_ps(pvec, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 4 * 16))), acc4);
                acc5 = _mm512_fmadd_ps(pvec, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 5 * 16))), acc5);
                acc6 = _mm512_fmadd_ps(pvec, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 6 * 16))), acc6);
                acc7 = _mm512_fmadd_ps(pvec, bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd + 7 * 16))), acc7);
            }

            _mm512_storeu_ps(optr + 0 * 16, acc0);
            _mm512_storeu_ps(optr + 1 * 16, acc1);
            _mm512_storeu_ps(optr + 2 * 16, acc2);
            _mm512_storeu_ps(optr + 3 * 16, acc3);
            _mm512_storeu_ps(optr + 4 * 16, acc4);
            _mm512_storeu_ps(optr + 5 * 16, acc5);
            _mm512_storeu_ps(optr + 6 * 16, acc6);
            _mm512_storeu_ps(optr + 7 * 16, acc7);
        }
    }

    for (; dd + 15 < d; dd += 16)
    {
        int i = 0;
        for (; i + 4 <= m; i += 4)
        {
            float* op[4];
            const float* pptr[4];
            for (int mi = 0; mi < 4; mi++)
            {
                op[mi] = O + (i + mi) * d + dd;
                pptr[mi] = P + (i + mi) * n;
            }
            __m512 acc[4];
            for (int mi = 0; mi < 4; mi++)
                acc[mi] = _mm512_loadu_ps(op[mi]);

            for (int j = 0; j < n; j++)
            {
                __m512 vvec = bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd)));
                for (int mi = 0; mi < 4; mi++)
                    acc[mi] = _mm512_fmadd_ps(_mm512_set1_ps(pptr[mi][j]), vvec, acc[mi]);
            }
            for (int mi = 0; mi < 4; mi++)
                _mm512_storeu_ps(op[mi], acc[mi]);
        }

        for (; i < m; i++)
        {
            float* optr = O + i * d + dd;
            const float* pptr = P + i * n;
            __m512 acc = _mm512_loadu_ps(optr);
            for (int j = 0; j < n; j++)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(pptr[j]), bfloat2float_avx512(_mm256_loadu_si256((const __m256i*)(V + j * d + dd))), acc);
            _mm512_storeu_ps(optr, acc);
        }
    }

    for (; dd < d; dd++)
    {
        for (int i = 0; i < m; i++)
        {
            float* optr = O + i * d + dd;
            const float* pptr = P + i * n;
            float acc = optr[0];
            for (int j = 0; j < n; j++)
                acc += pptr[j] * bfloat16_to_float32(V[j * d + dd]);
            optr[0] = acc;
        }
    }
}
#endif // __AVX512F__

#if __AVX__
static void pv_gemm_bf16s_avx(float* O, const float* P, const unsigned short* V, int m, int n, int d)
{
    int dd = 0;
    for (; dd + 31 < d; dd += 32)
    {
        int i = 0;
        for (; i + 2 <= m; i += 2)
        {
            float* op[2];
            const float* pptr[2];
            for (int mi = 0; mi < 2; mi++)
            {
                op[mi] = O + (i + mi) * d + dd;
                pptr[mi] = P + (i + mi) * n;
            }

            __m256 acc[2][4];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm256_loadu_ps(op[mi] + 0 * 8);
                acc[mi][1] = _mm256_loadu_ps(op[mi] + 1 * 8);
                acc[mi][2] = _mm256_loadu_ps(op[mi] + 2 * 8);
                acc[mi][3] = _mm256_loadu_ps(op[mi] + 3 * 8);
            }

            for (int j = 0; j < n; j++)
            {
                __m256 v0 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(V + j * d + dd + 0 * 8)));
                __m256 v1 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(V + j * d + dd + 1 * 8)));
                __m256 v2 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(V + j * d + dd + 2 * 8)));
                __m256 v3 = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(V + j * d + dd + 3 * 8)));

                for (int mi = 0; mi < 2; mi++)
                {
                    __m256 pvec = _mm256_set1_ps(pptr[mi][j]);
                    acc[mi][0] = _mm256_comp_fmadd_ps(pvec, v0, acc[mi][0]);
                    acc[mi][1] = _mm256_comp_fmadd_ps(pvec, v1, acc[mi][1]);
                    acc[mi][2] = _mm256_comp_fmadd_ps(pvec, v2, acc[mi][2]);
                    acc[mi][3] = _mm256_comp_fmadd_ps(pvec, v3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                _mm256_storeu_ps(op[mi] + 0 * 8, acc[mi][0]);
                _mm256_storeu_ps(op[mi] + 1 * 8, acc[mi][1]);
                _mm256_storeu_ps(op[mi] + 2 * 8, acc[mi][2]);
                _mm256_storeu_ps(op[mi] + 3 * 8, acc[mi][3]);
            }
        }

        for (; i < m; i++)
        {
            float* optr = O + i * d + dd;
            const float* pptr = P + i * n;
            __m256 acc0 = _mm256_loadu_ps(optr + 0 * 8);
            __m256 acc1 = _mm256_loadu_ps(optr + 1 * 8);
            __m256 acc2 = _mm256_loadu_ps(optr + 2 * 8);
            __m256 acc3 = _mm256_loadu_ps(optr + 3 * 8);

            for (int j = 0; j < n; j++)
            {
                __m256 pvec = _mm256_set1_ps(pptr[j]);
                acc0 = _mm256_comp_fmadd_ps(pvec, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(V + j * d + dd + 0 * 8))), acc0);
                acc1 = _mm256_comp_fmadd_ps(pvec, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(V + j * d + dd + 1 * 8))), acc1);
                acc2 = _mm256_comp_fmadd_ps(pvec, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(V + j * d + dd + 2 * 8))), acc2);
                acc3 = _mm256_comp_fmadd_ps(pvec, bfloat2float_avx(_mm_loadu_si128((const __m128i*)(V + j * d + dd + 3 * 8))), acc3);
            }

            _mm256_storeu_ps(optr + 0 * 8, acc0);
            _mm256_storeu_ps(optr + 1 * 8, acc1);
            _mm256_storeu_ps(optr + 2 * 8, acc2);
            _mm256_storeu_ps(optr + 3 * 8, acc3);
        }
    }

    for (; dd + 7 < d; dd += 8)
    {
        int i = 0;
        for (; i + 2 <= m; i += 2)
        {
            float* op[2];
            const float* pptr[2];
            for (int mi = 0; mi < 2; mi++)
            {
                op[mi] = O + (i + mi) * d + dd;
                pptr[mi] = P + (i + mi) * n;
            }
            __m256 acc[2];
            for (int mi = 0; mi < 2; mi++)
                acc[mi] = _mm256_loadu_ps(op[mi]);

            for (int j = 0; j < n; j++)
            {
                __m256 vvec = bfloat2float_avx(_mm_loadu_si128((const __m128i*)(V + j * d + dd)));
                for (int mi = 0; mi < 2; mi++)
                    acc[mi] = _mm256_comp_fmadd_ps(_mm256_set1_ps(pptr[mi][j]), vvec, acc[mi]);
            }
            for (int mi = 0; mi < 2; mi++)
                _mm256_storeu_ps(op[mi], acc[mi]);
        }

        for (; i < m; i++)
        {
            float* optr = O + i * d + dd;
            const float* pptr = P + i * n;
            __m256 acc = _mm256_loadu_ps(optr);
            for (int j = 0; j < n; j++)
                acc = _mm256_comp_fmadd_ps(_mm256_set1_ps(pptr[j]), bfloat2float_avx(_mm_loadu_si128((const __m128i*)(V + j * d + dd))), acc);
            _mm256_storeu_ps(optr, acc);
        }
    }

    for (; dd < d; dd++)
    {
        for (int i = 0; i < m; i++)
        {
            float* optr = O + i * d + dd;
            const float* pptr = P + i * n;
            float acc = optr[0];
            for (int j = 0; j < n; j++)
                acc += pptr[j] * bfloat16_to_float32(V[j * d + dd]);
            optr[0] = acc;
        }
    }
}
#endif // __AVX__

#if __SSE2__
static void pv_gemm_bf16s_sse2(float* O, const float* P, const unsigned short* V, int m, int n, int d)
{
    int dd = 0;
    for (; dd + 15 < d; dd += 16)
    {
        for (int i = 0; i < m; i++)
        {
            float* optr = O + i * d + dd;
            const float* pptr = P + i * n;
            __m128 acc0 = _mm_loadu_ps(optr + 0 * 4);
            __m128 acc1 = _mm_loadu_ps(optr + 1 * 4);
            __m128 acc2 = _mm_loadu_ps(optr + 2 * 4);
            __m128 acc3 = _mm_loadu_ps(optr + 3 * 4);

            for (int j = 0; j < n; j++)
            {
                __m128 pvec = _mm_set1_ps(pptr[j]);
                acc0 = _mm_comp_fmadd_ps(pvec, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(V + j * d + dd + 0 * 4))), acc0);
                acc1 = _mm_comp_fmadd_ps(pvec, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(V + j * d + dd + 1 * 4))), acc1);
                acc2 = _mm_comp_fmadd_ps(pvec, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(V + j * d + dd + 2 * 4))), acc2);
                acc3 = _mm_comp_fmadd_ps(pvec, bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(V + j * d + dd + 3 * 4))), acc3);
            }

            _mm_storeu_ps(optr + 0 * 4, acc0);
            _mm_storeu_ps(optr + 1 * 4, acc1);
            _mm_storeu_ps(optr + 2 * 4, acc2);
            _mm_storeu_ps(optr + 3 * 4, acc3);
        }
    }

    for (; dd + 3 < d; dd += 4)
    {
        for (int i = 0; i < m; i++)
        {
            float* optr = O + i * d + dd;
            const float* pptr = P + i * n;
            __m128 acc = _mm_loadu_ps(optr);
            for (int j = 0; j < n; j++)
                acc = _mm_comp_fmadd_ps(_mm_set1_ps(pptr[j]), bfloat2float_sse(_mm_loadl_epi64((const __m128i*)(V + j * d + dd))), acc);
            _mm_storeu_ps(optr, acc);
        }
    }

    for (; dd < d; dd++)
    {
        for (int i = 0; i < m; i++)
        {
            float* optr = O + i * d + dd;
            const float* pptr = P + i * n;
            float acc = optr[0];
            for (int j = 0; j < n; j++)
                acc += pptr[j] * bfloat16_to_float32(V[j * d + dd]);
            optr[0] = acc;
        }
    }
}
#endif // __SSE2__

static void pv_gemm_bf16s_scalar(float* O, const float* P, const unsigned short* V, int m, int n, int d)
{
    for (int i = 0; i < m; i++)
    {
        float* optr = O + i * d;
        const float* pptr = P + i * n;
        for (int j = 0; j < n; j++)
        {
            float p = pptr[j];
            const unsigned short* vptr = V + j * d;
            for (int k = 0; k < d; k++)
                optr[k] += p * bfloat16_to_float32(vptr[k]);
        }
    }
}


// ---------------------------------------------------------------------------
// Dispatch wrappers
// ---------------------------------------------------------------------------

static inline void decode_qk_dot_bf16s(float* s, const float* q, const unsigned short* K, int n_start, int block_n, int d, float scale)
{
#if __AVX512F__
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        decode_qk_dot_bf16s_avx512bf16(s, q, K, n_start, block_n, d, scale);
        return;
    }
#endif
    decode_qk_dot_bf16s_avx512_kernel(s, q, K, n_start, block_n, d, scale);
#elif __AVX__
    decode_qk_dot_bf16s_avx_kernel(s, q, K, n_start, block_n, d, scale);
#elif __SSE2__
    decode_qk_dot_bf16s_sse2_kernel(s, q, K, n_start, block_n, d, scale);
#else
    decode_qk_dot_bf16s_scalar_kernel(s, q, K, n_start, block_n, d, scale);
#endif
}

static inline void decode_pv_gemv_bf16s(float* out, const float* s, const unsigned short* V, int n_start, int block_n, int out_d)
{
#if __AVX512F__
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        decode_pv_gemv_bf16s_avx512bf16(out, s, V, n_start, block_n, out_d);
        return;
    }
#endif
    decode_pv_gemv_bf16s_avx512_kernel(out, s, V, n_start, block_n, out_d);
#elif __AVX__
    decode_pv_gemv_bf16s_avx_kernel(out, s, V, n_start, block_n, out_d);
#elif __SSE2__
    decode_pv_gemv_bf16s_sse2_kernel(out, s, V, n_start, block_n, out_d);
#else
    decode_pv_gemv_bf16s_scalar_kernel(out, s, V, n_start, block_n, out_d);
#endif
}

static inline void qk_gemm_bf16s_dispatch(float* S, const float* Q, const unsigned short* K, int m, int n, int d, float scale)
{
#if __AVX512F__
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        qk_gemm_bf16s_avx512bf16(S, Q, K, m, n, d, scale);
        return;
    }
#endif
    qk_gemm_bf16s_avx512(S, Q, K, m, n, d, scale);
#elif __AVX__
    qk_gemm_bf16s_avx(S, Q, K, m, n, d, scale);
#elif __SSE2__
    qk_gemm_bf16s_sse2(S, Q, K, m, n, d, scale);
#else
    qk_gemm_bf16s_scalar(S, Q, K, m, n, d, scale);
#endif
}

static inline void pv_gemm_bf16s_dispatch(float* O, const float* P, const unsigned short* V, int m, int n, int d)
{
#if __AVX512F__
#if NCNN_RUNTIME_CPU && NCNN_AVX512BF16 && !__AVX512BF16__
    if (ncnn::cpu_support_x86_avx512_bf16())
    {
        pv_gemm_bf16s_avx512bf16(O, P, V, m, n, d);
        return;
    }
#endif
    pv_gemm_bf16s_avx512(O, P, V, m, n, d);
#elif __AVX__
    pv_gemm_bf16s_avx(O, P, V, m, n, d);
#elif __SSE2__
    pv_gemm_bf16s_sse2(O, P, V, m, n, d);
#else
    pv_gemm_bf16s_scalar(O, P, V, m, n, d);
#endif
}

static inline void sdpa_decode_bf16s_dispatch(float* out, const float* q,
        const unsigned short* K, const unsigned short* V, const float* mask,
        int n, int d, int out_d, float scale)
{
    sdpa_decode_bf16s(out, q, K, V, mask, n, d, out_d, scale);
}

static inline void sdpa_decode_chunk_bf16s_dispatch(
    float* out, float* m_out, float* l_out,
    const float* q, const unsigned short* K, const unsigned short* V, const float* mask,
    int n_start, int n_end, int d, int out_d, float scale)
{
    sdpa_decode_chunk_bf16s(out, m_out, l_out, q, K, V, mask, n_start, n_end, d, out_d, scale);
}

static inline void sdpa_decode_reduce_bf16s_dispatch(
    float* out, int out_d,
    const float* partials, int num_chunks, int partial_stride)
{
    sdpa_decode_reduce_bf16s(out, out_d, partials, num_chunks, partial_stride);
}

#endif // SDPA_X86_BF16S_H
