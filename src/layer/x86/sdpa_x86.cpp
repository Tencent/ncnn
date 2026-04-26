// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_x86.h"

#include "layer_type.h"

#include "cpu.h"
#include "x86_usability.h"

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

#include <float.h>
#include <chrono>
#include <math.h>
#include <string.h>

namespace ncnn {

SDPA_x86::SDPA_x86()
{
#if NCNN_BF16
    support_bf16_storage = false;
#endif
}

int SDPA_x86::create_pipeline(const Option& /*_opt*/)
{
    if (int8_scale_term)
    {
        support_bf16_storage = false;
    }

    return 0;
}

int SDPA_x86::destroy_pipeline(const Option& /*_opt*/)
{
    return 0;
}

#if NCNN_INT8
static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static void dynamic_quantize_blockwise(const float* src, signed char* dst, float* scales, int width)
{
    const int block_size = 32;
    int num_blocks = (width + block_size - 1) / block_size;
    for (int b = 0; b < num_blocks; b++)
    {
        int start = b * block_size;
        int end = start + block_size < width ? start + block_size : width;
        float absmax = 0.f;
        for (int i = start; i < end; i++)
        {
            absmax = std::max(absmax, (float)fabs(src[i]));
        }
        float scale = absmax == 0.f ? 1.f : 127.f / absmax;
        scales[b] = scale;
        for (int i = start; i < end; i++)
        {
            dst[i] = float2int8(src[i] * scale);
        }
    }
}
#endif // NCNN_INT8


static inline void qk_gemm_scalar(float* S, const float* Q, const float* K,
        int m, int n, int d, float scale)
{
    for (int i = 0; i < m; i++)
    {
        const float* qptr = Q + i * d;
        for (int j = 0; j < n; j++)
        {
            const float* kptr = K + j * d;
            float sum = 0.f;
            for (int k = 0; k < d; k++)
                sum += qptr[k] * kptr[k];
            S[i * n + j] = sum * scale;
        }
    }
}

static inline void pv_gemm_scalar(float* O, const float* P, const float* V,
        int m, int n, int d)
{
    for (int i = 0; i < m; i++)
    {
        float* optr = O + i * d;
        const float* pptr = P + i * n;
        for (int j = 0; j < n; j++)
        {
            float p = pptr[j];
            const float* vptr = V + j * d;
            for (int k = 0; k < d; k++)
                optr[k] += p * vptr[k];
        }
    }
}

static inline void vec_scale_scalar(float* x, float s, int n)
{
    for (int i = 0; i < n; i++) x[i] *= s;
}

static inline void vec_zero_scalar(float* x, int n)
{
    for (int i = 0; i < n; i++) x[i] = 0.f;
}

static void sdpa_decode_scalar(float* out, const float* q,
    const float* K, const float* V, const float* mask,
    int n, int d, int out_d, float scale)
{
    const int BLOCK_N = 128;
    float s[BLOCK_N];

    for (int k = 0; k < out_d; k++) out[k] = 0.f;

    float m = -FLT_MAX;
    float l = 0.f;

    for (int n_start = 0; n_start < n; n_start += BLOCK_N)
    {
        int block_n = std::min(BLOCK_N, n - n_start);

        for (int j = 0; j < block_n; j++)
        {
            float sum = 0.f;
            for (int k = 0; k < d; k++)
                sum += q[k] * K[(n_start + j) * d + k];
            s[j] = sum * scale;
        }

        if (mask)
        {
            for (int j = 0; j < block_n; j++)
                s[j] += mask[n_start + j];
        }

        float tile_m = -FLT_MAX;
        for (int j = 0; j < block_n; j++)
            tile_m = std::max(tile_m, s[j]);

        float new_m = std::max(m, tile_m);
        if (m != new_m)
        {
            float scale_factor = expf(m - new_m);
            l *= scale_factor;
            for (int k = 0; k < out_d; k++)
                out[k] *= scale_factor;
        }

        float l_add = 0.f;
        for (int j = 0; j < block_n; j++)
        {
            s[j] = expf(s[j] - new_m);
            l_add += s[j];
        }
        l += l_add;

        for (int j = 0; j < block_n; j++)
        {
            for (int k = 0; k < out_d; k++)
                out[k] += s[j] * V[(n_start + j) * out_d + k];
        }

        m = new_m;
    }

    float inv_l = 1.f / l;
    for (int k = 0; k < out_d; k++)
        out[k] *= inv_l;
}

static inline void softmax_tile_scalar(float* P, const float* S,
        float* m_vec, float* l_vec, float* scale_out, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        const float* sptr = S + i * n;
        float* pptr = P + i * n;
        float m_new = m_vec[i];
        for (int j = 0; j < n; j++) m_new = std::max(m_new, sptr[j]);
        float scale_factor = expf(m_vec[i] - m_new);
        scale_out[i] = scale_factor;
        l_vec[i] *= scale_factor;
        float l_add = 0.f;
        for (int j = 0; j < n; j++)
        {
            pptr[j] = expf(sptr[j] - m_new);
            l_add += pptr[j];
        }
        l_vec[i] += l_add;
        m_vec[i] = m_new;
    }
}

#if __AVX512F__

static void qk_gemm_avx512(float* S, const float* Q, const float* K,
        int m, int n, int d, float scale)
{
    int i = 0;
    for (; i + 8 <= m; i += 8)
    {
        int j = 0;
        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * d;
            const float* k1 = K + (j + 1) * d;

            __m512 acc[8][2];
            for (int mi = 0; mi < 8; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
            }

            int k = 0;
            for (; k + 15 < d; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);

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
                __m512 kv0 = _mm512_maskz_loadu_ps(mask, k0 + k);
                __m512 kv1 = _mm512_maskz_loadu_ps(mask, k1 + k);

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
            const float* kptr = K + j * d;

            __m512 acc[8];
            for (int mi = 0; mi < 8; mi++)
                acc[mi] = _mm512_setzero_ps();

            int k = 0;
            for (; k + 15 < d; k += 16)
            {
                __m512 kvec = _mm512_loadu_ps(kptr + k);
                for (int mi = 0; mi < 8; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * d + k);
                    acc[mi] = _mm512_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            if (k < d)
            {
                __mmask16 mask = (__mmask16)((1u << (d - k)) - 1);
                __m512 kvec = _mm512_maskz_loadu_ps(mask, kptr + k);
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
            const float* k0 = K + (j + 0) * d;
            const float* k1 = K + (j + 1) * d;

            __m512 acc[4][2];
            for (int mi = 0; mi < 4; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
            }

            int k = 0;
            for (; k + 15 < d; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);

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
                __m512 kv0 = _mm512_maskz_loadu_ps(mask, k0 + k);
                __m512 kv1 = _mm512_maskz_loadu_ps(mask, k1 + k);

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
            const float* kptr = K + j * d;

            __m512 acc[4];
            for (int mi = 0; mi < 4; mi++)
                acc[mi] = _mm512_setzero_ps();

            int k = 0;
            for (; k + 15 < d; k += 16)
            {
                __m512 kvec = _mm512_loadu_ps(kptr + k);
                for (int mi = 0; mi < 4; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * d + k);
                    acc[mi] = _mm512_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            if (k < d)
            {
                __mmask16 mask = (__mmask16)((1u << (d - k)) - 1);
                __m512 kvec = _mm512_maskz_loadu_ps(mask, kptr + k);
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
            const float* k0 = K + (j + 0) * d;
            const float* k1 = K + (j + 1) * d;
            const float* k2 = K + (j + 2) * d;
            const float* k3 = K + (j + 3) * d;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            int k = 0;
            for (; k + 15 < d; k += 16)
            {
                __m512 qvec = _mm512_loadu_ps(qptr + k);
                acc0 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k0 + k), acc0);
                acc1 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k1 + k), acc1);
                acc2 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k2 + k), acc2);
                acc3 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k3 + k), acc3);
            }

            if (k < d)
            {
                __mmask16 mask = (__mmask16)((1u << (d - k)) - 1);
                __m512 qvec = _mm512_maskz_loadu_ps(mask, qptr + k);
                acc0 = _mm512_fmadd_ps(qvec, _mm512_maskz_loadu_ps(mask, k0 + k), acc0);
                acc1 = _mm512_fmadd_ps(qvec, _mm512_maskz_loadu_ps(mask, k1 + k), acc1);
                acc2 = _mm512_fmadd_ps(qvec, _mm512_maskz_loadu_ps(mask, k2 + k), acc2);
                acc3 = _mm512_fmadd_ps(qvec, _mm512_maskz_loadu_ps(mask, k3 + k), acc3);
            }

            S[i * n + j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
            S[i * n + j + 2] = _mm512_comp_reduce_add_ps(acc2) * scale;
            S[i * n + j + 3] = _mm512_comp_reduce_add_ps(acc3) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * d;
            const float* kptr = K + j * d;
            float sum = 0.f;
            int k = 0;
            __m512 vacc = _mm512_setzero_ps();
            for (; k + 15 < d; k += 16)
                vacc = _mm512_fmadd_ps(_mm512_loadu_ps(qptr + k), _mm512_loadu_ps(kptr + k), vacc);
            if (k < d)
            {
                __mmask16 mask = (__mmask16)((1u << (d - k)) - 1);
                vacc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask, qptr + k), _mm512_maskz_loadu_ps(mask, kptr + k), vacc);
            }
            S[i * n + j] = _mm512_comp_reduce_add_ps(vacc) * scale;
        }
    }
}


template<int D>
static inline void qk_gemm_specialized_avx512(float* S, const float* Q, const float* K,
        int m, int n, float scale)
{
    int i = 0;
    for (; i + 8 <= m; i += 8)
    {
        int j = 0;
        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * D;
            const float* k1 = K + (j + 1) * D;

            __m512 acc[8][2];
            for (int mi = 0; mi < 8; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
            }

            for (int k = 0; k < D; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);

                for (int mi = 0; mi < 8; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * D + k);
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
            const float* kptr = K + j * D;

            __m512 acc[8];
            for (int mi = 0; mi < 8; mi++)
                acc[mi] = _mm512_setzero_ps();

            for (int k = 0; k < D; k += 16)
            {
                __m512 kvec = _mm512_loadu_ps(kptr + k);
                for (int mi = 0; mi < 8; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * D + k);
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
            const float* k0 = K + (j + 0) * D;
            const float* k1 = K + (j + 1) * D;

            __m512 acc[4][2];
            for (int mi = 0; mi < 4; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
            }

            for (int k = 0; k < D; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);

                for (int mi = 0; mi < 4; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * D + k);
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
            __m512 acc[4];
            for (int mi = 0; mi < 4; mi++)
                acc[mi] = _mm512_setzero_ps();

            const float* kptr = K + j * D;
            for (int k = 0; k < D; k += 16)
            {
                __m512 kvec = _mm512_loadu_ps(kptr + k);
                for (int mi = 0; mi < 4; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * D + k);
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
            const float* qptr = Q + i * D;
            const float* k0 = K + (j + 0) * D;
            const float* k1 = K + (j + 1) * D;
            const float* k2 = K + (j + 2) * D;
            const float* k3 = K + (j + 3) * D;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int k = 0; k < D; k += 16)
            {
                __m512 qvec = _mm512_loadu_ps(qptr + k);
                acc0 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k0 + k), acc0);
                acc1 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k1 + k), acc1);
                acc2 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k2 + k), acc2);
                acc3 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k3 + k), acc3);
            }

            S[i * n + j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
            S[i * n + j + 2] = _mm512_comp_reduce_add_ps(acc2) * scale;
            S[i * n + j + 3] = _mm512_comp_reduce_add_ps(acc3) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * D;
            const float* kptr = K + j * D;
            __m512 vacc = _mm512_setzero_ps();
            for (int k = 0; k < D; k += 16)
                vacc = _mm512_fmadd_ps(_mm512_loadu_ps(qptr + k), _mm512_loadu_ps(kptr + k), vacc);
            S[i * n + j] = _mm512_comp_reduce_add_ps(vacc) * scale;
        }
    }
}

// Explicit specialization for D=128: 6x4 kernel to improve K-tile reuse
template<>
void qk_gemm_specialized_avx512<128>(float* S, const float* Q, const float* K,
        int m, int n, float scale)
{
    int i = 0;
    for (; i + 6 <= m; i += 6)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* k0 = K + (j + 0) * 128;
            const float* k1 = K + (j + 1) * 128;
            const float* k2 = K + (j + 2) * 128;
            const float* k3 = K + (j + 3) * 128;

            __m512 acc[6][4];
            for (int mi = 0; mi < 6; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
                acc[mi][2] = _mm512_setzero_ps();
                acc[mi][3] = _mm512_setzero_ps();
            }

            for (int k = 0; k < 128; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);
                __m512 kv2 = _mm512_loadu_ps(k2 + k);
                __m512 kv3 = _mm512_loadu_ps(k3 + k);

                for (int mi = 0; mi < 6; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 128 + k);
                    acc[mi][0] = _mm512_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(qvec, kv1, acc[mi][1]);
                    acc[mi][2] = _mm512_fmadd_ps(qvec, kv2, acc[mi][2]);
                    acc[mi][3] = _mm512_fmadd_ps(qvec, kv3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 6; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm512_comp_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm512_comp_reduce_add_ps(acc[mi][1]) * scale;
                S[(i + mi) * n + j + 2] = _mm512_comp_reduce_add_ps(acc[mi][2]) * scale;
                S[(i + mi) * n + j + 3] = _mm512_comp_reduce_add_ps(acc[mi][3]) * scale;
            }
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 128;
            const float* k1 = K + (j + 1) * 128;

            __m512 acc[6][2];
            for (int mi = 0; mi < 6; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
            }

            for (int k = 0; k < 128; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);

                for (int mi = 0; mi < 6; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 128 + k);
                    acc[mi][0] = _mm512_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 6; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm512_comp_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm512_comp_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * 128;

            __m512 acc[6];
            for (int mi = 0; mi < 6; mi++)
                acc[mi] = _mm512_setzero_ps();

            for (int k = 0; k < 128; k += 16)
            {
                __m512 kvec = _mm512_loadu_ps(kptr + k);
                for (int mi = 0; mi < 6; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 128 + k);
                    acc[mi] = _mm512_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 6; mi++)
                S[(i + mi) * n + j] = _mm512_comp_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i + 4 <= m; i += 4)
    {
        int j = 0;
        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 128;
            const float* k1 = K + (j + 1) * 128;

            __m512 acc[4][2];
            for (int mi = 0; mi < 4; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
            }

            for (int k = 0; k < 128; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);

                for (int mi = 0; mi < 4; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 128 + k);
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
            __m512 acc[4];
            for (int mi = 0; mi < 4; mi++)
                acc[mi] = _mm512_setzero_ps();

            const float* kptr = K + j * 128;
            for (int k = 0; k < 128; k += 16)
            {
                __m512 kvec = _mm512_loadu_ps(kptr + k);
                for (int mi = 0; mi < 4; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 128 + k);
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
        for (; j + 4 <= n; j += 4)
        {
            const float* qptr = Q + i * 128;
            const float* k0 = K + (j + 0) * 128;
            const float* k1 = K + (j + 1) * 128;
            const float* k2 = K + (j + 2) * 128;
            const float* k3 = K + (j + 3) * 128;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int k = 0; k < 128; k += 16)
            {
                __m512 qvec = _mm512_loadu_ps(qptr + k);
                acc0 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k0 + k), acc0);
                acc1 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k1 + k), acc1);
                acc2 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k2 + k), acc2);
                acc3 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k3 + k), acc3);
            }

            S[i * n + j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
            S[i * n + j + 2] = _mm512_comp_reduce_add_ps(acc2) * scale;
            S[i * n + j + 3] = _mm512_comp_reduce_add_ps(acc3) * scale;
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* qptr = Q + i * 128;
            const float* k0 = K + (j + 0) * 128;
            const float* k1 = K + (j + 1) * 128;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();

            for (int k = 0; k < 128; k += 16)
            {
                __m512 qvec = _mm512_loadu_ps(qptr + k);
                acc0 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k0 + k), acc0);
                acc1 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k1 + k), acc1);
            }

            S[i * n + j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * 128;
            const float* kptr = K + j * 128;
            __m512 vacc = _mm512_setzero_ps();
            for (int k = 0; k < 128; k += 16)
                vacc = _mm512_fmadd_ps(_mm512_loadu_ps(qptr + k), _mm512_loadu_ps(kptr + k), vacc);
            S[i * n + j] = _mm512_comp_reduce_add_ps(vacc) * scale;
        }
    }
}

template<>
void qk_gemm_specialized_avx512<1024>(float* S, const float* Q, const float* K,
        int m, int n, float scale)
{
    int i = 0;
    for (; i + 4 <= m; i += 4)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;
            const float* k2 = K + (j + 2) * 1024;
            const float* k3 = K + (j + 3) * 1024;

            __m512 acc[4][4];
            for (int mi = 0; mi < 4; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
                acc[mi][2] = _mm512_setzero_ps();
                acc[mi][3] = _mm512_setzero_ps();
            }

            for (int k = 0; k < 1024; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);
                __m512 kv2 = _mm512_loadu_ps(k2 + k);
                __m512 kv3 = _mm512_loadu_ps(k3 + k);

                for (int mi = 0; mi < 4; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi][0] = _mm512_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(qvec, kv1, acc[mi][1]);
                    acc[mi][2] = _mm512_fmadd_ps(qvec, kv2, acc[mi][2]);
                    acc[mi][3] = _mm512_fmadd_ps(qvec, kv3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 4; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm512_comp_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm512_comp_reduce_add_ps(acc[mi][1]) * scale;
                S[(i + mi) * n + j + 2] = _mm512_comp_reduce_add_ps(acc[mi][2]) * scale;
                S[(i + mi) * n + j + 3] = _mm512_comp_reduce_add_ps(acc[mi][3]) * scale;
            }
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;

            __m512 acc[4][2];
            for (int mi = 0; mi < 4; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
            }

            for (int k = 0; k < 1024; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);

                for (int mi = 0; mi < 4; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 1024 + k);
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
            const float* kptr = K + j * 1024;

            __m512 acc[4];
            for (int mi = 0; mi < 4; mi++)
                acc[mi] = _mm512_setzero_ps();

            for (int k = 0; k < 1024; k += 16)
            {
                __m512 kvec = _mm512_loadu_ps(kptr + k);
                for (int mi = 0; mi < 4; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi] = _mm512_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 4; mi++)
                S[(i + mi) * n + j] = _mm512_comp_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i + 2 <= m; i += 2)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;
            const float* k2 = K + (j + 2) * 1024;
            const float* k3 = K + (j + 3) * 1024;

            __m512 acc[2][4];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
                acc[mi][2] = _mm512_setzero_ps();
                acc[mi][3] = _mm512_setzero_ps();
            }

            for (int k = 0; k < 1024; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);
                __m512 kv2 = _mm512_loadu_ps(k2 + k);
                __m512 kv3 = _mm512_loadu_ps(k3 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi][0] = _mm512_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(qvec, kv1, acc[mi][1]);
                    acc[mi][2] = _mm512_fmadd_ps(qvec, kv2, acc[mi][2]);
                    acc[mi][3] = _mm512_fmadd_ps(qvec, kv3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm512_comp_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm512_comp_reduce_add_ps(acc[mi][1]) * scale;
                S[(i + mi) * n + j + 2] = _mm512_comp_reduce_add_ps(acc[mi][2]) * scale;
                S[(i + mi) * n + j + 3] = _mm512_comp_reduce_add_ps(acc[mi][3]) * scale;
            }
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;

            __m512 acc[2][2];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
            }

            for (int k = 0; k < 1024; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi][0] = _mm512_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm512_comp_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm512_comp_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * 1024;

            __m512 acc[2];
            for (int mi = 0; mi < 2; mi++)
                acc[mi] = _mm512_setzero_ps();

            for (int k = 0; k < 1024; k += 16)
            {
                __m512 kvec = _mm512_loadu_ps(kptr + k);
                for (int mi = 0; mi < 2; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi] = _mm512_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
                S[(i + mi) * n + j] = _mm512_comp_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i < m; i++)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* qptr = Q + i * 1024;
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;
            const float* k2 = K + (j + 2) * 1024;
            const float* k3 = K + (j + 3) * 1024;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int k = 0; k < 1024; k += 16)
            {
                __m512 qvec = _mm512_loadu_ps(qptr + k);
                acc0 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k0 + k), acc0);
                acc1 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k1 + k), acc1);
                acc2 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k2 + k), acc2);
                acc3 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k3 + k), acc3);
            }

            S[i * n + j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
            S[i * n + j + 2] = _mm512_comp_reduce_add_ps(acc2) * scale;
            S[i * n + j + 3] = _mm512_comp_reduce_add_ps(acc3) * scale;
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* qptr = Q + i * 1024;
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();

            for (int k = 0; k < 1024; k += 16)
            {
                __m512 qvec = _mm512_loadu_ps(qptr + k);
                acc0 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k0 + k), acc0);
                acc1 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k1 + k), acc1);
            }

            S[i * n + j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * 1024;
            const float* kptr = K + j * 1024;
            __m512 vacc = _mm512_setzero_ps();
            for (int k = 0; k < 1024; k += 16)
                vacc = _mm512_fmadd_ps(_mm512_loadu_ps(qptr + k), _mm512_loadu_ps(kptr + k), vacc);
            S[i * n + j] = _mm512_comp_reduce_add_ps(vacc) * scale;
        }
    }
}

template<>
void qk_gemm_specialized_avx512<2048>(float* S, const float* Q, const float* K,
        int m, int n, float scale)
{
    int i = 0;
    for (; i + 2 <= m; i += 2)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* k0 = K + (j + 0) * 2048;
            const float* k1 = K + (j + 1) * 2048;
            const float* k2 = K + (j + 2) * 2048;
            const float* k3 = K + (j + 3) * 2048;

            __m512 acc[2][4];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
                acc[mi][2] = _mm512_setzero_ps();
                acc[mi][3] = _mm512_setzero_ps();
            }

            for (int k = 0; k < 2048; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);
                __m512 kv2 = _mm512_loadu_ps(k2 + k);
                __m512 kv3 = _mm512_loadu_ps(k3 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 2048 + k);
                    acc[mi][0] = _mm512_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(qvec, kv1, acc[mi][1]);
                    acc[mi][2] = _mm512_fmadd_ps(qvec, kv2, acc[mi][2]);
                    acc[mi][3] = _mm512_fmadd_ps(qvec, kv3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm512_comp_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm512_comp_reduce_add_ps(acc[mi][1]) * scale;
                S[(i + mi) * n + j + 2] = _mm512_comp_reduce_add_ps(acc[mi][2]) * scale;
                S[(i + mi) * n + j + 3] = _mm512_comp_reduce_add_ps(acc[mi][3]) * scale;
            }
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 2048;
            const float* k1 = K + (j + 1) * 2048;

            __m512 acc[2][2];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
            }

            for (int k = 0; k < 2048; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 2048 + k);
                    acc[mi][0] = _mm512_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm512_comp_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm512_comp_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * 2048;

            __m512 acc[2];
            for (int mi = 0; mi < 2; mi++)
                acc[mi] = _mm512_setzero_ps();

            for (int k = 0; k < 2048; k += 16)
            {
                __m512 kvec = _mm512_loadu_ps(kptr + k);
                for (int mi = 0; mi < 2; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 2048 + k);
                    acc[mi] = _mm512_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
                S[(i + mi) * n + j] = _mm512_comp_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i < m; i++)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* qptr = Q + i * 2048;
            const float* k0 = K + (j + 0) * 2048;
            const float* k1 = K + (j + 1) * 2048;
            const float* k2 = K + (j + 2) * 2048;
            const float* k3 = K + (j + 3) * 2048;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int k = 0; k < 2048; k += 16)
            {
                __m512 qvec = _mm512_loadu_ps(qptr + k);
                acc0 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k0 + k), acc0);
                acc1 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k1 + k), acc1);
                acc2 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k2 + k), acc2);
                acc3 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k3 + k), acc3);
            }

            S[i * n + j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
            S[i * n + j + 2] = _mm512_comp_reduce_add_ps(acc2) * scale;
            S[i * n + j + 3] = _mm512_comp_reduce_add_ps(acc3) * scale;
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* qptr = Q + i * 2048;
            const float* k0 = K + (j + 0) * 2048;
            const float* k1 = K + (j + 1) * 2048;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();

            for (int k = 0; k < 2048; k += 16)
            {
                __m512 qvec = _mm512_loadu_ps(qptr + k);
                acc0 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k0 + k), acc0);
                acc1 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k1 + k), acc1);
            }

            S[i * n + j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * 2048;
            const float* kptr = K + j * 2048;
            __m512 vacc = _mm512_setzero_ps();
            for (int k = 0; k < 2048; k += 16)
                vacc = _mm512_fmadd_ps(_mm512_loadu_ps(qptr + k), _mm512_loadu_ps(kptr + k), vacc);
            S[i * n + j] = _mm512_comp_reduce_add_ps(vacc) * scale;
        }
    }
}

template<>
void qk_gemm_specialized_avx512<4096>(float* S, const float* Q, const float* K,
        int m, int n, float scale)
{
    int i = 0;
    for (; i + 2 <= m; i += 2)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* k0 = K + (j + 0) * 4096;
            const float* k1 = K + (j + 1) * 4096;
            const float* k2 = K + (j + 2) * 4096;
            const float* k3 = K + (j + 3) * 4096;

            __m512 acc[2][4];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
                acc[mi][2] = _mm512_setzero_ps();
                acc[mi][3] = _mm512_setzero_ps();
            }

            for (int k = 0; k < 4096; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);
                __m512 kv2 = _mm512_loadu_ps(k2 + k);
                __m512 kv3 = _mm512_loadu_ps(k3 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 4096 + k);
                    acc[mi][0] = _mm512_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(qvec, kv1, acc[mi][1]);
                    acc[mi][2] = _mm512_fmadd_ps(qvec, kv2, acc[mi][2]);
                    acc[mi][3] = _mm512_fmadd_ps(qvec, kv3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm512_comp_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm512_comp_reduce_add_ps(acc[mi][1]) * scale;
                S[(i + mi) * n + j + 2] = _mm512_comp_reduce_add_ps(acc[mi][2]) * scale;
                S[(i + mi) * n + j + 3] = _mm512_comp_reduce_add_ps(acc[mi][3]) * scale;
            }
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 4096;
            const float* k1 = K + (j + 1) * 4096;

            __m512 acc[2][2];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm512_setzero_ps();
                acc[mi][1] = _mm512_setzero_ps();
            }

            for (int k = 0; k < 4096; k += 16)
            {
                __m512 kv0 = _mm512_loadu_ps(k0 + k);
                __m512 kv1 = _mm512_loadu_ps(k1 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 4096 + k);
                    acc[mi][0] = _mm512_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm512_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm512_comp_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm512_comp_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * 4096;

            __m512 acc[2];
            for (int mi = 0; mi < 2; mi++)
                acc[mi] = _mm512_setzero_ps();

            for (int k = 0; k < 4096; k += 16)
            {
                __m512 kvec = _mm512_loadu_ps(kptr + k);
                for (int mi = 0; mi < 2; mi++)
                {
                    __m512 qvec = _mm512_loadu_ps(Q + (i + mi) * 4096 + k);
                    acc[mi] = _mm512_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
                S[(i + mi) * n + j] = _mm512_comp_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i < m; i++)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* qptr = Q + i * 4096;
            const float* k0 = K + (j + 0) * 4096;
            const float* k1 = K + (j + 1) * 4096;
            const float* k2 = K + (j + 2) * 4096;
            const float* k3 = K + (j + 3) * 4096;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int k = 0; k < 4096; k += 16)
            {
                __m512 qvec = _mm512_loadu_ps(qptr + k);
                acc0 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k0 + k), acc0);
                acc1 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k1 + k), acc1);
                acc2 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k2 + k), acc2);
                acc3 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k3 + k), acc3);
            }

            S[i * n + j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
            S[i * n + j + 2] = _mm512_comp_reduce_add_ps(acc2) * scale;
            S[i * n + j + 3] = _mm512_comp_reduce_add_ps(acc3) * scale;
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* qptr = Q + i * 4096;
            const float* k0 = K + (j + 0) * 4096;
            const float* k1 = K + (j + 1) * 4096;

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();

            for (int k = 0; k < 4096; k += 16)
            {
                __m512 qvec = _mm512_loadu_ps(qptr + k);
                acc0 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k0 + k), acc0);
                acc1 = _mm512_fmadd_ps(qvec, _mm512_loadu_ps(k1 + k), acc1);
            }

            S[i * n + j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * 4096;
            const float* kptr = K + j * 4096;
            __m512 vacc = _mm512_setzero_ps();
            for (int k = 0; k < 4096; k += 16)
                vacc = _mm512_fmadd_ps(_mm512_loadu_ps(qptr + k), _mm512_loadu_ps(kptr + k), vacc);
            S[i * n + j] = _mm512_comp_reduce_add_ps(vacc) * scale;
        }
    }
}


template<int M_BLOCK, int D_UNROLL>
static inline void pv_gemm_avx512(float* O, const float* P, const float* V, int m, int n, int d)
{
    const int VEC_PER_UNROLL = D_UNROLL / 16;
    int dd = 0;
    for (; dd + D_UNROLL - 1 < d; dd += D_UNROLL)
    {
        int i = 0;
        for (; i + M_BLOCK <= m; i += M_BLOCK)
        {
            float* op[M_BLOCK];
            const float* pptr[M_BLOCK];
            for (int mi = 0; mi < M_BLOCK; mi++)
            {
                op[mi] = O + (i + mi) * d + dd;
                pptr[mi] = P + (i + mi) * n;
            }

            __m512 acc[M_BLOCK][VEC_PER_UNROLL];
            for (int mi = 0; mi < M_BLOCK; mi++)
                for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                    acc[mi][vi] = _mm512_loadu_ps(op[mi] + vi * 16);

            for (int j = 0; j < n; j++)
            {
                __m512 vvec[VEC_PER_UNROLL];
                for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                    vvec[vi] = _mm512_loadu_ps(V + j * d + dd + vi * 16);

                for (int mi = 0; mi < M_BLOCK; mi++)
                {
                    __m512 pvec = _mm512_set1_ps(pptr[mi][j]);
                    for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                        acc[mi][vi] = _mm512_fmadd_ps(pvec, vvec[vi], acc[mi][vi]);
                }
            }

            for (int mi = 0; mi < M_BLOCK; mi++)
                for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                    _mm512_storeu_ps(op[mi] + vi * 16, acc[mi][vi]);
        }

        for (; i < m; i++)
        {
            float* optr = O + i * d + dd;
            const float* pptr = P + i * n;

            __m512 acc[VEC_PER_UNROLL];
            for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                acc[vi] = _mm512_loadu_ps(optr + vi * 16);

            for (int j = 0; j < n; j++)
            {
                __m512 pvec = _mm512_set1_ps(pptr[j]);
                for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                    acc[vi] = _mm512_fmadd_ps(pvec, _mm512_loadu_ps(V + j * d + dd + vi * 16), acc[vi]);
            }

            for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                _mm512_storeu_ps(optr + vi * 16, acc[vi]);
        }
    }

    for (; dd + 15 < d; dd += 16)
    {
        int i = 0;
        for (; i + M_BLOCK <= m; i += M_BLOCK)
        {
            float* op[M_BLOCK];
            const float* pptr[M_BLOCK];
            for (int mi = 0; mi < M_BLOCK; mi++)
            {
                op[mi] = O + (i + mi) * d + dd;
                pptr[mi] = P + (i + mi) * n;
            }

            __m512 acc[M_BLOCK];
            for (int mi = 0; mi < M_BLOCK; mi++)
                acc[mi] = _mm512_loadu_ps(op[mi]);

            for (int j = 0; j < n; j++)
            {
                __m512 vvec = _mm512_loadu_ps(V + j * d + dd);
                for (int mi = 0; mi < M_BLOCK; mi++)
                    acc[mi] = _mm512_fmadd_ps(_mm512_set1_ps(pptr[mi][j]), vvec, acc[mi]);
            }

            for (int mi = 0; mi < M_BLOCK; mi++)
                _mm512_storeu_ps(op[mi], acc[mi]);
        }

        for (; i < m; i++)
        {
            float* optr = O + i * d + dd;
            const float* pptr = P + i * n;
            __m512 acc = _mm512_loadu_ps(optr);
            for (int j = 0; j < n; j++)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(pptr[j]), _mm512_loadu_ps(V + j * d + dd), acc);
            _mm512_storeu_ps(optr, acc);
        }
    }

    for (; dd < d; dd++)
    {
        int i = 0;
        for (; i + M_BLOCK <= m; i += M_BLOCK)
        {
            float* op[M_BLOCK];
            const float* pptr[M_BLOCK];
            for (int mi = 0; mi < M_BLOCK; mi++)
            {
                op[mi] = O + (i + mi) * d + dd;
                pptr[mi] = P + (i + mi) * n;
            }
            for (int j = 0; j < n; j++)
            {
                for (int mi = 0; mi < M_BLOCK; mi++)
                    op[mi][0] += pptr[mi][j] * V[j * d + dd];
            }
        }
        for (; i < m; i++)
        {
            float* optr = O + i * d + dd;
            const float* pptr = P + i * n;
            for (int j = 0; j < n; j++)
                optr[0] += pptr[j] * V[j * d + dd];
        }
    }
}


template<int M_BLOCK, int D>
static void pv_gemm_avx512(float* O, const float* P, const float* V, int m, int n)
{
    const int VEC_PER_D = D / 16;
    int i = 0;
    for (; i + M_BLOCK <= m; i += M_BLOCK)
    {
        float* op[M_BLOCK];
        const float* pptr[M_BLOCK];
        for (int mi = 0; mi < M_BLOCK; mi++)
        {
            op[mi] = O + (i + mi) * D;
            pptr[mi] = P + (i + mi) * n;
        }

        int dd = 0;
        for (; dd + 127 < D; dd += 128)
        {
            __m512 acc[M_BLOCK][8];
            for (int mi = 0; mi < M_BLOCK; mi++)
            {
                acc[mi][0] = _mm512_loadu_ps(op[mi] + dd + 0 * 16);
                acc[mi][1] = _mm512_loadu_ps(op[mi] + dd + 1 * 16);
                acc[mi][2] = _mm512_loadu_ps(op[mi] + dd + 2 * 16);
                acc[mi][3] = _mm512_loadu_ps(op[mi] + dd + 3 * 16);
                acc[mi][4] = _mm512_loadu_ps(op[mi] + dd + 4 * 16);
                acc[mi][5] = _mm512_loadu_ps(op[mi] + dd + 5 * 16);
                acc[mi][6] = _mm512_loadu_ps(op[mi] + dd + 6 * 16);
                acc[mi][7] = _mm512_loadu_ps(op[mi] + dd + 7 * 16);
            }

            for (int j = 0; j < n; j++)
            {
                __m512 v0 = _mm512_loadu_ps(V + j * D + dd + 0 * 16);
                __m512 v1 = _mm512_loadu_ps(V + j * D + dd + 1 * 16);
                __m512 v2 = _mm512_loadu_ps(V + j * D + dd + 2 * 16);
                __m512 v3 = _mm512_loadu_ps(V + j * D + dd + 3 * 16);
                __m512 v4 = _mm512_loadu_ps(V + j * D + dd + 4 * 16);
                __m512 v5 = _mm512_loadu_ps(V + j * D + dd + 5 * 16);
                __m512 v6 = _mm512_loadu_ps(V + j * D + dd + 6 * 16);
                __m512 v7 = _mm512_loadu_ps(V + j * D + dd + 7 * 16);

                for (int mi = 0; mi < M_BLOCK; mi++)
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

            for (int mi = 0; mi < M_BLOCK; mi++)
            {
                _mm512_storeu_ps(op[mi] + dd + 0 * 16, acc[mi][0]);
                _mm512_storeu_ps(op[mi] + dd + 1 * 16, acc[mi][1]);
                _mm512_storeu_ps(op[mi] + dd + 2 * 16, acc[mi][2]);
                _mm512_storeu_ps(op[mi] + dd + 3 * 16, acc[mi][3]);
                _mm512_storeu_ps(op[mi] + dd + 4 * 16, acc[mi][4]);
                _mm512_storeu_ps(op[mi] + dd + 5 * 16, acc[mi][5]);
                _mm512_storeu_ps(op[mi] + dd + 6 * 16, acc[mi][6]);
                _mm512_storeu_ps(op[mi] + dd + 7 * 16, acc[mi][7]);
            }
        }

        for (; dd + 15 < D; dd += 16)
        {
            __m512 acc[M_BLOCK];
            for (int mi = 0; mi < M_BLOCK; mi++)
                acc[mi] = _mm512_loadu_ps(op[mi] + dd);

            for (int j = 0; j < n; j++)
            {
                __m512 vvec = _mm512_loadu_ps(V + j * D + dd);
                for (int mi = 0; mi < M_BLOCK; mi++)
                    acc[mi] = _mm512_fmadd_ps(_mm512_set1_ps(pptr[mi][j]), vvec, acc[mi]);
            }

            for (int mi = 0; mi < M_BLOCK; mi++)
                _mm512_storeu_ps(op[mi] + dd, acc[mi]);
        }

        for (; dd < D; dd++)
        {
            for (int mi = 0; mi < M_BLOCK; mi++)
            {
                float acc = op[mi][dd];
                for (int j = 0; j < n; j++)
                    acc += pptr[mi][j] * V[j * D + dd];
                op[mi][dd] = acc;
            }
        }
    }

    for (; i < m; i++)
    {
        float* optr = O + i * D;
        const float* pptr = P + i * n;

        int dd = 0;
        for (; dd + 127 < D; dd += 128)
        {
            __m512 acc0 = _mm512_loadu_ps(optr + dd + 0 * 16);
            __m512 acc1 = _mm512_loadu_ps(optr + dd + 1 * 16);
            __m512 acc2 = _mm512_loadu_ps(optr + dd + 2 * 16);
            __m512 acc3 = _mm512_loadu_ps(optr + dd + 3 * 16);
            __m512 acc4 = _mm512_loadu_ps(optr + dd + 4 * 16);
            __m512 acc5 = _mm512_loadu_ps(optr + dd + 5 * 16);
            __m512 acc6 = _mm512_loadu_ps(optr + dd + 6 * 16);
            __m512 acc7 = _mm512_loadu_ps(optr + dd + 7 * 16);

            for (int j = 0; j < n; j++)
            {
                __m512 pvec = _mm512_set1_ps(pptr[j]);
                acc0 = _mm512_fmadd_ps(pvec, _mm512_loadu_ps(V + j * D + dd + 0 * 16), acc0);
                acc1 = _mm512_fmadd_ps(pvec, _mm512_loadu_ps(V + j * D + dd + 1 * 16), acc1);
                acc2 = _mm512_fmadd_ps(pvec, _mm512_loadu_ps(V + j * D + dd + 2 * 16), acc2);
                acc3 = _mm512_fmadd_ps(pvec, _mm512_loadu_ps(V + j * D + dd + 3 * 16), acc3);
                acc4 = _mm512_fmadd_ps(pvec, _mm512_loadu_ps(V + j * D + dd + 4 * 16), acc4);
                acc5 = _mm512_fmadd_ps(pvec, _mm512_loadu_ps(V + j * D + dd + 5 * 16), acc5);
                acc6 = _mm512_fmadd_ps(pvec, _mm512_loadu_ps(V + j * D + dd + 6 * 16), acc6);
                acc7 = _mm512_fmadd_ps(pvec, _mm512_loadu_ps(V + j * D + dd + 7 * 16), acc7);
            }

            _mm512_storeu_ps(optr + dd + 0 * 16, acc0);
            _mm512_storeu_ps(optr + dd + 1 * 16, acc1);
            _mm512_storeu_ps(optr + dd + 2 * 16, acc2);
            _mm512_storeu_ps(optr + dd + 3 * 16, acc3);
            _mm512_storeu_ps(optr + dd + 4 * 16, acc4);
            _mm512_storeu_ps(optr + dd + 5 * 16, acc5);
            _mm512_storeu_ps(optr + dd + 6 * 16, acc6);
            _mm512_storeu_ps(optr + dd + 7 * 16, acc7);
        }

        for (; dd + 15 < D; dd += 16)
        {
            __m512 acc = _mm512_loadu_ps(optr + dd);
            for (int j = 0; j < n; j++)
                acc = _mm512_fmadd_ps(_mm512_set1_ps(pptr[j]), _mm512_loadu_ps(V + j * D + dd), acc);
            _mm512_storeu_ps(optr + dd, acc);
        }

        for (; dd < D; dd++)
        {
            float acc = optr[dd];
            for (int j = 0; j < n; j++)
                acc += pptr[j] * V[j * D + dd];
            optr[dd] = acc;
        }
    }
}


static inline void softmax_tile_avx512(float* P, const float* S,
        float* m_vec, float* l_vec, float* scale_out, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        const float* sptr = S + i * n;
        float* pptr = P + i * n;

        __m512 vmax = _mm512_set1_ps(m_vec[i]);
        int j = 0;
        for (; j + 15 < n; j += 16)
            vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(sptr + j));
        if (j < n)
        {
            __mmask16 mask = (__mmask16)((1u << (n - j)) - 1);
            __m512 tail = _mm512_mask_loadu_ps(_mm512_set1_ps(-FLT_MAX), mask, sptr + j);
            vmax = _mm512_max_ps(vmax, tail);
        }
        float m_new = _mm512_comp_reduce_max_ps(vmax);

        float scale_factor = expf(m_vec[i] - m_new);
        scale_out[i] = scale_factor;
        l_vec[i] *= scale_factor;

        __m512 vm_new = _mm512_set1_ps(m_new);
        __m512 vsum = _mm512_setzero_ps();
        j = 0;
        for (; j + 15 < n; j += 16)
        {
            __m512 svec = _mm512_loadu_ps(sptr + j);
            __m512 evec = exp512_ps(_mm512_sub_ps(svec, vm_new));
            _mm512_storeu_ps(pptr + j, evec);
            vsum = _mm512_add_ps(vsum, evec);
        }
        if (j < n)
        {
            __mmask16 mask = (__mmask16)((1u << (n - j)) - 1);
            __m512 svec = _mm512_maskz_loadu_ps(mask, sptr + j);
            __m512 evec = exp512_ps(_mm512_sub_ps(svec, vm_new));
            _mm512_mask_storeu_ps(pptr + j, mask, evec);
            vsum = _mm512_mask_add_ps(vsum, mask, vsum, evec);
        }
        float l_add = _mm512_comp_reduce_add_ps(vsum);
        l_vec[i] += l_add;
        m_vec[i] = m_new;
    }
}

static inline void vec_scale_avx512(float* x, float s, int n)
{
    __m512 vscale = _mm512_set1_ps(s);
    int i = 0;
    for (; i + 15 < n; i += 16)
        _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), vscale));
    if (i < n)
    {
        __mmask16 mask = (__mmask16)((1u << (n - i)) - 1);
        _mm512_mask_storeu_ps(x + i, mask, _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, x + i), vscale));
    }
}

static inline void vec_zero_avx512(float* x, int n)
{
    __m512 zero = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16)
        _mm512_storeu_ps(x + i, zero);
    if (i < n)
    {
        __mmask16 mask = (__mmask16)((1u << (n - i)) - 1);
        _mm512_mask_storeu_ps(x + i, mask, zero);
    }
}

static inline void decode_qk_dot_avx512(float* s, const float* q, const float* K, int n_start, int block_n, int d, float scale)
{
    int j = 0;
    for (; j + 3 < block_n; j += 4)
    {
        const float* k0 = K + (n_start + j + 0) * d;
        const float* k1 = K + (n_start + j + 1) * d;
        const float* k2 = K + (n_start + j + 2) * d;
        const float* k3 = K + (n_start + j + 3) * d;

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        int k = 0;
        for (; k + 15 < d; k += 16)
        {
            __m512 qv = _mm512_loadu_ps(q + k);
            acc0 = _mm512_fmadd_ps(qv, _mm512_loadu_ps(k0 + k), acc0);
            acc1 = _mm512_fmadd_ps(qv, _mm512_loadu_ps(k1 + k), acc1);
            acc2 = _mm512_fmadd_ps(qv, _mm512_loadu_ps(k2 + k), acc2);
            acc3 = _mm512_fmadd_ps(qv, _mm512_loadu_ps(k3 + k), acc3);
        }
        if (k < d)
        {
            __mmask16 mask_d = (__mmask16)((1u << (d - k)) - 1);
            __m512 qv = _mm512_maskz_loadu_ps(mask_d, q + k);
            acc0 = _mm512_fmadd_ps(qv, _mm512_maskz_loadu_ps(mask_d, k0 + k), acc0);
            acc1 = _mm512_fmadd_ps(qv, _mm512_maskz_loadu_ps(mask_d, k1 + k), acc1);
            acc2 = _mm512_fmadd_ps(qv, _mm512_maskz_loadu_ps(mask_d, k2 + k), acc2);
            acc3 = _mm512_fmadd_ps(qv, _mm512_maskz_loadu_ps(mask_d, k3 + k), acc3);
        }

        s[j + 0] = _mm512_comp_reduce_add_ps(acc0) * scale;
        s[j + 1] = _mm512_comp_reduce_add_ps(acc1) * scale;
        s[j + 2] = _mm512_comp_reduce_add_ps(acc2) * scale;
        s[j + 3] = _mm512_comp_reduce_add_ps(acc3) * scale;
    }

    for (; j < block_n; j++)
    {
        __m512 acc = _mm512_setzero_ps();
        int k = 0;
        for (; k + 15 < d; k += 16)
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(q + k), _mm512_loadu_ps(K + (n_start + j) * d + k), acc);
        if (k < d)
        {
            __mmask16 mask_d = (__mmask16)((1u << (d - k)) - 1);
            acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(mask_d, q + k), _mm512_maskz_loadu_ps(mask_d, K + (n_start + j) * d + k), acc);
        }
        s[j] = _mm512_comp_reduce_add_ps(acc) * scale;
    }
}

static inline void decode_pv_gemv_avx512(float* out, const float* s, const float* V, int n_start, int block_n, int out_d)
{
    for (int j = 0; j < block_n; j++)
    {
        __m512 pvec = _mm512_set1_ps(s[j]);
        int k = 0;
        for (; k + 15 < out_d; k += 16)
        {
            __m512 oval = _mm512_loadu_ps(out + k);
            __m512 vval = _mm512_loadu_ps(V + (n_start + j) * out_d + k);
            _mm512_storeu_ps(out + k, _mm512_fmadd_ps(pvec, vval, oval));
        }
        if (k < out_d)
        {
            __mmask16 mask_d = (__mmask16)((1u << (out_d - k)) - 1);
            __m512 oval = _mm512_maskz_loadu_ps(mask_d, out + k);
            __m512 vval = _mm512_maskz_loadu_ps(mask_d, V + (n_start + j) * out_d + k);
            _mm512_mask_storeu_ps(out + k, mask_d, _mm512_fmadd_ps(pvec, vval, oval));
        }
    }
}

static void sdpa_decode_avx512(float* out, const float* q,
    const float* K, const float* V, const float* mask,
    int n, int d, int out_d, float scale)
{
    const int BLOCK_N = 128;
    __attribute__((aligned(64))) float s[BLOCK_N];

    vec_zero_avx512(out, out_d);

    float m = -FLT_MAX;
    float l = 0.f;

    for (int n_start = 0; n_start < n; n_start += BLOCK_N)
    {
        int block_n = std::min(BLOCK_N, n - n_start);

        decode_qk_dot_avx512(s, q, K, n_start, block_n, d, scale);

        if (mask)
        {
            int j = 0;
            for (; j + 15 < block_n; j += 16)
                _mm512_storeu_ps(s + j, _mm512_add_ps(_mm512_loadu_ps(s + j), _mm512_loadu_ps(mask + n_start + j)));
            if (j < block_n)
            {
                __mmask16 mask_n = (__mmask16)((1u << (block_n - j)) - 1);
                _mm512_mask_storeu_ps(s + j, mask_n,
                    _mm512_add_ps(_mm512_maskz_loadu_ps(mask_n, s + j), _mm512_maskz_loadu_ps(mask_n, mask + n_start + j)));
            }
        }

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

        float new_m = std::max(m, tile_m);
        if (m != new_m)
        {
            float scale_factor = expf(m - new_m);
            l *= scale_factor;
            vec_scale_avx512(out, scale_factor, out_d);
        }

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

        decode_pv_gemv_avx512(out, s, V, n_start, block_n, out_d);

        m = new_m;
    }

    float inv_l = 1.f / l;
    vec_scale_avx512(out, inv_l, out_d);
}

#endif // __AVX512F__

#if __AVX__

static void qk_gemm_avx(float* S, const float* Q, const float* K,
        int m, int n, int d, float scale)
{
    int i = 0;
    for (; i + 6 <= m; i += 6)
    {
        int j = 0;
        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * d;
            const float* k1 = K + (j + 1) * d;

            __m256 acc[6][2];
            for (int mi = 0; mi < 6; mi++)
            {
                acc[mi][0] = _mm256_setzero_ps();
                acc[mi][1] = _mm256_setzero_ps();
            }

            int k = 0;
            for (; k + 7 < d; k += 8)
            {
                __m256 kv0 = _mm256_loadu_ps(k0 + k);
                __m256 kv1 = _mm256_loadu_ps(k1 + k);

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
                    sum0 += qv * k0[k];
                    sum1 += qv * k1[k];
                }

                S[(i + mi) * n + j + 0] = sum0 * scale;
                S[(i + mi) * n + j + 1] = sum1 * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * d;

            __m256 acc[6];
            for (int mi = 0; mi < 6; mi++)
                acc[mi] = _mm256_setzero_ps();

            int k = 0;
            for (; k + 7 < d; k += 8)
            {
                __m256 kvec = _mm256_loadu_ps(kptr + k);
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
                    sum += Q[(i + mi) * d + k] * kptr[k];
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
            const float* k0 = K + (j + 0) * d;
            const float* k1 = K + (j + 1) * d;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            int k = 0;
            for (; k + 7 < d; k += 8)
            {
                __m256 qvec = _mm256_loadu_ps(qptr + k);
                acc0 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k0 + k), acc0);
                acc1 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k1 + k), acc1);
            }

            float sum0 = _mm256_reduce_add_ps(acc0);
            float sum1 = _mm256_reduce_add_ps(acc1);

            for (; k < d; k++)
            {
                float qv = qptr[k];
                sum0 += qv * k0[k];
                sum1 += qv * k1[k];
            }

            S[i * n + j + 0] = sum0 * scale;
            S[i * n + j + 1] = sum1 * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * d;
            const float* kptr = K + j * d;
            float sum = 0.f;
            int k = 0;
            __m256 vacc = _mm256_setzero_ps();
            for (; k + 7 < d; k += 8)
                vacc = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr + k), _mm256_loadu_ps(kptr + k), vacc);
            sum = _mm256_reduce_add_ps(vacc);
            for (; k < d; k++)
                sum += qptr[k] * kptr[k];
            S[i * n + j] = sum * scale;
        }
    }
}


template<int D>
static inline void qk_gemm_specialized_avx(float* S, const float* Q, const float* K,
        int m, int n, float scale)
{
    int i = 0;
    for (; i + 6 <= m; i += 6)
    {
        int j = 0;
        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * D;
            const float* k1 = K + (j + 1) * D;

            __m256 acc[6][2];
            for (int mi = 0; mi < 6; mi++)
            {
                acc[mi][0] = _mm256_setzero_ps();
                acc[mi][1] = _mm256_setzero_ps();
            }

            for (int k = 0; k < D; k += 8)
            {
                __m256 kv0 = _mm256_loadu_ps(k0 + k);
                __m256 kv1 = _mm256_loadu_ps(k1 + k);

                for (int mi = 0; mi < 6; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * D + k);
                    acc[mi][0] = _mm256_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm256_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 6; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm256_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm256_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * D;

            __m256 acc[6];
            for (int mi = 0; mi < 6; mi++)
                acc[mi] = _mm256_setzero_ps();

            for (int k = 0; k < D; k += 8)
            {
                __m256 kvec = _mm256_loadu_ps(kptr + k);
                for (int mi = 0; mi < 6; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * D + k);
                    acc[mi] = _mm256_comp_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 6; mi++)
                S[(i + mi) * n + j] = _mm256_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i < m; i++)
    {
        int j = 0;
        for (; j + 1 < n; j += 2)
        {
            const float* qptr = Q + i * D;
            const float* k0 = K + (j + 0) * D;
            const float* k1 = K + (j + 1) * D;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (int k = 0; k < D; k += 8)
            {
                __m256 qvec = _mm256_loadu_ps(qptr + k);
                acc0 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k0 + k), acc0);
                acc1 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k1 + k), acc1);
            }

            S[i * n + j + 0] = _mm256_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm256_reduce_add_ps(acc1) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * D;
            const float* kptr = K + j * D;
            __m256 vacc = _mm256_setzero_ps();
            for (int k = 0; k < D; k += 8)
                vacc = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr + k), _mm256_loadu_ps(kptr + k), vacc);
            S[i * n + j] = _mm256_reduce_add_ps(vacc) * scale;
        }
    }
}

template<>
void qk_gemm_specialized_avx<1024>(float* S, const float* Q, const float* K,
        int m, int n, float scale)
{
    int i = 0;
    for (; i + 4 <= m; i += 4)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;
            const float* k2 = K + (j + 2) * 1024;
            const float* k3 = K + (j + 3) * 1024;

            __m256 acc[4][4];
            for (int mi = 0; mi < 4; mi++)
            {
                acc[mi][0] = _mm256_setzero_ps();
                acc[mi][1] = _mm256_setzero_ps();
                acc[mi][2] = _mm256_setzero_ps();
                acc[mi][3] = _mm256_setzero_ps();
            }

            for (int k = 0; k < 1024; k += 8)
            {
                __m256 kv0 = _mm256_loadu_ps(k0 + k);
                __m256 kv1 = _mm256_loadu_ps(k1 + k);
                __m256 kv2 = _mm256_loadu_ps(k2 + k);
                __m256 kv3 = _mm256_loadu_ps(k3 + k);

                for (int mi = 0; mi < 4; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi][0] = _mm256_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm256_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                    acc[mi][2] = _mm256_comp_fmadd_ps(qvec, kv2, acc[mi][2]);
                    acc[mi][3] = _mm256_comp_fmadd_ps(qvec, kv3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 4; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm256_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm256_reduce_add_ps(acc[mi][1]) * scale;
                S[(i + mi) * n + j + 2] = _mm256_reduce_add_ps(acc[mi][2]) * scale;
                S[(i + mi) * n + j + 3] = _mm256_reduce_add_ps(acc[mi][3]) * scale;
            }
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;

            __m256 acc[4][2];
            for (int mi = 0; mi < 4; mi++)
            {
                acc[mi][0] = _mm256_setzero_ps();
                acc[mi][1] = _mm256_setzero_ps();
            }

            for (int k = 0; k < 1024; k += 8)
            {
                __m256 kv0 = _mm256_loadu_ps(k0 + k);
                __m256 kv1 = _mm256_loadu_ps(k1 + k);

                for (int mi = 0; mi < 4; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi][0] = _mm256_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm256_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 4; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm256_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm256_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * 1024;

            __m256 acc[4];
            for (int mi = 0; mi < 4; mi++)
                acc[mi] = _mm256_setzero_ps();

            for (int k = 0; k < 1024; k += 8)
            {
                __m256 kvec = _mm256_loadu_ps(kptr + k);
                for (int mi = 0; mi < 4; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi] = _mm256_comp_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 4; mi++)
                S[(i + mi) * n + j] = _mm256_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i + 2 <= m; i += 2)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;
            const float* k2 = K + (j + 2) * 1024;
            const float* k3 = K + (j + 3) * 1024;

            __m256 acc[2][4];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm256_setzero_ps();
                acc[mi][1] = _mm256_setzero_ps();
                acc[mi][2] = _mm256_setzero_ps();
                acc[mi][3] = _mm256_setzero_ps();
            }

            for (int k = 0; k < 1024; k += 8)
            {
                __m256 kv0 = _mm256_loadu_ps(k0 + k);
                __m256 kv1 = _mm256_loadu_ps(k1 + k);
                __m256 kv2 = _mm256_loadu_ps(k2 + k);
                __m256 kv3 = _mm256_loadu_ps(k3 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi][0] = _mm256_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm256_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                    acc[mi][2] = _mm256_comp_fmadd_ps(qvec, kv2, acc[mi][2]);
                    acc[mi][3] = _mm256_comp_fmadd_ps(qvec, kv3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm256_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm256_reduce_add_ps(acc[mi][1]) * scale;
                S[(i + mi) * n + j + 2] = _mm256_reduce_add_ps(acc[mi][2]) * scale;
                S[(i + mi) * n + j + 3] = _mm256_reduce_add_ps(acc[mi][3]) * scale;
            }
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;

            __m256 acc[2][2];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm256_setzero_ps();
                acc[mi][1] = _mm256_setzero_ps();
            }

            for (int k = 0; k < 1024; k += 8)
            {
                __m256 kv0 = _mm256_loadu_ps(k0 + k);
                __m256 kv1 = _mm256_loadu_ps(k1 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi][0] = _mm256_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm256_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm256_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm256_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * 1024;

            __m256 acc[2];
            for (int mi = 0; mi < 2; mi++)
                acc[mi] = _mm256_setzero_ps();

            for (int k = 0; k < 1024; k += 8)
            {
                __m256 kvec = _mm256_loadu_ps(kptr + k);
                for (int mi = 0; mi < 2; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi] = _mm256_comp_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
                S[(i + mi) * n + j] = _mm256_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i < m; i++)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* qptr = Q + i * 1024;
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;
            const float* k2 = K + (j + 2) * 1024;
            const float* k3 = K + (j + 3) * 1024;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int k = 0; k < 1024; k += 8)
            {
                __m256 qvec = _mm256_loadu_ps(qptr + k);
                acc0 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k0 + k), acc0);
                acc1 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k1 + k), acc1);
                acc2 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k2 + k), acc2);
                acc3 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k3 + k), acc3);
            }

            S[i * n + j + 0] = _mm256_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm256_reduce_add_ps(acc1) * scale;
            S[i * n + j + 2] = _mm256_reduce_add_ps(acc2) * scale;
            S[i * n + j + 3] = _mm256_reduce_add_ps(acc3) * scale;
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* qptr = Q + i * 1024;
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (int k = 0; k < 1024; k += 8)
            {
                __m256 qvec = _mm256_loadu_ps(qptr + k);
                acc0 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k0 + k), acc0);
                acc1 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k1 + k), acc1);
            }

            S[i * n + j + 0] = _mm256_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm256_reduce_add_ps(acc1) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * 1024;
            const float* kptr = K + j * 1024;
            __m256 vacc = _mm256_setzero_ps();
            for (int k = 0; k < 1024; k += 8)
                vacc = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr + k), _mm256_loadu_ps(kptr + k), vacc);
            S[i * n + j] = _mm256_reduce_add_ps(vacc) * scale;
        }
    }
}

template<>
void qk_gemm_specialized_avx<2048>(float* S, const float* Q, const float* K,
        int m, int n, float scale)
{
    int i = 0;
    for (; i + 2 <= m; i += 2)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* k0 = K + (j + 0) * 2048;
            const float* k1 = K + (j + 1) * 2048;
            const float* k2 = K + (j + 2) * 2048;
            const float* k3 = K + (j + 3) * 2048;

            __m256 acc[2][4];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm256_setzero_ps();
                acc[mi][1] = _mm256_setzero_ps();
                acc[mi][2] = _mm256_setzero_ps();
                acc[mi][3] = _mm256_setzero_ps();
            }

            for (int k = 0; k < 2048; k += 8)
            {
                __m256 kv0 = _mm256_loadu_ps(k0 + k);
                __m256 kv1 = _mm256_loadu_ps(k1 + k);
                __m256 kv2 = _mm256_loadu_ps(k2 + k);
                __m256 kv3 = _mm256_loadu_ps(k3 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * 2048 + k);
                    acc[mi][0] = _mm256_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm256_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                    acc[mi][2] = _mm256_comp_fmadd_ps(qvec, kv2, acc[mi][2]);
                    acc[mi][3] = _mm256_comp_fmadd_ps(qvec, kv3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm256_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm256_reduce_add_ps(acc[mi][1]) * scale;
                S[(i + mi) * n + j + 2] = _mm256_reduce_add_ps(acc[mi][2]) * scale;
                S[(i + mi) * n + j + 3] = _mm256_reduce_add_ps(acc[mi][3]) * scale;
            }
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 2048;
            const float* k1 = K + (j + 1) * 2048;

            __m256 acc[2][2];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm256_setzero_ps();
                acc[mi][1] = _mm256_setzero_ps();
            }

            for (int k = 0; k < 2048; k += 8)
            {
                __m256 kv0 = _mm256_loadu_ps(k0 + k);
                __m256 kv1 = _mm256_loadu_ps(k1 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * 2048 + k);
                    acc[mi][0] = _mm256_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm256_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm256_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm256_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * 2048;

            __m256 acc[2];
            for (int mi = 0; mi < 2; mi++)
                acc[mi] = _mm256_setzero_ps();

            for (int k = 0; k < 2048; k += 8)
            {
                __m256 kvec = _mm256_loadu_ps(kptr + k);
                for (int mi = 0; mi < 2; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * 2048 + k);
                    acc[mi] = _mm256_comp_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
                S[(i + mi) * n + j] = _mm256_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i < m; i++)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* qptr = Q + i * 2048;
            const float* k0 = K + (j + 0) * 2048;
            const float* k1 = K + (j + 1) * 2048;
            const float* k2 = K + (j + 2) * 2048;
            const float* k3 = K + (j + 3) * 2048;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int k = 0; k < 2048; k += 8)
            {
                __m256 qvec = _mm256_loadu_ps(qptr + k);
                acc0 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k0 + k), acc0);
                acc1 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k1 + k), acc1);
                acc2 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k2 + k), acc2);
                acc3 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k3 + k), acc3);
            }

            S[i * n + j + 0] = _mm256_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm256_reduce_add_ps(acc1) * scale;
            S[i * n + j + 2] = _mm256_reduce_add_ps(acc2) * scale;
            S[i * n + j + 3] = _mm256_reduce_add_ps(acc3) * scale;
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* qptr = Q + i * 2048;
            const float* k0 = K + (j + 0) * 2048;
            const float* k1 = K + (j + 1) * 2048;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (int k = 0; k < 2048; k += 8)
            {
                __m256 qvec = _mm256_loadu_ps(qptr + k);
                acc0 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k0 + k), acc0);
                acc1 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k1 + k), acc1);
            }

            S[i * n + j + 0] = _mm256_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm256_reduce_add_ps(acc1) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * 2048;
            const float* kptr = K + j * 2048;
            __m256 vacc = _mm256_setzero_ps();
            for (int k = 0; k < 2048; k += 8)
                vacc = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr + k), _mm256_loadu_ps(kptr + k), vacc);
            S[i * n + j] = _mm256_reduce_add_ps(vacc) * scale;
        }
    }
}

template<>
void qk_gemm_specialized_avx<4096>(float* S, const float* Q, const float* K,
        int m, int n, float scale)
{
    int i = 0;
    for (; i + 2 <= m; i += 2)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* k0 = K + (j + 0) * 4096;
            const float* k1 = K + (j + 1) * 4096;
            const float* k2 = K + (j + 2) * 4096;
            const float* k3 = K + (j + 3) * 4096;

            __m256 acc[2][4];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm256_setzero_ps();
                acc[mi][1] = _mm256_setzero_ps();
                acc[mi][2] = _mm256_setzero_ps();
                acc[mi][3] = _mm256_setzero_ps();
            }

            for (int k = 0; k < 4096; k += 8)
            {
                __m256 kv0 = _mm256_loadu_ps(k0 + k);
                __m256 kv1 = _mm256_loadu_ps(k1 + k);
                __m256 kv2 = _mm256_loadu_ps(k2 + k);
                __m256 kv3 = _mm256_loadu_ps(k3 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * 4096 + k);
                    acc[mi][0] = _mm256_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm256_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                    acc[mi][2] = _mm256_comp_fmadd_ps(qvec, kv2, acc[mi][2]);
                    acc[mi][3] = _mm256_comp_fmadd_ps(qvec, kv3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm256_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm256_reduce_add_ps(acc[mi][1]) * scale;
                S[(i + mi) * n + j + 2] = _mm256_reduce_add_ps(acc[mi][2]) * scale;
                S[(i + mi) * n + j + 3] = _mm256_reduce_add_ps(acc[mi][3]) * scale;
            }
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 4096;
            const float* k1 = K + (j + 1) * 4096;

            __m256 acc[2][2];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm256_setzero_ps();
                acc[mi][1] = _mm256_setzero_ps();
            }

            for (int k = 0; k < 4096; k += 8)
            {
                __m256 kv0 = _mm256_loadu_ps(k0 + k);
                __m256 kv1 = _mm256_loadu_ps(k1 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * 4096 + k);
                    acc[mi][0] = _mm256_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm256_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm256_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm256_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * 4096;

            __m256 acc[2];
            for (int mi = 0; mi < 2; mi++)
                acc[mi] = _mm256_setzero_ps();

            for (int k = 0; k < 4096; k += 8)
            {
                __m256 kvec = _mm256_loadu_ps(kptr + k);
                for (int mi = 0; mi < 2; mi++)
                {
                    __m256 qvec = _mm256_loadu_ps(Q + (i + mi) * 4096 + k);
                    acc[mi] = _mm256_comp_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
                S[(i + mi) * n + j] = _mm256_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i < m; i++)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* qptr = Q + i * 4096;
            const float* k0 = K + (j + 0) * 4096;
            const float* k1 = K + (j + 1) * 4096;
            const float* k2 = K + (j + 2) * 4096;
            const float* k3 = K + (j + 3) * 4096;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int k = 0; k < 4096; k += 8)
            {
                __m256 qvec = _mm256_loadu_ps(qptr + k);
                acc0 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k0 + k), acc0);
                acc1 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k1 + k), acc1);
                acc2 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k2 + k), acc2);
                acc3 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k3 + k), acc3);
            }

            S[i * n + j + 0] = _mm256_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm256_reduce_add_ps(acc1) * scale;
            S[i * n + j + 2] = _mm256_reduce_add_ps(acc2) * scale;
            S[i * n + j + 3] = _mm256_reduce_add_ps(acc3) * scale;
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* qptr = Q + i * 4096;
            const float* k0 = K + (j + 0) * 4096;
            const float* k1 = K + (j + 1) * 4096;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (int k = 0; k < 4096; k += 8)
            {
                __m256 qvec = _mm256_loadu_ps(qptr + k);
                acc0 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k0 + k), acc0);
                acc1 = _mm256_comp_fmadd_ps(qvec, _mm256_loadu_ps(k1 + k), acc1);
            }

            S[i * n + j + 0] = _mm256_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm256_reduce_add_ps(acc1) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * 4096;
            const float* kptr = K + j * 4096;
            __m256 vacc = _mm256_setzero_ps();
            for (int k = 0; k < 4096; k += 8)
                vacc = _mm256_comp_fmadd_ps(_mm256_loadu_ps(qptr + k), _mm256_loadu_ps(kptr + k), vacc);
            S[i * n + j] = _mm256_reduce_add_ps(vacc) * scale;
        }
    }
}


template<int M_BLOCK, int D_UNROLL>
static inline void pv_gemm_avx(float* O, const float* P, const float* V, int m, int n, int d)
{
    const int VEC_PER_UNROLL = D_UNROLL / 8;
    int i = 0;
    for (; i + M_BLOCK <= m; i += M_BLOCK)
    {
        float* op[M_BLOCK];
        const float* pptr[M_BLOCK];
        for (int mi = 0; mi < M_BLOCK; mi++)
        {
            op[mi] = O + (i + mi) * d;
            pptr[mi] = P + (i + mi) * n;
        }

        int dd = 0;
        for (; dd + D_UNROLL - 1 < d; dd += D_UNROLL)
        {
            __m256 acc[M_BLOCK][VEC_PER_UNROLL];
            for (int mi = 0; mi < M_BLOCK; mi++)
                for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                    acc[mi][vi] = _mm256_loadu_ps(op[mi] + dd + vi * 8);

            for (int j = 0; j < n; j++)
            {
                __m256 vvec[VEC_PER_UNROLL];
                for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                    vvec[vi] = _mm256_loadu_ps(V + j * d + dd + vi * 8);

                for (int mi = 0; mi < M_BLOCK; mi++)
                {
                    __m256 pvec = _mm256_set1_ps(pptr[mi][j]);
                    for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                        acc[mi][vi] = _mm256_comp_fmadd_ps(pvec, vvec[vi], acc[mi][vi]);
                }
            }

            for (int mi = 0; mi < M_BLOCK; mi++)
                for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                    _mm256_storeu_ps(op[mi] + dd + vi * 8, acc[mi][vi]);
        }

        for (; dd + 7 < d; dd += 8)
        {
            __m256 acc[M_BLOCK];
            for (int mi = 0; mi < M_BLOCK; mi++)
                acc[mi] = _mm256_loadu_ps(op[mi] + dd);

            for (int j = 0; j < n; j++)
            {
                __m256 vvec = _mm256_loadu_ps(V + j * d + dd);
                for (int mi = 0; mi < M_BLOCK; mi++)
                    acc[mi] = _mm256_comp_fmadd_ps(_mm256_set1_ps(pptr[mi][j]), vvec, acc[mi]);
            }

            for (int mi = 0; mi < M_BLOCK; mi++)
                _mm256_storeu_ps(op[mi] + dd, acc[mi]);
        }

        for (; dd < d; dd++)
        {
            for (int mi = 0; mi < M_BLOCK; mi++)
            {
                float acc = op[mi][dd];
                for (int j = 0; j < n; j++)
                    acc += pptr[mi][j] * V[j * d + dd];
                op[mi][dd] = acc;
            }
        }
    }

    for (; i < m; i++)
    {
        float* optr = O + i * d;
        const float* pptr = P + i * n;

        int dd = 0;
        for (; dd + D_UNROLL - 1 < d; dd += D_UNROLL)
        {
            __m256 acc[VEC_PER_UNROLL];
            for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                acc[vi] = _mm256_loadu_ps(optr + dd + vi * 8);

            for (int j = 0; j < n; j++)
            {
                __m256 pvec = _mm256_set1_ps(pptr[j]);
                for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                    acc[vi] = _mm256_comp_fmadd_ps(pvec, _mm256_loadu_ps(V + j * d + dd + vi * 8), acc[vi]);
            }

            for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                _mm256_storeu_ps(optr + dd + vi * 8, acc[vi]);
        }

        for (; dd + 7 < d; dd += 8)
        {
            __m256 acc = _mm256_loadu_ps(optr + dd);
            for (int j = 0; j < n; j++)
                acc = _mm256_comp_fmadd_ps(_mm256_set1_ps(pptr[j]), _mm256_loadu_ps(V + j * d + dd), acc);
            _mm256_storeu_ps(optr + dd, acc);
        }

        for (; dd < d; dd++)
        {
            float acc = optr[dd];
            for (int j = 0; j < n; j++)
                acc += pptr[j] * V[j * d + dd];
            optr[dd] = acc;
        }
    }
}


template<int M_BLOCK, int D>
static inline void pv_gemm_avx(float* O, const float* P, const float* V, int m, int n)
{
    const int VEC_PER_D = D / 8;
    int i = 0;
    for (; i + M_BLOCK <= m; i += M_BLOCK)
    {
        float* op[M_BLOCK];
        const float* pptr[M_BLOCK];
        for (int mi = 0; mi < M_BLOCK; mi++)
        {
            op[mi] = O + (i + mi) * D;
            pptr[mi] = P + (i + mi) * n;
        }

        for (int mi = 0; mi < M_BLOCK; mi++)
        {
            __m256 acc[VEC_PER_D];
            for (int vi = 0; vi < VEC_PER_D; vi++)
                acc[vi] = _mm256_loadu_ps(op[mi] + vi * 8);

            for (int j = 0; j < n; j++)
            {
                __m256 pvec = _mm256_set1_ps(pptr[mi][j]);
                for (int vi = 0; vi < VEC_PER_D; vi++)
                    acc[vi] = _mm256_comp_fmadd_ps(pvec, _mm256_loadu_ps(V + j * D + vi * 8), acc[vi]);
            }

            for (int vi = 0; vi < VEC_PER_D; vi++)
                _mm256_storeu_ps(op[mi] + vi * 8, acc[vi]);
        }
    }

    for (; i < m; i++)
    {
        float* optr = O + i * D;
        const float* pptr = P + i * n;

        __m256 acc[VEC_PER_D];
        for (int vi = 0; vi < VEC_PER_D; vi++)
            acc[vi] = _mm256_loadu_ps(optr + vi * 8);

        for (int j = 0; j < n; j++)
        {
            __m256 pvec = _mm256_set1_ps(pptr[j]);
            for (int vi = 0; vi < VEC_PER_D; vi++)
                acc[vi] = _mm256_comp_fmadd_ps(pvec, _mm256_loadu_ps(V + j * D + vi * 8), acc[vi]);
        }

        for (int vi = 0; vi < VEC_PER_D; vi++)
            _mm256_storeu_ps(optr + vi * 8, acc[vi]);
    }
}


static inline void softmax_tile_avx(float* P, const float* S,
        float* m_vec, float* l_vec, float* scale_out, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        const float* sptr = S + i * n;
        float* pptr = P + i * n;

        __m256 vmax = _mm256_set1_ps(m_vec[i]);
        int j = 0;
        for (; j + 7 < n; j += 8)
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(sptr + j));
        float m_new = _mm256_reduce_max_ps(vmax);
        for (; j < n; j++)
            m_new = std::max(m_new, sptr[j]);

        float scale_factor = expf(m_vec[i] - m_new);
        scale_out[i] = scale_factor;
        l_vec[i] *= scale_factor;

        __m256 vm_new = _mm256_set1_ps(m_new);
        __m256 vsum = _mm256_setzero_ps();
        j = 0;
        for (; j + 7 < n; j += 8)
        {
            __m256 svec = _mm256_loadu_ps(sptr + j);
            __m256 evec = exp256_ps(_mm256_sub_ps(svec, vm_new));
            _mm256_storeu_ps(pptr + j, evec);
            vsum = _mm256_add_ps(vsum, evec);
        }
        float l_add = _mm256_reduce_add_ps(vsum);
        for (; j < n; j++)
        {
            pptr[j] = expf(sptr[j] - m_new);
            l_add += pptr[j];
        }
        l_vec[i] += l_add;
        m_vec[i] = m_new;
    }
}

static inline void vec_scale_avx(float* x, float s, int n)
{
    __m256 vscale = _mm256_set1_ps(s);
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vscale));
    for (; i < n; i++)
        x[i] *= s;
}

static inline void vec_zero_avx(float* x, int n)
{
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8)
        _mm256_storeu_ps(x + i, zero);
    for (; i < n; i++)
        x[i] = 0.f;
}

static inline void decode_qk_dot_avx(float* s, const float* q, const float* K, int n_start, int block_n, int d, float scale)
{
    int j = 0;
    for (; j + 1 < block_n; j += 2)
    {
        const float* k0 = K + (n_start + j + 0) * d;
        const float* k1 = K + (n_start + j + 1) * d;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();

        int k = 0;
        for (; k + 7 < d; k += 8)
        {
            __m256 qv = _mm256_loadu_ps(q + k);
            acc0 = _mm256_comp_fmadd_ps(qv, _mm256_loadu_ps(k0 + k), acc0);
            acc1 = _mm256_comp_fmadd_ps(qv, _mm256_loadu_ps(k1 + k), acc1);
        }

        float sum0 = _mm256_reduce_add_ps(acc0);
        float sum1 = _mm256_reduce_add_ps(acc1);

        for (; k < d; k++)
        {
            sum0 += q[k] * k0[k];
            sum1 += q[k] * k1[k];
        }

        s[j + 0] = sum0 * scale;
        s[j + 1] = sum1 * scale;
    }

    for (; j < block_n; j++)
    {
        const float* kptr = K + (n_start + j) * d;
        __m256 acc = _mm256_setzero_ps();
        int k = 0;
        for (; k + 7 < d; k += 8)
            acc = _mm256_comp_fmadd_ps(_mm256_loadu_ps(q + k), _mm256_loadu_ps(kptr + k), acc);
        float sum = _mm256_reduce_add_ps(acc);
        for (; k < d; k++)
            sum += q[k] * kptr[k];
        s[j] = sum * scale;
    }
}

static inline void decode_pv_gemv_avx(float* out, const float* s, const float* V, int n_start, int block_n, int out_d)
{
    for (int j = 0; j < block_n; j++)
    {
        __m256 pvec = _mm256_set1_ps(s[j]);
        int k = 0;
        for (; k + 7 < out_d; k += 8)
        {
            __m256 oval = _mm256_loadu_ps(out + k);
            __m256 vval = _mm256_loadu_ps(V + (n_start + j) * out_d + k);
            _mm256_storeu_ps(out + k, _mm256_comp_fmadd_ps(pvec, vval, oval));
        }
        for (; k < out_d; k++)
            out[k] += s[j] * V[(n_start + j) * out_d + k];
    }
}

static void sdpa_decode_avx(float* out, const float* q,
    const float* K, const float* V, const float* mask,
    int n, int d, int out_d, float scale)
{
    const int BLOCK_N = 128;
    __attribute__((aligned(32))) float s[BLOCK_N];

    vec_zero_avx(out, out_d);

    float m = -FLT_MAX;
    float l = 0.f;

    for (int n_start = 0; n_start < n; n_start += BLOCK_N)
    {
        int block_n = std::min(BLOCK_N, n - n_start);

        decode_qk_dot_avx(s, q, K, n_start, block_n, d, scale);

        if (mask)
        {
            int j = 0;
            for (; j + 7 < block_n; j += 8)
                _mm256_storeu_ps(s + j, _mm256_add_ps(_mm256_loadu_ps(s + j), _mm256_loadu_ps(mask + n_start + j)));
            for (; j < block_n; j++)
                s[j] += mask[n_start + j];
        }

        __m256 vmax = _mm256_set1_ps(-FLT_MAX);
        int j = 0;
        for (; j + 7 < block_n; j += 8)
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(s + j));
        float tile_m = _mm256_reduce_max_ps(vmax);
        for (; j < block_n; j++)
            tile_m = std::max(tile_m, s[j]);

        float new_m = std::max(m, tile_m);
        if (m != new_m)
        {
            float scale_factor = expf(m - new_m);
            l *= scale_factor;
            vec_scale_avx(out, scale_factor, out_d);
        }

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

        decode_pv_gemv_avx(out, s, V, n_start, block_n, out_d);

        m = new_m;
    }

    float inv_l = 1.f / l;
    vec_scale_avx(out, inv_l, out_d);
}

#endif // __AVX__

#if __SSE2__

static void qk_gemm_sse2(float* S, const float* Q, const float* K,
        int m, int n, int d, float scale)
{
    int i = 0;
    for (; i + 4 <= m; i += 4)
    {
        int j = 0;
        for (; j < n; j++)
        {
            const float* kptr = K + j * d;

            __m128 acc[4];
            for (int mi = 0; mi < 4; mi++)
                acc[mi] = _mm_setzero_ps();

            int k = 0;
            for (; k + 3 < d; k += 4)
            {
                __m128 kvec = _mm_loadu_ps(kptr + k);
                for (int mi = 0; mi < 4; mi++)
                {
                    __m128 qvec = _mm_loadu_ps(Q + (i + mi) * d + k);
                    acc[mi] = _mm_comp_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 4; mi++)
            {
                float sum = _mm_reduce_add_ps(acc[mi]);
                for (; k < d; k++)
                    sum += Q[(i + mi) * d + k] * kptr[k];
                S[(i + mi) * n + j] = sum * scale;
            }
        }
    }

    for (; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            const float* qptr = Q + i * d;
            const float* kptr = K + j * d;
            float sum = 0.f;
            int k = 0;
            __m128 vacc = _mm_setzero_ps();
            for (; k + 3 < d; k += 4)
                vacc = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr + k), _mm_loadu_ps(kptr + k), vacc);
            sum = _mm_reduce_add_ps(vacc);
            for (; k < d; k++)
                sum += qptr[k] * kptr[k];
            S[i * n + j] = sum * scale;
        }
    }
}


template<int D>
static inline void qk_gemm_specialized_sse2(float* S, const float* Q, const float* K,
        int m, int n, float scale)
{
    int i = 0;
    for (; i + 4 <= m; i += 4)
    {
        for (int j = 0; j < n; j++)
        {
            const float* kptr = K + j * D;

            const float* q0 = Q + (i + 0) * D;
            const float* q1 = Q + (i + 1) * D;
            const float* q2 = Q + (i + 2) * D;
            const float* q3 = Q + (i + 3) * D;

            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();
            __m128 acc2 = _mm_setzero_ps();
            __m128 acc3 = _mm_setzero_ps();

            for (int k = 0; k < D; k += 4)
            {
                __m128 kvec = _mm_loadu_ps(kptr + k);

                __m128 qvec = _mm_loadu_ps(q0 + k);
                acc0 = _mm_comp_fmadd_ps(qvec, kvec, acc0);

                qvec = _mm_loadu_ps(q1 + k);
                acc1 = _mm_comp_fmadd_ps(qvec, kvec, acc1);

                qvec = _mm_loadu_ps(q2 + k);
                acc2 = _mm_comp_fmadd_ps(qvec, kvec, acc2);

                qvec = _mm_loadu_ps(q3 + k);
                acc3 = _mm_comp_fmadd_ps(qvec, kvec, acc3);
            }

            S[(i + 0) * n + j] = _mm_reduce_add_ps(acc0) * scale;
            S[(i + 1) * n + j] = _mm_reduce_add_ps(acc1) * scale;
            S[(i + 2) * n + j] = _mm_reduce_add_ps(acc2) * scale;
            S[(i + 3) * n + j] = _mm_reduce_add_ps(acc3) * scale;
        }
    }

    for (; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            const float* qptr = Q + i * D;
            const float* kptr = K + j * D;
            __m128 vacc = _mm_setzero_ps();
            for (int k = 0; k < D; k += 4)
                vacc = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr + k), _mm_loadu_ps(kptr + k), vacc);
            S[i * n + j] = _mm_reduce_add_ps(vacc) * scale;
        }
    }
}

template<>
void qk_gemm_specialized_sse2<1024>(float* S, const float* Q, const float* K,
        int m, int n, float scale)
{
    int i = 0;
    for (; i + 4 <= m; i += 4)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;
            const float* k2 = K + (j + 2) * 1024;
            const float* k3 = K + (j + 3) * 1024;

            __m128 acc[4][4];
            for (int mi = 0; mi < 4; mi++)
            {
                acc[mi][0] = _mm_setzero_ps();
                acc[mi][1] = _mm_setzero_ps();
                acc[mi][2] = _mm_setzero_ps();
                acc[mi][3] = _mm_setzero_ps();
            }

            for (int k = 0; k < 1024; k += 4)
            {
                __m128 kv0 = _mm_loadu_ps(k0 + k);
                __m128 kv1 = _mm_loadu_ps(k1 + k);
                __m128 kv2 = _mm_loadu_ps(k2 + k);
                __m128 kv3 = _mm_loadu_ps(k3 + k);

                for (int mi = 0; mi < 4; mi++)
                {
                    __m128 qvec = _mm_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi][0] = _mm_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                    acc[mi][2] = _mm_comp_fmadd_ps(qvec, kv2, acc[mi][2]);
                    acc[mi][3] = _mm_comp_fmadd_ps(qvec, kv3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 4; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm_reduce_add_ps(acc[mi][1]) * scale;
                S[(i + mi) * n + j + 2] = _mm_reduce_add_ps(acc[mi][2]) * scale;
                S[(i + mi) * n + j + 3] = _mm_reduce_add_ps(acc[mi][3]) * scale;
            }
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;

            __m128 acc[4][2];
            for (int mi = 0; mi < 4; mi++)
            {
                acc[mi][0] = _mm_setzero_ps();
                acc[mi][1] = _mm_setzero_ps();
            }

            for (int k = 0; k < 1024; k += 4)
            {
                __m128 kv0 = _mm_loadu_ps(k0 + k);
                __m128 kv1 = _mm_loadu_ps(k1 + k);

                for (int mi = 0; mi < 4; mi++)
                {
                    __m128 qvec = _mm_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi][0] = _mm_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 4; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * 1024;

            __m128 acc[4];
            for (int mi = 0; mi < 4; mi++)
                acc[mi] = _mm_setzero_ps();

            for (int k = 0; k < 1024; k += 4)
            {
                __m128 kvec = _mm_loadu_ps(kptr + k);
                for (int mi = 0; mi < 4; mi++)
                {
                    __m128 qvec = _mm_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi] = _mm_comp_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 4; mi++)
                S[(i + mi) * n + j] = _mm_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i + 2 <= m; i += 2)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;
            const float* k2 = K + (j + 2) * 1024;
            const float* k3 = K + (j + 3) * 1024;

            __m128 acc[2][4];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm_setzero_ps();
                acc[mi][1] = _mm_setzero_ps();
                acc[mi][2] = _mm_setzero_ps();
                acc[mi][3] = _mm_setzero_ps();
            }

            for (int k = 0; k < 1024; k += 4)
            {
                __m128 kv0 = _mm_loadu_ps(k0 + k);
                __m128 kv1 = _mm_loadu_ps(k1 + k);
                __m128 kv2 = _mm_loadu_ps(k2 + k);
                __m128 kv3 = _mm_loadu_ps(k3 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m128 qvec = _mm_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi][0] = _mm_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                    acc[mi][2] = _mm_comp_fmadd_ps(qvec, kv2, acc[mi][2]);
                    acc[mi][3] = _mm_comp_fmadd_ps(qvec, kv3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm_reduce_add_ps(acc[mi][1]) * scale;
                S[(i + mi) * n + j + 2] = _mm_reduce_add_ps(acc[mi][2]) * scale;
                S[(i + mi) * n + j + 3] = _mm_reduce_add_ps(acc[mi][3]) * scale;
            }
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;

            __m128 acc[2][2];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm_setzero_ps();
                acc[mi][1] = _mm_setzero_ps();
            }

            for (int k = 0; k < 1024; k += 4)
            {
                __m128 kv0 = _mm_loadu_ps(k0 + k);
                __m128 kv1 = _mm_loadu_ps(k1 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m128 qvec = _mm_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi][0] = _mm_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * 1024;

            __m128 acc[2];
            for (int mi = 0; mi < 2; mi++)
                acc[mi] = _mm_setzero_ps();

            for (int k = 0; k < 1024; k += 4)
            {
                __m128 kvec = _mm_loadu_ps(kptr + k);
                for (int mi = 0; mi < 2; mi++)
                {
                    __m128 qvec = _mm_loadu_ps(Q + (i + mi) * 1024 + k);
                    acc[mi] = _mm_comp_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
                S[(i + mi) * n + j] = _mm_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i < m; i++)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* qptr = Q + i * 1024;
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;
            const float* k2 = K + (j + 2) * 1024;
            const float* k3 = K + (j + 3) * 1024;

            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();
            __m128 acc2 = _mm_setzero_ps();
            __m128 acc3 = _mm_setzero_ps();

            for (int k = 0; k < 1024; k += 4)
            {
                __m128 qvec = _mm_loadu_ps(qptr + k);
                acc0 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k0 + k), acc0);
                acc1 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k1 + k), acc1);
                acc2 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k2 + k), acc2);
                acc3 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k3 + k), acc3);
            }

            S[i * n + j + 0] = _mm_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm_reduce_add_ps(acc1) * scale;
            S[i * n + j + 2] = _mm_reduce_add_ps(acc2) * scale;
            S[i * n + j + 3] = _mm_reduce_add_ps(acc3) * scale;
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* qptr = Q + i * 1024;
            const float* k0 = K + (j + 0) * 1024;
            const float* k1 = K + (j + 1) * 1024;

            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();

            for (int k = 0; k < 1024; k += 4)
            {
                __m128 qvec = _mm_loadu_ps(qptr + k);
                acc0 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k0 + k), acc0);
                acc1 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k1 + k), acc1);
            }

            S[i * n + j + 0] = _mm_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm_reduce_add_ps(acc1) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * 1024;
            const float* kptr = K + j * 1024;
            __m128 vacc = _mm_setzero_ps();
            for (int k = 0; k < 1024; k += 4)
                vacc = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr + k), _mm_loadu_ps(kptr + k), vacc);
            S[i * n + j] = _mm_reduce_add_ps(vacc) * scale;
        }
    }
}

template<>
void qk_gemm_specialized_sse2<2048>(float* S, const float* Q, const float* K,
        int m, int n, float scale)
{
    int i = 0;
    for (; i + 2 <= m; i += 2)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* k0 = K + (j + 0) * 2048;
            const float* k1 = K + (j + 1) * 2048;
            const float* k2 = K + (j + 2) * 2048;
            const float* k3 = K + (j + 3) * 2048;

            __m128 acc[2][4];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm_setzero_ps();
                acc[mi][1] = _mm_setzero_ps();
                acc[mi][2] = _mm_setzero_ps();
                acc[mi][3] = _mm_setzero_ps();
            }

            for (int k = 0; k < 2048; k += 4)
            {
                __m128 kv0 = _mm_loadu_ps(k0 + k);
                __m128 kv1 = _mm_loadu_ps(k1 + k);
                __m128 kv2 = _mm_loadu_ps(k2 + k);
                __m128 kv3 = _mm_loadu_ps(k3 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m128 qvec = _mm_loadu_ps(Q + (i + mi) * 2048 + k);
                    acc[mi][0] = _mm_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                    acc[mi][2] = _mm_comp_fmadd_ps(qvec, kv2, acc[mi][2]);
                    acc[mi][3] = _mm_comp_fmadd_ps(qvec, kv3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm_reduce_add_ps(acc[mi][1]) * scale;
                S[(i + mi) * n + j + 2] = _mm_reduce_add_ps(acc[mi][2]) * scale;
                S[(i + mi) * n + j + 3] = _mm_reduce_add_ps(acc[mi][3]) * scale;
            }
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 2048;
            const float* k1 = K + (j + 1) * 2048;

            __m128 acc[2][2];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm_setzero_ps();
                acc[mi][1] = _mm_setzero_ps();
            }

            for (int k = 0; k < 2048; k += 4)
            {
                __m128 kv0 = _mm_loadu_ps(k0 + k);
                __m128 kv1 = _mm_loadu_ps(k1 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m128 qvec = _mm_loadu_ps(Q + (i + mi) * 2048 + k);
                    acc[mi][0] = _mm_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * 2048;

            __m128 acc[2];
            for (int mi = 0; mi < 2; mi++)
                acc[mi] = _mm_setzero_ps();

            for (int k = 0; k < 2048; k += 4)
            {
                __m128 kvec = _mm_loadu_ps(kptr + k);
                for (int mi = 0; mi < 2; mi++)
                {
                    __m128 qvec = _mm_loadu_ps(Q + (i + mi) * 2048 + k);
                    acc[mi] = _mm_comp_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
                S[(i + mi) * n + j] = _mm_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i < m; i++)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* qptr = Q + i * 2048;
            const float* k0 = K + (j + 0) * 2048;
            const float* k1 = K + (j + 1) * 2048;
            const float* k2 = K + (j + 2) * 2048;
            const float* k3 = K + (j + 3) * 2048;

            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();
            __m128 acc2 = _mm_setzero_ps();
            __m128 acc3 = _mm_setzero_ps();

            for (int k = 0; k < 2048; k += 4)
            {
                __m128 qvec = _mm_loadu_ps(qptr + k);
                acc0 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k0 + k), acc0);
                acc1 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k1 + k), acc1);
                acc2 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k2 + k), acc2);
                acc3 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k3 + k), acc3);
            }

            S[i * n + j + 0] = _mm_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm_reduce_add_ps(acc1) * scale;
            S[i * n + j + 2] = _mm_reduce_add_ps(acc2) * scale;
            S[i * n + j + 3] = _mm_reduce_add_ps(acc3) * scale;
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* qptr = Q + i * 2048;
            const float* k0 = K + (j + 0) * 2048;
            const float* k1 = K + (j + 1) * 2048;

            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();

            for (int k = 0; k < 2048; k += 4)
            {
                __m128 qvec = _mm_loadu_ps(qptr + k);
                acc0 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k0 + k), acc0);
                acc1 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k1 + k), acc1);
            }

            S[i * n + j + 0] = _mm_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm_reduce_add_ps(acc1) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * 2048;
            const float* kptr = K + j * 2048;
            __m128 vacc = _mm_setzero_ps();
            for (int k = 0; k < 2048; k += 4)
                vacc = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr + k), _mm_loadu_ps(kptr + k), vacc);
            S[i * n + j] = _mm_reduce_add_ps(vacc) * scale;
        }
    }
}

template<>
void qk_gemm_specialized_sse2<4096>(float* S, const float* Q, const float* K,
        int m, int n, float scale)
{
    int i = 0;
    for (; i + 2 <= m; i += 2)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* k0 = K + (j + 0) * 4096;
            const float* k1 = K + (j + 1) * 4096;
            const float* k2 = K + (j + 2) * 4096;
            const float* k3 = K + (j + 3) * 4096;

            __m128 acc[2][4];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm_setzero_ps();
                acc[mi][1] = _mm_setzero_ps();
                acc[mi][2] = _mm_setzero_ps();
                acc[mi][3] = _mm_setzero_ps();
            }

            for (int k = 0; k < 4096; k += 4)
            {
                __m128 kv0 = _mm_loadu_ps(k0 + k);
                __m128 kv1 = _mm_loadu_ps(k1 + k);
                __m128 kv2 = _mm_loadu_ps(k2 + k);
                __m128 kv3 = _mm_loadu_ps(k3 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m128 qvec = _mm_loadu_ps(Q + (i + mi) * 4096 + k);
                    acc[mi][0] = _mm_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                    acc[mi][2] = _mm_comp_fmadd_ps(qvec, kv2, acc[mi][2]);
                    acc[mi][3] = _mm_comp_fmadd_ps(qvec, kv3, acc[mi][3]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm_reduce_add_ps(acc[mi][1]) * scale;
                S[(i + mi) * n + j + 2] = _mm_reduce_add_ps(acc[mi][2]) * scale;
                S[(i + mi) * n + j + 3] = _mm_reduce_add_ps(acc[mi][3]) * scale;
            }
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* k0 = K + (j + 0) * 4096;
            const float* k1 = K + (j + 1) * 4096;

            __m128 acc[2][2];
            for (int mi = 0; mi < 2; mi++)
            {
                acc[mi][0] = _mm_setzero_ps();
                acc[mi][1] = _mm_setzero_ps();
            }

            for (int k = 0; k < 4096; k += 4)
            {
                __m128 kv0 = _mm_loadu_ps(k0 + k);
                __m128 kv1 = _mm_loadu_ps(k1 + k);

                for (int mi = 0; mi < 2; mi++)
                {
                    __m128 qvec = _mm_loadu_ps(Q + (i + mi) * 4096 + k);
                    acc[mi][0] = _mm_comp_fmadd_ps(qvec, kv0, acc[mi][0]);
                    acc[mi][1] = _mm_comp_fmadd_ps(qvec, kv1, acc[mi][1]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
            {
                S[(i + mi) * n + j + 0] = _mm_reduce_add_ps(acc[mi][0]) * scale;
                S[(i + mi) * n + j + 1] = _mm_reduce_add_ps(acc[mi][1]) * scale;
            }
        }

        for (; j < n; j++)
        {
            const float* kptr = K + j * 4096;

            __m128 acc[2];
            for (int mi = 0; mi < 2; mi++)
                acc[mi] = _mm_setzero_ps();

            for (int k = 0; k < 4096; k += 4)
            {
                __m128 kvec = _mm_loadu_ps(kptr + k);
                for (int mi = 0; mi < 2; mi++)
                {
                    __m128 qvec = _mm_loadu_ps(Q + (i + mi) * 4096 + k);
                    acc[mi] = _mm_comp_fmadd_ps(qvec, kvec, acc[mi]);
                }
            }

            for (int mi = 0; mi < 2; mi++)
                S[(i + mi) * n + j] = _mm_reduce_add_ps(acc[mi]) * scale;
        }
    }

    for (; i < m; i++)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            const float* qptr = Q + i * 4096;
            const float* k0 = K + (j + 0) * 4096;
            const float* k1 = K + (j + 1) * 4096;
            const float* k2 = K + (j + 2) * 4096;
            const float* k3 = K + (j + 3) * 4096;

            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();
            __m128 acc2 = _mm_setzero_ps();
            __m128 acc3 = _mm_setzero_ps();

            for (int k = 0; k < 4096; k += 4)
            {
                __m128 qvec = _mm_loadu_ps(qptr + k);
                acc0 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k0 + k), acc0);
                acc1 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k1 + k), acc1);
                acc2 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k2 + k), acc2);
                acc3 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k3 + k), acc3);
            }

            S[i * n + j + 0] = _mm_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm_reduce_add_ps(acc1) * scale;
            S[i * n + j + 2] = _mm_reduce_add_ps(acc2) * scale;
            S[i * n + j + 3] = _mm_reduce_add_ps(acc3) * scale;
        }

        for (; j + 2 <= n; j += 2)
        {
            const float* qptr = Q + i * 4096;
            const float* k0 = K + (j + 0) * 4096;
            const float* k1 = K + (j + 1) * 4096;

            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();

            for (int k = 0; k < 4096; k += 4)
            {
                __m128 qvec = _mm_loadu_ps(qptr + k);
                acc0 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k0 + k), acc0);
                acc1 = _mm_comp_fmadd_ps(qvec, _mm_loadu_ps(k1 + k), acc1);
            }

            S[i * n + j + 0] = _mm_reduce_add_ps(acc0) * scale;
            S[i * n + j + 1] = _mm_reduce_add_ps(acc1) * scale;
        }

        for (; j < n; j++)
        {
            const float* qptr = Q + i * 4096;
            const float* kptr = K + j * 4096;
            __m128 vacc = _mm_setzero_ps();
            for (int k = 0; k < 4096; k += 4)
                vacc = _mm_comp_fmadd_ps(_mm_loadu_ps(qptr + k), _mm_loadu_ps(kptr + k), vacc);
            S[i * n + j] = _mm_reduce_add_ps(vacc) * scale;
        }
    }
}


template<int M_BLOCK, int D_UNROLL>
static inline void pv_gemm_sse2(float* O, const float* P, const float* V, int m, int n, int d)
{
    const int VEC_PER_UNROLL = D_UNROLL / 4;
    int i = 0;
    for (; i + M_BLOCK <= m; i += M_BLOCK)
    {
        float* op[M_BLOCK];
        const float* pptr[M_BLOCK];
        for (int mi = 0; mi < M_BLOCK; mi++)
        {
            op[mi] = O + (i + mi) * d;
            pptr[mi] = P + (i + mi) * n;
        }

        int dd = 0;
        for (; dd + D_UNROLL - 1 < d; dd += D_UNROLL)
        {
            __m128 acc[M_BLOCK][VEC_PER_UNROLL];
            for (int mi = 0; mi < M_BLOCK; mi++)
                for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                    acc[mi][vi] = _mm_loadu_ps(op[mi] + dd + vi * 4);

            for (int j = 0; j < n; j++)
            {
                __m128 vvec[VEC_PER_UNROLL];
                for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                    vvec[vi] = _mm_loadu_ps(V + j * d + dd + vi * 4);

                for (int mi = 0; mi < M_BLOCK; mi++)
                {
                    __m128 pvec = _mm_set1_ps(pptr[mi][j]);
                    for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                        acc[mi][vi] = _mm_comp_fmadd_ps(pvec, vvec[vi], acc[mi][vi]);
                }
            }

            for (int mi = 0; mi < M_BLOCK; mi++)
                for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                    _mm_storeu_ps(op[mi] + dd + vi * 4, acc[mi][vi]);
        }

        for (; dd + 3 < d; dd += 4)
        {
            __m128 acc[M_BLOCK];
            for (int mi = 0; mi < M_BLOCK; mi++)
                acc[mi] = _mm_loadu_ps(op[mi] + dd);

            for (int j = 0; j < n; j++)
            {
                __m128 vvec = _mm_loadu_ps(V + j * d + dd);
                for (int mi = 0; mi < M_BLOCK; mi++)
                    acc[mi] = _mm_comp_fmadd_ps(_mm_set1_ps(pptr[mi][j]), vvec, acc[mi]);
            }

            for (int mi = 0; mi < M_BLOCK; mi++)
                _mm_storeu_ps(op[mi] + dd, acc[mi]);
        }

        for (; dd < d; dd++)
        {
            for (int mi = 0; mi < M_BLOCK; mi++)
            {
                float acc = op[mi][dd];
                for (int j = 0; j < n; j++)
                    acc += pptr[mi][j] * V[j * d + dd];
                op[mi][dd] = acc;
            }
        }
    }

    for (; i < m; i++)
    {
        float* optr = O + i * d;
        const float* pptr = P + i * n;

        int dd = 0;
        for (; dd + D_UNROLL - 1 < d; dd += D_UNROLL)
        {
            __m128 acc[VEC_PER_UNROLL];
            for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                acc[vi] = _mm_loadu_ps(optr + dd + vi * 4);

            for (int j = 0; j < n; j++)
            {
                __m128 pvec = _mm_set1_ps(pptr[j]);
                for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                    acc[vi] = _mm_comp_fmadd_ps(pvec, _mm_loadu_ps(V + j * d + dd + vi * 4), acc[vi]);
            }

            for (int vi = 0; vi < VEC_PER_UNROLL; vi++)
                _mm_storeu_ps(optr + dd + vi * 4, acc[vi]);
        }

        for (; dd + 3 < d; dd += 4)
        {
            __m128 acc = _mm_loadu_ps(optr + dd);
            for (int j = 0; j < n; j++)
                acc = _mm_comp_fmadd_ps(_mm_set1_ps(pptr[j]), _mm_loadu_ps(V + j * d + dd), acc);
            _mm_storeu_ps(optr + dd, acc);
        }

        for (; dd < d; dd++)
        {
            float acc = optr[dd];
            for (int j = 0; j < n; j++)
                acc += pptr[j] * V[j * d + dd];
            optr[dd] = acc;
        }
    }
}


template<int M_BLOCK, int D>
static inline void pv_gemm_sse2(float* O, const float* P, const float* V, int m, int n)
{
    const int VEC_PER_D = D / 4;
    int i = 0;
    for (; i + M_BLOCK <= m; i += M_BLOCK)
    {
        float* op[M_BLOCK];
        const float* pptr[M_BLOCK];
        for (int mi = 0; mi < M_BLOCK; mi++)
        {
            op[mi] = O + (i + mi) * D;
            pptr[mi] = P + (i + mi) * n;
        }

        for (int mi = 0; mi < M_BLOCK; mi++)
        {
            __m128 acc[VEC_PER_D];
            for (int vi = 0; vi < VEC_PER_D; vi++)
                acc[vi] = _mm_loadu_ps(op[mi] + vi * 4);

            for (int j = 0; j < n; j++)
            {
                __m128 pvec = _mm_set1_ps(pptr[mi][j]);
                for (int vi = 0; vi < VEC_PER_D; vi++)
                    acc[vi] = _mm_comp_fmadd_ps(pvec, _mm_loadu_ps(V + j * D + vi * 4), acc[vi]);
            }

            for (int vi = 0; vi < VEC_PER_D; vi++)
                _mm_storeu_ps(op[mi] + vi * 4, acc[vi]);
        }
    }

    for (; i < m; i++)
    {
        float* optr = O + i * D;
        const float* pptr = P + i * n;

        __m128 acc[VEC_PER_D];
        for (int vi = 0; vi < VEC_PER_D; vi++)
            acc[vi] = _mm_loadu_ps(optr + vi * 4);

        for (int j = 0; j < n; j++)
        {
            __m128 pvec = _mm_set1_ps(pptr[j]);
            for (int vi = 0; vi < VEC_PER_D; vi++)
                acc[vi] = _mm_comp_fmadd_ps(pvec, _mm_loadu_ps(V + j * D + vi * 4), acc[vi]);
        }

        for (int vi = 0; vi < VEC_PER_D; vi++)
            _mm_storeu_ps(optr + vi * 4, acc[vi]);
    }
}


static inline void softmax_tile_sse2(float* P, const float* S,
        float* m_vec, float* l_vec, float* scale_out, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        const float* sptr = S + i * n;
        float* pptr = P + i * n;

        __m128 vmax = _mm_set1_ps(m_vec[i]);
        int j = 0;
        for (; j + 3 < n; j += 4)
            vmax = _mm_max_ps(vmax, _mm_loadu_ps(sptr + j));
        float m_new = _mm_reduce_max_ps(vmax);
        for (; j < n; j++)
            m_new = std::max(m_new, sptr[j]);

        float scale_factor = expf(m_vec[i] - m_new);
        scale_out[i] = scale_factor;
        l_vec[i] *= scale_factor;

        __m128 vm_new = _mm_set1_ps(m_new);
        __m128 vsum = _mm_setzero_ps();
        j = 0;
        for (; j + 3 < n; j += 4)
        {
            __m128 svec = _mm_loadu_ps(sptr + j);
            __m128 evec = exp_ps(_mm_sub_ps(svec, vm_new));
            _mm_storeu_ps(pptr + j, evec);
            vsum = _mm_add_ps(vsum, evec);
        }
        float l_add = _mm_reduce_add_ps(vsum);
        for (; j < n; j++)
        {
            pptr[j] = expf(sptr[j] - m_new);
            l_add += pptr[j];
        }
        l_vec[i] += l_add;
        m_vec[i] = m_new;
    }
}

static inline void vec_scale_sse2(float* x, float s, int n)
{
    __m128 vscale = _mm_set1_ps(s);
    int i = 0;
    for (; i + 3 < n; i += 4)
        _mm_storeu_ps(x + i, _mm_mul_ps(_mm_loadu_ps(x + i), vscale));
    for (; i < n; i++)
        x[i] *= s;
}

static inline void vec_zero_sse2(float* x, int n)
{
    __m128 zero = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < n; i += 4)
        _mm_storeu_ps(x + i, zero);
    for (; i < n; i++)
        x[i] = 0.f;
}

static inline void decode_qk_dot_sse2(float* s, const float* q, const float* K, int n_start, int block_n, int d, float scale)
{
    for (int j = 0; j < block_n; j++)
    {
        __m128 acc = _mm_setzero_ps();
        int k = 0;
        for (; k + 3 < d; k += 4)
            acc = _mm_comp_fmadd_ps(_mm_loadu_ps(q + k), _mm_loadu_ps(K + (n_start + j) * d + k), acc);
        float sum = _mm_reduce_add_ps(acc);
        for (; k < d; k++)
            sum += q[k] * K[(n_start + j) * d + k];
        s[j] = sum * scale;
    }
}

static inline void decode_pv_gemv_sse2(float* out, const float* s, const float* V, int n_start, int block_n, int out_d)
{
    for (int j = 0; j < block_n; j++)
    {
        __m128 pvec = _mm_set1_ps(s[j]);
        int k = 0;
        for (; k + 3 < out_d; k += 4)
        {
            __m128 oval = _mm_loadu_ps(out + k);
            __m128 vval = _mm_loadu_ps(V + (n_start + j) * out_d + k);
            _mm_storeu_ps(out + k, _mm_comp_fmadd_ps(pvec, vval, oval));
        }
        for (; k < out_d; k++)
            out[k] += s[j] * V[(n_start + j) * out_d + k];
    }
}

static void sdpa_decode_sse2(float* out, const float* q,
    const float* K, const float* V, const float* mask,
    int n, int d, int out_d, float scale)
{
    const int BLOCK_N = 128;
    __attribute__((aligned(16))) float s[BLOCK_N];

    vec_zero_sse2(out, out_d);

    float m = -FLT_MAX;
    float l = 0.f;

    for (int n_start = 0; n_start < n; n_start += BLOCK_N)
    {
        int block_n = std::min(BLOCK_N, n - n_start);

        decode_qk_dot_sse2(s, q, K, n_start, block_n, d, scale);

        if (mask)
        {
            int j = 0;
            for (; j + 3 < block_n; j += 4)
                _mm_storeu_ps(s + j, _mm_add_ps(_mm_loadu_ps(s + j), _mm_loadu_ps(mask + n_start + j)));
            for (; j < block_n; j++)
                s[j] += mask[n_start + j];
        }

        __m128 vmax = _mm_set1_ps(-FLT_MAX);
        int j = 0;
        for (; j + 3 < block_n; j += 4)
            vmax = _mm_max_ps(vmax, _mm_loadu_ps(s + j));
        float tile_m = _mm_reduce_max_ps(vmax);
        for (; j < block_n; j++)
            tile_m = std::max(tile_m, s[j]);

        float new_m = std::max(m, tile_m);
        if (m != new_m)
        {
            float scale_factor = expf(m - new_m);
            l *= scale_factor;
            vec_scale_sse2(out, scale_factor, out_d);
        }

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

        decode_pv_gemv_sse2(out, s, V, n_start, block_n, out_d);

        m = new_m;
    }

    float inv_l = 1.f / l;
    vec_scale_sse2(out, inv_l, out_d);
}

#endif // __SSE2__


static inline void qk_gemm_dispatch(float* S, const float* Q, const float* K,
        int m, int n, int d, float scale)
{
    if (d == 128)
    {
#if __AVX512F__
        qk_gemm_specialized_avx512<128>(S, Q, K, m, n, scale);
        return;
#elif __AVX__
        qk_gemm_specialized_avx<128>(S, Q, K, m, n, scale);
        return;
#elif __SSE2__
        qk_gemm_specialized_sse2<128>(S, Q, K, m, n, scale);
        return;
#endif
    }
    if (d == 64)
    {
#if __AVX512F__
        qk_gemm_specialized_avx512<64>(S, Q, K, m, n, scale);
        return;
#elif __AVX__
        qk_gemm_specialized_avx<64>(S, Q, K, m, n, scale);
        return;
#elif __SSE2__
        qk_gemm_specialized_sse2<64>(S, Q, K, m, n, scale);
        return;
#endif
    }
    if (d == 512)
    {
#if __AVX512F__
        qk_gemm_specialized_avx512<512>(S, Q, K, m, n, scale);
        return;
#elif __AVX__
        qk_gemm_specialized_avx<512>(S, Q, K, m, n, scale);
        return;
#elif __SSE2__
        qk_gemm_specialized_sse2<512>(S, Q, K, m, n, scale);
        return;
#endif
    }
    if (d == 256)
    {
#if __AVX512F__
        qk_gemm_specialized_avx512<256>(S, Q, K, m, n, scale);
        return;
#elif __AVX__
        qk_gemm_specialized_avx<256>(S, Q, K, m, n, scale);
        return;
#elif __SSE2__
        qk_gemm_specialized_sse2<256>(S, Q, K, m, n, scale);
        return;
#endif
    }
    if (d == 32)
    {
#if __AVX512F__
        qk_gemm_specialized_avx512<32>(S, Q, K, m, n, scale);
        return;
#elif __AVX__
        qk_gemm_specialized_avx<32>(S, Q, K, m, n, scale);
        return;
#elif __SSE2__
        qk_gemm_specialized_sse2<32>(S, Q, K, m, n, scale);
        return;
#endif
    }
    if (d == 80)
    {
#if __AVX512F__
        qk_gemm_specialized_avx512<80>(S, Q, K, m, n, scale);
        return;
#elif __AVX__
        qk_gemm_specialized_avx<80>(S, Q, K, m, n, scale);
        return;
#elif __SSE2__
        qk_gemm_specialized_sse2<80>(S, Q, K, m, n, scale);
        return;
#endif
    }
    if (d == 96)
    {
#if __AVX512F__
        qk_gemm_specialized_avx512<96>(S, Q, K, m, n, scale);
        return;
#elif __AVX__
        qk_gemm_specialized_avx<96>(S, Q, K, m, n, scale);
        return;
#elif __SSE2__
        qk_gemm_specialized_sse2<96>(S, Q, K, m, n, scale);
        return;
#endif
    }
    if (d == 160)
    {
#if __AVX512F__
        qk_gemm_specialized_avx512<160>(S, Q, K, m, n, scale);
        return;
#elif __AVX__
        qk_gemm_specialized_avx<160>(S, Q, K, m, n, scale);
        return;
#elif __SSE2__
        qk_gemm_specialized_sse2<160>(S, Q, K, m, n, scale);
        return;
#endif
    }
    if (d == 1024)
    {
#if __AVX512F__
        qk_gemm_specialized_avx512<1024>(S, Q, K, m, n, scale);
        return;
#elif __AVX__
        qk_gemm_specialized_avx<1024>(S, Q, K, m, n, scale);
        return;
#elif __SSE2__
        qk_gemm_specialized_sse2<1024>(S, Q, K, m, n, scale);
        return;
#endif
    }
    if (d == 2048)
    {
#if __AVX512F__
        qk_gemm_specialized_avx512<2048>(S, Q, K, m, n, scale);
        return;
#elif __AVX__
        qk_gemm_specialized_avx<2048>(S, Q, K, m, n, scale);
        return;
#elif __SSE2__
        qk_gemm_specialized_sse2<2048>(S, Q, K, m, n, scale);
        return;
#endif
    }
    if (d == 4096)
    {
#if __AVX512F__
        qk_gemm_specialized_avx512<4096>(S, Q, K, m, n, scale);
        return;
#elif __AVX__
        qk_gemm_specialized_avx<4096>(S, Q, K, m, n, scale);
        return;
#elif __SSE2__
        qk_gemm_specialized_sse2<4096>(S, Q, K, m, n, scale);
        return;
#endif
    }

#if __AVX512F__
    qk_gemm_avx512(S, Q, K, m, n, d, scale);
#elif __AVX__
    qk_gemm_avx(S, Q, K, m, n, d, scale);
#elif __SSE2__
    qk_gemm_sse2(S, Q, K, m, n, d, scale);
#else
    qk_gemm_scalar(S, Q, K, m, n, d, scale);
#endif
}

static inline void pv_gemm_dispatch(float* O, const float* P, const float* V,
        int m, int n, int d)
{
    if (d == 128)
    {
#if __AVX512F__
        pv_gemm_avx512<2, 128>(O, P, V, m, n);
        return;
#elif __AVX__
        pv_gemm_avx<2, 32>(O, P, V, m, n, d);
        return;
#elif __SSE2__
        pv_gemm_sse2<2, 16>(O, P, V, m, n, d);
        return;
#endif
    }
    if (d == 64)
    {
#if __AVX512F__
        pv_gemm_avx512<4, 64>(O, P, V, m, n);
        return;
#elif __AVX__
        pv_gemm_avx<2, 32>(O, P, V, m, n, d);
        return;
#elif __SSE2__
        pv_gemm_sse2<2, 16>(O, P, V, m, n, d);
        return;
#endif
    }
    if (d == 256)
    {
#if __AVX512F__
        pv_gemm_avx512<2, 256>(O, P, V, m, n);
        return;
#elif __AVX__
        pv_gemm_avx<2, 32>(O, P, V, m, n, d);
        return;
#elif __SSE2__
        pv_gemm_sse2<2, 16>(O, P, V, m, n, d);
        return;
#endif
    }
    if (d == 1024)
    {
#if __AVX512F__
        pv_gemm_avx512<2, 1024>(O, P, V, m, n);
        return;
#elif __AVX__
        pv_gemm_avx<2, 32>(O, P, V, m, n, d);
        return;
#elif __SSE2__
        pv_gemm_sse2<2, 16>(O, P, V, m, n, d);
        return;
#endif
    }
    if (d == 2048)
    {
#if __AVX512F__
        pv_gemm_avx512<2, 2048>(O, P, V, m, n);
        return;
#elif __AVX__
        pv_gemm_avx<2, 32>(O, P, V, m, n, d);
        return;
#elif __SSE2__
        pv_gemm_sse2<2, 16>(O, P, V, m, n, d);
        return;
#endif
    }
    if (d == 4096)
    {
#if __AVX512F__
        pv_gemm_avx512<2, 4096>(O, P, V, m, n);
        return;
#elif __AVX__
        pv_gemm_avx<2, 32>(O, P, V, m, n, d);
        return;
#elif __SSE2__
        pv_gemm_sse2<2, 16>(O, P, V, m, n, d);
        return;
#endif
    }

#if __AVX512F__
    pv_gemm_avx512<4, 64>(O, P, V, m, n, d);
#elif __AVX__
    pv_gemm_avx<2, 32>(O, P, V, m, n, d);
#elif __SSE2__
    pv_gemm_sse2<2, 16>(O, P, V, m, n, d);
#else
    pv_gemm_scalar(O, P, V, m, n, d);
#endif
}

static inline void vec_scale_dispatch(float* x, float s, int n)
{
#if __AVX512F__
    vec_scale_avx512(x, s, n);
#elif __AVX__
    vec_scale_avx(x, s, n);
#elif __SSE2__
    vec_scale_sse2(x, s, n);
#else
    vec_scale_scalar(x, s, n);
#endif
}

static inline void vec_zero_dispatch(float* x, int n)
{
#if __AVX512F__
    vec_zero_avx512(x, n);
#elif __AVX__
    vec_zero_avx(x, n);
#elif __SSE2__
    vec_zero_sse2(x, n);
#else
    vec_zero_scalar(x, n);
#endif
}

static inline void softmax_tile_dispatch(float* P, const float* S,
        float* m_vec, float* l_vec, float* scale_out, int m, int n)
{
#if __AVX512F__
    softmax_tile_avx512(P, S, m_vec, l_vec, scale_out, m, n);
#elif __AVX__
    softmax_tile_avx(P, S, m_vec, l_vec, scale_out, m, n);
#elif __SSE2__
    softmax_tile_sse2(P, S, m_vec, l_vec, scale_out, m, n);
#else
    softmax_tile_scalar(P, S, m_vec, l_vec, scale_out, m, n);
#endif
}

static inline void sdpa_decode_dispatch(float* out, const float* q,
    const float* K, const float* V, const float* mask,
    int n, int d, int out_d, float scale)
{
#if __AVX512F__
    sdpa_decode_avx512(out, q, K, V, mask, n, d, out_d, scale);
#elif __AVX__
    sdpa_decode_avx(out, q, K, V, mask, n, d, out_d, scale);
#elif __SSE2__
    sdpa_decode_sse2(out, q, K, V, mask, n, d, out_d, scale);
#else
    sdpa_decode_scalar(out, q, K, V, mask, n, d, out_d, scale);
#endif
}

// Timing instrumentation removed

int SDPA_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& _opt) const
{
    Option opt = _opt;
    if (int8_scale_term)
    {
        opt.use_packing_layout = false; // TODO enable packing
    }

    const Mat& query = bottom_blobs[0];
    const Mat& cur_key = bottom_blobs[1];
    const Mat& cur_value = bottom_blobs[2];
    const Mat& attn_mask_blob = attn_mask ? bottom_blobs[3] : Mat();
    const Mat& past_key = kv_cache ? bottom_blobs[attn_mask ? 4 : 3] : Mat();
    const Mat& past_value = kv_cache ? bottom_blobs[attn_mask ? 5 : 4] : Mat();

    const int embed_dim = query.w;
    const int src_seqlen = query.h;
    const int num_heads = query.c;
    const int cur_seqlen = cur_key.h;
    const int num_group = cur_key.c;
    const int out_embed_dim = cur_value.w;
    const int past_seqlen = kv_cache ? past_key.h : 0;
    const int dst_seqlen = past_seqlen + cur_seqlen;

    const size_t elemsize = query.elemsize;

    Mat key;
    if (past_seqlen > 0)
    {
        key.create(embed_dim, dst_seqlen, num_group, elemsize, opt.blob_allocator);
        if (key.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_group; q++)
        {
            const Mat past_key_head = past_key.channel(q);
            const Mat cur_key_head = cur_key.channel(q);
            Mat key_head = key.channel(q);

            memcpy(key_head.row(0), past_key_head, embed_dim * past_seqlen * elemsize);
            memcpy(key_head.row(past_seqlen), cur_key_head, embed_dim * cur_seqlen * elemsize);
        }
    }
    else
    {
        key = cur_key;
    }

    Mat value;
    if (past_seqlen > 0)
    {
        value.create(out_embed_dim, dst_seqlen, num_group, elemsize, opt.blob_allocator);
        if (value.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_group; q++)
        {
            const Mat past_value_head = past_value.channel(q);
            const Mat cur_value_head = cur_value.channel(q);
            Mat value_head = value.channel(q);

            memcpy(value_head.row(0), past_value_head, out_embed_dim * past_seqlen * elemsize);
            memcpy(value_head.row(past_seqlen), cur_value_head, out_embed_dim * cur_seqlen * elemsize);
        }
    }
    else
    {
        value = cur_value;
    }

    const int num_heads_per_group = num_heads / num_group;
    const float _scale = scale == 0.f ? 1.f / sqrtf((float)embed_dim) : scale;

    const int BLOCK_M = 64;
    const int BLOCK_N = 128;

    Mat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if NCNN_INT8
    if (int8_scale_term)
    {
        const int qk_num_blocks = (embed_dim + 31) / 32;
        const int v_num_blocks = (out_embed_dim + 31) / 32;

        Mat key_int8(embed_dim, dst_seqlen, num_group, 1u, opt.blob_allocator);
        Mat key_scales(qk_num_blocks, dst_seqlen, num_group, 4u, opt.blob_allocator);
        Mat value_int8(out_embed_dim, dst_seqlen, num_group, 1u, opt.blob_allocator);
        Mat value_scales(v_num_blocks, dst_seqlen, num_group, 4u, opt.blob_allocator);

        if (key_int8.empty() || key_scales.empty() || value_int8.empty() || value_scales.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < num_group; g++)
        {
            const Mat key_head = key.channel(g);
            Mat key_int8_head = key_int8.channel(g);
            Mat key_scales_head = key_scales.channel(g);
            for (int j = 0; j < dst_seqlen; j++)
            {
                dynamic_quantize_blockwise(key_head.row(j), key_int8_head.row<signed char>(j), key_scales_head.row(j), embed_dim);
            }

            const Mat value_head = value.channel(g);
            Mat value_int8_head = value_int8.channel(g);
            Mat value_scales_head = value_scales.channel(g);
            for (int j = 0; j < dst_seqlen; j++)
            {
                dynamic_quantize_blockwise(value_head.row(j), value_int8_head.row<signed char>(j), value_scales_head.row(j), out_embed_dim);
            }
        }

        Mat o_accum(out_embed_dim, BLOCK_M, opt.num_threads, 4u, opt.workspace_allocator);
        Mat s_vec(BLOCK_N, opt.num_threads, 4u, opt.workspace_allocator);
        Mat q_int8_tile(embed_dim, BLOCK_M, opt.num_threads, 1u, opt.workspace_allocator);
        Mat q_scales_tile(qk_num_blocks, BLOCK_M, opt.num_threads, 4u, opt.workspace_allocator);

        if (o_accum.empty() || s_vec.empty() || q_int8_tile.empty() || q_scales_tile.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < num_heads; q++)
        {
            const Mat query_head = query.channel(q);
            const Mat key_int8_head = key_int8.channel(q / num_heads_per_group);
            const Mat key_scales_head = key_scales.channel(q / num_heads_per_group);
            const Mat value_int8_head = value_int8.channel(q / num_heads_per_group);
            const Mat value_scales_head = value_scales.channel(q / num_heads_per_group);
            Mat top_blob_head = top_blob.channel(q);

            Mat mask_head;
            if (attn_mask)
            {
                const Mat& maskm = attn_mask_blob;
                if (maskm.dims == 3)
                {
                    mask_head = maskm.c > 1 ? maskm.channel(q) : maskm.channel(0);
                }
                else
                {
                    mask_head = maskm;
                }
            }

            Mat o_accum_head = o_accum.channel(get_omp_thread_num());
            float* s_vec_ptr = s_vec.row(get_omp_thread_num());
            Mat q_int8_tile_head = q_int8_tile.channel(get_omp_thread_num());
            Mat q_scales_tile_head = q_scales_tile.channel(get_omp_thread_num());

            for (int m_start = 0; m_start < src_seqlen; m_start += BLOCK_M)
            {
                int m_end = m_start + BLOCK_M < src_seqlen ? m_start + BLOCK_M : src_seqlen;
                int block_m = m_end - m_start;

                for (int i = 0; i < block_m; i++)
                {
                    dynamic_quantize_blockwise(query_head.row(m_start + i), q_int8_tile_head.row<signed char>(i), q_scales_tile_head.row(i), embed_dim);
                }

                for (int i = 0; i < block_m; i++)
                {
                    float* optr = o_accum_head.row(i);
                    for (int k = 0; k < out_embed_dim; k++)
                    {
                        optr[k] = 0.f;
                    }
                }

                float m_vec[BLOCK_M];
                float l_vec[BLOCK_M];
                for (int i = 0; i < block_m; i++)
                {
                    m_vec[i] = -FLT_MAX;
                    l_vec[i] = 0.f;
                }

                for (int n_start = 0; n_start < dst_seqlen; n_start += BLOCK_N)
                {
                    int n_end = n_start + BLOCK_N < dst_seqlen ? n_start + BLOCK_N : dst_seqlen;
                    int block_n = n_end - n_start;

                    for (int i = 0; i < block_m; i++)
                    {
                        const signed char* qptr = q_int8_tile_head.row<const signed char>(i);
                        const float* qscales = q_scales_tile_head.row(i);

                        for (int j = 0; j < block_n; j++)
                        {
                            const signed char* kptr = key_int8_head.row<const signed char>(n_start + j);
                            const float* kscales = key_scales_head.row(n_start + j);

                            float sum = 0.f;
                            for (int b = 0; b < qk_num_blocks; b++)
                            {
                                int k_start = b * 32;
                                int k_end = k_start + 32 < embed_dim ? k_start + 32 : embed_dim;
                                int block_sum = 0;
                                for (int k = k_start; k < k_end; k++)
                                {
                                    block_sum += qptr[k] * kptr[k];
                                }
                                sum += (float)block_sum / (qscales[b] * kscales[b]);
                            }
                            s_vec_ptr[j] = sum * _scale;
                        }

                        if (attn_mask)
                        {
                            const float* mptr = mask_head.row(m_start + i) + n_start;
                            for (int j = 0; j < block_n; j++)
                            {
                                s_vec_ptr[j] += mptr[j];
                            }
                        }

                        float m_new = m_vec[i];
                        for (int j = 0; j < block_n; j++)
                        {
                            m_new = std::max(m_new, s_vec_ptr[j]);
                        }

                        float scale_factor = expf(m_vec[i] - m_new);
                        float l_new = l_vec[i] * scale_factor;

                        float* optr = o_accum_head.row(i);
                        for (int k = 0; k < out_embed_dim; k++)
                        {
                            optr[k] *= scale_factor;
                        }

                        for (int j = 0; j < block_n; j++)
                        {
                            float p = expf(s_vec_ptr[j] - m_new);
                            l_new += p;

                            const signed char* vptr = value_int8_head.row<const signed char>(n_start + j);
                            const float* vscales = value_scales_head.row(n_start + j);
                            for (int vb = 0; vb < v_num_blocks; vb++)
                            {
                                float inv_scale = 1.f / vscales[vb];
                                int k_start = vb * 32;
                                int k_end = k_start + 32 < out_embed_dim ? k_start + 32 : out_embed_dim;
                                for (int k = k_start; k < k_end; k++)
                                {
                                    optr[k] += p * (float)vptr[k] * inv_scale;
                                }
                            }
                        }

                        m_vec[i] = m_new;
                        l_vec[i] = l_new;
                    }
                }

                for (int i = 0; i < block_m; i++)
                {
                    float* optr = o_accum_head.row(i);
                    float* outptr = top_blob_head.row(m_start + i);
                    float inv_l = 1.f / l_vec[i];
                    for (int k = 0; k < out_embed_dim; k++)
                    {
                        outptr[k] = optr[k] * inv_l;
                    }
                }
            }
        }

        if (kv_cache)
        {
            top_blobs[1] = key;
            top_blobs[2] = value;
        }

        return 0;
    }
#endif // NCNN_INT8

    // FP32 optimized path using tiled GEMM + online softmax
    if (src_seqlen == 1)
    {
        // Decode path: fused GEMV kernel for single-query attention
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < num_group; g++)
        {
            const Mat key_head = key.channel(g);
            const Mat value_head = value.channel(g);

            for (int hq = 0; hq < num_heads_per_group; hq++)
            {
                int q = g * num_heads_per_group + hq;
                const Mat query_head = query.channel(q);
                Mat top_blob_head = top_blob.channel(q);

                const float* qptr = query_head.row(0);
                float* outptr = top_blob_head.row(0);
                const float* Kptr = key_head;
                const float* Vptr = value_head;

                const float* mask_ptr = nullptr;
                if (attn_mask)
                {
                    const Mat& maskm = attn_mask_blob;
                    Mat mask_head;
                    if (maskm.dims == 3)
                    {
                        mask_head = maskm.c > 1 ? maskm.channel(q) : maskm.channel(0);
                    }
                    else
                    {
                        mask_head = maskm;
                    }
                    mask_ptr = mask_head.row(0);
                }

                sdpa_decode_dispatch(outptr, qptr, Kptr, Vptr, mask_ptr, dst_seqlen, embed_dim, out_embed_dim, _scale);
            }
        }

        if (kv_cache)
        {
            top_blobs[1] = key;
            top_blobs[2] = value;
        }

        return 0;
    }

    Mat s_vec(BLOCK_N * BLOCK_M * num_heads_per_group, opt.num_threads, 4u, opt.workspace_allocator);
    Mat o_accum(out_embed_dim, BLOCK_M * num_heads_per_group, opt.num_threads, 4u, opt.workspace_allocator);
    const bool large_dim = embed_dim > 512;
    Mat q_batch(embed_dim, large_dim ? BLOCK_M : BLOCK_M * num_heads_per_group, opt.num_threads, 4u, opt.workspace_allocator);

    if (s_vec.empty() || o_accum.empty() || q_batch.empty())
        return -100;

    int num_m_tiles = (src_seqlen + BLOCK_M - 1) / BLOCK_M;

    // Per-head per-M-tile softmax state for cross-N-tile accumulation
    Mat m_state(BLOCK_M, num_heads_per_group, num_group * num_m_tiles, 4u, opt.workspace_allocator);
    Mat l_state(BLOCK_M, num_heads_per_group, num_group * num_m_tiles, 4u, opt.workspace_allocator);

    if (m_state.empty() || l_state.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int idx = 0; idx < num_group * num_m_tiles; idx++)
    {
        int g = idx / num_m_tiles;
        int m_tile = idx % num_m_tiles;
        int m_start = m_tile * BLOCK_M;
        int block_m = m_start + BLOCK_M < src_seqlen ? BLOCK_M : src_seqlen - m_start;

        const Mat key_head = key.channel(g);
        const Mat value_head = value.channel(g);

        Mat s_vec_thread = s_vec.channel(get_omp_thread_num());
        Mat o_accum_thread = o_accum.channel(get_omp_thread_num());
        Mat q_batch_thread = q_batch.channel(get_omp_thread_num());

        // Pre-resolve mask pointers for all heads in this group
        const float* mask_data[num_heads_per_group];
        int mask_stride[num_heads_per_group];
        for (int hq = 0; hq < num_heads_per_group; hq++)
        {
            int q = g * num_heads_per_group + hq;
            mask_data[hq] = nullptr;
            mask_stride[hq] = 0;
            if (attn_mask)
            {
                const Mat& maskm = attn_mask_blob;
                Mat mh = (maskm.dims == 3 && maskm.c > 1) ? maskm.channel(q)
                         : (maskm.dims == 3 ? maskm.channel(0) : maskm);
                mask_data[hq] = mh;
                mask_stride[hq] = mh.w;
            }
        }

        Mat m_state_tile = m_state.channel(idx);
        Mat l_state_tile = l_state.channel(idx);

        if (!large_dim)
        {
            for (int hq = 0; hq < num_heads_per_group; hq++)
            {
                int q = g * num_heads_per_group + hq;
                const Mat query_head = query.channel(q);
                float* q_dst = q_batch_thread.row(hq * block_m);
                for (int i = 0; i < block_m; i++)
                {
                    memcpy(q_dst + i * embed_dim, query_head.row(m_start + i), embed_dim * sizeof(float));
                }
            }
        }

        // Initialize softmax state and zero output accumulator for all Q heads in this group
        for (int hq = 0; hq < num_heads_per_group; hq++)
        {
            float* m_vec = m_state_tile.row(hq);
            float* l_vec = l_state_tile.row(hq);
            for (int i = 0; i < block_m; i++)
            {
                m_vec[i] = -FLT_MAX;
                l_vec[i] = 0.f;
            }

            float* o_ptr = o_accum_thread.row(hq * block_m);
            vec_zero_dispatch(o_ptr, out_embed_dim * block_m);
        }

        // N-outer loop: K/V N-tile is loaded once and reused by all Q heads in this group
        for (int n_start = 0; n_start < dst_seqlen; n_start += BLOCK_N)
        {
            int n_end = n_start + BLOCK_N < dst_seqlen ? n_start + BLOCK_N : dst_seqlen;
            int block_n = n_end - n_start;

            float* s_ptr = s_vec_thread.row(0);

            if (!large_dim)
            {
                qk_gemm_dispatch(s_ptr,
                                 q_batch_thread.row(0),
                                 key_head.row(n_start),
                                 block_m * num_heads_per_group, block_n, embed_dim, _scale);

                for (int hq = 0; hq < num_heads_per_group; hq++)
                {
                    float* s_head = s_ptr + hq * block_m * block_n;

                    if (attn_mask && mask_data[hq])
                    {
                        for (int i = 0; i < block_m; i++)
                        {
                            const float* mptr = mask_data[hq] + (m_start + i) * mask_stride[hq] + n_start;
                            float* sptr = s_head + i * block_n;
                            int j = 0;
#if __AVX512F__
                            for (; j + 15 < block_n; j += 16)
                            {
                                _mm512_storeu_ps(sptr + j, _mm512_add_ps(_mm512_loadu_ps(sptr + j), _mm512_loadu_ps(mptr + j)));
                            }
                            if (j < block_n)
                            {
                                __mmask16 mask = (__mmask16)((1u << (block_n - j)) - 1);
                                __m512 _s = _mm512_maskz_loadu_ps(mask, sptr + j);
                                __m512 _m = _mm512_maskz_loadu_ps(mask, mptr + j);
                                _mm512_mask_storeu_ps(sptr + j, mask, _mm512_add_ps(_s, _m));
                                j = block_n;
                            }
#elif __AVX__
                            for (; j + 7 < block_n; j += 8)
                            {
                                _mm256_storeu_ps(sptr + j, _mm256_add_ps(_mm256_loadu_ps(sptr + j), _mm256_loadu_ps(mptr + j)));
                            }
#elif __SSE2__
                            for (; j + 3 < block_n; j += 4)
                            {
                                _mm_storeu_ps(sptr + j, _mm_add_ps(_mm_loadu_ps(sptr + j), _mm_loadu_ps(mptr + j)));
                            }
#endif
                            for (; j < block_n; j++)
                            {
                                sptr[j] += mptr[j];
                            }
                        }
                    }

                    float* m_vec = m_state_tile.row(hq);
                    float* l_vec = l_state_tile.row(hq);
                    float* o_ptr = o_accum_thread.row(hq * block_m);

                    float m_old[BLOCK_M];
                    float scale_factors[BLOCK_M];
                    for (int i = 0; i < block_m; i++)
                    {
                        m_old[i] = m_vec[i];
                    }
                    softmax_tile_dispatch(s_head, s_head, m_vec, l_vec, scale_factors, block_m, block_n);

                    for (int i = 0; i < block_m; i++)
                    {
                        if (m_old[i] != m_vec[i])
                        {
                            vec_scale_dispatch(o_ptr + i * out_embed_dim, scale_factors[i], out_embed_dim);
                        }
                    }
                }

                pv_gemm_dispatch(o_accum_thread.row(0), s_ptr, value_head.row(n_start),
                                 block_m * num_heads_per_group, block_n, out_embed_dim);
            }
            else
            {
                for (int hq = 0; hq < num_heads_per_group; hq++)
                {
                    int q = g * num_heads_per_group + hq;
                    const Mat query_head = query.channel(q);

                    float* q_dst = q_batch_thread.row(0);
                    for (int i = 0; i < block_m; i++)
                    {
                        memcpy(q_dst + i * embed_dim, query_head.row(m_start + i), embed_dim * sizeof(float));
                    }

                    float* s_head = s_ptr;

                    qk_gemm_dispatch(s_head,
                                     q_dst,
                                     key_head.row(n_start),
                                     block_m, block_n, embed_dim, _scale);

                    if (attn_mask && mask_data[hq])
                    {
                        for (int i = 0; i < block_m; i++)
                        {
                            const float* mptr = mask_data[hq] + (m_start + i) * mask_stride[hq] + n_start;
                            float* sptr = s_head + i * block_n;
                            int j = 0;
#if __AVX512F__
                            for (; j + 15 < block_n; j += 16)
                            {
                                _mm512_storeu_ps(sptr + j, _mm512_add_ps(_mm512_loadu_ps(sptr + j), _mm512_loadu_ps(mptr + j)));
                            }
                            if (j < block_n)
                            {
                                __mmask16 mask = (__mmask16)((1u << (block_n - j)) - 1);
                                __m512 _s = _mm512_maskz_loadu_ps(mask, sptr + j);
                                __m512 _m = _mm512_maskz_loadu_ps(mask, mptr + j);
                                _mm512_mask_storeu_ps(sptr + j, mask, _mm512_add_ps(_s, _m));
                                j = block_n;
                            }
#elif __AVX__
                            for (; j + 7 < block_n; j += 8)
                            {
                                _mm256_storeu_ps(sptr + j, _mm256_add_ps(_mm256_loadu_ps(sptr + j), _mm256_loadu_ps(mptr + j)));
                            }
#elif __SSE2__
                            for (; j + 3 < block_n; j += 4)
                            {
                                _mm_storeu_ps(sptr + j, _mm_add_ps(_mm_loadu_ps(sptr + j), _mm_loadu_ps(mptr + j)));
                            }
#endif
                            for (; j < block_n; j++)
                            {
                                sptr[j] += mptr[j];
                            }
                        }
                    }

                    float* m_vec = m_state_tile.row(hq);
                    float* l_vec = l_state_tile.row(hq);
                    float* o_ptr = o_accum_thread.row(hq * block_m);

                    float m_old[BLOCK_M];
                    float scale_factors[BLOCK_M];
                    for (int i = 0; i < block_m; i++)
                    {
                        m_old[i] = m_vec[i];
                    }
                    softmax_tile_dispatch(s_head, s_head, m_vec, l_vec, scale_factors, block_m, block_n);

                    for (int i = 0; i < block_m; i++)
                    {
                        if (m_old[i] != m_vec[i])
                        {
                            vec_scale_dispatch(o_ptr + i * out_embed_dim, scale_factors[i], out_embed_dim);
                        }
                    }

                    pv_gemm_dispatch(o_ptr, s_head, value_head.row(n_start),
                                     block_m, block_n, out_embed_dim);
                }
            }
        }

        // Normalize all Q heads for this M tile and write back to top_blob
        for (int hq = 0; hq < num_heads_per_group; hq++)
        {
            int q = g * num_heads_per_group + hq;
            Mat top_blob_head = top_blob.channel(q);
            float* l_vec = l_state_tile.row(hq);
            float* o_ptr = o_accum_thread.row(hq * block_m);

            for (int i = 0; i < block_m; i++)
            {
                float* outptr = top_blob_head.row(m_start + i);
                float inv_l = 1.f / l_vec[i];
                int k = 0;
#if __AVX512F__
                __m512 vinv_l = _mm512_set1_ps(inv_l);
                for (; k + 15 < out_embed_dim; k += 16)
                {
                    _mm512_storeu_ps(outptr + k, _mm512_mul_ps(_mm512_loadu_ps(o_ptr + i * out_embed_dim + k), vinv_l));
                }
                if (k < out_embed_dim)
                {
                    __mmask16 mask = (__mmask16)((1u << (out_embed_dim - k)) - 1);
                    _mm512_mask_storeu_ps(outptr + k, mask, _mm512_mul_ps(_mm512_maskz_loadu_ps(mask, o_ptr + i * out_embed_dim + k), vinv_l));
                    k = out_embed_dim;
                }
#elif __AVX__
                __m256 vinv_l = _mm256_set1_ps(inv_l);
                for (; k + 7 < out_embed_dim; k += 8)
                {
                    _mm256_storeu_ps(outptr + k, _mm256_mul_ps(_mm256_loadu_ps(o_ptr + i * out_embed_dim + k), vinv_l));
                }
#elif __SSE2__
                __m128 vinv_l = _mm_set1_ps(inv_l);
                for (; k + 3 < out_embed_dim; k += 4)
                {
                    _mm_storeu_ps(outptr + k, _mm_mul_ps(_mm_loadu_ps(o_ptr + i * out_embed_dim + k), vinv_l));
                }
#endif
                for (; k < out_embed_dim; k++)
                {
                    outptr[k] = o_ptr[i * out_embed_dim + k] * inv_l;
                }
            }
        }
    }

    if (kv_cache)
    {
        top_blobs[1] = key;
        top_blobs[2] = value;
    }

    return 0;
}

} // namespace ncnn
