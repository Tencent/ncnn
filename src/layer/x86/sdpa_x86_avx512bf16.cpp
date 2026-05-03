// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

#include "sdpa_x86_bf16s.h"

#if __AVX512BF16__

void decode_qk_dot_bf16s_avx512bf16(float* s, const float* q, const unsigned short* K, int n_start, int block_n, int d, float scale)
{
    decode_qk_dot_bf16s_avx512_kernel(s, q, K, n_start, block_n, d, scale);
}

void decode_pv_gemv_bf16s_avx512bf16(float* out, const float* s, const unsigned short* V, int n_start, int block_n, int out_d)
{
    decode_pv_gemv_bf16s_avx512_kernel(out, s, V, n_start, block_n, out_d);
}

void qk_gemm_bf16s_avx512bf16(float* S, const float* Q, const unsigned short* K, int m, int n, int d, float scale)
{
    qk_gemm_bf16s_avx512(S, Q, K, m, n, d, scale);
}

void pv_gemm_bf16s_avx512bf16(float* O, const float* P, const unsigned short* V, int m, int n, int d)
{
    pv_gemm_bf16s_avx512(O, P, V, m, n, d);
}

#endif // __AVX512BF16__

} // namespace ncnn
