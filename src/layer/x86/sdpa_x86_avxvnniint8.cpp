// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

#include "sdpa_x86_int8.h"

#if __AVXVNNIINT8__

void decode_qk_dot_int8_avxvnniint8(float* s, const signed char* q, const signed char* K, const float* qscales, const float* kscales, int n_start, int block_n, int d, float scale)
{
    decode_qk_dot_int8_avxvnni_kernel(s, q, K, qscales, kscales, n_start, block_n, d, scale);
}

void qk_int8_gemm_row_avxvnniint8(float* s_row, const signed char* q_row, const signed char* K, float qscale, const float* kscales, int n, int d, float scale)
{
    qk_int8_gemm_row_avx2_kernel(s_row, q_row, K, qscale, kscales, n, d, scale);
}

void qk_int8_gemm_tiled_avxvnniint8(float* S, const signed char* Q, const signed char* K, const float* qscales, const float* kscales, int m, int n, int d, float scale)
{
    qk_int8_gemm_tiled_avx2_kernel(S, Q, K, qscales, kscales, m, n, d, scale);
}

#endif // __AVXVNNIINT8__

} // namespace ncnn
