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

#if __AVX2__

void dynamic_quantize_blockwise_avx2(const float* src, signed char* dst, float* scales, int width)
{
    dynamic_quantize_blockwise_avx2_kernel(src, dst, scales, width);
}

void dynamic_quantize_rowwise_avx2(const float* src, signed char* dst, float* scale, int width)
{
    dynamic_quantize_rowwise_avx2_kernel(src, dst, scale, width);
}

int qk_int8_dot_block_avx2(const signed char* a, const signed char* b, int len)
{
    return qk_int8_dot_block_avx2_kernel(a, b, len);
}

void decode_qk_dot_int8_avx2(float* s, const signed char* q, const signed char* K, const float* qscales, const float* kscales, int n_start, int block_n, int d, float scale)
{
    decode_qk_dot_int8_avx2_kernel(s, q, K, qscales, kscales, n_start, block_n, d, scale);
}

void qk_int8_gemm_row_avx2(float* s_row, const signed char* q_row, const signed char* K, float qscale, const float* kscales, int n, int d, float scale)
{
    qk_int8_gemm_row_avx2_kernel(s_row, q_row, K, qscale, kscales, n, d, scale);
}

void qk_int8_gemm_tiled_avx2(float* S, const signed char* Q, const signed char* K, const float* qscales, const float* kscales, int m, int n, int d, float scale)
{
    qk_int8_gemm_tiled_avx2_kernel(S, Q, K, qscales, kscales, m, n, d, scale);
}

void decode_pv_gemv_int8_avx2(float* out, const float* s, const signed char* V, const float* vscales, int n_start, int block_n, int out_d)
{
    decode_pv_gemv_int8_avx2_kernel(out, s, V, vscales, n_start, block_n, out_d);
}

void pv_float_int8_gemm_row_avx2(float* out, const float* p_row, const signed char* V, const float* vscales, int n, int out_d)
{
    pv_float_int8_gemm_row_avx2_kernel(out, p_row, V, vscales, n, out_d);
}

void pv_float_int8_fma_block_avx2(float* out, float p_invscale, const signed char* v, int len)
{
    pv_float_int8_fma_block_avx2_kernel(out, p_invscale, v, len);
}

void pv_float_int8_gemm_tile_avx2(float* O, const float* P, const signed char* V, const float* vscales, int block_m, int block_n, int out_embed_dim)
{
    pv_float_int8_gemm_tile_avx2_kernel(O, P, V, vscales, block_m, block_n, out_embed_dim);
}

#endif // __AVX2__

} // namespace ncnn
