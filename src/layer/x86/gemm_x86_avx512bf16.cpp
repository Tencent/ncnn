// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
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
#include "x86_usability.h"

namespace ncnn {

#include "gemm_bf16s.h"

void pack_A_tile_bf16_avx512bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    pack_A_tile_bf16(A, AT, i, max_ii, k, max_kk);
}

void transpose_pack_A_tile_bf16_avx512bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    transpose_pack_A_tile_bf16(A, AT, i, max_ii, k, max_kk);
}

void pack_B_tile_bf16_avx512bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    pack_B_tile_bf16(B, BT, j, max_jj, k, max_kk);
}

void transpose_pack_B_tile_bf16_avx512bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    transpose_pack_B_tile_bf16(B, BT, j, max_jj, k, max_kk);
}

void gemm_transB_packed_tile_bf16s_avx512bf16(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
}

#if NCNN_WEIGHT_QUANT
#if NCNN_BF16
#include "gemm_wq_int8_bf16s.h"

void quantize_A_tile_wq_int8_bf16s_avx512bf16(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    quantize_A_tile_wq_int8_bf16s(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
}

void transpose_quantize_A_tile_wq_int8_bf16s_avx512bf16(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    transpose_quantize_A_tile_wq_int8_bf16s(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
}

void unpack_output_tile_wq_int8_bf16s_avx512bf16(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_transpose)
{
    unpack_output_tile_wq_int8_bf16s(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, output_transpose);
}
#endif // NCNN_BF16
#endif // NCNN_WEIGHT_QUANT

} // namespace ncnn
