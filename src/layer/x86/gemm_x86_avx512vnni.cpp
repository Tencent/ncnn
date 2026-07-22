// Copyright 2024 Tencent
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

#include "gemm_int8.h"

void pack_A_tile_int8_avx512vnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
}

void transpose_pack_A_tile_int8_avx512vnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    transpose_pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
}

void pack_B_tile_int8_avx512vnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    pack_B_tile_int8(B, BT, j, max_jj, k, max_kk);
}

void transpose_pack_B_tile_int8_avx512vnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    transpose_pack_B_tile_int8(B, BT, j, max_jj, k, max_kk);
}

void pack_A_tile_fp32_to_int8_avx512vnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    pack_A_tile_fp32_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

void transpose_pack_A_tile_fp32_to_int8_avx512vnni(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    transpose_pack_A_tile_fp32_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

void pack_B_tile_fp32_to_int8_avx512vnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    pack_B_tile_fp32_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

void transpose_pack_B_tile_fp32_to_int8_avx512vnni(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    transpose_pack_B_tile_fp32_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

void gemm_transB_packed_tile_int8_avx512vnni(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
}

#if NCNN_WEIGHT_QUANT
#include "gemm_wq_int8.h"

void pack_B_tile_wq_int8_avx512vnni(const Mat& B, const Mat& B_scales, Mat& BT_tile, Mat& BT_descales_tile, int j, int max_jj, int K, int block_size)
{
    pack_B_tile_wq_int8(B, B_scales, BT_tile, BT_descales_tile, j, max_jj, K, block_size);
}

void quantize_A_tile_wq_int8_avx512vnni(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    quantize_A_tile_wq_int8(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
}

void transpose_quantize_A_tile_wq_int8_avx512vnni(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    transpose_quantize_A_tile_wq_int8(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
}

void gemm_transB_packed_tile_wq_int8_avx512vnni(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk, int K, int block_size)
{
    gemm_transB_packed_tile_wq_int8(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, k, max_kk, K, block_size);
}

void unpack_output_tile_wq_int8_avx512vnni(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    unpack_output_tile_wq_int8(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta);
}

void transpose_unpack_output_tile_wq_int8_avx512vnni(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta)
{
    transpose_unpack_output_tile_wq_int8(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta);
}
#endif // NCNN_WEIGHT_QUANT

} // namespace ncnn
