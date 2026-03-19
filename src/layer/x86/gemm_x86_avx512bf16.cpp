// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
#include "x86_usability.h"

namespace ncnn {

#include "gemm_bf16.h"

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

void gemm_transB_packed_tile_bf16_avx512bf16(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    gemm_transB_packed_tile_bf16(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
}

void unpack_output_tile_fp32_to_bf16_avx512bf16(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, int output_transpose)
{
    unpack_output_tile_fp32_to_bf16(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, output_transpose);
}

void get_optimal_tile_mnk_bf16_avx512bf16(int M, int N, int K, int constant_TILE_M, int constant_TILE_N, int constant_TILE_K, int& TILE_M, int& TILE_N, int& TILE_K, int nT)
{
    get_optimal_tile_mnk_bf16(M, N, K, constant_TILE_M, constant_TILE_N, constant_TILE_K, TILE_M, TILE_N, TILE_K, nT);
}

} // namespace ncnn
