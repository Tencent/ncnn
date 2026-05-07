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

} // namespace ncnn
