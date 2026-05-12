// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "arm_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "gemm_bf16s.h"

void pack_A_tile_bf16_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    pack_A_tile_bf16(A, AT, i, max_ii, k, max_kk);
}

void transpose_pack_A_tile_bf16_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    transpose_pack_A_tile_bf16(A, AT, i, max_ii, k, max_kk);
}

void pack_B_tile_bf16_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    pack_B_tile_bf16(B, BT, j, max_jj, k, max_kk);
}

void transpose_pack_B_tile_bf16_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    transpose_pack_B_tile_bf16(B, BT, j, max_jj, k, max_kk);
}

void pack_A_tile_fp32_to_bf16_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    pack_A_tile_fp32_to_bf16(A, AT, i, max_ii, k, max_kk);
}

void transpose_pack_A_tile_fp32_to_bf16_bf16(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    transpose_pack_A_tile_fp32_to_bf16(A, AT, i, max_ii, k, max_kk);
}

void pack_B_tile_fp32_to_bf16_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    pack_B_tile_fp32_to_bf16(B, BT, j, max_jj, k, max_kk);
}

void transpose_pack_B_tile_fp32_to_bf16_bf16(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    transpose_pack_B_tile_fp32_to_bf16(B, BT, j, max_jj, k, max_kk);
}

void unpack_output_tile_fp32_to_bf16_bf16(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_transpose)
{
    unpack_output_tile_fp32_to_bf16(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, output_transpose);
}

void gemm_transB_packed_tile_bf16s_bf16(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int max_ii, int max_jj, int k, int max_kk)
{
    gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, topT_tile, max_ii, max_jj, k, max_kk);
}
#endif // NCNN_BF16

} // namespace ncnn
