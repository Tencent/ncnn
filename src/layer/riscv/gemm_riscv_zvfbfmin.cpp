// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16 && __riscv_zvfbfmin
#include "gemm_bf16s.h"

void pack_A_tile_fp32_to_bf16_zvfbfmin(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    pack_A_tile_fp32_to_bf16(A, AT, i, max_ii, k, max_kk);
}

void transpose_pack_A_tile_fp32_to_bf16_zvfbfmin(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    transpose_pack_A_tile_fp32_to_bf16(A, AT, i, max_ii, k, max_kk);
}

void pack_B_tile_fp32_to_bf16_zvfbfmin(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    pack_B_tile_fp32_to_bf16(B, BT, j, max_jj, k, max_kk);
}

void transpose_pack_B_tile_fp32_to_bf16_zvfbfmin(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    transpose_pack_B_tile_fp32_to_bf16(B, BT, j, max_jj, k, max_kk);
}

void gemm_transB_packed_tile_bf16s_zvfbfmin(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, float alpha, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
}

void unpack_output_tile_bf16s_zvfbfmin(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj, int output_transpose)
{
    unpack_output_tile_bf16s(topT, top_blob, i, max_ii, j, max_jj, output_transpose);
}

#endif // NCNN_BF16 && __riscv_zvfbfmin

} // namespace ncnn
