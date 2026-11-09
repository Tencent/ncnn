// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_BF16 && __riscv_zvfbfwma
#include "gemm_bf16s.h"

void gemm_transB_packed_tile_bf16s_zvfbfwma(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, float alpha, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    gemm_transB_packed_tile_bf16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
}
#endif // NCNN_BF16 && __riscv_zvfbfwma

} // namespace ncnn
