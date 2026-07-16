// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <arm_sve.h>

#include "cpu.h"
#include "mat.h"
#include "arm_usability.h"

namespace ncnn {

#if NCNN_WEIGHT_QUANT
#include "gemm_wq_int8.h"

int pack_B_wq_int8_svei8mm(const Mat& B, const Mat& B_scales, Mat& BT, Mat& BT_descales, int N, int K, int block_size, int num_threads)
{
    return pack_B_wq_int8(B, B_scales, BT, BT_descales, N, K, block_size, num_threads);
}

void gemm_transB_packed_tile_wq_int8_svei8mm(const Mat& AT_tile, const Mat& AT_descales_tile, const Mat& BT_tile, const Mat& BT_descales_tile, Mat& topT_tile, int max_ii, int max_jj, int K, int block_size)
{
    gemm_transB_packed_tile_wq_int8(AT_tile, AT_descales_tile, BT_tile, BT_descales_tile, topT_tile, max_ii, max_jj, K, block_size);
}
#endif // NCNN_WEIGHT_QUANT

} // namespace ncnn
