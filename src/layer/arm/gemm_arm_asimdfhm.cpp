// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_usability.h"

#include "cpu.h"

namespace ncnn {

#include "gemm_fp16s.h"

void gemm_transB_packed_tile_fp16s_asimdfhm(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, float alpha, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    gemm_transB_packed_tile_fp16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, alpha, i, max_ii, j, max_jj, k, max_kk, k_end);
}

} // namespace ncnn
