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
#include "gemm_fp16sa.h"

#if NCNN_INT8
#include "gemm_int8_fp16s.h"
#endif

void gemm_transB_packed_tile_fp16sa_asimdhp(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    gemm_transB_packed_tile_fp16sa(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, i, max_ii, j, max_jj, k, max_kk, k_end);
}

#if NCNN_INT8
void compute_A_tile_fp16_int8_scales_asimdhp(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    compute_A_tile_fp16_int8_scales(A, scales, B_scale, out_descales, i, max_ii);
}

void transpose_compute_A_tile_fp16_int8_scales_asimdhp(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    transpose_compute_A_tile_fp16_int8_scales(A, scales, B_scale, out_descales, i, max_ii);
}

void compute_B_fp16_int8_scale_asimdhp(const Mat& B, float& scale)
{
    compute_B_fp16_int8_scale(B, scale);
}

#endif // NCNN_INT8

} // namespace ncnn
