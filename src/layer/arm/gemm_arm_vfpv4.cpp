// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gemm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

#include "arm_usability.h"

#include "cpu.h"

namespace ncnn {

#include "gemm_fp16s.h"

#if NCNN_INT8
#include "gemm_int8_fp16s.h"
#endif

#if NCNN_WEIGHT_QUANT
#include "gemm_wq_int8.h"
#include "gemm_wq_int8_fp16s.h"
#endif

void pack_A_tile_fp32_to_fp16_vfpv4(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    pack_A_tile_fp32_to_fp16(A, AT, i, max_ii, k, max_kk);
}

void transpose_pack_A_tile_fp32_to_fp16_vfpv4(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    transpose_pack_A_tile_fp32_to_fp16(A, AT, i, max_ii, k, max_kk);
}

void pack_B_tile_fp32_to_fp16_vfpv4(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    pack_B_tile_fp32_to_fp16(B, BT, j, max_jj, k, max_kk);
}

void transpose_pack_B_tile_fp32_to_fp16_vfpv4(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    transpose_pack_B_tile_fp32_to_fp16(B, BT, j, max_jj, k, max_kk);
}

void transpose_unpack_output_tile_fp32_to_fp16_vfpv4(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    transpose_unpack_output_tile_fp32_to_fp16(topT, top_blob, i, max_ii, j, max_jj);
}

void gemm_transB_packed_tile_fp16s_vfpv4(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, float alpha, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end, int output_elemtype)
{
    gemm_transB_packed_tile_fp16s(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, broadcast_type_C, alpha, i, max_ii, j, max_jj, k, max_kk, k_end, output_elemtype);
}

#if NCNN_INT8
void compute_A_tile_fp16_int8_scales_vfpv4(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    compute_A_tile_fp16_int8_scales(A, scales, B_scale, out_descales, i, max_ii);
}

void transpose_compute_A_tile_fp16_int8_scales_vfpv4(const Mat& A, Mat& scales, float B_scale, Mat& out_descales, int i, int max_ii)
{
    transpose_compute_A_tile_fp16_int8_scales(A, scales, B_scale, out_descales, i, max_ii);
}

void pack_A_tile_fp16_to_int8_vfpv4(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    pack_A_tile_fp16_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

void transpose_pack_A_tile_fp16_to_int8_vfpv4(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    transpose_pack_A_tile_fp16_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

void compute_B_fp16_int8_scale_vfpv4(const Mat& B, float& scale)
{
    compute_B_fp16_int8_scale(B, scale);
}

void pack_B_tile_fp16_to_int8_vfpv4(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    pack_B_tile_fp16_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

void transpose_pack_B_tile_fp16_to_int8_vfpv4(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    transpose_pack_B_tile_fp16_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

void unpack_output_tile_int32_to_fp16_vfpv4(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta)
{
    unpack_output_tile_int32_to_fp16(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta);
}

void transpose_unpack_output_tile_int32_to_fp16_vfpv4(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta)
{
    transpose_unpack_output_tile_int32_to_fp16(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta);
}
#endif // NCNN_INT8

#if NCNN_WEIGHT_QUANT
void quantize_A_tile_wq_int8_fp16s_vfpv4(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    quantize_A_tile_wq_int8_fp16s(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
}

void transpose_quantize_A_tile_wq_int8_fp16s_vfpv4(const Mat& A, Mat& AT_tile, Mat& AT_descales_tile, int i, int max_ii, int k, int max_kk, int block_size, const Mat& input_scales)
{
    transpose_quantize_A_tile_wq_int8_fp16s(A, AT_tile, AT_descales_tile, i, max_ii, k, max_kk, block_size, input_scales);
}

void unpack_output_tile_wq_int8_vfpv4(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, float alpha, float beta, int output_elemtype, int output_transpose)
{
    unpack_output_tile_wq_int8(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, alpha, beta, output_elemtype, output_transpose);
}
#endif // NCNN_WEIGHT_QUANT

} // namespace ncnn
