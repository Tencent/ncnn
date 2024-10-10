// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "cpu.h"
#include "mat.h"
#include "arm_usability.h"

namespace ncnn {

#include "gemm_int8.h"
#include "gemm_int8_fp16s.h"

#if NCNN_BF16
#include "gemm_int8_bf16s.h"
#endif

void pack_A_tile_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
}

void transpose_pack_A_tile_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    transpose_pack_A_tile_int8(A, AT, i, max_ii, k, max_kk);
}

void pack_B_tile_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    pack_B_tile_int8(B, BT, j, max_jj, k, max_kk);
}

void transpose_pack_B_tile_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    transpose_pack_B_tile_int8(B, BT, j, max_jj, k, max_kk);
}

void pack_A_tile_fp32_to_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    pack_A_tile_fp32_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

void transpose_pack_A_tile_fp32_to_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    transpose_pack_A_tile_fp32_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

void pack_B_tile_fp32_to_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    pack_B_tile_fp32_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

void transpose_pack_B_tile_fp32_to_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    transpose_pack_B_tile_fp32_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

void unpack_output_tile_int32_to_fp32_asimddp(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta)
{
    unpack_output_tile_int32_to_fp32(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta);
}

void transpose_unpack_output_tile_int32_to_fp32_asimddp(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta)
{
    transpose_unpack_output_tile_int32_to_fp32(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta);
}

void gemm_transB_packed_tile_int8_asimddp(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
}

void pack_A_tile_fp16_to_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    pack_A_tile_fp16_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

void transpose_pack_A_tile_fp16_to_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    transpose_pack_A_tile_fp16_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

void pack_B_tile_fp16_to_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    pack_B_tile_fp16_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

void transpose_pack_B_tile_fp16_to_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    transpose_pack_B_tile_fp16_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

void unpack_output_tile_int32_to_fp16_asimddp(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta)
{
    unpack_output_tile_int32_to_fp16(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta);
}

void transpose_unpack_output_tile_int32_to_fp16_asimddp(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta)
{
    transpose_unpack_output_tile_int32_to_fp16(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta);
}

#if NCNN_BF16
void pack_A_tile_bf16_to_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    pack_A_tile_bf16_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

void transpose_pack_A_tile_bf16_to_int8_asimddp(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk, const Mat& scales)
{
    transpose_pack_A_tile_bf16_to_int8(A, AT, i, max_ii, k, max_kk, scales);
}

void pack_B_tile_bf16_to_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    pack_B_tile_bf16_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

void transpose_pack_B_tile_bf16_to_int8_asimddp(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk, float scale)
{
    transpose_pack_B_tile_bf16_to_int8(B, BT, j, max_jj, k, max_kk, scale);
}

void unpack_output_tile_int32_to_bf16_asimddp(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta)
{
    unpack_output_tile_int32_to_bf16(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta);
}

void transpose_unpack_output_tile_int32_to_bf16_asimddp(const Mat& topT, const Mat& C, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, const Mat& descales, float alpha, float beta)
{
    transpose_unpack_output_tile_int32_to_bf16(topT, C, top_blob, broadcast_type_C, i, max_ii, j, max_jj, descales, alpha, beta);
}
#endif // NCNN_BF16

} // namespace ncnn
