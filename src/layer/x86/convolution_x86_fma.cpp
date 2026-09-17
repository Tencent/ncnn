// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution_x86.h"

#include <immintrin.h>

#include "cpu.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "convolution_packed.h"
#include "convolution_im2col_gemm.h"
#include "convolution_3x3_winograd.h"
#include "convolution_2x2_pack8.h"
#include "convolution_3x3_pack8.h"
#include "convolution_3x3_pack1to8.h"
#include "convolution_3x3_pack8to1.h"

#if NCNN_BF16
#include "convolution_packed_bf16s.h"
#include "convolution_im2col_gemm_bf16s.h"
#include "convolution_3x3_winograd_bf16s.h"
#endif // NCNN_BF16

void convolution_packed_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    convolution_packed(bottom_blob, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
}

void convolution_gemm_transB_packed_tile_fma(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    convolution_gemm_transB_packed_tile(AT_tile, BT_tile, CT_tile, topT_tile, top_blob, i, max_ii, j, max_jj, k, max_kk, k_end);
}

void convolution_winograd_gemm_transB_packed_tile_fma(const Mat& AT_tile, const Mat& BT_tile, Mat& top_blob, int batch, int max_ii, int max_jj, int k, int max_kk)
{
    gemm_transB_packed_tile(AT_tile, BT_tile, top_blob, batch, max_ii, max_jj, k, max_kk);
}

void conv2x2s1_pack8_avx_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& bias, const Option& opt)
{
    conv2x2s1_pack8_avx(bottom_blob, top_blob, kernel, bias, opt);
}

void conv3x3s1_pack8_avx_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& bias, const Option& opt)
{
    conv3x3s1_pack8_avx(bottom_blob, top_blob, kernel, bias, opt);
}

void conv3x3s1_pack1to8_avx_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& bias, const Option& opt)
{
    conv3x3s1_pack1to8_avx(bottom_blob, top_blob, kernel, bias, opt);
}

void conv3x3s2_pack1to8_avx_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& bias, const Option& opt)
{
    conv3x3s2_pack1to8_avx(bottom_blob, top_blob, kernel, bias, opt);
}

void conv3x3s1_pack8to1_avx_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Mat& bias, const Option& opt)
{
    conv3x3s1_pack8to1_avx(bottom_blob, top_blob, kernel, bias, opt);
}

#if NCNN_BF16
void convolution_packed_bf16s_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    convolution_packed_bf16s(bottom_blob, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
}

int convolution_im2col_gemm_bf16s_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, int nT, const Option& opt)
{
    return convolution_im2col_gemm_bf16s(bottom_blob, top_blob, AT, bias, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, nT, opt);
}

int conv3x3s1_winograd23_bf16s_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, int activation_type, const Mat& activation_params, const Option& opt)
{
    return conv3x3s1_winograd23_bf16s(bottom_blob, top_blob, AT, bias, nT, activation_type, activation_params, opt);
}

int conv3x3s1_winograd43_bf16s_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, int activation_type, const Mat& activation_params, const Option& opt)
{
    return conv3x3s1_winograd43_bf16s(bottom_blob, top_blob, AT, bias, nT, activation_type, activation_params, opt);
}

int conv3x3s1_winograd63_bf16s_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int nT, int activation_type, const Mat& activation_params, const Option& opt)
{
    return conv3x3s1_winograd63_bf16s(bottom_blob, top_blob, AT, bias, nT, activation_type, activation_params, opt);
}
#endif // NCNN_BF16

} // namespace ncnn
