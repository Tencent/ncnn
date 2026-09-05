// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution_x86.h"

#include "cpu.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "convolution_packed_int8.h"
#include "convolution_im2col_gemm_int8.h"
#include "convolution_3x3_winograd_int8.h"

#include "convolution_packed_bf16s.h"
#include "convolution_im2col_gemm_bf16s.h"
// convolution_3x3_winograd.h provides gemm_transB_packed_tile() used by convolution_3x3_winograd_bf16s.h
#include "convolution_3x3_winograd.h"
#include "convolution_3x3_winograd_bf16s.h"

// packed
void convolution_transform_kernel_packed_int8_avx2(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    convolution_transform_kernel_packed_int8(kernel, kernel_tm, inch, outch, kernel_w, kernel_h);
}

void convolution_packed_int8_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    convolution_packed_int8(bottom_blob, top_blob, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
}

// gemm
void convolution_im2col_input_tile_int8_avx2(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    convolution_im2col_input_tile_int8(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

void unpack_output_tile_int32_avx2(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    unpack_output_tile_int32(topT, top_blob, i, max_ii, j, max_jj);
}

// winograd
void conv3x3s1_winograd23_transform_kernel_int8_avx2(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    conv3x3s1_winograd23_transform_kernel_int8(kernel, AT, inch, outch, opt);
}

int conv3x3s1_winograd23_int8_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt)
{
    return conv3x3s1_winograd23_int8(bottom_blob, top_blob, AT, nT, opt);
}

void conv3x3s1_winograd43_transform_kernel_int8_avx2(const Mat& kernel, Mat& AT, int inch, int outch, const Option& opt)
{
    conv3x3s1_winograd43_transform_kernel_int8(kernel, AT, inch, outch, opt);
}

int conv3x3s1_winograd43_int8_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt)
{
    return conv3x3s1_winograd43_int8(bottom_blob, top_blob, AT, nT, opt);
}

void convolution_transform_kernel_packed_bf16s_avx2(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h)
{
    convolution_transform_kernel_packed_bf16s(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
}

void convolution_packed_bf16s_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    convolution_packed_bf16s(bottom_blob, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
}

void convolution_im2col_gemm_transform_kernel_bf16s_avx2(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    convolution_im2col_gemm_transform_kernel_bf16s(kernel, AT, inch, outch, kernel_w, kernel_h, opt);
}

int convolution_im2col_gemm_bf16s_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, int nT, const Option& opt)
{
    return convolution_im2col_gemm_bf16s(bottom_blob, top_blob, AT, bias, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, nT, opt);
}

void conv3x3s1_winograd23_transform_input_tile_bf16s_avx2(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    conv3x3s1_winograd23_transform_input_tile_bf16s(bottom_blob, B, j, max_jj, k, max_kk, nT);
}

void conv3x3s1_winograd23_transform_output_tile_bf16s_avx2(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj, int activation_type, const Mat& activation_params)
{
    conv3x3s1_winograd23_transform_output_tile_bf16s(top_tile, top_blob, bias, i, max_ii, j, max_jj, activation_type, activation_params);
}

int conv3x3s1_winograd23_bf16s_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias_data, int nT, int activation_type, const Mat& activation_params, const Option& opt)
{
    return conv3x3s1_winograd23_bf16s(bottom_blob, top_blob, AT, bias_data, nT, activation_type, activation_params, opt);
}

void conv3x3s1_winograd43_transform_input_tile_bf16s_avx2(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    conv3x3s1_winograd43_transform_input_tile_bf16s(bottom_blob, B, j, max_jj, k, max_kk, nT);
}

void conv3x3s1_winograd43_transform_output_tile_bf16s_avx2(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj, int activation_type, const Mat& activation_params)
{
    conv3x3s1_winograd43_transform_output_tile_bf16s(top_tile, top_blob, bias, i, max_ii, j, max_jj, activation_type, activation_params);
}

int conv3x3s1_winograd43_bf16s_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias_data, int nT, int activation_type, const Mat& activation_params, const Option& opt)
{
    return conv3x3s1_winograd43_bf16s(bottom_blob, top_blob, AT, bias_data, nT, activation_type, activation_params, opt);
}

void conv3x3s1_winograd63_transform_input_tile_bf16s_avx2(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int nT)
{
    conv3x3s1_winograd63_transform_input_tile_bf16s(bottom_blob, B, j, max_jj, k, max_kk, nT);
}

void conv3x3s1_winograd63_transform_output_tile_bf16s_avx2(const Mat& top_tile, Mat& top_blob, const Mat& bias, int i, int max_ii, int j, int max_jj, int activation_type, const Mat& activation_params)
{
    conv3x3s1_winograd63_transform_output_tile_bf16s(top_tile, top_blob, bias, i, max_ii, j, max_jj, activation_type, activation_params);
}

int conv3x3s1_winograd63_bf16s_avx2(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias_data, int nT, int activation_type, const Mat& activation_params, const Option& opt)
{
    return conv3x3s1_winograd63_bf16s(bottom_blob, top_blob, AT, bias_data, nT, activation_type, activation_params, opt);
}

} // namespace ncnn
