// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "convolution_packed_int8.h"
#include "convolution_im2col_gemm_int8.h"
#include "convolution_3x3_winograd_int8.h"

// packed
void convolution_packed_int8_avxvnni(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    convolution_packed_int8(bottom_blob, top_blob, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
}

// gemm
void convolution_im2col_input_tile_int8_avxvnni(const Mat& bottom_blob, Mat& B, int j, int max_jj, int k, int max_kk, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h)
{
    convolution_im2col_input_tile_int8(bottom_blob, B, j, max_jj, k, max_kk, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h);
}

// winograd
int conv3x3s1_winograd23_int8_avxvnni(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt)
{
    return conv3x3s1_winograd23_int8(bottom_blob, top_blob, AT, nT, opt);
}

int conv3x3s1_winograd43_int8_avxvnni(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int nT, const Option& opt)
{
    return conv3x3s1_winograd43_int8(bottom_blob, top_blob, AT, nT, opt);
}

} // namespace ncnn
