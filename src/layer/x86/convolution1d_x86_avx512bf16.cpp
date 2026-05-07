// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "convolution1d_packed_bf16s.h"

void convolution1d_packed_bf16s_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int dilation_w, int stride_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    convolution1d_packed_bf16s(bottom_blob, top_blob, weight_data_tm, bias_data, kernel_w, dilation_w, stride_w, activation_type, activation_params, opt);
}

void convolution1d_transform_kernel_packed_bf16s_avx512bf16(const Mat& kernel, Mat& kernel_tm, int inh, int outh, int kernel_w)
{
    convolution1d_transform_kernel_packed_bf16s(kernel, kernel_tm, inh, outh, kernel_w);
}

} // namespace ncnn
