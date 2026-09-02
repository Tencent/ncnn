// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deconvolutiondepthwise_x86.h"

#include "cpu.h"
#include "layer.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "deconvolutiondepthwise_fp32.h"

void deconvolutiondepthwise_fp32_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int bias_term, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    deconvolutiondepthwise_fp32(bottom_blob, top_blob, weight_data, bias_data, bias_term, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
}

} // namespace ncnn
