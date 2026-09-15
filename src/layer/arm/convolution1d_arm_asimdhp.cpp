// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution1d_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#include "convolution1d_packed_fp16s.h"

void convolution1d_transform_kernel_packed_fp16s_asimdhp(const Mat& kernel, Mat& kernel_tm, int inh, int outh, int kernel_w)
{
    convolution1d_transform_kernel_packed_fp16s(kernel, kernel_tm, inh, outh, kernel_w);
}

void convolution1d_packed_fp16s_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int dilation_w, int stride_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    convolution1d_packed_fp16s(bottom_blob, top_blob, weight_data_tm, bias_data, kernel_w, dilation_w, stride_w, activation_type, activation_params, opt);
}

void convolution1d_packed_fp16sa_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int dilation_w, int stride_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    convolution1d_packed_fp16sa(bottom_blob, top_blob, weight_data_tm, bias_data, kernel_w, dilation_w, stride_w, activation_type, activation_params, opt);
}

} // namespace ncnn
