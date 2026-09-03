// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deconvolutiondepthwise_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"

namespace ncnn {

#include "deconvolutiondepthwise_fp16s.h"

void deconvolutiondepthwise_fp16s_asimdhp(const Mat& bottom_blob, Mat& top_blob_bordered, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    deconvolutiondepthwise_fp16s(bottom_blob, top_blob_bordered, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, bias_term, activation_type, activation_params, opt);
}

void deconvolutiondepthwise_fp16sa_asimdhp(const Mat& bottom_blob, Mat& top_blob_bordered, const Mat& weight_data_tm, const Mat& bias_data, const Mat& bias_data_fp16, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    deconvolutiondepthwise_fp16sa(bottom_blob, top_blob_bordered, weight_data_tm, bias_data, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, bias_term, activation_type, activation_params, opt);
}

} // namespace ncnn
