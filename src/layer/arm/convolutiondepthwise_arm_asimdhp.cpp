// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolutiondepthwise_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#include "convolutiondepthwise_fp16s.h"

int convolutiondepthwise_fp16s_asimdhp(const Mat& bottom_blob_bordered, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int group, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    return convolutiondepthwise_fp16s(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, group, bias_term, activation_type, activation_params, opt);
}

int convolutiondepthwise_fp16sa_asimdhp(const Mat& bottom_blob_bordered, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, const Mat& bias_data_fp16, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int group, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    return convolutiondepthwise_fp16sa(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, group, bias_term, activation_type, activation_params, opt);
}

} // namespace ncnn
