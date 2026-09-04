// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution_arm.h"

#include "cpu.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#include "convolution_fp16s.h"

void convolution_transform_kernel_fp16s_asimdhp(const Mat& weight_data, Mat& weight_data_tm, Mat& weight_sgemm_data, Mat& weight_winograd23_data, Mat& weight_winograd43_data, Mat& weight_winograd63_data, int weight_data_size, int num_output, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    convolution_transform_kernel_fp16s(weight_data, weight_data_tm, weight_sgemm_data, weight_winograd23_data, weight_winograd43_data, weight_winograd63_data, weight_data_size, num_output, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
}

void convolution_fp16s_asimdhp(const Mat& bottom_blob_bordered, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, const Option& opt)
{
    convolution_fp16s(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
}

int convolution_fp16sa_asimdhp(const Mat& bottom_blob_bordered, Mat& top_blob, const Mat& weight_data_tm, const Mat& weight_sgemm_data, const Mat& weight_winograd23_data, const Mat& weight_winograd43_data, const Mat& weight_winograd63_data, const Mat& bias_data_fp16, int num_output, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int activation_type, const Mat& activation_params, int nT, bool& activation_unfused, const Option& opt)
{
    return convolution_fp16sa(bottom_blob_bordered, top_blob, weight_data_tm, weight_sgemm_data, weight_winograd23_data, weight_winograd43_data, weight_winograd63_data, bias_data_fp16, num_output, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, nT, activation_unfused, opt);
}

} // namespace ncnn
