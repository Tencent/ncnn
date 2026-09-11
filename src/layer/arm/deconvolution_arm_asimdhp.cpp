// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deconvolution_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"

namespace ncnn {

#include "deconvolution_fp16s.h"

void deconvolution_transform_kernel_fp16s_asimdhp(const Mat& kernel, Mat& kernel_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    deconvolution_transform_kernel_fp16s(kernel, kernel_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
}

void deconvolution_fp16s_asimdhp(const Mat& bottom_blob, Mat& top_blob_bordered, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int num_output, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    deconvolution_fp16s(bottom_blob, top_blob_bordered, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, num_output, bias_term, activation_type, activation_params, opt);
}

void deconvolution_col2im_fp16sa_asimdhp(const Mat& top_col2im, Mat& top_blob_bordered, const Mat& bias_data, const Mat& bias_data_fp16, int input_w, int input_h, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    deconvolution_col2im_fp16sa(top_col2im, top_blob_bordered, bias_data, bias_data_fp16, input_w, input_h, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
}

bool deconvolution_fp16sa_asimdhp(const Mat& bottom_blob, Mat& top_blob_bordered, const Mat& weight_data_tm, const Mat& bias_data, const Mat& bias_data_fp16, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int num_output, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    return deconvolution_fp16sa(bottom_blob, top_blob_bordered, weight_data_tm, bias_data, bias_data_fp16, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, num_output, bias_term, activation_type, activation_params, opt);
}

} // namespace ncnn
