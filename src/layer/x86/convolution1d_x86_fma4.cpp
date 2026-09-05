// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution1d_x86.h"

#include <immintrin.h>

#include "cpu.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "convolution1d_packed.h"
#if NCNN_BF16
#include "convolution1d_packed_bf16s.h"
#endif // NCNN_BF16

void convolution1d_packed_fma4(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int dilation_w, int stride_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    convolution1d_packed(bottom_blob, top_blob, weight_data_tm, bias_data, kernel_w, dilation_w, stride_w, activation_type, activation_params, opt);
}

#if NCNN_BF16
void convolution1d_packed_bf16s_fma4(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int kernel_w, int dilation_w, int stride_w, int activation_type, const Mat& activation_params, const Option& opt)
{
    convolution1d_packed_bf16s(bottom_blob, top_blob, weight_data_tm, bias_data, kernel_w, dilation_w, stride_w, activation_type, activation_params, opt);
}
#endif // NCNN_BF16

} // namespace ncnn
