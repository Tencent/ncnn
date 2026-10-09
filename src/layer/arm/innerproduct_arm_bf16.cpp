// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "innerproduct_arm.h"

#include "cpu.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "innerproduct_bf16s.h"
#include "innerproduct_gemm_bf16s.h"

void innerproduct_pack4_bf16s_neon_bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_bf16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_pack4_bf16s_neon(bottom_blob, top_blob, weight_data_bf16, bias_data, activation_type, activation_params, opt);
}

void innerproduct_bf16s_neon_bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_bf16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_bf16s_neon(bottom_blob, top_blob, weight_data_bf16, bias_data, activation_type, activation_params, opt);
}

void innerproduct_transform_kernel_bf16s_neon_bf16(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
{
    innerproduct_transform_kernel_bf16s_neon(weight_data, weight_data_tm, num_input, num_output, opt);
}

void innerproduct_gemm_bf16s_neon_bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_bf16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_gemm_bf16s_neon(bottom_blob, top_blob, weight_data_bf16, bias_data, activation_type, activation_params, opt);
}
#endif // NCNN_BF16

} // namespace ncnn
