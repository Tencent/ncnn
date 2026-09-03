// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "innerproduct_arm.h"

#include "cpu.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#include "innerproduct_fp16s.h"
#include "innerproduct_gemm_fp16s.h"
#include "innerproduct_fp16sa.h"

void innerproduct_pack4_fp16s_neon_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_pack4_fp16s_neon(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
}

void innerproduct_fp16s_neon_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_fp16s_neon(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
}

void innerproduct_gemm_fp16s_neon_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_gemm_fp16s_neon(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
}

void innerproduct_transform_kernel_fp16s_neon_asimdhp(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
{
    innerproduct_transform_kernel_fp16s_neon(weight_data, weight_data_tm, num_input, num_output, opt);
}

void innerproduct_gemm_fp16sa_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, const Mat& bias_data_fp16, int num_input, int num_output, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_gemm_fp16sa(bottom_blob, top_blob, weight_data_tm, bias_data, bias_data_fp16, num_input, num_output, bias_term, activation_type, activation_params, opt);
}

void innerproduct_pack8_fp16sa_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data_fp16, int num_input, int num_output, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_pack8_fp16sa(bottom_blob, top_blob, weight_data_tm, bias_data_fp16, num_input, num_output, bias_term, activation_type, activation_params, opt);
}

void innerproduct_pack4_fp16sa_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data_fp16, int num_input, int num_output, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_pack4_fp16sa(bottom_blob, top_blob, weight_data_tm, bias_data_fp16, num_input, num_output, bias_term, activation_type, activation_params, opt);
}

void innerproduct_fp16sa_asimdhp(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int num_input, int num_output, int bias_term, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_fp16sa(bottom_blob, top_blob, weight_data_tm, bias_data, num_input, num_output, bias_term, activation_type, activation_params, opt);
}
} // namespace ncnn
