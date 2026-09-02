// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "innerproduct_x86.h"

#include <immintrin.h>

#include "cpu.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "innerproduct_fp.h"
#include "innerproduct_gemm_fp.h"

#define NCNN_IMPL_FP16S 1
#include "innerproduct_fp.h"
#include "innerproduct_gemm_fp.h"
#undef NCNN_IMPL_FP16S

#if NCNN_BF16
#include "innerproduct_bf16s.h"
#include "innerproduct_gemm_bf16s.h"
#endif // NCNN_BF16

void innerproduct_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct(bottom_blob, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
}

void innerproduct_gemm_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_gemm(bottom_blob, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
}

void innerproduct_fp16s_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_fp16s(bottom_blob, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
}

void innerproduct_gemm_fp16s_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_gemm_fp16s(bottom_blob, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
}

#if NCNN_BF16
void innerproduct_bf16s_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_bf16s(bottom_blob, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
}

void innerproduct_gemm_bf16s_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_gemm_bf16s(bottom_blob, top_blob, weight_data_tm, bias_data, activation_type, activation_params, opt);
}
#endif // NCNN_BF16

} // namespace ncnn
