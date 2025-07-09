// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "innerproduct_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE2__

#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#define NCNN_IMPL_FP16S 1
#include "innerproduct_fp.h"
#include "innerproduct_gemm_fp.h"
#undef NCNN_IMPL_FP16S

void innerproduct_fp16s_sse_f16c(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_fp16s_sse(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
}

void innerproduct_gemm_fp16s_sse_f16c(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_fp16, const Mat& bias_data, int activation_type, const Mat& activation_params, const Option& opt)
{
    innerproduct_gemm_fp16s_sse(bottom_blob, top_blob, weight_data_fp16, bias_data, activation_type, activation_params, opt);
}

void innerproduct_transform_kernel_fp16s_sse_f16c(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, const Option& opt)
{
    innerproduct_transform_kernel_fp16s_sse(weight_data, weight_data_tm, num_input, num_output, opt);
}

} // namespace ncnn
