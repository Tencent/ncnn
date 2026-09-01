// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "dequantize_bf16s.h"
#include "dequantize_fp32.h"

void dequantize_forward_bf16s_fma(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_data, int scale_data_size, const Mat& bias_data, int bias_data_size, const Option& opt)
{
    dequantize_forward_bf16s(bottom_blob, top_blob, scale_data, scale_data_size, bias_data, bias_data_size, opt);
}

void dequantize_fma(const int* intptr, float* ptr, const Mat& scale_data, const Mat& bias_data, int elemcount, int elempack)
{
    dequantize(intptr, ptr, scale_data, bias_data, elemcount, elempack);
}
} // namespace ncnn
