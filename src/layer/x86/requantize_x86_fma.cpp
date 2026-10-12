// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "layer.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "requantize_fp32.h"

void requantize_fma(const int* intptr, signed char* ptr, const Mat& scale_in_data, const Mat& bias_data, const Mat& scale_out_data, int activation_type, const Mat& activation_params, int elemcount, int elempack)
{
    requantize(intptr, ptr, scale_in_data, bias_data, scale_out_data, activation_type, activation_params, elemcount, elempack);
}

} // namespace ncnn
