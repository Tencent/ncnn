// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "hardswish_bf16s.h"
#include "hardswish_fp32.h"

void hardswish_bf16s_fma(Mat& a, float alpha, float beta, float lower, float upper, const Option& opt)
{
    hardswish_bf16s(a, alpha, beta, lower, upper, opt);
}

void hardswish_fp32_fma(Mat& bottom_top_blob, float alpha, float beta, float lower, float upper, const Option& opt)
{
    hardswish_fp32(bottom_top_blob, alpha, beta, lower, upper, opt);
}
} // namespace ncnn
