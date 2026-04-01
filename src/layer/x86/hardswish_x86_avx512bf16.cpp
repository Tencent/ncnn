// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "hardswish_x86.h"

#include "x86_usability.h"

#include "cpu.h"
#include "mat.h"

namespace ncnn {

#include "hardswish_bf16s.h"

void hardswish_bf16s_avx512bf16(Mat& a, float alpha, float beta, float lower, float upper, const Option& opt)
{
    hardswish_bf16s(a, alpha, beta, lower, upper, opt);
}

} // namespace ncnn
