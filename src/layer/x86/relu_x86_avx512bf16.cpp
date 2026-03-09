// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "relu_x86.h"

#include "x86_usability.h"

#include "cpu.h"
#include "mat.h"

namespace ncnn {

#include "relu_bf16s.h"

void relu_bf16s_avx512bf16(Mat& a, float slope, const Option& opt)
{
    relu_bf16s(a, slope, opt);
}

} // namespace ncnn
