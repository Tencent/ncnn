// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "gelu_bf16s.h"

void gelu_bf16s_avx512bf16(Mat& a, int fast_gelu, const Option& opt)
{
    gelu_bf16s(a, fast_gelu, opt);
}

} // namespace ncnn
