// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "tanh_x86.h"

#include "x86_activation.h"

#include "cpu.h"

namespace ncnn {

#include "tanh_bf16s.h"

void tanh_bf16s_avx512bf16(Mat& a, const Option& opt)
{
    tanh_bf16s(a, opt);
}

} // namespace ncnn
