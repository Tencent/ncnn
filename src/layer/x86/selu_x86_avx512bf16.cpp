// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "selu_bf16s.h"

void selu_bf16s_avx512bf16(Mat& a, float alphaxlambda, float lambda, const Option& opt)
{
    selu_bf16s(a, alphaxlambda, lambda, opt);
}

} // namespace ncnn
