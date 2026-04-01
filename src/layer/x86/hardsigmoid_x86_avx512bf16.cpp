// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "hardsigmoid_x86.h"

#include "x86_usability.h"

#include "cpu.h"
#include "mat.h"

namespace ncnn {

#include "hardsigmoid_bf16s.h"

void hardsigmoid_bf16s_avx512bf16(Mat& a, float alpha, float beta, const Option& opt)
{
    hardsigmoid_bf16s(a, alpha, beta, opt);
}

} // namespace ncnn
