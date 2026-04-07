// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "dropout_bf16s.h"

void dropout_bf16s_avx512bf16(Mat& a, float scale, const Option& opt)
{
    dropout_bf16s(a, scale, opt);
}

} // namespace ncnn
