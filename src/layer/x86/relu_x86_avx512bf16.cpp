// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "relu_x86.h"

#include "x86_usability.h"

#include "cpu.h"
#include "mat.h"

namespace ncnn {

#include "relu_bf16s.h"

int ReLU_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    return relu_bf16s(bottom_top_blob, slope, opt);
}

} // namespace ncnn
