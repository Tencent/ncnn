// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "gelu_bf16s.h"
#include "gelu_fp32.h"

void gelu_bf16s_fma(Mat& a, int fast_gelu, const Option& opt)
{
    gelu_bf16s(a, fast_gelu, opt);
}

void gelu_fp32_fma(Mat& bottom_top_blob, int fast_gelu, const Option& opt)
{
    gelu_fp32(bottom_top_blob, fast_gelu, opt);
}
} // namespace ncnn
