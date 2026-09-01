// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "erf_bf16s.h"
#include "erf_fp32.h"

void erf_bf16s_fma(Mat& a, const Option& opt)
{
    erf_bf16s(a, opt);
}

void erf_fp32_fma(Mat& bottom_top_blob, const Option& opt)
{
    erf_fp32(bottom_top_blob, opt);
}
} // namespace ncnn
