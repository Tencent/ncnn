// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "selu_bf16s.h"
#include "selu_fp32.h"

void selu_bf16s_fma(Mat& a, float alphaxlambda, float lambda, const Option& opt)
{
    selu_bf16s(a, alphaxlambda, lambda, opt);
}

void selu_fp32_fma(Mat& bottom_top_blob, float alpha, float lambda, const Option& opt)
{
    selu_fp32(bottom_top_blob, alpha, lambda, opt);
}
} // namespace ncnn
