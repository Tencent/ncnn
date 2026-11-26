// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "celu_bf16s.h"
#endif // NCNN_BF16
#include "celu_fp32.h"

#if NCNN_BF16
void celu_bf16s_fma(Mat& a, float alpha, const Option& opt)
{
    celu_bf16s(a, alpha, opt);
}
#endif // NCNN_BF16

void celu_fp32_fma(Mat& bottom_top_blob, float alpha, const Option& opt)
{
    celu_fp32(bottom_top_blob, alpha, opt);
}
} // namespace ncnn
