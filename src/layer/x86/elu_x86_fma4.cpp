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
#include "elu_bf16s.h"
#endif // NCNN_BF16
#include "elu_fp32.h"

#if NCNN_BF16
void elu_bf16s_fma4(Mat& a, float alpha, const Option& opt)
{
    elu_bf16s(a, alpha, opt);
}
#endif // NCNN_BF16

void elu_fp32_fma4(Mat& bottom_top_blob, float alpha, const Option& opt)
{
    elu_fp32(bottom_top_blob, alpha, opt);
}
} // namespace ncnn
