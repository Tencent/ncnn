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
#include "bnll_bf16s.h"
#endif // NCNN_BF16
#include "bnll_fp32.h"

#if NCNN_BF16
void bnll_bf16s_fma(Mat& a, const Option& opt)
{
    bnll_bf16s(a, opt);
}
#endif // NCNN_BF16

void bnll_fp32_fma(Mat& bottom_top_blob, const Option& opt)
{
    bnll_fp32(bottom_top_blob, opt);
}
} // namespace ncnn
