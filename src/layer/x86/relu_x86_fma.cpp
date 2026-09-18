// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "relu_x86.h"

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "relu_fp32.h"

#if NCNN_BF16
#include "relu_bf16s.h"
#endif

void relu_fp32_fma(Mat& bottom_top_blob, float slope, const Option& opt)
{
    relu_fp32(bottom_top_blob, slope, opt);
}

#if NCNN_BF16
void relu_bf16s_fma(Mat& a, float slope, const Option& opt)
{
    relu_bf16s(a, slope, opt);
}
#endif

} // namespace ncnn
