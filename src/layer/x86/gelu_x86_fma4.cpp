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
#include "gelu_bf16s.h"
#endif // NCNN_BF16
#include "gelu_fp32.h"

#if NCNN_BF16
void gelu_bf16s_fma4(Mat& a, int fast_gelu, const Option& opt)
{
    gelu_bf16s(a, fast_gelu, opt);
}
#endif // NCNN_BF16

void gelu_fp32_fma4(Mat& bottom_top_blob, int fast_gelu, const Option& opt)
{
    gelu_fp32(bottom_top_blob, fast_gelu, opt);
}
} // namespace ncnn
