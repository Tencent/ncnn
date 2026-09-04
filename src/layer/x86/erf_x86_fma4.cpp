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
#include "erf_bf16s.h"
#endif // NCNN_BF16
#include "erf_fp32.h"

#if NCNN_BF16
void erf_bf16s_fma4(Mat& a, const Option& opt)
{
    erf_bf16s(a, opt);
}
#endif // NCNN_BF16

void erf_fp32_fma4(Mat& bottom_top_blob, const Option& opt)
{
    erf_fp32(bottom_top_blob, opt);
}
} // namespace ncnn
