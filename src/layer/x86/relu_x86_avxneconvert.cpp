// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "relu_bf16s.h"

void relu_bf16s_avxneconvert(Mat& a, float slope, const Option& opt)
{
    relu_bf16s(a, slope, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
