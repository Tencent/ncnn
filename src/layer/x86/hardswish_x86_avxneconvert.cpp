// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "hardswish_bf16s.h"

void hardswish_bf16s_avxneconvert(Mat& a, float alpha, float beta, float lower, float upper, const Option& opt)
{
    hardswish_bf16s(a, alpha, beta, lower, upper, opt);
}

#endif // NCNN_BF16

} // namespace ncnn
