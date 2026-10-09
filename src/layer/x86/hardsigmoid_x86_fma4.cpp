// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "hardsigmoid_bf16s.h"
#endif // NCNN_BF16
#include "hardsigmoid_fp32.h"

#if NCNN_BF16
void hardsigmoid_bf16s_fma4(Mat& a, float alpha, float beta, const Option& opt)
{
    hardsigmoid_bf16s(a, alpha, beta, opt);
}
#endif // NCNN_BF16

void hardsigmoid_fp32_fma4(Mat& bottom_top_blob, float alpha, float beta, float lower, float upper, const Option& opt)
{
    hardsigmoid_fp32(bottom_top_blob, alpha, beta, lower, upper, opt);
}
} // namespace ncnn
