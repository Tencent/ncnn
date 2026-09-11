// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "hardsigmoid_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#include "neon_mathfun.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "neon_mathfun_fp16s.h"
#endif
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#include "hardsigmoid_fp16s.h"

int hardsigmoid_fp16s_asimdhp(Mat& bottom_top_blob, float alpha, float beta, float lower, float upper, const Option& opt)
{
    return hardsigmoid_fp16s(bottom_top_blob, alpha, beta, lower, upper, opt);
}

int hardsigmoid_fp16sa_asimdhp(Mat& bottom_top_blob, float alpha, float beta, float lower, float upper, const Option& opt)
{
    return hardsigmoid_fp16sa(bottom_top_blob, alpha, beta, lower, upper, opt);
}

} // namespace ncnn
