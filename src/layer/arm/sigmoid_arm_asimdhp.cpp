// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sigmoid_arm.h"

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

#include "sigmoid_fp16s.h"

int sigmoid_fp16s_asimdhp(Mat& bottom_top_blob, const Option& opt)
{
    return sigmoid_fp16s(bottom_top_blob, opt);
}

int sigmoid_fp16sa_asimdhp(Mat& bottom_top_blob, const Option& opt)
{
    return sigmoid_fp16sa(bottom_top_blob, opt);
}

} // namespace ncnn
