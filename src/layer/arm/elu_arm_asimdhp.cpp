// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "elu_arm.h"

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

#include "elu_fp16s.h"

int elu_fp16s_asimdhp(Mat& bottom_top_blob, float alpha, const Option& opt)
{
    return elu_fp16s(bottom_top_blob, alpha, opt);
}

int elu_fp16sa_asimdhp(Mat& bottom_top_blob, float alpha, const Option& opt)
{
    return elu_fp16sa(bottom_top_blob, alpha, opt);
}

} // namespace ncnn
