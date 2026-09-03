// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "unaryop_arm.h"

// #include <fenv.h>
#include <float.h>

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "neon_mathfun_fp16s.h"
#endif
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#include "unaryop_fp16s.h"

int unaryop_fp16s_asimdhp(Mat& bottom_top_blob, int op_type, const Option& opt)
{
    return unaryop_fp16s(bottom_top_blob, op_type, opt);
}

} // namespace ncnn
