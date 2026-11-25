// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "glu_arm.h"

#include "arm_16bit.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

#include "glu_arm_16bit.h"

#if NCNN_ARM82
int GLU_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    return glu_forward_16bit_dispatch<ArmFP16Converter>(bottom_blob, top_blob, axis, opt);
}
#endif // NCNN_ARM82

} // namespace ncnn
