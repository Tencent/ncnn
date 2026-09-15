// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "softmax_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#include "neon_mathfun_fp16s.h"
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#include "softmax_fp16s.h"

int softmax_inplace_fp16s_asimdhp(Mat& bottom_top_blob, int axis, const Option& opt)
{
    return softmax_inplace_fp16s(bottom_top_blob, axis, opt);
}

} // namespace ncnn
