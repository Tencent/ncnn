// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "relu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#include "relu_fp16s.h"

int relu_fp16s_asimdhp(Mat& bottom_top_blob, float slope, const Option& opt)
{
    return relu_fp16s(bottom_top_blob, slope, opt);
}

} // namespace ncnn
