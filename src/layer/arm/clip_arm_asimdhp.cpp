// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "clip_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

#if __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif // __ARM_FEATURE_SVE

#include "cpu.h"

namespace ncnn {

#include "clip_fp16s.h"

int clip_fp16s_asimdhp(Mat& bottom_top_blob, float min, float max, const Option& opt)
{
    return clip_fp16s(bottom_top_blob, min, max, opt);
}
} // namespace ncnn
