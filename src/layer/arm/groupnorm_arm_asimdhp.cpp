// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "groupnorm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#include "groupnorm_fp16s.h"

int groupnorm_inplace_fp16s_asimdhp(Mat& bottom_top_blob, int group, int channels, float eps, int affine, const Mat& gamma_data, const Mat& beta_data, const Option& opt)
{
    return groupnorm_inplace_fp16s(bottom_top_blob, group, channels, eps, affine, gamma_data, beta_data, opt);
}

} // namespace ncnn
