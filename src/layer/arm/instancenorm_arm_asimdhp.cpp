// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "instancenorm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#include "instancenorm_fp16s.h"

int instancenorm_fp16s_asimdhp(Mat& bottom_top_blob, float eps, int affine, const Mat& gamma_data, const Mat& beta_data, const Option& opt)
{
    return instancenorm_fp16s(bottom_top_blob, eps, affine, gamma_data, beta_data, opt);
}

} // namespace ncnn
