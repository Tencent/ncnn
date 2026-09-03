// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layernorm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#include "layernorm_fp16s.h"

int layernorm_inplace_fp16s_asimdhp(Mat& bottom_top_blob, int affine_size, float eps, const Mat& gamma_data, const Mat& beta_data, const Option& opt)
{
    return layernorm_inplace_fp16s(bottom_top_blob, affine_size, eps, gamma_data, beta_data, opt);
}

} // namespace ncnn
