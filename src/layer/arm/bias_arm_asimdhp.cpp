// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "bias_arm.h"

#include "arm_16bit.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

#include "bias_arm_16bit.h"

#if NCNN_ARM82
int Bias_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    return bias_forward_inplace_16bit<ArmFP16Converter>(bottom_top_blob, bias_data, opt);
}
#endif // NCNN_ARM82

} // namespace ncnn
