// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dropout_arm.h"

#include "arm_16bit.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

#include "dropout_arm_16bit.h"

#if NCNN_ARM82
int Dropout_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    return dropout_forward_inplace_16bit<ArmFP16Converter>(bottom_top_blob, scale, opt);
}
#endif // NCNN_ARM82

} // namespace ncnn
