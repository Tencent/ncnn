// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "lrn_arm.h"

#include "arm_16bit.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "neon_mathfun.h"
#endif // __ARM_NEON

namespace ncnn {

#include "lrn_arm_16bit.h"

#if NCNN_ARM82
int LRN_arm::forward_inplace_fp16s(Mat& bottom_top_blob, const Option& opt) const
{
    return lrn_forward_inplace_16bit<ArmFP16Converter>(bottom_top_blob, opt, region_type, local_size, alpha, beta, bias);
}
#endif // NCNN_ARM82

} // namespace ncnn
