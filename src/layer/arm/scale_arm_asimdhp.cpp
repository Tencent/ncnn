// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_arm.h"

#include "arm_16bit.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

#include "scale_arm_16bit.h"

#if NCNN_ARM82
int Scale_arm::forward_inplace_fp16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    return scale_forward_inplace_16bit<ArmFP16Converter>(bottom_top_blobs[0], bottom_top_blobs[1], bias_data, bias_term, opt);
}
#endif // NCNN_ARM82

} // namespace ncnn
