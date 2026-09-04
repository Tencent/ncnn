// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prelu_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#include "prelu_fp16s.h"

int prelu_fp16s_asimdhp(Mat& bottom_top_blob, int num_slope, const Mat& slope_data, const Option& opt)
{
    return prelu_fp16s(bottom_top_blob, num_slope, slope_data, opt);
}

int prelu_fp16sa_asimdhp(Mat& bottom_top_blob, int num_slope, const Mat& slope_data, const Option& opt)
{
    return prelu_fp16sa(bottom_top_blob, num_slope, slope_data, opt);
}

} // namespace ncnn
