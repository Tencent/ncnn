// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "batchnorm_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#include "batchnorm_fp16s.h"

int batchnorm_fp16s_asimdhp(Mat& bottom_top_blob, const Mat& a_data, const Mat& b_data, const Option& opt)
{
    return batchnorm_fp16s(bottom_top_blob, a_data, b_data, opt);
}

int batchnorm_fp16sa_asimdhp(Mat& bottom_top_blob, const Mat& a_data, const Mat& b_data, const Option& opt)
{
    return batchnorm_fp16sa(bottom_top_blob, a_data, b_data, opt);
}

} // namespace ncnn
