// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eltwise_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

#include "cpu.h"

namespace ncnn {

#include "eltwise_fp16s.h"

int eltwise_fp16s_asimdhp(const std::vector<Mat>& bottom_blobs, Mat& top_blob, int op_type, const Mat& coeffs, const Option& opt)
{
    return eltwise_fp16s(bottom_blobs, top_blob, op_type, coeffs, opt);
}

int eltwise_fp16sa_asimdhp(const std::vector<Mat>& bottom_blobs, Mat& top_blob, int op_type, const Mat& coeffs, const Option& opt)
{
    return eltwise_fp16sa(bottom_blobs, top_blob, op_type, coeffs, opt);
}

} // namespace ncnn
