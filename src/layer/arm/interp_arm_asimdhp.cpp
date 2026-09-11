// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "interp_arm.h"

#if __ARM_NEON
#include <arm_neon.h>
#include "arm_usability.h"
#endif // __ARM_NEON

namespace ncnn {

#include "interp_bicubic.h"
#include "interp_bilinear.h"
#include "interp_fp16s.h"

int interp_fp16s_asimdhp(const Mat& bottom_blob, Mat& top_blob, int outw, int outh, int resize_type, float width_scale, float height_scale, int align_corner, bool use_output_width, bool use_output_height, const Option& opt)
{
    return interp_fp16s(bottom_blob, top_blob, outw, outh, resize_type, width_scale, height_scale, align_corner, use_output_width, use_output_height, opt);
}

int interp_fp16sa_asimdhp(const Mat& bottom_blob, Mat& top_blob, int outw, int outh, int resize_type, float width_scale, float height_scale, int align_corner, bool use_output_width, bool use_output_height, const Option& opt)
{
    return interp_fp16sa(bottom_blob, top_blob, outw, outh, resize_type, width_scale, height_scale, align_corner, use_output_width, use_output_height, opt);
}

} // namespace ncnn
