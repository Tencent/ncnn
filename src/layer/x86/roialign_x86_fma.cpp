// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "roialign_x86.h"

#include "cpu.h"

namespace ncnn {

#include "roialign_fp32.h"

void roialign_fp32_fma(const Mat& bottom_blob, const Mat& roi_blob, Mat& top_blob, int pooled_width, int pooled_height, float spatial_scale, int sampling_ratio, int aligned, int version, const Option& opt)
{
    roialign_fp32(bottom_blob, roi_blob, top_blob, pooled_width, pooled_height, spatial_scale, sampling_ratio, aligned, version, opt);
}

} // namespace ncnn
