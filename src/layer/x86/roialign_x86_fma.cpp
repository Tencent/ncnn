// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "roialign_x86.h"

#include "cpu.h"

namespace ncnn {

#include "roialign_fp32.h"

int roialign_fp32_fma(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, int pooled_width, int pooled_height, float spatial_scale, int sampling_ratio, int aligned, int version, const Option& opt)
{
    return roialign_fp32(bottom_blobs, top_blobs, pooled_width, pooled_height, spatial_scale, sampling_ratio, aligned, version, opt);
}

} // namespace ncnn
