// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "roialign_x86.h"

#include "cpu.h"

namespace ncnn {

#include "roialign_fp32.h"

ROIAlign_x86::ROIAlign_x86()
{
}

int ROIAlign_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    return roialign_fp32(bottom_blobs, top_blobs, pooled_width, pooled_height, spatial_scale, sampling_ratio, aligned, version, opt);
}

} // namespace ncnn
