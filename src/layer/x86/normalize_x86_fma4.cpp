// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "normalize_fp32.h"

int normalize_fp32_fma4(Mat& bottom_top_blob, const Mat& scale_data, int across_spatial, int across_channel, int channel_shared, float eps, int eps_mode, const Option& opt)
{
    return normalize_fp32(bottom_top_blob, scale_data, across_spatial, across_channel, channel_shared, eps, eps_mode, opt);
}

} // namespace ncnn
