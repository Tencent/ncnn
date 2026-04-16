// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gatherelements_arm.h"

namespace ncnn {

int GatherElements_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    return GatherElements::forward(bottom_blobs, top_blobs, opt);
}

} // namespace ncnn
