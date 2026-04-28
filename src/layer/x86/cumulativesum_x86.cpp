// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cumulativesum_x86.h"

#include "x86_usability.h"

#include "cpu.h"

namespace ncnn {

#include "cumulativesum_x86_packed.h"

CumulativeSum_x86::CumulativeSum_x86()
{
}

int CumulativeSum_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    return cumulative_sum_forward_inplace(bottom_top_blob, axis, opt);
}

} // namespace ncnn
