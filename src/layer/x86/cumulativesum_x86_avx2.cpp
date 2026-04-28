// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "cumulativesum_x86_packed.h"

int cumulative_sum_forward_inplace_avx2(Mat& bottom_top_blob, int axis, const Option& opt)
{
    return cumulative_sum_forward_inplace(bottom_top_blob, axis, opt);
}

} // namespace ncnn
