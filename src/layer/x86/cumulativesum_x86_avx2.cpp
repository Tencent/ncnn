// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "cumulativesum_fp32.h"

void cumulative_sum_avx2(float* ptr, int w)
{
    cumulative_sum(ptr, w);
}

} // namespace ncnn
