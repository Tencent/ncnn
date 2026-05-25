// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "cumulativesum_x86_helper.h"

void cumulative_sum_prefix_sum_row_avx2(float* ptr, int w)
{
    cumulative_sum_prefix_sum_row_avx2_impl(ptr, w);
}

void cumulative_sum_add_avx2(const float* ptr, float* outptr, int size)
{
    cumulative_sum_add_avx2_impl(ptr, outptr, size);
}

} // namespace ncnn
