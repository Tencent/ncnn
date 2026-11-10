// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_COOPERATIVE_MATRIX_VULKAN_H
#define NCNN_COOPERATIVE_MATRIX_VULKAN_H

#include "gpu.h"
#include "option.h"

namespace ncnn {

static inline bool use_cooperative_matrix_conversion(const GpuInfo& info, const Option& opt, int M, int N, int K, int subgroup_size, bool int8 = false)
{
    if (!info.support_cooperative_matrix_conversion() || !opt.use_subgroup_ops)
        return false;

    const uint32_t required_subgroup_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    if ((info.support_subgroup_ops() & required_subgroup_ops) != required_subgroup_ops)
        return false;

    // the conversion extension does not support bf16 operands
    if (!int8 && (opt.use_bf16_storage || opt.use_bf16_packed || !info.support_fp16_arithmetic()))
        return false;

    // each lane supplies one A row and one B column; output rows use typed arrays
    if (M < 4 || (M & (M - 1)) || M > subgroup_size || (subgroup_size & (subgroup_size - 1))
            || N < 8 || N > 64 || (N & (N - 1)))
        return false;
    if (N != subgroup_size && N * 2 != subgroup_size && N * 4 != subgroup_size)
        return false;

    return K == (int8 ? 32 : 16);
}

} // namespace ncnn

#endif // NCNN_COOPERATIVE_MATRIX_VULKAN_H
