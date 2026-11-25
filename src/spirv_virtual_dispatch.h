// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_SPIRV_VIRTUAL_DISPATCH_H
#define NCNN_SPIRV_VIRTUAL_DISPATCH_H

#include "gpu.h"

#if NCNN_VULKAN
namespace ncnn {

enum { VIRTUAL_DISPATCH_CONSTANT_COUNT = 6 };

// Append six scalar push constants (workgroup base xyz, logical count xyz).
// Keep all original push constant offsets and specialization IDs unchanged.
int inject_virtual_dispatch(const uint32_t* code, size_t size, int push_constant_count,
                            uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                            std::vector<uint32_t>& out);

} // namespace ncnn
#endif // NCNN_VULKAN

#endif // NCNN_SPIRV_VIRTUAL_DISPATCH_H
