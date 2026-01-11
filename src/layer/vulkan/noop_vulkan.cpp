// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "noop_vulkan.h"

namespace ncnn {

Noop_vulkan::Noop_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
}

int Noop_vulkan::forward_inplace(std::vector<VkMat>& /*bottom_top_blobs*/, VkCompute& /*cmd*/, const Option& /*opt*/) const
{
    return 0;
}

} // namespace ncnn
