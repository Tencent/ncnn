// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "split_vulkan.h"

namespace ncnn {

Split_vulkan::Split_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
}

int Split_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& /*cmd*/, const Option& /*opt*/) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    for (size_t i = 0; i < top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blob;
    }

    return 0;
}

} // namespace ncnn
