// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_SPLIT_VULKAN_H
#define LAYER_SPLIT_VULKAN_H

#include "split.h"

namespace ncnn {

class Split_vulkan : public Split
{
public:
    Split_vulkan();

    using Split::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_SPLIT_VULKAN_H
