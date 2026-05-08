// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_NOOP_VULKAN_H
#define LAYER_NOOP_VULKAN_H

#include "noop.h"

namespace ncnn {

class Noop_vulkan : public Noop
{
public:
    Noop_vulkan();

    using Noop::forward;
    virtual int forward_inplace(std::vector<VkMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_NOOP_VULKAN_H
