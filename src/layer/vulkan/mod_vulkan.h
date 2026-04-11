// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_MOD_VULKAN_H
#define LAYER_MOD_VULKAN_H

#include "mod.h"

namespace ncnn {

class Mod_vulkan : public virtual Mod
{
public:
    Mod_vulkan(vkcom::VulkanDevice* _vkdev);
    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_mod;
};

} // namespace ncnn

#endif // LAYER_MOD_VULKAN_H
