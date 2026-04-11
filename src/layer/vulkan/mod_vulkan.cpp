// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "mod_vulkan.h"
#include "command.h"

namespace ncnn {

Mod_vulkan::Mod_vulkan(vkcom::VulkanDevice* _vkdev)
    : Mod(), pipeline_mod(0)
{
    vkdev = _vkdev;
}

int Mod_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(1 + 1);
    specializations[0] = 0; // fmode
    specializations[1] = 0; // placeholder

    pipeline_mod = new Pipeline(vkdev, opt.shader_blob_option());
    pipeline_mod->create("mod_comp", specializations);

    return 0;
}

int Mod_vulkan::destroy_pipeline(const Option& opt)
{
    if (pipeline_mod)
    {
        delete pipeline_mod;
        pipeline_mod = 0;
    }

    return 0;
}

int Mod_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    if (bottom_blobs.size() < 2)
        return -1;

    const VkMat& a_blob = bottom_blobs[0];
    const VkMat& b_blob = bottom_blobs[1];

    // Output has same shape as a_blob
    VkMat& top_blob = top_blobs[0];
    top_blob.create(a_blob.w, a_blob.h, a_blob.c, a_blob.elemsize, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    // Record command buffer
    // The mod_comp shader would compute: out[i] = a[i] % b[i]
    
    // TODO: Implement actual Vulkan dispatch
    // Requires mod_comp shader with modulo operation
    // For now, placeholder implementation

    return 0;
}

int Mod_vulkan::forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    return -1; // Not supported for image format yet
}

} // namespace ncnn
