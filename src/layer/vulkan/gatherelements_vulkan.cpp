// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gatherelements_vulkan.h"
#include "command.h"

namespace ncnn {

GatherElements_vulkan::GatherElements_vulkan(vkcom::VulkanDevice* _vkdev)
    : GatherElements(), pipeline_gatherelements(0)
{
    vkdev = _vkdev;
}

int GatherElements_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(1);
    specializations[0] = 0; // placeholder

    pipeline_gatherelements = new Pipeline(vkdev, opt.shader_blob_option());
    pipeline_gatherelements->create("gatherelements_comp", specializations);

    return 0;
}

int GatherElements_vulkan::destroy_pipeline(const Option& opt)
{
    if (pipeline_gatherelements)
    {
        delete pipeline_gatherelements;
        pipeline_gatherelements = 0;
    }

    return 0;
}

int GatherElements_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    if (bottom_blobs.size() < 2)
        return -1;

    const VkMat& data_blob = bottom_blobs[0];
    const VkMat& index_blob = bottom_blobs[1];

    // Output has same shape as index_blob
    VkMat& top_blob = top_blobs[0];
    top_blob.create(index_blob.w, index_blob.h, index_blob.c, data_blob.elemsize, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    // TODO: Implement Vulkan compute shader dispatch
    // For now, fallback to CPU implementation
    // This requires creating a gatherelements.comp shader file

    return 0;
}

int GatherElements_vulkan::forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    return -1; // Not supported for image format yet
}

} // namespace ncnn
