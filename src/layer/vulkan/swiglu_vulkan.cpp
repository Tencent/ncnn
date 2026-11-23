// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "swiglu_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

SwiGLU_vulkan::SwiGLU_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_swiglu = 0;
}

int SwiGLU_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    std::vector<vk_specialization_type> specializations(1 + 3);
    specializations[0].u32 = shape.total() * shape.elempack / 4;
    specializations[1].f = sigmoid_scale;
    specializations[2].f = up_offset;
    specializations[3].f = activation_limit;

    const int local_size_x = std::max(16, (int)vkdev->info.subgroup_size());

    pipeline_swiglu = new Pipeline(vkdev);
    pipeline_swiglu->set_optimal_local_size_xyz(local_size_x, 1, 1);
    pipeline_swiglu->create(LayerShaderType::swiglu, opt, specializations);

    return 0;
}

int SwiGLU_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_swiglu;
    pipeline_swiglu = 0;

    return 0;
}

int SwiGLU_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& gate = bottom_blobs[0];
    const VkMat& up = bottom_blobs[1];

    VkMat& top_blob = top_blobs[0];
    top_blob.create_like(gate, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    const uint32_t n = (uint32_t)(gate.total() * gate.elempack / 4);

    std::vector<VkMat> bindings(3);
    bindings[0] = gate;
    bindings[1] = up;
    bindings[2] = top_blob;

    std::vector<vk_constant_type> constants(1);
    constants[0].u32 = n;

    VkMat dispatcher;
    dispatcher.w = (int)n;
    dispatcher.h = 1;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_swiglu, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
