// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "tanh_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

TanH_vulkan::TanH_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_tanh = 0;
}

int TanH_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    std::vector<vk_specialization_type> specializations(1);
    specializations[0].u32 = shape.total() * shape.elempack / 4;

    const int local_size_x = vkdev->info.subgroup_size();

    pipeline_tanh = new Pipeline(vkdev);
    pipeline_tanh->set_optimal_local_size_xyz(local_size_x, 1, 1);
    pipeline_tanh->create(LayerShaderType::tanh, opt, specializations);

    return 0;
}

int TanH_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_tanh;
    pipeline_tanh = 0;

    return 0;
}

int TanH_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& /*opt*/) const
{
    const size_t n = bottom_top_blob.total() * bottom_top_blob.elempack / 4;

    std::vector<VkMat> bindings(1);
    bindings[0] = bottom_top_blob;

    std::vector<vk_constant_type> constants(1);
    constants[0].u32 = n;

    VkMat dispatcher;
    dispatcher.w = n;
    dispatcher.h = 1;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline_tanh, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
