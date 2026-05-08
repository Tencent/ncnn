// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "elu_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

ELU_vulkan::ELU_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_elu = 0;
}

int ELU_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    std::vector<vk_specialization_type> specializations(1 + 1);
    specializations[0].f = alpha;
    specializations[1 + 0].u32 = shape.total() * shape.elempack / 4;

    const int local_size_x = vkdev->info.subgroup_size();

    pipeline_elu = new Pipeline(vkdev);
    pipeline_elu->set_optimal_local_size_xyz(local_size_x, 1, 1);
    pipeline_elu->create(LayerShaderType::elu, opt, specializations);

    return 0;
}

int ELU_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_elu;
    pipeline_elu = 0;

    return 0;
}

int ELU_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& /*opt*/) const
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
    cmd.record_pipeline(pipeline_elu, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
