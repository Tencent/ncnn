// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "unaryop_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

UnaryOp_vulkan::UnaryOp_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_unaryop = 0;
}

int UnaryOp_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    std::vector<vk_specialization_type> specializations(2);
    specializations[0].i = op_type;
    specializations[1].u32 = shape.total() / 4;

    const int local_size_x = vkdev->info.subgroup_size();

    pipeline_unaryop = new Pipeline(vkdev);
    pipeline_unaryop->set_optimal_local_size_xyz(local_size_x, 1, 1);
    pipeline_unaryop->create(LayerShaderType::unaryop, opt, specializations);

    return 0;
}

int UnaryOp_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_unaryop;
    pipeline_unaryop = 0;

    return 0;
}

int UnaryOp_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& /*opt*/) const
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
    cmd.record_pipeline(pipeline_unaryop, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
