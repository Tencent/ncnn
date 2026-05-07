// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deepcopy_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

DeepCopy_vulkan::DeepCopy_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_deepcopy = 0;
    pipeline_deepcopy_pack4 = 0;
}

int DeepCopy_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    std::vector<vk_specialization_type> specializations(0 + 5);
    specializations[0 + 0].i = shape.dims;
    specializations[0 + 1].i = shape.w;
    specializations[0 + 2].i = shape.h;
    specializations[0 + 3].i = shape.c;
    specializations[0 + 4].i = shape.cstep;

    Mat local_size_xyz;
    if (out_shape.dims == 1)
    {
        local_size_xyz.w = std::min(64, out_shape.w);
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }
    if (out_shape.dims == 2)
    {
        local_size_xyz.w = std::min(8, out_shape.w);
        local_size_xyz.h = std::min(8, out_shape.h);
        local_size_xyz.c = 1;
    }
    if (out_shape.dims == 3)
    {
        local_size_xyz.w = std::min(4, out_shape.w);
        local_size_xyz.h = std::min(4, out_shape.h);
        local_size_xyz.c = std::min(4, out_shape.c);
    }

    // pack1
    if (shape.dims == 0 || shape.elempack == 1)
    {
        pipeline_deepcopy = new Pipeline(vkdev);
        pipeline_deepcopy->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_deepcopy->create(LayerShaderType::deepcopy, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || shape.elempack == 4)
    {
        pipeline_deepcopy_pack4 = new Pipeline(vkdev);
        pipeline_deepcopy_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_deepcopy_pack4->create(LayerShaderType::deepcopy_pack4, opt, specializations);
    }

    return 0;
}

int DeepCopy_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_deepcopy;
    pipeline_deepcopy = 0;

    delete pipeline_deepcopy_pack4;
    pipeline_deepcopy_pack4 = 0;

    return 0;
}

int DeepCopy_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int elempack = bottom_blob.elempack;

    top_blob.create_like(bottom_blob, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;

    const Pipeline* pipeline = elempack == 4 ? pipeline_deepcopy_pack4 : pipeline_deepcopy;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn
