// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "flip_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Flip_vulkan::Flip_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_flip = 0;
    pipeline_flip_pack4 = 0;
}

int Flip_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];

    std::vector<vk_specialization_type> specializations(0);

    // pack1
    if (shape.dims == 0 || shape.elempack == 1)
    {
        pipeline_flip = new Pipeline(vkdev);
        pipeline_flip->set_optimal_local_size_xyz(vkdev->info.subgroup_size(), 1, 1);
        pipeline_flip->create(LayerShaderType::flip, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || shape.elempack == 4)
    {
        pipeline_flip_pack4 = new Pipeline(vkdev);
        pipeline_flip_pack4->set_optimal_local_size_xyz(vkdev->info.subgroup_size(), 1, 1);
        pipeline_flip_pack4->create(LayerShaderType::flip_pack4, opt, specializations);
    }

    return 0;
}

int Flip_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_flip;
    pipeline_flip = 0;

    delete pipeline_flip_pack4;
    pipeline_flip_pack4 = 0;

    return 0;
}

int Flip_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    if (axes.empty())
    {
        top_blob = bottom_blob;
        return 0;
    }

    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;

    int axes_flag[4] = {0};
    {
        const int* axes_ptr = axes;
        for (int i = 0; i < axes.w; i++)
        {
            int axis = axes_ptr[i];
            // handle negative axis
            if (axis < 0)
                axis += dims;
            axes_flag[axis] = 1;
        }
    }

    top_blob.create_like(bottom_blob, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    const int elempack = bottom_blob.elempack;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    // flip flags and packlane reversal are derived from dims and axes_flag in the shader
    std::vector<vk_constant_type> constants(10);
    constants[0].i = dims;
    constants[1].i = w;
    constants[2].i = h;
    constants[3].i = d;
    constants[4].i = channels;
    constants[5].i = bottom_blob.cstep;
    constants[6].i = axes_flag[0];
    constants[7].i = axes_flag[1];
    constants[8].i = axes_flag[2];
    constants[9].i = axes_flag[3];

    VkMat dispatcher;
    dispatcher.w = w * h * d * channels;
    dispatcher.h = 1;
    dispatcher.c = 1;

    const Pipeline* pipeline = elempack == 4 ? pipeline_flip_pack4 : pipeline_flip;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
