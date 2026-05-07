// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "softmax_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Softmax_vulkan::Softmax_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_softmax_reduce_max = 0;
    pipeline_softmax_exp_sub_max = 0;
    pipeline_softmax_reduce_sum = 0;
    pipeline_softmax_div_sum = 0;

    pipeline_softmax_reduce_max_pack4 = 0;
    pipeline_softmax_exp_sub_max_pack4 = 0;
    pipeline_softmax_reduce_sum_pack4 = 0;
    pipeline_softmax_div_sum_pack4 = 0;
}

int Softmax_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];
    int positive_axis = axis < 0 ? shape.dims + axis : axis;

    int elempack = shape.elempack;
    size_t elemsize = shape.elemsize;

    Mat workspace_shape;
    if (shape.dims == 1) // positive_axis == 0
    {
        workspace_shape = Mat(1, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 2 && positive_axis == 0)
    {
        workspace_shape = Mat(shape.w, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 2 && positive_axis == 1)
    {
        workspace_shape = Mat(shape.h, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 3 && positive_axis == 0)
    {
        workspace_shape = Mat(shape.w, shape.h, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 3 && positive_axis == 1)
    {
        workspace_shape = Mat(shape.w, shape.c, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 3 && positive_axis == 2)
    {
        workspace_shape = Mat(shape.h, shape.c, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 4 && positive_axis == 0)
    {
        workspace_shape = Mat(shape.w, shape.h, shape.d, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 4 && positive_axis == 1)
    {
        workspace_shape = Mat(shape.w, shape.h, shape.c, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 4 && positive_axis == 2)
    {
        workspace_shape = Mat(shape.w, shape.d, shape.c, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 4 && positive_axis == 3)
    {
        workspace_shape = Mat(shape.h, shape.d, shape.c, (void*)0, elemsize, elempack);
    }

    std::vector<vk_specialization_type> specializations(1 + 12);
    specializations[0].i = axis;
    specializations[1 + 0].i = shape.dims;
    specializations[1 + 1].i = shape.w;
    specializations[1 + 2].i = shape.h;
    specializations[1 + 3].i = shape.d;
    specializations[1 + 4].i = shape.c;
    specializations[1 + 5].i = shape.cstep;
    specializations[1 + 6].i = workspace_shape.dims;
    specializations[1 + 7].i = workspace_shape.w;
    specializations[1 + 8].i = workspace_shape.h;
    specializations[1 + 9].i = workspace_shape.d;
    specializations[1 + 10].i = workspace_shape.c;
    specializations[1 + 11].i = workspace_shape.cstep;

    {
        Mat local_size_xyz;
        if (workspace_shape.dims == 1)
        {
            local_size_xyz.w = std::min(64, workspace_shape.w);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }
        if (workspace_shape.dims == 2)
        {
            local_size_xyz.w = std::min(8, workspace_shape.w);
            local_size_xyz.h = std::min(8, workspace_shape.h);
            local_size_xyz.c = 1;
        }
        if (workspace_shape.dims != 0)
        {
            local_size_xyz.w = std::min(4, workspace_shape.w);
            local_size_xyz.h = std::min(4, workspace_shape.h * workspace_shape.d);
            local_size_xyz.c = std::min(4, workspace_shape.c);
        }

        // pack1
        {
            pipeline_softmax_reduce_max = new Pipeline(vkdev);
            pipeline_softmax_reduce_sum = new Pipeline(vkdev);

            pipeline_softmax_reduce_max->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_softmax_reduce_sum->set_optimal_local_size_xyz(local_size_xyz);

            pipeline_softmax_reduce_max->create(LayerShaderType::softmax_reduce_max, opt, specializations);
            pipeline_softmax_reduce_sum->create(LayerShaderType::softmax_reduce_sum, opt, specializations);
        }

        // pack4
        {
            pipeline_softmax_reduce_max_pack4 = new Pipeline(vkdev);
            pipeline_softmax_reduce_sum_pack4 = new Pipeline(vkdev);

            pipeline_softmax_reduce_max_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_softmax_reduce_sum_pack4->set_optimal_local_size_xyz(local_size_xyz);

            pipeline_softmax_reduce_max_pack4->create(LayerShaderType::softmax_reduce_max_pack4, opt, specializations);
            pipeline_softmax_reduce_sum_pack4->create(LayerShaderType::softmax_reduce_sum_pack4, opt, specializations);
        }
    }

    {
        Mat local_size_xyz;
        if (shape.dims == 1)
        {
            local_size_xyz.w = std::min(64, shape.w);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }
        if (shape.dims == 2)
        {
            local_size_xyz.w = std::min(8, shape.w);
            local_size_xyz.h = std::min(8, shape.h);
            local_size_xyz.c = 1;
        }
        if (shape.dims == 3)
        {
            local_size_xyz.w = std::min(4, shape.w);
            local_size_xyz.h = std::min(4, shape.h);
            local_size_xyz.c = std::min(4, shape.c);
        }
        if (shape.dims == 4)
        {
            local_size_xyz.w = std::min(4, shape.w);
            local_size_xyz.h = std::min(4, shape.h * shape.d);
            local_size_xyz.c = std::min(4, shape.c);
        }

        // pack1
        {
            pipeline_softmax_exp_sub_max = new Pipeline(vkdev);
            pipeline_softmax_div_sum = new Pipeline(vkdev);

            pipeline_softmax_exp_sub_max->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_softmax_div_sum->set_optimal_local_size_xyz(local_size_xyz);

            pipeline_softmax_exp_sub_max->create(LayerShaderType::softmax_exp_sub_max, opt, specializations);
            pipeline_softmax_div_sum->create(LayerShaderType::softmax_div_sum, opt, specializations);
        }

        // pack4
        {
            pipeline_softmax_exp_sub_max_pack4 = new Pipeline(vkdev);
            pipeline_softmax_div_sum_pack4 = new Pipeline(vkdev);

            pipeline_softmax_exp_sub_max_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_softmax_div_sum_pack4->set_optimal_local_size_xyz(local_size_xyz);

            pipeline_softmax_exp_sub_max_pack4->create(LayerShaderType::softmax_exp_sub_max_pack4, opt, specializations);
            pipeline_softmax_div_sum_pack4->create(LayerShaderType::softmax_div_sum_pack4, opt, specializations);
        }
    }

    return 0;
}

int Softmax_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_softmax_reduce_max;
    pipeline_softmax_reduce_max = 0;

    delete pipeline_softmax_exp_sub_max;
    pipeline_softmax_exp_sub_max = 0;

    delete pipeline_softmax_reduce_sum;
    pipeline_softmax_reduce_sum = 0;

    delete pipeline_softmax_div_sum;
    pipeline_softmax_div_sum = 0;

    delete pipeline_softmax_reduce_max_pack4;
    pipeline_softmax_reduce_max_pack4 = 0;

    delete pipeline_softmax_exp_sub_max_pack4;
    pipeline_softmax_exp_sub_max_pack4 = 0;

    delete pipeline_softmax_reduce_sum_pack4;
    pipeline_softmax_reduce_sum_pack4 = 0;

    delete pipeline_softmax_div_sum_pack4;
    pipeline_softmax_div_sum_pack4 = 0;

    return 0;
}

int Softmax_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;
    int positive_axis = axis < 0 ? dims + axis : axis;

    VkMat max_workspace;
    VkMat sum_workspace;

    if (dims == 1) // positive_axis == 0
    {
        max_workspace.create(1, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(1, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 2 && positive_axis == 0)
    {
        max_workspace.create(w, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 2 && positive_axis == 1)
    {
        max_workspace.create(h, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(h, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 3 && positive_axis == 0)
    {
        max_workspace.create(w, h, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, h, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 3 && positive_axis == 1)
    {
        max_workspace.create(w, channels, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, channels, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 3 && positive_axis == 2)
    {
        max_workspace.create(h, channels, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(h, channels, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 4 && positive_axis == 0)
    {
        max_workspace.create(w, h, d, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, h, d, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 4 && positive_axis == 1)
    {
        max_workspace.create(w, h, channels, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, h, channels, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 4 && positive_axis == 2)
    {
        max_workspace.create(w, d, channels, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, d, channels, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 4 && positive_axis == 3)
    {
        max_workspace.create(h, d, channels, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(h, d, channels, elemsize, elempack, opt.workspace_vkallocator);
    }

    // reduce max
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = max_workspace;

        std::vector<vk_constant_type> constants(12);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.d;
        constants[4].i = bottom_top_blob.c;
        constants[5].i = bottom_top_blob.cstep;
        constants[6].i = max_workspace.dims;
        constants[7].i = max_workspace.w;
        constants[8].i = max_workspace.h;
        constants[9].i = max_workspace.d;
        constants[10].i = max_workspace.c;
        constants[11].i = max_workspace.cstep;

        const Pipeline* pipeline = elempack == 4 ? pipeline_softmax_reduce_max_pack4 : pipeline_softmax_reduce_max;

        cmd.record_pipeline(pipeline, bindings, constants, max_workspace);
    }

    // exp( v - max )
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = max_workspace;

        std::vector<vk_constant_type> constants(12);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.d;
        constants[4].i = bottom_top_blob.c;
        constants[5].i = bottom_top_blob.cstep;
        constants[6].i = max_workspace.dims;
        constants[7].i = max_workspace.w;
        constants[8].i = max_workspace.h;
        constants[9].i = max_workspace.d;
        constants[10].i = max_workspace.c;
        constants[11].i = max_workspace.cstep;

        const Pipeline* pipeline = elempack == 4 ? pipeline_softmax_exp_sub_max_pack4 : pipeline_softmax_exp_sub_max;

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    // reduce sum
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = sum_workspace;

        std::vector<vk_constant_type> constants(12);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.d;
        constants[4].i = bottom_top_blob.c;
        constants[5].i = bottom_top_blob.cstep;
        constants[6].i = sum_workspace.dims;
        constants[7].i = sum_workspace.w;
        constants[8].i = sum_workspace.h;
        constants[9].i = sum_workspace.d;
        constants[10].i = sum_workspace.c;
        constants[11].i = sum_workspace.cstep;

        const Pipeline* pipeline = elempack == 4 ? pipeline_softmax_reduce_sum_pack4 : pipeline_softmax_reduce_sum;

        cmd.record_pipeline(pipeline, bindings, constants, sum_workspace);
    }

    // div sum
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = sum_workspace;

        std::vector<vk_constant_type> constants(12);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.d;
        constants[4].i = bottom_top_blob.c;
        constants[5].i = bottom_top_blob.cstep;
        constants[6].i = sum_workspace.dims;
        constants[7].i = sum_workspace.w;
        constants[8].i = sum_workspace.h;
        constants[9].i = sum_workspace.d;
        constants[10].i = sum_workspace.c;
        constants[11].i = sum_workspace.cstep;

        const Pipeline* pipeline = elempack == 4 ? pipeline_softmax_div_sum_pack4 : pipeline_softmax_div_sum;

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

} // namespace ncnn
