// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "softmax_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Softmax_vulkan::Softmax_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    pipeline_softmax_reduce_max = 0;
    pipeline_softmax_exp_sub_max = 0;
    pipeline_softmax_reduce_sum = 0;
    pipeline_softmax_div_sum = 0;

    pipeline_softmax_reduce_max_pack4 = 0;
    pipeline_softmax_exp_sub_max_pack4 = 0;
    pipeline_softmax_reduce_sum_pack4 = 0;
    pipeline_softmax_div_sum_pack4 = 0;

    pipeline_softmax_reduce_max_pack8 = 0;
    pipeline_softmax_exp_sub_max_pack8 = 0;
    pipeline_softmax_reduce_sum_pack8 = 0;
    pipeline_softmax_div_sum_pack8 = 0;
}

int Softmax_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = opt.use_shader_pack8 && shape.w % 8 == 0 ? 8 : shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;

    size_t elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
    }

    Mat shape_packed;
    if (shape.dims == 1) shape_packed = Mat(shape.w / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat workspace_shape_packed;
    if (shape.dims == 1) // axis == 0
    {
        workspace_shape_packed = Mat(1, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 2 && axis == 0)
    {
        workspace_shape_packed = Mat(shape.w, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 2 && axis == 1)
    {
        workspace_shape_packed = Mat(shape.h / elempack, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 3 && axis == 0)
    {
        workspace_shape_packed = Mat(shape.w, shape.h, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 3 && axis == 1)
    {
        workspace_shape_packed = Mat(shape.w, shape.c / elempack, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 3 && axis == 2)
    {
        workspace_shape_packed = Mat(shape.h, shape.c / elempack, (void*)0, elemsize, elempack);
    }

    std::vector<vk_specialization_type> specializations(1 + 10);
    specializations[0].i = axis;
    specializations[1 + 0].i = shape_packed.dims;
    specializations[1 + 1].i = shape_packed.w;
    specializations[1 + 2].i = shape_packed.h;
    specializations[1 + 3].i = shape_packed.c;
    specializations[1 + 4].i = shape_packed.cstep;
    specializations[1 + 5].i = workspace_shape_packed.dims;
    specializations[1 + 6].i = workspace_shape_packed.w;
    specializations[1 + 7].i = workspace_shape_packed.h;
    specializations[1 + 8].i = workspace_shape_packed.c;
    specializations[1 + 9].i = workspace_shape_packed.cstep;

    {
        Mat local_size_xyz;
        if (workspace_shape_packed.dims == 1)
        {
            local_size_xyz.w = std::min(64, workspace_shape_packed.w);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }
        if (workspace_shape_packed.dims == 2)
        {
            local_size_xyz.w = std::min(8, workspace_shape_packed.w);
            local_size_xyz.h = std::min(8, workspace_shape_packed.h);
            local_size_xyz.c = 1;
        }
        if (workspace_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(4, workspace_shape_packed.w);
            local_size_xyz.h = std::min(4, workspace_shape_packed.h);
            local_size_xyz.c = std::min(4, workspace_shape_packed.c);
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

        // pack8
        if (opt.use_shader_pack8)
        {
            pipeline_softmax_reduce_max_pack8 = new Pipeline(vkdev);
            pipeline_softmax_reduce_sum_pack8 = new Pipeline(vkdev);

            pipeline_softmax_reduce_max_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_softmax_reduce_sum_pack8->set_optimal_local_size_xyz(local_size_xyz);

            pipeline_softmax_reduce_max_pack8->create(LayerShaderType::softmax_reduce_max_pack8, opt, specializations);
            pipeline_softmax_reduce_sum_pack8->create(LayerShaderType::softmax_reduce_sum_pack8, opt, specializations);
        }
    }

    {
        Mat local_size_xyz;
        if (shape_packed.dims == 1)
        {
            local_size_xyz.w = std::min(64, shape_packed.w);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }
        if (shape_packed.dims == 2)
        {
            local_size_xyz.w = std::min(8, shape_packed.w);
            local_size_xyz.h = std::min(8, shape_packed.h);
            local_size_xyz.c = 1;
        }
        if (shape_packed.dims == 3)
        {
            local_size_xyz.w = std::min(4, shape_packed.w);
            local_size_xyz.h = std::min(4, shape_packed.h);
            local_size_xyz.c = std::min(4, shape_packed.c);
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

        // pack8
        if (opt.use_shader_pack8)
        {
            pipeline_softmax_exp_sub_max_pack8 = new Pipeline(vkdev);
            pipeline_softmax_div_sum_pack8 = new Pipeline(vkdev);

            pipeline_softmax_exp_sub_max_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_softmax_div_sum_pack8->set_optimal_local_size_xyz(local_size_xyz);

            pipeline_softmax_exp_sub_max_pack8->create(LayerShaderType::softmax_exp_sub_max_pack8, opt, specializations);
            pipeline_softmax_div_sum_pack8->create(LayerShaderType::softmax_div_sum_pack8, opt, specializations);
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

    delete pipeline_softmax_reduce_max_pack8;
    pipeline_softmax_reduce_max_pack8 = 0;

    delete pipeline_softmax_exp_sub_max_pack8;
    pipeline_softmax_exp_sub_max_pack8 = 0;

    delete pipeline_softmax_reduce_sum_pack8;
    pipeline_softmax_reduce_sum_pack8 = 0;

    delete pipeline_softmax_div_sum_pack8;
    pipeline_softmax_div_sum_pack8 = 0;

    return 0;
}

int Softmax_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;

    VkMat max_workspace;
    VkMat sum_workspace;

    if (dims == 1) // axis == 0
    {
        max_workspace.create(1, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(1, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 2 && axis == 0)
    {
        max_workspace.create(w, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 2 && axis == 1)
    {
        max_workspace.create(h, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(h, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 3 && axis == 0)
    {
        max_workspace.create(w, h, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, h, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 3 && axis == 1)
    {
        max_workspace.create(w, channels, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, channels, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 3 && axis == 2)
    {
        max_workspace.create(h, channels, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(h, channels, elemsize, elempack, opt.workspace_vkallocator);
    }

    // reduce max
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = max_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = bottom_top_blob.cstep;
        constants[5].i = max_workspace.dims;
        constants[6].i = max_workspace.w;
        constants[7].i = max_workspace.h;
        constants[8].i = max_workspace.c;
        constants[9].i = max_workspace.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_softmax_reduce_max_pack8
                                   : elempack == 4 ? pipeline_softmax_reduce_max_pack4
                                   : pipeline_softmax_reduce_max;

        cmd.record_pipeline(pipeline, bindings, constants, max_workspace);
    }

    // exp( v - max )
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = max_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = bottom_top_blob.cstep;
        constants[5].i = max_workspace.dims;
        constants[6].i = max_workspace.w;
        constants[7].i = max_workspace.h;
        constants[8].i = max_workspace.c;
        constants[9].i = max_workspace.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_softmax_exp_sub_max_pack8
                                   : elempack == 4 ? pipeline_softmax_exp_sub_max_pack4
                                   : pipeline_softmax_exp_sub_max;

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    // reduce sum
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = sum_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = bottom_top_blob.cstep;
        constants[5].i = sum_workspace.dims;
        constants[6].i = sum_workspace.w;
        constants[7].i = sum_workspace.h;
        constants[8].i = sum_workspace.c;
        constants[9].i = sum_workspace.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_softmax_reduce_sum_pack8
                                   : elempack == 4 ? pipeline_softmax_reduce_sum_pack4
                                   : pipeline_softmax_reduce_sum;

        cmd.record_pipeline(pipeline, bindings, constants, sum_workspace);
    }

    // div sum
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = sum_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = bottom_top_blob.cstep;
        constants[5].i = sum_workspace.dims;
        constants[6].i = sum_workspace.w;
        constants[7].i = sum_workspace.h;
        constants[8].i = sum_workspace.c;
        constants[9].i = sum_workspace.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_softmax_div_sum_pack8
                                   : elempack == 4 ? pipeline_softmax_div_sum_pack4
                                   : pipeline_softmax_div_sum;

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

int Softmax_vulkan::forward_inplace(VkImageMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;

    VkImageMat max_workspace;
    VkImageMat sum_workspace;

    if (dims == 1) // axis == 0
    {
        max_workspace.create(1, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(1, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 2 && axis == 0)
    {
        max_workspace.create(w, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 2 && axis == 1)
    {
        max_workspace.create(h, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(h, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 3 && axis == 0)
    {
        max_workspace.create(w, h, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, h, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 3 && axis == 1)
    {
        max_workspace.create(w, channels, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, channels, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 3 && axis == 2)
    {
        max_workspace.create(h, channels, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(h, channels, elemsize, elempack, opt.workspace_vkallocator);
    }

    // reduce max
    {
        std::vector<VkImageMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = max_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = 0; //bottom_top_blob.cstep;
        constants[5].i = max_workspace.dims;
        constants[6].i = max_workspace.w;
        constants[7].i = max_workspace.h;
        constants[8].i = max_workspace.c;
        constants[9].i = 0; //max_workspace.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_softmax_reduce_max_pack8
                                   : elempack == 4 ? pipeline_softmax_reduce_max_pack4
                                   : pipeline_softmax_reduce_max;

        cmd.record_pipeline(pipeline, bindings, constants, max_workspace);
    }

    // exp( v - max )
    {
        std::vector<VkImageMat> bindings(3);
        bindings[0] = bottom_top_blob;
        bindings[1] = bottom_top_blob;
        bindings[2] = max_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = 0; //bottom_top_blob.cstep;
        constants[5].i = max_workspace.dims;
        constants[6].i = max_workspace.w;
        constants[7].i = max_workspace.h;
        constants[8].i = max_workspace.c;
        constants[9].i = 0; //max_workspace.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_softmax_exp_sub_max_pack8
                                   : elempack == 4 ? pipeline_softmax_exp_sub_max_pack4
                                   : pipeline_softmax_exp_sub_max;

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    // reduce sum
    {
        std::vector<VkImageMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = sum_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = 0; //bottom_top_blob.cstep;
        constants[5].i = sum_workspace.dims;
        constants[6].i = sum_workspace.w;
        constants[7].i = sum_workspace.h;
        constants[8].i = sum_workspace.c;
        constants[9].i = 0; //sum_workspace.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_softmax_reduce_sum_pack8
                                   : elempack == 4 ? pipeline_softmax_reduce_sum_pack4
                                   : pipeline_softmax_reduce_sum;

        cmd.record_pipeline(pipeline, bindings, constants, sum_workspace);
    }

    // div sum
    {
        std::vector<VkImageMat> bindings(3);
        bindings[0] = bottom_top_blob;
        bindings[1] = bottom_top_blob;
        bindings[2] = sum_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = 0; //bottom_top_blob.cstep;
        constants[5].i = sum_workspace.dims;
        constants[6].i = sum_workspace.w;
        constants[7].i = sum_workspace.h;
        constants[8].i = sum_workspace.c;
        constants[9].i = 0; //sum_workspace.cstep;

        const Pipeline* pipeline = elempack == 8 ? pipeline_softmax_div_sum_pack8
                                   : elempack == 4 ? pipeline_softmax_div_sum_pack4
                                   : pipeline_softmax_div_sum;

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

} // namespace ncnn
