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

#include "lrn_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

LRN_vulkan::LRN_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    pipeline_lrn_square_pad = 0;
    pipeline_lrn_norm = 0;
    pipeline_lrn_square_pad_across_channel_pack4 = 0;
    pipeline_lrn_norm_across_channel_pack4 = 0;
    pipeline_lrn_square_pad_within_channel_pack4 = 0;
    pipeline_lrn_norm_within_channel_pack4 = 0;
    pipeline_lrn_square_pad_across_channel_pack8 = 0;
    pipeline_lrn_norm_across_channel_pack8 = 0;
    pipeline_lrn_square_pad_within_channel_pack8 = 0;
    pipeline_lrn_norm_within_channel_pack8 = 0;
}

int LRN_vulkan::create_pipeline(const Option& opt)
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
    if (shape.dims != 0)
    {
        if (region_type == NormRegion_ACROSS_CHANNELS)
        {
            workspace_shape_packed = Mat(shape.w, shape.h, shape.c + local_size - 1, (void*)0);
        }
        else if (region_type == NormRegion_WITHIN_CHANNEL)
        {
            workspace_shape_packed = Mat(shape.w + local_size - 1, shape.h + local_size - 1, shape.c / elempack, (void*)0, elempack * 4u, elempack);
        }
    }

    {
        int pad = local_size / 2;

        std::vector<vk_specialization_type> specializations(3 + 10);
        specializations[0].i = region_type;
        specializations[1].i = pad;
        specializations[2].i = local_size - pad - 1;
        specializations[3 + 0].i = shape_packed.dims;
        specializations[3 + 1].i = shape_packed.w;
        specializations[3 + 2].i = shape_packed.h;
        specializations[3 + 3].i = shape_packed.c;
        specializations[3 + 4].i = shape_packed.cstep;
        specializations[3 + 5].i = workspace_shape_packed.dims;
        specializations[3 + 6].i = workspace_shape_packed.w;
        specializations[3 + 7].i = workspace_shape_packed.h;
        specializations[3 + 8].i = workspace_shape_packed.c;
        specializations[3 + 9].i = workspace_shape_packed.cstep;

        Mat local_size_xyz;
        if (workspace_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(4, workspace_shape_packed.w);
            local_size_xyz.h = std::min(4, workspace_shape_packed.h);
            local_size_xyz.c = std::min(4, workspace_shape_packed.c);
        }

        // pack1
        if (shape.dims == 0 || elempack == 1)
        {
            pipeline_lrn_square_pad = new Pipeline(vkdev);
            pipeline_lrn_square_pad->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_lrn_square_pad->create(LayerShaderType::lrn_square_pad, opt, specializations);
        }

        // pack4
        if (region_type == 0 && (shape.dims == 0 || elempack == 4))
        {
            pipeline_lrn_square_pad_across_channel_pack4 = new Pipeline(vkdev);
            pipeline_lrn_square_pad_across_channel_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_lrn_square_pad_across_channel_pack4->create(LayerShaderType::lrn_square_pad_across_channel_pack4, opt, specializations);
        }
        if (region_type == 1 && (shape.dims == 0 || elempack == 4))
        {
            pipeline_lrn_square_pad_within_channel_pack4 = new Pipeline(vkdev);
            pipeline_lrn_square_pad_within_channel_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_lrn_square_pad_within_channel_pack4->create(LayerShaderType::lrn_square_pad_within_channel_pack4, opt, specializations);
        }

        // pack8
        if (region_type == 0 && ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8))
        {
            pipeline_lrn_square_pad_across_channel_pack8 = new Pipeline(vkdev);
            pipeline_lrn_square_pad_across_channel_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_lrn_square_pad_across_channel_pack8->create(LayerShaderType::lrn_square_pad_across_channel_pack8, opt, specializations);
        }
        if (region_type == 1 && ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8))
        {
            pipeline_lrn_square_pad_within_channel_pack8 = new Pipeline(vkdev);
            pipeline_lrn_square_pad_within_channel_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_lrn_square_pad_within_channel_pack8->create(LayerShaderType::lrn_square_pad_within_channel_pack8, opt, specializations);
        }
    }

    {
        std::vector<vk_specialization_type> specializations(5 + 10);
        specializations[0].i = region_type;
        specializations[1].i = local_size;
        specializations[2].f = alpha;
        specializations[3].f = beta;
        specializations[4].f = bias;
        specializations[5 + 0].i = workspace_shape_packed.dims;
        specializations[5 + 1].i = workspace_shape_packed.w;
        specializations[5 + 2].i = workspace_shape_packed.h;
        specializations[5 + 3].i = workspace_shape_packed.c;
        specializations[5 + 4].i = workspace_shape_packed.cstep;
        specializations[5 + 5].i = shape_packed.dims;
        specializations[5 + 6].i = shape_packed.w;
        specializations[5 + 7].i = shape_packed.h;
        specializations[5 + 8].i = shape_packed.c;
        specializations[5 + 9].i = shape_packed.cstep;

        Mat local_size_xyz;
        if (shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(4, shape_packed.w);
            local_size_xyz.h = std::min(4, shape_packed.h);
            local_size_xyz.c = std::min(4, shape_packed.c);
        }

        // pack1
        if (shape.dims == 0 || elempack == 1)
        {
            pipeline_lrn_norm = new Pipeline(vkdev);
            pipeline_lrn_norm->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_lrn_norm->create(LayerShaderType::lrn_norm, opt, specializations);
        }

        // pack4
        if (region_type == 0 && (shape.dims == 0 || elempack == 4))
        {
            pipeline_lrn_norm_across_channel_pack4 = new Pipeline(vkdev);
            pipeline_lrn_norm_across_channel_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_lrn_norm_across_channel_pack4->create(LayerShaderType::lrn_norm_across_channel_pack4, opt, specializations);
        }
        if (region_type == 1 && (shape.dims == 0 || elempack == 4))
        {
            pipeline_lrn_norm_within_channel_pack4 = new Pipeline(vkdev);
            pipeline_lrn_norm_within_channel_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_lrn_norm_within_channel_pack4->create(LayerShaderType::lrn_norm_within_channel_pack4, opt, specializations);
        }

        // pack8
        if (region_type == 0 && ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8))
        {
            pipeline_lrn_norm_across_channel_pack8 = new Pipeline(vkdev);
            pipeline_lrn_norm_across_channel_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_lrn_norm_across_channel_pack8->create(LayerShaderType::lrn_norm_across_channel_pack8, opt, specializations);
        }
        if (region_type == 1 && ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8))
        {
            pipeline_lrn_norm_within_channel_pack8 = new Pipeline(vkdev);
            pipeline_lrn_norm_within_channel_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_lrn_norm_within_channel_pack8->create(LayerShaderType::lrn_norm_within_channel_pack8, opt, specializations);
        }
    }

    return 0;
}

int LRN_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_lrn_square_pad;
    pipeline_lrn_square_pad = 0;

    delete pipeline_lrn_norm;
    pipeline_lrn_norm = 0;

    delete pipeline_lrn_square_pad_across_channel_pack4;
    pipeline_lrn_square_pad_across_channel_pack4 = 0;

    delete pipeline_lrn_norm_across_channel_pack4;
    pipeline_lrn_norm_across_channel_pack4 = 0;

    delete pipeline_lrn_square_pad_within_channel_pack4;
    pipeline_lrn_square_pad_within_channel_pack4 = 0;

    delete pipeline_lrn_norm_within_channel_pack4;
    pipeline_lrn_norm_within_channel_pack4 = 0;

    delete pipeline_lrn_square_pad_across_channel_pack8;
    pipeline_lrn_square_pad_across_channel_pack8 = 0;

    delete pipeline_lrn_norm_across_channel_pack8;
    pipeline_lrn_norm_across_channel_pack8 = 0;

    delete pipeline_lrn_square_pad_within_channel_pack8;
    pipeline_lrn_square_pad_within_channel_pack8 = 0;

    delete pipeline_lrn_norm_within_channel_pack8;
    pipeline_lrn_norm_within_channel_pack8 = 0;

    return 0;
}

int LRN_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    VkMat square_workspace;

    if (region_type == NormRegion_ACROSS_CHANNELS)
    {
        // always create scalar square workspace blob for norm across channel
        square_workspace.create(w, h, channels * elempack + local_size - 1, 4u, 1, opt.workspace_vkallocator);
    }
    else if (region_type == NormRegion_WITHIN_CHANNEL)
    {
        square_workspace.create(w + local_size - 1, h + local_size - 1, channels, elempack * 4u, elempack, opt.workspace_vkallocator);
    }

    // square pad
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = square_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = bottom_top_blob.cstep;
        constants[5].i = square_workspace.dims;
        constants[6].i = square_workspace.w;
        constants[7].i = square_workspace.h;
        constants[8].i = square_workspace.c;
        constants[9].i = square_workspace.cstep;

        const Pipeline* pipeline = 0;
        if (elempack == 8)
        {
            if (region_type == 0) pipeline = pipeline_lrn_square_pad_across_channel_pack8;
            if (region_type == 1) pipeline = pipeline_lrn_square_pad_within_channel_pack8;
        }
        else if (elempack == 4)
        {
            if (region_type == 0) pipeline = pipeline_lrn_square_pad_across_channel_pack4;
            if (region_type == 1) pipeline = pipeline_lrn_square_pad_within_channel_pack4;
        }
        else
        {
            pipeline = pipeline_lrn_square_pad;
        }

        cmd.record_pipeline(pipeline, bindings, constants, square_workspace);
    }

    // norm
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = square_workspace;
        bindings[1] = bottom_top_blob;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = square_workspace.dims;
        constants[1].i = square_workspace.w;
        constants[2].i = square_workspace.h;
        constants[3].i = square_workspace.c;
        constants[4].i = square_workspace.cstep;
        constants[5].i = bottom_top_blob.dims;
        constants[6].i = bottom_top_blob.w;
        constants[7].i = bottom_top_blob.h;
        constants[8].i = bottom_top_blob.c;
        constants[9].i = bottom_top_blob.cstep;

        const Pipeline* pipeline = 0;
        if (elempack == 8)
        {
            if (region_type == 0) pipeline = pipeline_lrn_norm_across_channel_pack8;
            if (region_type == 1) pipeline = pipeline_lrn_norm_within_channel_pack8;
        }
        else if (elempack == 4)
        {
            if (region_type == 0) pipeline = pipeline_lrn_norm_across_channel_pack4;
            if (region_type == 1) pipeline = pipeline_lrn_norm_within_channel_pack4;
        }
        else
        {
            pipeline = pipeline_lrn_norm;
        }

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

int LRN_vulkan::forward_inplace(VkImageMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    VkImageMat square_workspace;

    if (region_type == NormRegion_ACROSS_CHANNELS)
    {
        // always create scalar square workspace blob for norm across channel
        square_workspace.create(w, h, channels * elempack + local_size - 1, 4u, 1, opt.workspace_vkallocator);
    }
    else if (region_type == NormRegion_WITHIN_CHANNEL)
    {
        square_workspace.create(w + local_size - 1, h + local_size - 1, channels, elempack * 4u, elempack, opt.workspace_vkallocator);
    }

    // square pad
    {
        std::vector<VkImageMat> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = square_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = 0; //bottom_top_blob.cstep;
        constants[5].i = square_workspace.dims;
        constants[6].i = square_workspace.w;
        constants[7].i = square_workspace.h;
        constants[8].i = square_workspace.c;
        constants[9].i = 0; //square_workspace.cstep;

        const Pipeline* pipeline = 0;
        if (elempack == 8)
        {
            if (region_type == 0) pipeline = pipeline_lrn_square_pad_across_channel_pack8;
            if (region_type == 1) pipeline = pipeline_lrn_square_pad_within_channel_pack8;
        }
        else if (elempack == 4)
        {
            if (region_type == 0) pipeline = pipeline_lrn_square_pad_across_channel_pack4;
            if (region_type == 1) pipeline = pipeline_lrn_square_pad_within_channel_pack4;
        }
        else
        {
            pipeline = pipeline_lrn_square_pad;
        }

        cmd.record_pipeline(pipeline, bindings, constants, square_workspace);
    }

    // norm
    {
        std::vector<VkImageMat> bindings(3);
        bindings[0] = square_workspace;
        bindings[1] = bottom_top_blob;
        bindings[2] = bottom_top_blob;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = square_workspace.dims;
        constants[1].i = square_workspace.w;
        constants[2].i = square_workspace.h;
        constants[3].i = square_workspace.c;
        constants[4].i = 0; //square_workspace.cstep;
        constants[5].i = bottom_top_blob.dims;
        constants[6].i = bottom_top_blob.w;
        constants[7].i = bottom_top_blob.h;
        constants[8].i = bottom_top_blob.c;
        constants[9].i = 0; //bottom_top_blob.cstep;

        const Pipeline* pipeline = 0;
        if (elempack == 8)
        {
            if (region_type == 0) pipeline = pipeline_lrn_norm_across_channel_pack8;
            if (region_type == 1) pipeline = pipeline_lrn_norm_within_channel_pack8;
        }
        else if (elempack == 4)
        {
            if (region_type == 0) pipeline = pipeline_lrn_norm_across_channel_pack4;
            if (region_type == 1) pipeline = pipeline_lrn_norm_within_channel_pack4;
        }
        else
        {
            pipeline = pipeline_lrn_norm;
        }

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

} // namespace ncnn
