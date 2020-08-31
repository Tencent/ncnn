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

#include "unaryop_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

UnaryOp_vulkan::UnaryOp_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    pipeline_unaryop = 0;
    pipeline_unaryop_pack4 = 0;
    pipeline_unaryop_pack8 = 0;
}

int UnaryOp_vulkan::create_pipeline(const Option& opt)
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

    std::vector<vk_specialization_type> specializations(1 + 5);
    specializations[0].i = op_type;
    specializations[1 + 0].i = shape_packed.dims;
    specializations[1 + 1].i = shape_packed.w;
    specializations[1 + 2].i = shape_packed.h;
    specializations[1 + 3].i = shape_packed.c;
    specializations[1 + 4].i = shape_packed.cstep;

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
    if (shape.dims == 0 || elempack == 1)
    {
        pipeline_unaryop = new Pipeline(vkdev);
        pipeline_unaryop->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_unaryop->create(LayerShaderType::unaryop, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || elempack == 4)
    {
        pipeline_unaryop_pack4 = new Pipeline(vkdev);
        pipeline_unaryop_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_unaryop_pack4->create(LayerShaderType::unaryop_pack4, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
    {
        pipeline_unaryop_pack8 = new Pipeline(vkdev);
        pipeline_unaryop_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_unaryop_pack8->create(LayerShaderType::unaryop_pack8, opt, specializations);
    }

    return 0;
}

int UnaryOp_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_unaryop;
    pipeline_unaryop = 0;

    delete pipeline_unaryop_pack4;
    pipeline_unaryop_pack4 = 0;

    delete pipeline_unaryop_pack8;
    pipeline_unaryop_pack8 = 0;

    return 0;
}

int UnaryOp_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& /*opt*/) const
{
    int elempack = bottom_top_blob.elempack;

    std::vector<VkMat> bindings(1);
    bindings[0] = bottom_top_blob;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = elempack == 8 ? pipeline_unaryop_pack8
                               : elempack == 4 ? pipeline_unaryop_pack4
                               : pipeline_unaryop;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

int UnaryOp_vulkan::forward_inplace(VkImageMat& bottom_top_blob, VkCompute& cmd, const Option& /*opt*/) const
{
    int elempack = bottom_top_blob.elempack;

    std::vector<VkImageMat> bindings(2);
    bindings[0] = bottom_top_blob;
    bindings[1] = bottom_top_blob;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = 0; //bottom_top_blob.cstep;

    const Pipeline* pipeline = elempack == 8 ? pipeline_unaryop_pack8
                               : elempack == 4 ? pipeline_unaryop_pack4
                               : pipeline_unaryop;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

} // namespace ncnn
