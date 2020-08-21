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

#include "reorg_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Reorg_vulkan::Reorg_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    pipeline_reorg = 0;
    pipeline_reorg_pack4 = 0;
    pipeline_reorg_pack1to4 = 0;
    pipeline_reorg_pack8 = 0;
    pipeline_reorg_pack1to8 = 0;
    pipeline_reorg_pack4to8 = 0;
}

int Reorg_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = opt.use_shader_pack8 && shape.w % 8 == 0 ? 8 : shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = opt.use_shader_pack8 && out_shape.w % 8 == 0 ? 8 : out_shape.w % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 2) out_elempack = opt.use_shader_pack8 && out_shape.h % 8 == 0 ? 8 : out_shape.h % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 3) out_elempack = opt.use_shader_pack8 && out_shape.c % 8 == 0 ? 8 : out_shape.c % 4 == 0 ? 4 : 1;

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

    Mat shape_packed;
    if (shape.dims == 1) shape_packed = Mat(shape.w / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    // check blob shape
    if (!vkdev->shape_support_image_storage(shape_packed) || !vkdev->shape_support_image_storage(out_shape_packed))
    {
        support_image_storage = false;
        opt.use_image_storage = false;
    }

    std::vector<vk_specialization_type> specializations(1 + 10);
    specializations[0].i = stride;
    specializations[1 + 0].i = shape_packed.dims;
    specializations[1 + 1].i = shape_packed.w;
    specializations[1 + 2].i = shape_packed.h;
    specializations[1 + 3].i = shape_packed.c;
    specializations[1 + 4].i = shape_packed.cstep;
    specializations[1 + 5].i = out_shape_packed.dims;
    specializations[1 + 6].i = out_shape_packed.w;
    specializations[1 + 7].i = out_shape_packed.h;
    specializations[1 + 8].i = out_shape_packed.c;
    specializations[1 + 9].i = out_shape_packed.cstep;

    Mat local_size_xyz;
    if (out_shape_packed.dims != 0)
    {
        local_size_xyz.w = std::min(4, out_shape_packed.w);
        local_size_xyz.h = std::min(4, out_shape_packed.h);
        local_size_xyz.c = std::min(4, out_shape_packed.c);
    }

    // pack1
    if (shape.dims == 0 || (elempack == 1 && out_elempack == 1))
    {
        pipeline_reorg = new Pipeline(vkdev);
        pipeline_reorg->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reorg->create(LayerShaderType::reorg, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || (elempack == 4 && out_elempack == 4))
    {
        pipeline_reorg_pack4 = new Pipeline(vkdev);
        pipeline_reorg_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reorg_pack4->create(LayerShaderType::reorg_pack4, opt, specializations);
    }

    // pack1to4
    if (shape.dims == 0 || (elempack == 1 && out_elempack == 4))
    {
        pipeline_reorg_pack1to4 = new Pipeline(vkdev);
        pipeline_reorg_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reorg_pack1to4->create(LayerShaderType::reorg_pack1to4, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && shape.dims == 0) || (elempack == 8 && out_elempack == 8))
    {
        pipeline_reorg_pack8 = new Pipeline(vkdev);
        pipeline_reorg_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reorg_pack8->create(LayerShaderType::reorg_pack8, opt, specializations);
    }

    // pack1to8
    if ((opt.use_shader_pack8 && shape.dims == 0) || (elempack == 1 && out_elempack == 8))
    {
        pipeline_reorg_pack1to8 = new Pipeline(vkdev);
        pipeline_reorg_pack1to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reorg_pack1to8->create(LayerShaderType::reorg_pack1to8, opt, specializations);
    }

    // pack4to8
    if ((opt.use_shader_pack8 && shape.dims == 0) || (elempack == 4 && out_elempack == 8))
    {
        pipeline_reorg_pack4to8 = new Pipeline(vkdev);
        pipeline_reorg_pack4to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reorg_pack4to8->create(LayerShaderType::reorg_pack4to8, opt, specializations);
    }

    return 0;
}

int Reorg_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_reorg;
    pipeline_reorg = 0;

    delete pipeline_reorg_pack4;
    pipeline_reorg_pack4 = 0;

    delete pipeline_reorg_pack1to4;
    pipeline_reorg_pack1to4 = 0;

    delete pipeline_reorg_pack8;
    pipeline_reorg_pack8 = 0;

    delete pipeline_reorg_pack1to8;
    pipeline_reorg_pack1to8 = 0;

    delete pipeline_reorg_pack4to8;
    pipeline_reorg_pack4to8 = 0;

    return 0;
}

int Reorg_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = w / stride;
    int outh = h / stride;
    int outc = channels * elempack * stride * stride;

    int out_elempack = opt.use_shader_pack8 && outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_reorg;
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_reorg_pack4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_reorg_pack1to4;
    }
    else if (elempack == 8) // assert out_elempack == 8
    {
        pipeline = pipeline_reorg_pack8;
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_reorg_pack1to8;
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_reorg_pack4to8;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int Reorg_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = w / stride;
    int outh = h / stride;
    int outc = channels * elempack * stride * stride;

    int out_elempack = opt.use_shader_pack8 && outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkImageMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = 0; //bottom_blob.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = 0; //top_blob.cstep;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_reorg;
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_reorg_pack4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_reorg_pack1to4;
    }
    else if (elempack == 8) // assert out_elempack == 8
    {
        pipeline = pipeline_reorg_pack8;
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_reorg_pack1to8;
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_reorg_pack4to8;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn
