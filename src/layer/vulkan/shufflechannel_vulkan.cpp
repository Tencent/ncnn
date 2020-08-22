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

#include "shufflechannel_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

ShuffleChannel_vulkan::ShuffleChannel_vulkan()
{
    support_vulkan = true;
    support_image_storage = true;

    pipeline_shufflechannel = 0;
    pipeline_shufflechannel_pack4 = 0;
    pipeline_shufflechannel_pack8 = 0;
}

int ShuffleChannel_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 3) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
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
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    std::vector<vk_specialization_type> specializations(2 + 10);
    specializations[0].i = reverse ? shape.c / group : group;
    specializations[1].i = vkdev->info.bug_implicit_fp16_arithmetic;
    specializations[2 + 0].i = shape_packed.dims;
    specializations[2 + 1].i = shape_packed.w;
    specializations[2 + 2].i = shape_packed.h;
    specializations[2 + 3].i = shape_packed.c;
    specializations[2 + 4].i = shape_packed.cstep;
    specializations[2 + 5].i = out_shape_packed.dims;
    specializations[2 + 6].i = out_shape_packed.w;
    specializations[2 + 7].i = out_shape_packed.h;
    specializations[2 + 8].i = out_shape_packed.c;
    specializations[2 + 9].i = out_shape_packed.cstep;

    Mat local_size_xyz;
    if (out_shape_packed.dims != 0)
    {
        local_size_xyz.w = std::min(4, out_shape_packed.w);
        local_size_xyz.h = std::min(4, out_shape_packed.h);
        local_size_xyz.c = std::min(4, out_shape_packed.c);
    }

    // pack1
    if (shape.dims == 0 || elempack == 1)
    {
        pipeline_shufflechannel = new Pipeline(vkdev);
        pipeline_shufflechannel->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_shufflechannel->create(LayerShaderType::shufflechannel, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || elempack == 4)
    {
        pipeline_shufflechannel_pack4 = new Pipeline(vkdev);
        pipeline_shufflechannel_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_shufflechannel_pack4->create(LayerShaderType::shufflechannel_pack4, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
    {
        pipeline_shufflechannel_pack8 = new Pipeline(vkdev);
        pipeline_shufflechannel_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_shufflechannel_pack8->create(LayerShaderType::shufflechannel_pack8, opt, specializations);
    }

    return 0;
}

int ShuffleChannel_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_shufflechannel;
    pipeline_shufflechannel = 0;

    delete pipeline_shufflechannel_pack4;
    pipeline_shufflechannel_pack4 = 0;

    delete pipeline_shufflechannel_pack8;
    pipeline_shufflechannel_pack8 = 0;

    return 0;
}

int ShuffleChannel_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    top_blob.create(w, h, channels, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(11);
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
    constants[10].i = reverse ? channels * elempack / group : group;

    const Pipeline* pipeline = elempack == 8 ? pipeline_shufflechannel_pack8
                               : elempack == 4 ? pipeline_shufflechannel_pack4
                               : pipeline_shufflechannel;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

int ShuffleChannel_vulkan::forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    top_blob.create(w, h, channels, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkImageMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(11);
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
    constants[10].i = reverse ? channels * elempack / group : group;

    const Pipeline* pipeline = elempack == 8 ? pipeline_shufflechannel_pack8
                               : elempack == 4 ? pipeline_shufflechannel_pack4
                               : pipeline_shufflechannel;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn
