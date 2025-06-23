// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "quantize_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Quantize_vulkan::Quantize_vulkan()
{
    support_vulkan = true;

    pipeline_quantize = 0;
    pipeline_quantize_pack4 = 0;
    pipeline_quantize_pack8 = 0;
}

int Quantize_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = opt.use_shader_pack8 && shape.w % 8 == 0 ? 8 : shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3 || shape.dims == 4) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = opt.use_shader_pack8 && out_shape.w % 8 == 0 ? 8 : out_shape.w % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 2) out_elempack = opt.use_shader_pack8 && out_shape.h % 8 == 0 ? 8 : out_shape.h % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 3 || out_shape.dims == 4) out_elempack = opt.use_shader_pack8 && out_shape.c % 8 == 0 ? 8 : out_shape.c % 4 == 0 ? 4 : 1;

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

    size_t out_elemsize;
    if (opt.use_int8_storage)
    {
        out_elemsize = out_elempack * 1u;
    }
    else if (opt.use_int8_packed)
    {
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 1u;
    }
    else
    {
        out_elemsize = out_elempack * 4u;
    }

    Mat shape_packed;
    if (shape.dims == 1) shape_packed = Mat(shape.w / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 4) shape_packed = Mat(shape.w, shape.h, shape.d, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 4) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.d, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    std::vector<vk_specialization_type> specializations(2 + 10);
    specializations[0].i = scale_data_size;
    specializations[1].f = scale_data_size == 1 ? scale_data[0] : 1.f;
    specializations[2 + 0].i = shape_packed.dims;
    specializations[2 + 1].i = shape_packed.w;
    specializations[2 + 2].i = shape_packed.h * shape_packed.d;
    specializations[2 + 3].i = shape_packed.c;
    specializations[2 + 4].i = shape_packed.cstep;
    specializations[2 + 5].i = out_shape_packed.dims;
    specializations[2 + 6].i = out_shape_packed.w;
    specializations[2 + 7].i = out_shape_packed.h * out_shape_packed.d;
    specializations[2 + 8].i = out_shape_packed.c;
    specializations[2 + 9].i = out_shape_packed.cstep;

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
    if (shape_packed.dims == 4)
    {
        local_size_xyz.w = std::min(4, shape_packed.w);
        local_size_xyz.h = std::min(4, shape_packed.h * shape_packed.d);
        local_size_xyz.c = std::min(4, shape_packed.c);
    }

    // pack1
    if (shape.dims == 0 || elempack == 1)
    {
        pipeline_quantize = new Pipeline(vkdev);
        pipeline_quantize->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_quantize->create(LayerShaderType::quantize, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || elempack == 4)
    {
        pipeline_quantize_pack4 = new Pipeline(vkdev);
        pipeline_quantize_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_quantize_pack4->create(LayerShaderType::quantize_pack4, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
    {
        pipeline_quantize_pack8 = new Pipeline(vkdev);
        pipeline_quantize_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_quantize_pack8->create(LayerShaderType::quantize_pack8, opt, specializations);
    }

    return 0;
}

int Quantize_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_quantize;
    pipeline_quantize = 0;

    delete pipeline_quantize_pack4;
    pipeline_quantize_pack4 = 0;

    delete pipeline_quantize_pack8;
    pipeline_quantize_pack8 = 0;

    return 0;
}

int Quantize_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (scale_data_size > 1)
    {
        int elempack = opt.use_shader_pack8 && scale_data_size % 8 == 0 ? 8 : scale_data_size % 4 == 0 ? 4 : 1;

        Mat scale_data_packed;
        convert_packing(scale_data, scale_data_packed, elempack, opt);

        cmd.record_upload(scale_data_packed, scale_data_gpu, opt);
    }

    return 0;
}

int Quantize_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blob.dims;
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int elempack = bottom_blob.elempack;

    size_t out_elemsize = 1u * elempack;

    if (opt.use_int8_storage)
    {
        out_elemsize = 1u * elempack;
    }
    else if (opt.use_int8_packed)
    {
        if (elempack == 8) out_elemsize = 8 * 1u;
        if (elempack == 4) out_elemsize = 4 * 1u;
        if (elempack == 1) out_elemsize = 4u;
    }
    else
    {
        out_elemsize = 4u * elempack;
    }

    if (dims == 1)
        top_blob.create(w, out_elemsize, elempack, opt.blob_vkallocator);
    if (dims == 2)
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_vkallocator);
    if (dims == 3)
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;
    bindings[2] = scale_data_gpu;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h * bottom_blob.d;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h * top_blob.d;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    const Pipeline* pipeline = elempack == 8 ? pipeline_quantize_pack8
                               : elempack == 4 ? pipeline_quantize_pack4
                               : pipeline_quantize;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_blob);

    return 0;
}

} // namespace ncnn
