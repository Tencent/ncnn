// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "dequantize_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Dequantize_vulkan::Dequantize_vulkan()
{
    support_vulkan = true;

    pipeline_dequantize = 0;
    pipeline_dequantize_pack4 = 0;
    pipeline_dequantize_pack8 = 0;
}

int Dequantize_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    const int dims = shape.dims;

    int elempack = 1;
    if (dims == 1) elempack = opt.use_shader_pack8 && shape.w % 8 == 0 ? 8 : shape.w % 4 == 0 ? 4 : 1;
    if (dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4 : 1;
    if (dims == 3 || dims == 4) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;

    const size_t elemsize = elempack * 4u;
    size_t out_elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed)
    {
        out_elemsize = elempack * 2u;
    }
    else
    {
        out_elemsize = elempack * 4u;
    }

    Mat shape_packed;
    if (dims == 1) shape_packed = Mat(shape.w / elempack, (void*)0, elemsize, elempack);
    if (dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);
    if (dims == 4) shape_packed = Mat(shape.w, shape.h, shape.d, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (dims == 1) out_shape_packed = Mat(out_shape.w / elempack, (void*)0, out_elemsize, elempack);
    if (dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / elempack, (void*)0, out_elemsize, elempack);
    if (dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / elempack, (void*)0, out_elemsize, elempack);
    if (dims == 4) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.d, out_shape.c / elempack, (void*)0, out_elemsize, elempack);

    size_t c = 0;
    size_t in_stride = 0;
    size_t out_stride = 0;
    if (dims == 1)
    {
        c = 1;
        in_stride = shape_packed.w;
        out_stride = out_shape_packed.w;
    }
    if (dims == 2)
    {
        c = shape_packed.h;
        in_stride = shape_packed.w;
        out_stride = out_shape_packed.w;
    }
    if (dims == 3 || dims == 4)
    {
        c = shape_packed.c;
        in_stride = shape_packed.cstep;
        out_stride = out_shape_packed.cstep;
    }

    std::vector<vk_specialization_type> specializations(4 + 3);
    specializations[0].i = scale_data_size;
    specializations[1].f = scale_data_size == 1 ? scale_data[0] : 1.f;
    specializations[2].i = bias_data_size;
    specializations[3].f = bias_data_size == 1 ? bias_data[0] : 0.f;
    specializations[4 + 0].u32 = c;
    specializations[4 + 1].u32 = in_stride;
    specializations[4 + 2].u32 = out_stride;

    const int local_size_x = vkdev->info.subgroup_size();

    // pack1
    if (shape.dims == 0 || elempack == 1)
    {
        pipeline_dequantize = new Pipeline(vkdev);
        pipeline_dequantize->set_optimal_local_size_xyz(local_size_x, 1, 1);
        pipeline_dequantize->create(LayerShaderType::dequantize, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || elempack == 4)
    {
        pipeline_dequantize_pack4 = new Pipeline(vkdev);
        pipeline_dequantize_pack4->set_optimal_local_size_xyz(local_size_x, 1, 1);
        pipeline_dequantize_pack4->create(LayerShaderType::dequantize_pack4, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
    {
        pipeline_dequantize_pack8 = new Pipeline(vkdev);
        pipeline_dequantize_pack8->set_optimal_local_size_xyz(local_size_x, 1, 1);
        pipeline_dequantize_pack8->create(LayerShaderType::dequantize_pack8, opt, specializations);
    }

    return 0;
}

int Dequantize_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_dequantize;
    pipeline_dequantize = 0;

    delete pipeline_dequantize_pack4;
    pipeline_dequantize_pack4 = 0;

    delete pipeline_dequantize_pack8;
    pipeline_dequantize_pack8 = 0;

    return 0;
}

int Dequantize_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (scale_data_size > 1)
    {
        cmd.record_upload(scale_data, scale_data_gpu, opt);
    }

    if (bias_data_size > 1)
    {
        cmd.record_upload(bias_data, bias_data_gpu, opt);
    }

    return 0;
}

int Dequantize_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    size_t out_elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed)
    {
        out_elemsize = elempack * 2u;
    }
    else
    {
        out_elemsize = elempack * 4u;
    }

    if (dims == 1)
        top_blob.create(w, out_elemsize, elempack, opt.blob_vkallocator);
    if (dims == 2)
        top_blob.create(w, h, out_elemsize, elempack, opt.blob_vkallocator);
    if (dims == 3)
        top_blob.create(w, h, channels, out_elemsize, elempack, opt.blob_vkallocator);
    if (dims == 4)
        top_blob.create(w, h, d, channels, out_elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    size_t c = 0;
    size_t in_stride = 0;
    size_t out_stride = 0;
    if (dims == 1)
    {
        c = 1;
        in_stride = bottom_blob.w;
        out_stride = top_blob.w;
    }
    if (dims == 2)
    {
        c = bottom_blob.h;
        in_stride = bottom_blob.w;
        out_stride = top_blob.w;
    }
    if (dims == 3 || dims == 4)
    {
        c = bottom_blob.c;
        in_stride = bottom_blob.cstep;
        out_stride = top_blob.cstep;
    }

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;
    bindings[2] = scale_data_gpu;
    bindings[3] = bias_data_gpu;

    std::vector<vk_constant_type> constants(3);
    constants[0].u32 = c;
    constants[1].u32 = in_stride;
    constants[2].u32 = out_stride;

    VkMat dispatcher;
    dispatcher.w = in_stride * c;
    dispatcher.h = 1;
    dispatcher.c = 1;

    const Pipeline* pipeline = elempack == 8 ? pipeline_dequantize_pack8
                               : elempack == 4 ? pipeline_dequantize_pack4
                               : pipeline_dequantize;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
