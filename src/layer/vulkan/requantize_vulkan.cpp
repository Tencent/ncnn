// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "requantize_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Requantize_vulkan::Requantize_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_requantize = 0;
    pipeline_requantize_pack4 = 0;
}

int Requantize_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    const int dims = shape.dims;

    int elempack = 1;
    if (dims == 1) elempack = shape.w % 4 == 0 ? 4 : 1;
    if (dims == 2) elempack = shape.h % 4 == 0 ? 4 : 1;
    if (dims == 3 || dims == 4) elempack = shape.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
    if (dims == 1) out_elempack = out_shape.w % 4 == 0 ? 4 : 1;
    if (dims == 2) out_elempack = out_shape.h % 4 == 0 ? 4 : 1;
    if (dims == 3 || dims == 4) out_elempack = out_shape.c % 4 == 0 ? 4 : 1;

    const size_t elemsize = elempack * 4u;
    const size_t out_elemsize = out_elempack * 1u;

    Mat shape_packed;
    if (dims == 1) shape_packed = Mat(shape.w / elempack, (void*)0, elemsize, elempack);
    if (dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);
    if (dims == 4) shape_packed = Mat(shape.w, shape.h, shape.d, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (dims == 4) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.d, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

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

    std::vector<vk_specialization_type> specializations(9 + 3);
    specializations[0].i = scale_in_data_size;
    specializations[1].f = scale_in_data_size == 1 ? scale_in_data[0] : 1.f;
    specializations[2].i = scale_out_data_size;
    specializations[3].f = scale_out_data_size == 1 ? scale_out_data[0] : 1.f;
    specializations[4].i = bias_data_size;
    specializations[5].f = bias_data_size == 1 ? bias_data[0] : 0.f;
    specializations[6].i = activation_type;
    specializations[7].f = activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[8].f = activation_params.w == 2 ? activation_params[1] : 0.f;
    specializations[9 + 0].u32 = c;
    specializations[9 + 1].u32 = in_stride;
    specializations[9 + 2].u32 = out_stride;

    const int local_size_x = vkdev->info.subgroup_size();

    // pack1
    if (shape.dims == 0 || elempack == 1)
    {
        pipeline_requantize = new Pipeline(vkdev);
        pipeline_requantize->set_optimal_local_size_xyz(local_size_x, 1, 1);
        pipeline_requantize->create(LayerShaderType::requantize, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || elempack == 4)
    {
        pipeline_requantize_pack4 = new Pipeline(vkdev);
        pipeline_requantize_pack4->set_optimal_local_size_xyz(local_size_x, 1, 1);
        pipeline_requantize_pack4->create(LayerShaderType::requantize_pack4, opt, specializations);
    }

    return 0;
}

int Requantize_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_requantize;
    pipeline_requantize = 0;

    delete pipeline_requantize_pack4;
    pipeline_requantize_pack4 = 0;

    return 0;
}

int Requantize_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (scale_in_data_size > 1)
    {
        cmd.record_upload(scale_in_data, scale_in_data_gpu, opt);
    }

    if (scale_out_data_size > 1)
    {
        cmd.record_upload(scale_out_data, scale_out_data_gpu, opt);
    }

    if (bias_data_size > 1)
    {
        cmd.record_upload(bias_data, bias_data_gpu, opt);
    }

    return 0;
}

int Requantize_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int channels = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    size_t out_elemsize = 1u * elempack;

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

    std::vector<VkMat> bindings(5);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;
    bindings[2] = scale_in_data_gpu;
    bindings[3] = scale_out_data_gpu;
    bindings[4] = bias_data_gpu;

    std::vector<vk_constant_type> constants(3);
    constants[0].u32 = c;
    constants[1].u32 = in_stride;
    constants[2].u32 = out_stride;

    VkMat dispatcher;
    dispatcher.w = in_stride * c;
    dispatcher.h = 1;
    dispatcher.c = 1;

    const Pipeline* pipeline = elempack == 4 ? pipeline_requantize_pack4 : pipeline_requantize;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
