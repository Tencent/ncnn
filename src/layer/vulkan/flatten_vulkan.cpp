// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "flatten_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Flatten_vulkan::Flatten_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_flatten = 0;
    pipeline_flatten_pack4 = 0;
    pipeline_flatten_pack1to4 = 0;
}

int Flatten_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3 || shape.dims == 4) elempack = shape.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = out_shape.w % 4 == 0 ? 4 : 1;

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
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
    if (shape.dims == 4) shape_packed = Mat(shape.w, shape.h, shape.d, shape.c / elempack, (void*)0, elemsize, elempack);

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);

    std::vector<vk_specialization_type> specializations(0 + 10);
    specializations[0 + 0].i = std::min(3, shape_packed.dims);
    specializations[0 + 1].i = shape_packed.w;
    specializations[0 + 2].i = shape_packed.h * shape_packed.d;
    specializations[0 + 3].i = shape_packed.c;
    specializations[0 + 4].i = shape_packed.cstep;
    specializations[0 + 5].i = std::min(3, out_shape_packed.dims);
    specializations[0 + 6].i = out_shape_packed.w;
    specializations[0 + 7].i = out_shape_packed.h * out_shape_packed.d;
    specializations[0 + 8].i = out_shape_packed.c;
    specializations[0 + 9].i = out_shape_packed.cstep;

    Mat local_size_xyz(64, 1, 1, (void*)0);
    if (out_shape_packed.dims != 0)
    {
        local_size_xyz.w = std::min(64, out_shape_packed.w);
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }

    // pack1
    if (shape.dims == 0 || (elempack == 1 && out_elempack == 1))
    {
        pipeline_flatten = new Pipeline(vkdev);
        pipeline_flatten->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_flatten->create(LayerShaderType::flatten, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || (elempack == 4 && out_elempack == 4))
    {
        pipeline_flatten_pack4 = new Pipeline(vkdev);
        pipeline_flatten_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_flatten_pack4->create(LayerShaderType::flatten_pack4, opt, specializations);
    }

    // pack1to4
    if (shape.dims == 0 || (elempack == 1 && out_elempack == 4))
    {
        pipeline_flatten_pack1to4 = new Pipeline(vkdev);
        pipeline_flatten_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_flatten_pack1to4->create(LayerShaderType::flatten_pack1to4, opt, specializations);
    }

    return 0;
}

int Flatten_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_flatten;
    pipeline_flatten = 0;

    delete pipeline_flatten_pack4;
    pipeline_flatten_pack4 = 0;

    delete pipeline_flatten_pack1to4;
    pipeline_flatten_pack1to4 = 0;

    return 0;
}

int Flatten_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int total = w * h * d * channels * elempack;

    int out_elempack = total % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (dims == 2 && elempack == 1)
    {
        top_blob = bottom_blob;
        top_blob.dims = 1;
        top_blob.w = total / out_elempack;
        top_blob.h = 1;
        top_blob.cstep = bottom_blob.cstep * elempack / out_elempack;
        top_blob.elemsize = out_elemsize;
        top_blob.elempack = out_elempack;
        return 0;
    }

    top_blob.create(total / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = std::min(3, bottom_blob.dims);
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h * bottom_blob.d;
    constants[3].i = bottom_blob.c;
    constants[4].i = bottom_blob.cstep;
    constants[5].i = std::min(3, top_blob.dims);
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h * top_blob.d;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_flatten;
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_flatten_pack4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_flatten_pack1to4;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn
