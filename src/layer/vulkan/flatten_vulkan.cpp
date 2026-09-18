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
#if NCNN_INT8
    pipeline_flatten_int8 = 0;
    pipeline_flatten_pack4_int8 = 0;
    pipeline_flatten_pack1to4_int8 = 0;
#endif
}

int Flatten_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    std::vector<vk_specialization_type> specializations(0 + 10);
    specializations[0 + 0].i = std::min(3, shape.dims);
    specializations[0 + 1].i = shape.w;
    specializations[0 + 2].i = shape.h * shape.d;
    specializations[0 + 3].i = shape.c;
    specializations[0 + 4].i = shape.cstep;
    specializations[0 + 5].i = std::min(3, out_shape.dims);
    specializations[0 + 6].i = out_shape.w;
    specializations[0 + 7].i = out_shape.h * out_shape.d;
    specializations[0 + 8].i = out_shape.c;
    specializations[0 + 9].i = out_shape.cstep;

    Mat local_size_xyz(64, 1, 1, (void*)0);
    if (out_shape.dims != 0)
    {
        local_size_xyz.w = std::min(64, out_shape.w);
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }

#if NCNN_INT8
    Mat shape_int8;
    if (shape.dims == 1) shape_int8 = Mat(shape.w, (void*)0, (size_t)shape.elempack, shape.elempack);
    if (shape.dims == 2) shape_int8 = Mat(shape.w, shape.h, (void*)0, (size_t)shape.elempack, shape.elempack);
    if (shape.dims == 3) shape_int8 = Mat(shape.w, shape.h, shape.c, (void*)0, (size_t)shape.elempack, shape.elempack);
    if (shape.dims == 4) shape_int8 = Mat(shape.w, shape.h, shape.d, shape.c, (void*)0, (size_t)shape.elempack, shape.elempack);

    Mat out_shape_int8;
    if (out_shape.dims == 1) out_shape_int8 = Mat(out_shape.w, (void*)0, (size_t)out_shape.elempack, out_shape.elempack);
    if (out_shape.dims == 2) out_shape_int8 = Mat(out_shape.w, out_shape.h, (void*)0, (size_t)out_shape.elempack, out_shape.elempack);
    if (out_shape.dims == 3) out_shape_int8 = Mat(out_shape.w, out_shape.h, out_shape.c, (void*)0, (size_t)out_shape.elempack, out_shape.elempack);
    if (out_shape.dims == 4) out_shape_int8 = Mat(out_shape.w, out_shape.h, out_shape.d, out_shape.c, (void*)0, (size_t)out_shape.elempack, out_shape.elempack);

    std::vector<vk_specialization_type> specializations_int8 = specializations;
    specializations_int8[0 + 0].i = std::min(3, shape_int8.dims);
    specializations_int8[0 + 1].i = shape_int8.w;
    specializations_int8[0 + 2].i = shape_int8.h * shape_int8.d;
    specializations_int8[0 + 3].i = shape_int8.c;
    specializations_int8[0 + 4].i = shape_int8.cstep;
    specializations_int8[0 + 5].i = std::min(3, out_shape_int8.dims);
    specializations_int8[0 + 6].i = out_shape_int8.w;
    specializations_int8[0 + 7].i = out_shape_int8.h * out_shape_int8.d;
    specializations_int8[0 + 8].i = out_shape_int8.c;
    specializations_int8[0 + 9].i = out_shape_int8.cstep;

    const bool use_int8_pipeline = opt.use_int8_packed || opt.use_int8_storage;
#endif

    // pack1
    if (shape.dims == 0 || (shape.elempack == 1 && out_shape.elempack == 1))
    {
        pipeline_flatten = new Pipeline(vkdev);
        pipeline_flatten->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_flatten->create(LayerShaderType::flatten, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || (shape.elempack == 4 && out_shape.elempack == 4))
    {
        pipeline_flatten_pack4 = new Pipeline(vkdev);
        pipeline_flatten_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_flatten_pack4->create(LayerShaderType::flatten_pack4, opt, specializations);
    }

    // pack1to4
    if (shape.dims == 0 || (shape.elempack == 1 && out_shape.elempack == 4))
    {
        pipeline_flatten_pack1to4 = new Pipeline(vkdev);
        pipeline_flatten_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_flatten_pack1to4->create(LayerShaderType::flatten_pack1to4, opt, specializations);
    }

#if NCNN_INT8
    if (use_int8_pipeline)
    {
        if (shape.dims == 0 || (shape.elempack == 1 && out_shape.elempack == 1))
        {
            pipeline_flatten_int8 = new Pipeline(vkdev);
            pipeline_flatten_int8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_flatten_int8->create(LayerShaderType::flatten_int8, opt, specializations_int8);
        }

        if (shape.dims == 0 || (shape.elempack == 4 && out_shape.elempack == 4))
        {
            pipeline_flatten_pack4_int8 = new Pipeline(vkdev);
            pipeline_flatten_pack4_int8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_flatten_pack4_int8->create(LayerShaderType::flatten_pack4_int8, opt, specializations_int8);
        }

        if (shape.dims == 0 || (shape.elempack == 1 && out_shape.elempack == 4))
        {
            pipeline_flatten_pack1to4_int8 = new Pipeline(vkdev);
            pipeline_flatten_pack1to4_int8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_flatten_pack1to4_int8->create(LayerShaderType::flatten_pack1to4_int8, opt, specializations_int8);
        }
    }
#endif

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

#if NCNN_INT8
    delete pipeline_flatten_int8;
    pipeline_flatten_int8 = 0;

    delete pipeline_flatten_pack4_int8;
    pipeline_flatten_pack4_int8 = 0;

    delete pipeline_flatten_pack1to4_int8;
    pipeline_flatten_pack1to4_int8 = 0;
#endif

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
#if NCNN_INT8
    if (bottom_blob.elembits() == 8 && elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_flatten_int8;
    }
    else if (bottom_blob.elembits() == 8 && elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_flatten_pack4_int8;
    }
    else if (bottom_blob.elembits() == 8 && elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_flatten_pack1to4_int8;
    }
    else if (elempack == 1 && out_elempack == 1)
#else
    if (bottom_blob.elembits() == 8)
    {
        return -1;
    }

    if (elempack == 1 && out_elempack == 1)
#endif
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

    if (!pipeline)
        return -1;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn
