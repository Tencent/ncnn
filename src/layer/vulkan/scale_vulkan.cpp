// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Scale_vulkan::Scale_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_scale = 0;
    pipeline_scale_pack4 = 0;
}

int Scale_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3) elempack = shape.c % 4 == 0 ? 4 : 1;

    size_t elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed)
    {
        elemsize = elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
    }

    Mat shape_packed;
    if (shape.dims == 1) shape_packed = Mat(shape.w / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    if (scale_data_size == -233)
    {
        std::vector<vk_specialization_type> specializations(1 + 5);
        specializations[0].i = 0;
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
            pipeline_scale = new Pipeline(vkdev);
            pipeline_scale->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_scale->create(LayerShaderType::scale, opt, specializations);
        }

        // pack4
        if (shape.dims == 0 || elempack == 4)
        {
            pipeline_scale_pack4 = new Pipeline(vkdev);
            pipeline_scale_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_scale_pack4->create(LayerShaderType::scale_pack4, opt, specializations);
        }

        return 0;
    }

    if (shape.dims == 0) elempack = scale_data_size % 4 == 0 ? 4 : 1;

    std::vector<vk_specialization_type> specializations(1 + 5);
    specializations[0].i = bias_term;
    specializations[1 + 0].i = shape_packed.dims;
    specializations[1 + 1].i = shape_packed.w;
    specializations[1 + 2].i = shape_packed.h;
    specializations[1 + 3].i = shape_packed.c;
    specializations[1 + 4].i = shape_packed.cstep;

    Mat local_size_xyz(4, 4, std::min(4, scale_data_size / elempack), (void*)0);
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
    if (elempack == 1)
    {
        pipeline_scale = new Pipeline(vkdev);
        pipeline_scale->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_scale->create(LayerShaderType::scale, opt, specializations);
    }

    // pack4
    if (elempack == 4)
    {
        pipeline_scale_pack4 = new Pipeline(vkdev);
        pipeline_scale_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_scale_pack4->create(LayerShaderType::scale_pack4, opt, specializations);
    }

    return 0;
}

int Scale_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_scale;
    pipeline_scale = 0;

    delete pipeline_scale_pack4;
    pipeline_scale_pack4 = 0;

    return 0;
}

int Scale_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (scale_data_size == -233)
        return 0;

    cmd.record_upload(scale_data, scale_data_gpu, opt);

    if (bias_term)
    {
        cmd.record_upload(bias_data, bias_data_gpu, opt);
    }

    if (opt.lightmode)
    {
        scale_data.release();
        bias_data.release();
    }

    return 0;
}

int Scale_vulkan::forward_inplace(std::vector<VkMat>& bottom_top_blobs, VkCompute& cmd, const Option& /*opt*/) const
{
    VkMat& bottom_top_blob = bottom_top_blobs[0];
    const VkMat& scale_blob = bottom_top_blobs[1];

    int elempack = bottom_top_blob.elempack;

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_top_blob;
    bindings[1] = scale_blob;
    bindings[2] = bias_data_gpu;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = elempack == 4 ? pipeline_scale_pack4 : pipeline_scale;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

int Scale_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    std::vector<VkMat> bottom_top_blobs(2);
    bottom_top_blobs[0] = bottom_top_blob;
    bottom_top_blobs[1] = scale_data_gpu;

    return forward_inplace(bottom_top_blobs, cmd, opt);
}

} // namespace ncnn
