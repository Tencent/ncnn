// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "batchnorm_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

BatchNorm_vulkan::BatchNorm_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_batchnorm = 0;
    pipeline_batchnorm_pack4 = 0;
}

int BatchNorm_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = channels % 4 == 0 ? 4 : 1;

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
    if (shape.dims == 4) shape_packed = Mat(shape.w, shape.h, shape.d, shape.c / elempack, (void*)0, elemsize, elempack);

    std::vector<vk_specialization_type> specializations(0 + 5);
    specializations[0 + 0].i = std::min(3, shape_packed.dims);
    specializations[0 + 1].i = shape_packed.w;
    specializations[0 + 2].i = shape_packed.h * shape_packed.d;
    specializations[0 + 3].i = shape_packed.c;
    specializations[0 + 4].i = shape_packed.cstep;

    Mat local_size_xyz(4, 4, std::min(4, channels / elempack), (void*)0);
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
    if (elempack == 1)
    {
        pipeline_batchnorm = new Pipeline(vkdev);
        pipeline_batchnorm->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_batchnorm->create(LayerShaderType::batchnorm, opt, specializations);
    }

    // pack4
    if (elempack == 4)
    {
        pipeline_batchnorm_pack4 = new Pipeline(vkdev);
        pipeline_batchnorm_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_batchnorm_pack4->create(LayerShaderType::batchnorm_pack4, opt, specializations);
    }

    return 0;
}

int BatchNorm_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_batchnorm;
    pipeline_batchnorm = 0;

    delete pipeline_batchnorm_pack4;
    pipeline_batchnorm_pack4 = 0;

    return 0;
}

int BatchNorm_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    cmd.record_upload(a_data, a_data_gpu, opt);

    cmd.record_upload(b_data, b_data_gpu, opt);

    if (opt.lightmode)
    {
        a_data.release();
        b_data.release();
    }

    return 0;
}

int BatchNorm_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& /*opt*/) const
{
    int elempack = bottom_top_blob.elempack;

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_top_blob;
    bindings[1] = a_data_gpu;
    bindings[2] = b_data_gpu;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = std::min(3, bottom_top_blob.dims);
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h * bottom_top_blob.d;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = elempack == 4 ? pipeline_batchnorm_pack4 : pipeline_batchnorm;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

} // namespace ncnn
