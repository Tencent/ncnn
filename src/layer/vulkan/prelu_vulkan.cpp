// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "prelu_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

PReLU_vulkan::PReLU_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_prelu = 0;
    pipeline_prelu_pack4 = 0;
}

int PReLU_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 0) elempack = num_slope % 4 == 0 ? 4 : 1;
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

    std::vector<vk_specialization_type> specializations(2 + 5);
    specializations[0].i = num_slope;
    specializations[1].f = num_slope == 1 ? slope_data[0] : 1.f;
    specializations[2 + 0].i = shape_packed.dims;
    specializations[2 + 1].i = shape_packed.w;
    specializations[2 + 2].i = shape_packed.h;
    specializations[2 + 3].i = shape_packed.c;
    specializations[2 + 4].i = shape_packed.cstep;

    Mat local_size_xyz(4, 4, std::min(4, num_slope / elempack), (void*)0);
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
    if (num_slope == 1 || elempack == 1)
    {
        pipeline_prelu = new Pipeline(vkdev);
        pipeline_prelu->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_prelu->create(LayerShaderType::prelu, opt, specializations);
    }

    // pack4
    if (num_slope == 1 || elempack == 4)
    {
        pipeline_prelu_pack4 = new Pipeline(vkdev);
        pipeline_prelu_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_prelu_pack4->create(LayerShaderType::prelu_pack4, opt, specializations);
    }

    return 0;
}

int PReLU_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_prelu;
    pipeline_prelu = 0;

    delete pipeline_prelu_pack4;
    pipeline_prelu_pack4 = 0;

    return 0;
}

int PReLU_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (num_slope > 1)
    {
        cmd.record_upload(slope_data, slope_data_gpu, opt);

        if (opt.lightmode)
        {
            slope_data.release();
        }
    }

    return 0;
}

int PReLU_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& /*opt*/) const
{
    int elempack = bottom_top_blob.elempack;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_top_blob;
    bindings[1] = slope_data_gpu;

    std::vector<vk_constant_type> constants(5);
    constants[0].i = bottom_top_blob.dims;
    constants[1].i = bottom_top_blob.w;
    constants[2].i = bottom_top_blob.h;
    constants[3].i = bottom_top_blob.c;
    constants[4].i = bottom_top_blob.cstep;

    const Pipeline* pipeline = elempack == 4 ? pipeline_prelu_pack4 : pipeline_prelu;

    cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);

    return 0;
}

} // namespace ncnn
