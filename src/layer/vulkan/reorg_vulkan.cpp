// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reorg_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Reorg_vulkan::Reorg_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_reorg = 0;
    pipeline_reorg_pack4 = 0;
    pipeline_reorg_pack1to4 = 0;
}

int Reorg_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    std::vector<vk_specialization_type> specializations(2 + 10);
    specializations[0].i = stride;
    specializations[1].i = mode;
    specializations[2 + 0].i = shape.dims;
    specializations[2 + 1].i = shape.w;
    specializations[2 + 2].i = shape.h;
    specializations[2 + 3].i = shape.c;
    specializations[2 + 4].i = shape.cstep;
    specializations[2 + 5].i = out_shape.dims;
    specializations[2 + 6].i = out_shape.w;
    specializations[2 + 7].i = out_shape.h;
    specializations[2 + 8].i = out_shape.c;
    specializations[2 + 9].i = out_shape.cstep;

    Mat local_size_xyz;
    if (out_shape.dims != 0)
    {
        local_size_xyz.w = std::min(4, out_shape.w);
        local_size_xyz.h = std::min(4, out_shape.h);
        local_size_xyz.c = std::min(4, out_shape.c);
    }

    // pack1
    if (shape.dims == 0 || (shape.elempack == 1 && out_shape.elempack == 1))
    {
        pipeline_reorg = new Pipeline(vkdev);
        pipeline_reorg->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reorg->create(LayerShaderType::reorg, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || (shape.elempack == 4 && out_shape.elempack == 4))
    {
        pipeline_reorg_pack4 = new Pipeline(vkdev);
        pipeline_reorg_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reorg_pack4->create(LayerShaderType::reorg_pack4, opt, specializations);
    }

    // pack1to4
    if (shape.dims == 0 || (shape.elempack == 1 && out_shape.elempack == 4))
    {
        pipeline_reorg_pack1to4 = new Pipeline(vkdev);
        pipeline_reorg_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reorg_pack1to4->create(LayerShaderType::reorg_pack1to4, opt, specializations);
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

    int out_elempack = outc % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

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

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace ncnn
