// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "gridsample_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

GridSample_vulkan::GridSample_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;

    pipeline_gridsample = 0;
}

int GridSample_vulkan::create_pipeline(const Option& opt)
{
    std::vector<vk_specialization_type> specializations(4 + 18);

    specializations[0].i = sample_type;
    specializations[1].i = padding_mode;
    specializations[2].i = align_corner;
    specializations[3].i = permute_fusion;

    for (int i = 0; i < 18; i++)
    {
        specializations[4 + i].i = 0;
    }

    pipeline_gridsample = new Pipeline(vkdev);
    pipeline_gridsample->set_local_size_xyz(8, 8, 1);
    pipeline_gridsample->create(LayerShaderType::gridsample, opt, specializations);

    return 0;
}

int GridSample_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_gridsample;
    pipeline_gridsample = 0;

    return 0;
}

int GridSample_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob0 = bottom_blobs[0];
    const VkMat& grid_blob0 = bottom_blobs[1];

    VkMat bottom_blob;
    VkMat grid_blob;

    vkdev->convert_packing(bottom_blob0, bottom_blob, 1, cmd, opt);
    vkdev->convert_packing(grid_blob0, grid_blob, 1, cmd, opt);

    if (bottom_blob.empty() || grid_blob.empty())
        return -100;

    int outw = 0;
    int outh = 0;
    int outd = 1;

    if (bottom_blob.dims == 3)
    {
        if (permute_fusion == 0)
        {
            outw = grid_blob.h;
            outh = grid_blob.c;
        }
        else
        {
            outw = grid_blob.w;
            outh = grid_blob.h;
        }
    }
    else if (bottom_blob.dims == 4)
    {
        if (permute_fusion == 0)
        {
            outw = grid_blob.h;
            outh = grid_blob.d;
            outd = grid_blob.c;
        }
        else
        {
            outw = grid_blob.w;
            outh = grid_blob.h;
            outd = grid_blob.d;
        }
    }
    else
    {
        return -100;
    }

    VkMat& top_blob = top_blobs[0];

    if (bottom_blob.dims == 3)
    {
        top_blob.create(outw, outh, bottom_blob.c, bottom_blob.elemsize, 1, opt.blob_vkallocator);
    }
    else
    {
        top_blob.create(outw, outh, outd, bottom_blob.c, bottom_blob.elemsize, 1, opt.blob_vkallocator);
    }

    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(3);
    bindings[0] = top_blob;
    bindings[1] = bottom_blob;
    bindings[2] = grid_blob;

    std::vector<vk_constant_type> constants(18);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.d;
    constants[4].i = bottom_blob.c;
    constants[5].i = bottom_blob.cstep;
    constants[6].i = grid_blob.dims;
    constants[7].i = grid_blob.w;
    constants[8].i = grid_blob.h;
    constants[9].i = grid_blob.d;
    constants[10].i = grid_blob.c;
    constants[11].i = grid_blob.cstep;
    constants[12].i = top_blob.dims;
    constants[13].i = top_blob.w;
    constants[14].i = top_blob.h;
    constants[15].i = top_blob.d;
    constants[16].i = top_blob.c;
    constants[17].i = top_blob.cstep;

    VkMat dispatcher;
    dispatcher.w = top_blob.w;
    dispatcher.h = top_blob.h * (top_blob.dims == 4 ? top_blob.d : 1);
    dispatcher.c = top_blob.c;

    cmd.record_pipeline(pipeline_gridsample, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
