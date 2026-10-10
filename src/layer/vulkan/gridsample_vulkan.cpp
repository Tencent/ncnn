// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "gridsample_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

GridSample_vulkan::GridSample_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_gridsample = 0;
    pipeline_gridsample_pack4 = 0;
}

int GridSample_vulkan::create_pipeline(const Option& opt)
{
    const Mat& bottom_shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& grid_shape = bottom_shapes.size() > 1 ? bottom_shapes[1] : Mat();
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (bottom_shape.dims == 3 || bottom_shape.dims == 4)
        elempack = bottom_shape.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
    if (out_shape.dims == 3 || out_shape.dims == 4)
        out_elempack = out_shape.c % 4 == 0 ? 4 : 1;

    size_t elemsize = 4u;
    size_t out_elemsize = 4u;
    size_t grid_elemsize = 4u;
    if (opt.use_fp16_storage || opt.use_fp16_packed)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
        grid_elemsize = 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        out_elemsize = out_elempack * 4u;
        grid_elemsize = 4u;
    }

    Mat bottom_shape_packed;
    if (bottom_shape.dims == 3)
        bottom_shape_packed = Mat(bottom_shape.w, bottom_shape.h, bottom_shape.c / elempack, (void*)0, elemsize, elempack);
    if (bottom_shape.dims == 4)
        bottom_shape_packed = Mat(bottom_shape.w, bottom_shape.h, bottom_shape.d, bottom_shape.c / elempack, (void*)0, elemsize, elempack);

    Mat grid_shape_packed;
    if (grid_shape.dims == 3)
        grid_shape_packed = Mat(grid_shape.w, grid_shape.h, grid_shape.c, (void*)0, grid_elemsize, 1);
    if (grid_shape.dims == 4)
        grid_shape_packed = Mat(grid_shape.w, grid_shape.h, grid_shape.d, grid_shape.c, (void*)0, grid_elemsize, 1);

    Mat out_shape_packed;
    if (out_shape.dims == 3)
        out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 4)
        out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.d, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    std::vector<vk_specialization_type> specializations(4 + 18);
    specializations[0].i = sample_type;
    specializations[1].i = padding_mode;
    specializations[2].i = align_corner;
    specializations[3].i = permute_fusion;

    if (bottom_shape_packed.dims == 0 || grid_shape_packed.dims == 0 || out_shape_packed.dims == 0)
    {
        for (int i = 0; i < 18; i++)
            specializations[4 + i].i = 0;
    }
    else
    {
        specializations[4 + 0].i = bottom_shape_packed.dims;
        specializations[4 + 1].i = bottom_shape_packed.w;
        specializations[4 + 2].i = bottom_shape_packed.h;
        specializations[4 + 3].i = bottom_shape_packed.d;
        specializations[4 + 4].i = bottom_shape_packed.c;
        specializations[4 + 5].i = bottom_shape_packed.cstep;

        specializations[4 + 6].i = grid_shape_packed.dims;
        specializations[4 + 7].i = grid_shape_packed.w;
        specializations[4 + 8].i = grid_shape_packed.h;
        specializations[4 + 9].i = grid_shape_packed.d;
        specializations[4 + 10].i = grid_shape_packed.c;
        specializations[4 + 11].i = grid_shape_packed.cstep;

        specializations[4 + 12].i = out_shape_packed.dims;
        specializations[4 + 13].i = out_shape_packed.w;
        specializations[4 + 14].i = out_shape_packed.h;
        specializations[4 + 15].i = out_shape_packed.d;
        specializations[4 + 16].i = out_shape_packed.c;
        specializations[4 + 17].i = out_shape_packed.cstep;
    }

    Mat local_size_xyz;
    if (out_shape_packed.dims == 3)
    {
        local_size_xyz.w = std::min(4, out_shape_packed.w);
        local_size_xyz.h = std::min(4, out_shape_packed.h);
        local_size_xyz.c = std::min(4, out_shape_packed.c);
    }
    if (out_shape_packed.dims == 4)
    {
        local_size_xyz.w = std::min(4, out_shape_packed.w);
        local_size_xyz.h = std::min(4, out_shape_packed.h * out_shape_packed.d);
        local_size_xyz.c = std::min(4, out_shape_packed.c);
    }

    if (bottom_shape.dims == 0 || (elempack == 1 && out_elempack == 1))
    {
        pipeline_gridsample = new Pipeline(vkdev);
        pipeline_gridsample->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_gridsample->create(LayerShaderType::gridsample, opt, specializations);
    }

    if (bottom_shape.dims == 0 || (elempack == 4 && out_elempack == 4))
    {
        pipeline_gridsample_pack4 = new Pipeline(vkdev);
        pipeline_gridsample_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_gridsample_pack4->create(LayerShaderType::gridsample_pack4, opt, specializations);
    }

    return 0;
}

int GridSample_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_gridsample;
    pipeline_gridsample = 0;

    delete pipeline_gridsample_pack4;
    pipeline_gridsample_pack4 = 0;

    return 0;
}

int GridSample_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& grid_blob = bottom_blobs[1];

    if (bottom_blob.empty() || grid_blob.empty())
        return -100;

    if (bottom_blob.dims != 3 && bottom_blob.dims != 4)
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
    else
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

    VkMat& top_blob = top_blobs[0];

    const int elempack = bottom_blob.elempack;
    const size_t elemsize = bottom_blob.elemsize;

    if (bottom_blob.dims == 3)
    {
        top_blob.create(outw, outh, bottom_blob.c, elemsize, elempack, opt.blob_vkallocator);
    }
    else
    {
        top_blob.create(outw, outh, outd, bottom_blob.c, elemsize, elempack, opt.blob_vkallocator);
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

    const Pipeline* pipeline = 0;
    if (elempack == 4)
        pipeline = pipeline_gridsample_pack4;
    else
        pipeline = pipeline_gridsample;

    if (!pipeline)
        return -100;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
