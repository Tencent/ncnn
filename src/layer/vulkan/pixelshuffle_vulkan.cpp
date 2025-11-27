// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pixelshuffle_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

PixelShuffle_vulkan::PixelShuffle_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_pixelshuffle = 0;
    pipeline_pixelshuffle_pack4 = 0;
    pipeline_pixelshuffle_pack4to1 = 0;
}

int PixelShuffle_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3) elempack = shape.c % 4 == 0 ? 4 : 1;

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = out_shape.w % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 2) out_elempack = out_shape.h % 4 == 0 ? 4 : 1;
    if (out_shape.dims == 3) out_elempack = out_shape.c % 4 == 0 ? 4 : 1;

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

    Mat out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    std::vector<vk_specialization_type> specializations(2 + 10);
    specializations[0].i = upscale_factor;
    specializations[1].i = mode;
    specializations[2 + 0].i = shape_packed.dims;
    specializations[2 + 1].i = shape_packed.w;
    specializations[2 + 2].i = shape_packed.h;
    specializations[2 + 3].i = shape_packed.c;
    specializations[2 + 4].i = shape_packed.cstep;
    specializations[2 + 5].i = out_shape_packed.dims;
    specializations[2 + 6].i = out_shape_packed.w;
    specializations[2 + 7].i = out_shape_packed.h;
    specializations[2 + 8].i = out_shape_packed.c;
    specializations[2 + 9].i = out_shape_packed.cstep;

    Mat local_size_xyz_bottom; // pack4to1
    if (shape_packed.dims != 3)
    {
        local_size_xyz_bottom.w = std::min(4, shape_packed.w);
        local_size_xyz_bottom.h = std::min(4, shape_packed.h);
        local_size_xyz_bottom.c = std::min(4, shape_packed.c);
    }

    Mat local_size_xyz;
    if (out_shape_packed.dims != 0)
    {
        local_size_xyz.w = std::min(4, out_shape_packed.w);
        local_size_xyz.h = std::min(4, out_shape_packed.h);
        local_size_xyz.c = std::min(4, out_shape_packed.c);
    }

    // pack1
    if (shape.dims == 0 || (elempack == 1 && out_elempack == 1))
    {
        pipeline_pixelshuffle = new Pipeline(vkdev);
        pipeline_pixelshuffle->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_pixelshuffle->create(LayerShaderType::pixelshuffle, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || (elempack == 4 && out_elempack == 4))
    {
        pipeline_pixelshuffle_pack4 = new Pipeline(vkdev);
        pipeline_pixelshuffle_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_pixelshuffle_pack4->create(LayerShaderType::pixelshuffle_pack4, opt, specializations);
    }

    // pack4to1
    if (shape.dims == 0 || (elempack == 4 && out_elempack == 1))
    {
        pipeline_pixelshuffle_pack4to1 = new Pipeline(vkdev);
        pipeline_pixelshuffle_pack4to1->set_optimal_local_size_xyz(local_size_xyz_bottom);
        pipeline_pixelshuffle_pack4to1->create(LayerShaderType::pixelshuffle_pack4to1, opt, specializations);
    }

    return 0;
}

int PixelShuffle_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_pixelshuffle;
    pipeline_pixelshuffle = 0;

    delete pipeline_pixelshuffle_pack4;
    pipeline_pixelshuffle_pack4 = 0;

    delete pipeline_pixelshuffle_pack4to1;
    pipeline_pixelshuffle_pack4to1 = 0;

    return 0;
}

int PixelShuffle_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = w * upscale_factor;
    int outh = h * upscale_factor;
    int outc = channels * elempack / (upscale_factor * upscale_factor);

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

    if (elempack == 1 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_pixelshuffle, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_pixelshuffle_pack4, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_pixelshuffle_pack4to1, bindings, constants, bottom_blob);
    }

    return 0;
}

} // namespace ncnn
