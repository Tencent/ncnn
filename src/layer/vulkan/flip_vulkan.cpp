// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "flip_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Flip_vulkan::Flip_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = false;
    support_any_packing = false;

    pipeline_flip = 0;
}

int Flip_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;

    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    // pack1 only
    const int elempack = 1;
    const int out_elempack = 1;

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
    if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 4) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.d, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    std::vector<vk_specialization_type> specializations(12);
    specializations[0].i = shape_packed.dims;
    specializations[1].i = shape_packed.w;
    specializations[2].i = shape_packed.h;
    specializations[3].i = shape_packed.d;
    specializations[4].i = shape_packed.c;
    specializations[5].i = shape_packed.cstep;
    specializations[6].i = out_shape_packed.dims;
    specializations[7].i = out_shape_packed.w;
    specializations[8].i = out_shape_packed.h;
    specializations[9].i = out_shape_packed.d;
    specializations[10].i = out_shape_packed.c;
    specializations[11].i = out_shape_packed.cstep;

    Mat local_size_xyz;
    if (out_shape_packed.dims == 1)
    {
        local_size_xyz.w = std::min(64, out_shape_packed.w);
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }
    if (out_shape_packed.dims == 2)
    {
        local_size_xyz.w = std::min(8, out_shape_packed.w);
        local_size_xyz.h = std::min(8, out_shape_packed.h);
        local_size_xyz.c = 1;
    }
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

    pipeline_flip = new Pipeline(vkdev);
    pipeline_flip->set_optimal_local_size_xyz(local_size_xyz);
    pipeline_flip->create(LayerShaderType::flip, opt, specializations);

    return 0;
}

int Flip_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_flip;
    pipeline_flip = 0;

    return 0;
}

int Flip_vulkan::forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const
{
    if (axes.empty())
    {
        top_blob = bottom_blob;
        return 0;
    }

    const VkMat& bottom = bottom_blob;

    const int dims = bottom.dims;
    const int w = bottom.w;
    const int h = bottom.h;
    const int d = bottom.d;
    const int channels = bottom.c;
    const size_t elemsize = bottom.elemsize;

    // compute flip flags exactly like cpu version
    int axes_flag[4] = {0};
    int flip_w = 0;
    int flip_h = 0;
    int flip_d = 0;
    int flip_c = 0;
    {
        const int* axes_ptr = axes;
        for (int i = 0; i < axes.w; i++)
        {
            int axis = axes_ptr[i];
            if (axis < 0)
                axis += dims;

            if (axis < 0 || axis >= 4)
                continue;

            axes_flag[axis] = 1;
        }

        if (dims == 1)
        {
            flip_w = 1;
        }
        else if (dims == 2)
        {
            if (axes_flag[0] == 1) flip_h = 1;
            if (axes_flag[1] == 1) flip_w = 1;
        }
        else if (dims == 3)
        {
            if (axes_flag[0] == 1) flip_c = 1;
            if (axes_flag[1] == 1) flip_h = 1;
            if (axes_flag[2] == 1) flip_w = 1;
        }
        else if (dims == 4)
        {
            if (axes_flag[0] == 1) flip_c = 1;
            if (axes_flag[1] == 1) flip_d = 1;
            if (axes_flag[2] == 1) flip_h = 1;
            if (axes_flag[3] == 1) flip_w = 1;
        }
    }

    // output same shape
    if (dims == 1)
        top_blob.create(w, elemsize, 1, opt.blob_vkallocator);
    else if (dims == 2)
        top_blob.create(w, h, elemsize, 1, opt.blob_vkallocator);
    else if (dims == 3)
        top_blob.create(w, h, channels, elemsize, 1, opt.blob_vkallocator);
    else
        top_blob.create(w, h, d, channels, elemsize, 1, opt.blob_vkallocator);

    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom;
    bindings[1] = top_blob;

    // push constants match flip.comp struct order
    std::vector<vk_constant_type> constants(16);
    constants[0].i = bottom.dims;
    constants[1].i = bottom.w;
    constants[2].i = bottom.h;
    constants[3].i = bottom.d;
    constants[4].i = bottom.c;
    constants[5].i = bottom.cstep;
    constants[6].i = top_blob.dims;
    constants[7].i = top_blob.w;
    constants[8].i = top_blob.h;
    constants[9].i = top_blob.d;
    constants[10].i = top_blob.c;
    constants[11].i = top_blob.cstep;
    constants[12].i = flip_w;
    constants[13].i = flip_h;
    constants[14].i = flip_d;
    constants[15].i = flip_c;

    VkMat dispatcher;
    if (dims == 1)
    {
        dispatcher.w = top_blob.w;
        dispatcher.h = 1;
        dispatcher.c = 1;
    }
    else if (dims == 2)
    {
        dispatcher.w = top_blob.w;
        dispatcher.h = top_blob.h;
        dispatcher.c = 1;
    }
    else if (dims == 3)
    {
        dispatcher.w = top_blob.w;
        dispatcher.h = top_blob.h;
        dispatcher.c = top_blob.c;
    }
    else
    {
        dispatcher.w = top_blob.w;
        dispatcher.h = top_blob.h * top_blob.d;
        dispatcher.c = top_blob.c;
    }

    cmd.record_pipeline(pipeline_flip, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
