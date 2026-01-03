// Copyright 2026
// SPDX-License-Identifier: BSD-3-Clause

#include "softmax_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Softmax_vulkan::Softmax_vulkan()
{
    support_vulkan = true;

    support_vulkan_packing = false;
    support_vulkan_any_packing = false;

    pipeline_softmax = 0;
}

int Softmax_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;

    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];

    size_t elemsize = 4u;
    if (opt.use_fp16_storage || opt.use_fp16_packed)
        elemsize = 2u;

    Mat shape_packed;
    if (shape.dims == 1) shape_packed = Mat(shape.w, (void*)0, elemsize, 1);
    if (shape.dims == 2) shape_packed = Mat(shape.w, shape.h, (void*)0, elemsize, 1);
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c, (void*)0, elemsize, 1);
    if (shape.dims == 4) shape_packed = Mat(shape.w, shape.h, shape.d, shape.c, (void*)0, elemsize, 1);

    std::vector<vk_specialization_type> specializations(1 + 7);
    specializations[0].i = axis;
    specializations[1 + 0].i = shape_packed.dims;
    specializations[1 + 1].i = shape_packed.w;
    specializations[1 + 2].i = shape_packed.h;
    specializations[1 + 3].i = shape_packed.d;
    specializations[1 + 4].i = shape_packed.c;
    specializations[1 + 5].i = 0;
    specializations[1 + 6].i = 0;

    pipeline_softmax = new Pipeline(vkdev);
    pipeline_softmax->set_local_size_xyz(256, 1, 1);
    pipeline_softmax->create(LayerShaderType::softmax, opt, specializations);

    return 0;
}

int Softmax_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_softmax;
    pipeline_softmax = 0;
    return 0;
}

int Softmax_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_top_blob;

    const int dims = bottom_blob.dims;
    const int w = bottom_blob.w;
    const int h = bottom_blob.h;
    const int d = bottom_blob.d;
    const int c = bottom_blob.c;

    VkMat top_blob;
    if (dims == 1)
        top_blob.create(w, bottom_blob.elemsize, 1, opt.blob_vkallocator);
    else if (dims == 2)
        top_blob.create(w, h, bottom_blob.elemsize, 1, opt.blob_vkallocator);
    else if (dims == 3)
        top_blob.create(w, h, c, bottom_blob.elemsize, 1, opt.blob_vkallocator);
    else
        top_blob.create(w, h, d, c, bottom_blob.elemsize, 1, opt.blob_vkallocator);

    if (top_blob.empty())
        return -100;

    int pa = axis < 0 ? dims + axis : axis;

    int slices = 1;
    if (dims == 1)
    {
        slices = 1;
    }
    else if (dims == 2)
    {
        slices = pa == 0 ? w : h;
    }
    else if (dims == 3)
    {
        if (pa == 0) slices = w * h;
        else if (pa == 1) slices = c * w;
        else slices = c * h;
    }
    else
    {
        const int plane = w * h;
        if (pa == 0) slices = plane * d;
        else if (pa == 1) slices = c * plane;
        else if (pa == 2) slices = c * d * w;
        else slices = c * d * h;
    }

    std::vector<VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(7);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.d;
    constants[4].i = bottom_blob.c;
    constants[5].i = bottom_blob.cstep;
    constants[6].i = top_blob.cstep;

    const int wg = 256;

    VkMat dispatcher;
    dispatcher.w = slices * wg;
    dispatcher.h = 1;
    dispatcher.c = 1;

    cmd.record_pipeline(pipeline_softmax, bindings, constants, dispatcher);

    bottom_top_blob = top_blob;
    return 0;
}

} // namespace ncnn
