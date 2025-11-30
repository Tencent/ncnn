// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eltwise_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

Eltwise_vulkan::Eltwise_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_eltwise[0] = 0;
    pipeline_eltwise[1] = 0;
}

int Eltwise_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    const int dims = shape.dims;

    int elempack = 0;
    if (dims == 1) elempack = shape.w % 4 == 0 ? 4 : 1;
    if (dims == 2) elempack = shape.h % 4 == 0 ? 4 : 1;
    if (dims == 3 || dims == 4) elempack = shape.c % 4 == 0 ? 4 : 1;

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
    if (dims == 1) shape_packed = Mat(shape.w / elempack, (void*)0, elemsize, elempack);
    if (dims == 2) shape_packed = Mat(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);
    if (dims == 4) shape_packed = Mat(shape.w, shape.h, shape.d, shape.c / elempack, (void*)0, elemsize, elempack);

    std::vector<vk_specialization_type> specializations(3);
    specializations[0].i = op_type;
    specializations[1].i = coeffs.w == 0 ? 0 : 1;
    specializations[2].u32 = shape_packed.total() * elempack / 4;

    const int local_size_x = vkdev->info.subgroup_size();

    pipeline_eltwise[0] = new Pipeline(vkdev);
    pipeline_eltwise[0]->set_optimal_local_size_xyz(local_size_x, 1, 1);
    pipeline_eltwise[0]->create(LayerShaderType::eltwise, opt, specializations);
    pipeline_eltwise[1] = new Pipeline(vkdev);
    pipeline_eltwise[1]->set_optimal_local_size_xyz(local_size_x, 1, 1);
    pipeline_eltwise[1]->create(LayerShaderType::eltwise, opt, specializations);

    return 0;
}

int Eltwise_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_eltwise[0];
    delete pipeline_eltwise[1];
    pipeline_eltwise[0] = 0;
    pipeline_eltwise[1] = 0;

    return 0;
}

int Eltwise_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& bottom_blob1 = bottom_blobs[1];

    VkMat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    const size_t n = top_blob.total() * top_blob.elempack / 4;

    std::vector<VkMat> bindings(3);
    bindings[0] = bottom_blob;
    bindings[1] = bottom_blob1;
    bindings[2] = top_blob;

    std::vector<vk_constant_type> constants(3);
    constants[0].u32 = n;
    constants[1].f = coeffs.w == 0 ? 1.f : coeffs[0];
    constants[2].f = coeffs.w == 0 ? 1.f : coeffs[1];

    VkMat dispatcher;
    dispatcher.w = n;
    dispatcher.h = 1;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline_eltwise[1], bindings, constants, dispatcher);

    for (size_t b = 2; b < bottom_blobs.size(); b++)
    {
        std::vector<VkMat> bindings(3);
        bindings[0] = top_blob;
        bindings[1] = bottom_blobs[b];
        bindings[2] = top_blob; // TODO use separated pipeline ?

        std::vector<vk_constant_type> constants(3);
        constants[0].u32 = n;
        constants[1].f = 1.f;
        constants[2].f = coeffs.w == 0 ? 1 : coeffs[b];

        cmd.record_pipeline(pipeline_eltwise[b % 2], bindings, constants, dispatcher);
    }

    return 0;
}

} // namespace ncnn
