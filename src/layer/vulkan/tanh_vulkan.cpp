// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "tanh_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

TanH_vulkan::TanH_vulkan()
{
    support_vulkan = true;

    pipeline_tanh = 0;
}

int TanH_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = top_shapes.empty() ? Mat() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = opt.use_shader_pack8 && shape.w % 8 == 0 ? 8 : shape.w % 4 == 0 ? 4 : 1;
    if (shape.dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4 : 1;
    if (shape.dims == 3 || shape.dims == 4) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4 : 1;

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

    std::vector<vk_specialization_type> specializations(1);
    specializations[0 + 0].u32 = shape_packed.total() * elempack / 4;

    const int local_size_x = vkdev->info.subgroup_size();

    pipeline_tanh = new Pipeline(vkdev);
    pipeline_tanh->set_optimal_local_size_xyz(local_size_x, 1, 1);
    pipeline_tanh->create(LayerShaderType::tanh, opt, specializations);

    return 0;
}

int TanH_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_tanh;
    pipeline_tanh = 0;

    return 0;
}

int TanH_vulkan::forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& /*opt*/) const
{
    const size_t n = bottom_top_blob.total() * bottom_top_blob.elempack / 4;

    std::vector<VkMat> bindings(1);
    bindings[0] = bottom_top_blob;

    std::vector<vk_constant_type> constants(1);
    constants[0].u32 = n;

    VkMat dispatcher;
    dispatcher.w = n;
    dispatcher.h = 1;
    dispatcher.c = 1;
    cmd.record_pipeline(pipeline_tanh, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
