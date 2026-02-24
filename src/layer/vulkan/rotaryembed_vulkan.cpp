// Copyright 2026 Futz12 <pchar.cn>
// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

RotaryEmbed_vulkan::RotaryEmbed_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    pipeline_rotaryembed = 0;
    pipeline_rotaryembed_pack4 = 0;
}

int RotaryEmbed_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];

    int elempack = 1;
    if (shape.dims == 3) elempack = shape.c % 4 == 0 ? 4 : 1;

    size_t elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
        elemsize = elempack * 2u;
    else
        elemsize = elempack * 4u;

    Mat shape_packed;
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    std::vector<vk_specialization_type> specializations(1 + 4);
    specializations[0].i = interleaved;
    specializations[1 + 0].i = shape_packed.w;
    specializations[1 + 1].i = shape_packed.h;
    specializations[1 + 2].i = shape_packed.c;
    specializations[1 + 3].i = (int)shape_packed.cstep;

    Mat local_size_xyz;
    if (shape_packed.dims == 3)
    {
        const int halfdim = shape_packed.w / 2;
        local_size_xyz.w = std::min(64, std::max(1, halfdim));
        local_size_xyz.h = std::min(4, std::max(1, shape_packed.h));
        local_size_xyz.c = std::min(4, std::max(1, shape_packed.c));
    }
    else
    {
        local_size_xyz.w = 8;
        local_size_xyz.h = 8;
        local_size_xyz.c = 1;
    }

    // pack1
    if (shape.dims == 0 || elempack == 1)
    {
        pipeline_rotaryembed = new Pipeline(vkdev);
        pipeline_rotaryembed->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_rotaryembed->create(LayerShaderType::rotaryembed, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || elempack == 4)
    {
        pipeline_rotaryembed_pack4 = new Pipeline(vkdev);
        pipeline_rotaryembed_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_rotaryembed_pack4->create(LayerShaderType::rotaryembed_pack4, opt, specializations);
    }

    return 0;
}

int RotaryEmbed_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_rotaryembed;
    pipeline_rotaryembed = 0;

    delete pipeline_rotaryembed_pack4;
    pipeline_rotaryembed_pack4 = 0;

    return 0;
}

int RotaryEmbed_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& cos_cache = bottom_blobs[1];
    const VkMat& sin_cache = bottom_blobs[2];

    const int embed_dim = bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int num_heads = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    VkMat cos_cache_unpacked = cos_cache;
    if (cos_cache.elempack != 1)
    {
        vkdev->convert_packing(cos_cache, cos_cache_unpacked, 1, cmd, opt);
        if (cos_cache_unpacked.empty())
            return -100;
    }

    VkMat sin_cache_unpacked = sin_cache;
    if (sin_cache.elempack != 1)
    {
        vkdev->convert_packing(sin_cache, sin_cache_unpacked, 1, cmd, opt);
        if (sin_cache_unpacked.empty())
            return -100;
    }

    VkMat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob;
    bindings[1] = cos_cache_unpacked;
    bindings[2] = sin_cache_unpacked;
    bindings[3] = top_blob;

    std::vector<vk_constant_type> constants(4);
    constants[0].i = embed_dim;
    constants[1].i = seqlen;
    constants[2].i = num_heads;
    constants[3].i = (int)bottom_blob.cstep;

    VkMat dispatcher;
    dispatcher.w = embed_dim / 2;
    dispatcher.h = seqlen;
    dispatcher.c = num_heads;

    const Pipeline* pipeline = elempack == 4 ? pipeline_rotaryembed_pack4 : pipeline_rotaryembed;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
