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
    pipeline_rotaryembed_2d_pack4 = 0;
}

int RotaryEmbed_vulkan::create_pipeline(const Option& opt)
{
    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];

    std::vector<vk_specialization_type> specializations(1 + 4);
    specializations[0].i = interleaved;
    specializations[1 + 0].i = shape.dims == 1 ? shape.w * shape.elempack : shape.w;
    specializations[1 + 1].i = shape.h;
    specializations[1 + 2].i = shape.c;
    specializations[1 + 3].i = (int)shape.cstep;

    Mat local_size_xyz;
    if (shape.dims == 3)
    {
        const int halfdim = shape.w / 2;
        local_size_xyz.w = std::min(64, std::max(1, halfdim));
        local_size_xyz.h = std::min(4, std::max(1, shape.h));
        local_size_xyz.c = std::min(4, std::max(1, shape.c));
    }
    else
    {
        local_size_xyz.w = 8;
        local_size_xyz.h = 8;
        local_size_xyz.c = 1;
    }

    // pack1 or 1d packed along embed_dim
    if (shape.dims == 0 || shape.dims == 1 || shape.elempack == 1)
    {
        pipeline_rotaryembed = new Pipeline(vkdev);
        pipeline_rotaryembed->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_rotaryembed->create(LayerShaderType::rotaryembed, opt, specializations);
    }

    // pack4 along num_heads
    if (shape.dims == 0 || (shape.dims >= 3 && shape.elempack == 4))
    {
        pipeline_rotaryembed_pack4 = new Pipeline(vkdev);
        pipeline_rotaryembed_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_rotaryembed_pack4->create(LayerShaderType::rotaryembed_pack4, opt, specializations);
    }

    // pack4 along seqlen
    if (shape.dims == 0 || (shape.dims == 2 && shape.elempack == 4))
    {
        pipeline_rotaryembed_2d_pack4 = new Pipeline(vkdev);
        pipeline_rotaryembed_2d_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_rotaryembed_2d_pack4->create(LayerShaderType::rotaryembed_2d_pack4, opt, specializations);
    }

    return 0;
}

int RotaryEmbed_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_rotaryembed;
    pipeline_rotaryembed = 0;

    delete pipeline_rotaryembed_pack4;
    pipeline_rotaryembed_pack4 = 0;

    delete pipeline_rotaryembed_2d_pack4;
    pipeline_rotaryembed_2d_pack4 = 0;

    return 0;
}

int RotaryEmbed_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    const VkMat& cos_cache = bottom_blobs[1];
    const VkMat& sin_cache = bottom_blobs[2];

    const int dims = bottom_blob.dims;
    const int elempack = bottom_blob.elempack;
    const bool pack4_2d = dims == 2 && elempack == 4;

    // 1d pack4 has the same scalar order as pack1
    const int embed_dim = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int num_heads = bottom_blob.c;

    VkMat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkMat> bindings(pack4_2d ? 6 : 4);
    bindings[0] = bottom_blob;
    bindings[1] = cos_cache;
    bindings[2] = sin_cache;
    bindings[3] = top_blob;
    if (pack4_2d)
    {
        bindings[4] = cos_cache;
        bindings[5] = sin_cache;
    }

    std::vector<vk_constant_type> constants(6);
    constants[0].i = embed_dim;
    constants[1].i = seqlen;
    constants[2].i = num_heads;
    constants[3].i = (int)bottom_blob.cstep;
    // cos and sin share the same cache layout
    // 1d cache is contiguous in embed_dim, 2d cache is packed along seqlen
    constants[4].i = cos_cache.dims == 1 ? cos_cache.w * cos_cache.elempack : cos_cache.w;
    constants[5].i = cos_cache.dims == 2 ? cos_cache.elempack : 1;

    VkMat dispatcher;
    dispatcher.w = embed_dim / 2;
    dispatcher.h = seqlen;
    dispatcher.c = num_heads;

    const Pipeline* pipeline = pipeline_rotaryembed;
    if (pack4_2d)
        pipeline = pipeline_rotaryembed_2d_pack4;
    else if (dims >= 3 && elempack == 4)
        pipeline = pipeline_rotaryembed_pack4;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
