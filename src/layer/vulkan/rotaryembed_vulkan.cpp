// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "rotaryembed_vulkan.h"

#include "layer_shader_type.h"

namespace ncnn {

RotaryEmbed_vulkan::RotaryEmbed_vulkan()
{
    one_blob_only = false;
    support_inplace = false;

    support_vulkan = true;
    support_vulkan_packing = true;
    support_vulkan_any_packing = true;

    pipeline_rotaryembed = 0;
    pipeline_rotaryembed_pack4 = 0;
}

int RotaryEmbed_vulkan::load_param(const ParamDict& pd)
{
    return RotaryEmbed::load_param(pd);
}

int RotaryEmbed_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;

    const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& cos_shape = bottom_shapes.size() > 1 ? bottom_shapes[1] : Mat();

    int elempack = 1;
    if (shape.dims == 3) elempack = shape.c % 4 == 0 ? 4 : 1;

    size_t elemsize;
    if (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed)
        elemsize = elempack * 2u;
    else
        elemsize = elempack * 4u;

    Mat shape_packed;
    if (shape.dims == 3) shape_packed = Mat(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    const int cache_w = cos_shape.dims == 2 ? cos_shape.w : 0;

    std::vector<vk_specialization_type> specializations(1 + 7);
    specializations[0].i = interleaved;

    specializations[1 + 0].i = shape_packed.dims;
    specializations[1 + 1].i = shape_packed.w;
    specializations[1 + 2].i = shape_packed.h;
    specializations[1 + 3].i = shape_packed.c;
    specializations[1 + 4].i = (int)shape_packed.cstep;
    specializations[1 + 5].i = shape_packed.dims ? (int)shape_packed.cstep : 0; // outcstep == cstep
    specializations[1 + 6].i = cache_w;

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

    // pack1 pipeline
    if (shape.dims == 0 || elempack == 1)
    {
        pipeline_rotaryembed = new Pipeline(vkdev);
        pipeline_rotaryembed->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_rotaryembed->create(LayerShaderType::rotaryembed, opt, specializations);
    }

    // pack4 pipeline (do not depend on out_shape)
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
    const VkMat& bottom_blob0 = bottom_blobs[0];
    const VkMat& cos_cache0 = bottom_blobs[1];
    const VkMat& sin_cache0 = bottom_blobs[2];

    if (bottom_blob0.dims != 3)
        return -100;

    VkMat bottom_blob = bottom_blob0;
    if (bottom_blob.elempack != 1 && bottom_blob.elempack != 4)
        vkdev->convert_packing(bottom_blob0, bottom_blob, 1, cmd, opt);

    VkMat cos_cache = cos_cache0;
    if (cos_cache.elempack != 1)
        vkdev->convert_packing(cos_cache0, cos_cache, 1, cmd, opt);

    VkMat sin_cache = sin_cache0;
    if (sin_cache.elempack != 1)
        vkdev->convert_packing(sin_cache0, sin_cache, 1, cmd, opt);

    if (cos_cache.dims != 2 || sin_cache.dims != 2)
        return -100;

    const int embed_dim = bottom_blob.w;
    const int seqlen = bottom_blob.h;
    const int heads_packed = bottom_blob.c;
    const int elempack = bottom_blob.elempack;

    if (embed_dim % 2 != 0)
        return -100;

    const int halfdim = embed_dim / 2;

    if (cos_cache.w < halfdim || sin_cache.w < halfdim)
        return -100;
    if (cos_cache.h < seqlen || sin_cache.h < seqlen)
        return -100;

    VkMat& top_blob = top_blobs[0];
    top_blob.create(embed_dim, seqlen, heads_packed, bottom_blob.elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    const Pipeline* pipeline = 0;
    if (elempack == 4)
        pipeline = pipeline_rotaryembed_pack4;
    else
        pipeline = pipeline_rotaryembed;

    if (!pipeline)
        return -100;

    std::vector<VkMat> bindings(4);
    bindings[0] = bottom_blob;
    bindings[1] = cos_cache;
    bindings[2] = sin_cache;
    bindings[3] = top_blob;

    std::vector<vk_constant_type> constants(7);
    constants[0].i = bottom_blob.dims;
    constants[1].i = bottom_blob.w;
    constants[2].i = bottom_blob.h;
    constants[3].i = bottom_blob.c;
    constants[4].i = (int)bottom_blob.cstep;
    constants[5].i = (int)top_blob.cstep;
    constants[6].i = cos_cache.w;

    VkMat dispatcher;
    dispatcher.w = halfdim;
    dispatcher.h = seqlen;
    dispatcher.c = heads_packed;

    cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn
