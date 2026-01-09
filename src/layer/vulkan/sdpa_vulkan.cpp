// sdpa_vulkan.cpp
// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_vulkan.h"
#include "layer_shader_type.h"
#include <cmath> // for sqrt

namespace ncnn {

SDPA_vulkan::SDPA_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;

    pipeline_sdpa = 0;
    pipeline_sdpa_kv_concat = 0;
}

int SDPA_vulkan::load_param(const ParamDict& pd)
{
    int ret = SDPA::load_param(pd);

    if (int8_scale_term)
    {
        support_vulkan = false;
    }

    return ret;
}

int SDPA_vulkan::create_pipeline(const Option& opt)
{
    const Mat& qshape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Mat& vshape = bottom_shapes.size() > 2 ? bottom_shapes[2] : Mat();

    int head_dim = 0;
    int out_head_dim = 0;

    if (qshape.dims == 3) head_dim = qshape.w;
    if (vshape.dims == 3) out_head_dim = vshape.w;

    // SDPA Pipeline
    // Spec constants: 0=head_dim, 1=out_head_dim.
    // Scale removed from spec constants as it is passed dynamically via push constants.
    std::vector<vk_specialization_type> spec_sdpa(2);
    spec_sdpa[0].i = head_dim;
    spec_sdpa[1].i = out_head_dim;

    pipeline_sdpa = new Pipeline(vkdev);
    pipeline_sdpa->set_local_size_xyz(256, 1, 1);
    pipeline_sdpa->create(LayerShaderType::sdpa, opt, spec_sdpa);

    // KV Concat Pipeline
    std::vector<vk_specialization_type> spec_kv(2);
    spec_kv[0].i = head_dim;
    spec_kv[1].i = out_head_dim;

    pipeline_sdpa_kv_concat = new Pipeline(vkdev);
    pipeline_sdpa_kv_concat->set_local_size_xyz(64, 1, 1);
    pipeline_sdpa_kv_concat->create(LayerShaderType::sdpa_kv_concat, opt, spec_kv);

    return 0;
}

int SDPA_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_sdpa;
    pipeline_sdpa = 0;

    delete pipeline_sdpa_kv_concat;
    pipeline_sdpa_kv_concat = 0;

    return 0;
}

static int sdpa_make_dispatcher(VkMat& dispatcher, int tiles_q, int heads)
{
    dispatcher.w = tiles_q * 256;
    dispatcher.h = heads;
    dispatcher.c = 1;
    return 0;
}

int SDPA_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    if (bottom_blobs.size() < 3 || top_blobs.empty())
        return -100;

    const VkMat& query = bottom_blobs[0];
    VkMat key = bottom_blobs[1];
    VkMat value = bottom_blobs[2];
    VkMat mask;

    // --- KV Cache Concat Path ---
    if (bottom_blobs.size() >= 5 && top_blobs.size() >= 3)
    {
        const VkMat& past_key = bottom_blobs[1];
        const VkMat& past_value = bottom_blobs[2];
        const VkMat& cur_key = bottom_blobs[3];
        const VkMat& cur_value = bottom_blobs[4];
        mask = bottom_blobs.size() >= 6 ? bottom_blobs[5] : VkMat();

        const int d = query.w;
        const int dv = cur_value.w;
        const int past_seqlen = past_key.h;
        const int cur_seqlen = cur_key.h;
        const int num_group = cur_key.c;

        VkMat& out_key = top_blobs[1];
        VkMat& out_value = top_blobs[2];

        out_key.create(d, past_seqlen + cur_seqlen, num_group, cur_key.elemsize, 1, opt.blob_vkallocator);
        if (out_key.empty()) return -100;

        out_value.create(dv, past_seqlen + cur_seqlen, num_group, cur_value.elemsize, 1, opt.blob_vkallocator);
        if (out_value.empty()) return -100;

        std::vector<VkMat> bindings_kv(6);
        bindings_kv[0] = past_key;
        bindings_kv[1] = past_value;
        bindings_kv[2] = cur_key;
        bindings_kv[3] = cur_value;
        bindings_kv[4] = out_key;
        bindings_kv[5] = out_value;

        std::vector<vk_constant_type> constants_kv(11);
        constants_kv[0].i = d;
        constants_kv[1].i = dv;
        constants_kv[2].i = past_seqlen;
        constants_kv[3].i = cur_seqlen;
        constants_kv[4].i = num_group;
        constants_kv[5].i = past_key.cstep;
        constants_kv[6].i = past_value.cstep;
        constants_kv[7].i = cur_key.cstep;
        constants_kv[8].i = cur_value.cstep;
        constants_kv[9].i = out_key.cstep;
        constants_kv[10].i = out_value.cstep;

        VkMat dispatcher_kv;
        const int dst_seqlen = past_seqlen + cur_seqlen;
        const int maxw = d > dv ? d : dv;
        dispatcher_kv.w = maxw;
        dispatcher_kv.h = dst_seqlen;
        dispatcher_kv.c = num_group;

        cmd.record_pipeline(pipeline_sdpa_kv_concat, bindings_kv, constants_kv, dispatcher_kv);

        key = out_key;
        value = out_value;
    }
    else
    {
        mask = bottom_blobs.size() >= 4 ? bottom_blobs[3] : VkMat();
    }

    // --- Main SDPA Path ---
    const int d = query.w;
    const int src_seqlen = query.h;
    const int num_heads = query.c;
    const int dv = value.w;
    const int dst_seqlen = key.h;

    if (d <= 0 || dv <= 0 || src_seqlen <= 0 || dst_seqlen <= 0 || num_heads <= 0)
        return -100;

    int num_heads_per_group = 1;
    if (key.c > 0 && num_heads % key.c == 0)
        num_heads_per_group = num_heads / key.c;

    VkMat& top_blob = top_blobs[0];
    top_blob.create(dv, src_seqlen, num_heads, query.elemsize, 1, opt.blob_vkallocator);
    if (top_blob.empty()) return -100;

    // Mask info
    int mask_dims = 0;
    int mask_w = 0;
    int mask_c = 0;
    int mask_cstep = 0;
    if (!mask.empty())
    {
        mask_dims = mask.dims;
        mask_w = mask.w;
        mask_c = mask.c;
        mask_cstep = mask.cstep;
        if (mask_dims != 2 && mask_dims != 3)
        {
            mask_dims = 0; mask_w = 0; mask_c = 0; mask_cstep = 0;
        }
    }

    float final_scale = this->scale;
    if (final_scale == 0.f)
    {
        final_scale = 1.0f / std::sqrt((float)d);
    }

    int qw = query.w;
    int kw = key.w;
    int vw = value.w;
    int ow = top_blob.w;

    std::vector<VkMat> bindings(5);
    bindings[0] = query;
    bindings[1] = key;
    bindings[2] = value;
    bindings[3] = mask;
    bindings[4] = top_blob;

    std::vector<vk_constant_type> constants(18);
    constants[0].i = d;
    constants[1].i = dv;
    constants[2].i = src_seqlen;
    constants[3].i = dst_seqlen;
    constants[4].i = num_heads_per_group;
    constants[5].i = query.cstep;
    constants[6].i = key.cstep;
    constants[7].i = value.cstep;
    constants[8].i = top_blob.cstep;
    constants[9].i = mask_dims;
    constants[10].i = mask_w;
    constants[11].i = mask_c;
    constants[12].i = mask_cstep;
    constants[13].f = final_scale;
    constants[14].i = qw;
    constants[15].i = kw;
    constants[16].i = vw;
    constants[17].i = ow;

    VkMat dispatcher;
    const int tiles_q = (src_seqlen + 16 - 1) / 16;
    sdpa_make_dispatcher(dispatcher, tiles_q, num_heads);

    cmd.record_pipeline(pipeline_sdpa, bindings, constants, dispatcher);

    return 0;
}

} // namespace ncnn