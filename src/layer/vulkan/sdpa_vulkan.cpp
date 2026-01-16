// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sdpa_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

SDPA_vulkan::SDPA_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;

    qk_softmax = 0;
    kvcache_concat = 0;

    pipeline_sdpa_qk_cross = 0;
    pipeline_sdpa_qkv_cross = 0;
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
    {
        std::vector<vk_specialization_type> specializations(12);
        specializations[0].i = attn_mask;
        specializations[1].f = 0.f; // scale
        specializations[2].i = 0;   // src_seqlen
        specializations[3].i = 0;   // dst_seqlen
        specializations[4].i = 0;   // embed_dim
        specializations[5].i = 0;   // num_heads
        specializations[6].i = 0;   // attn_mask.dims
        specializations[7].i = 0;   // num_heads / num_group
        specializations[8].i = 0;   // q_cstep
        specializations[9].i = 0;   // k_cstep
        specializations[10].i = 0;  // qk_cstep
        specializations[11].i = 0;  // mask_cstep

        {
            pipeline_sdpa_qk_cross = new Pipeline(vkdev);
            pipeline_sdpa_qk_cross->set_local_size_xyz(8, 8, 1);
            pipeline_sdpa_qk_cross->create(LayerShaderType::sdpa_qk_cross, opt, specializations);
        }
    }
    {
        std::vector<vk_specialization_type> specializations(8);
        specializations[0].i = 0; // src_seqlen
        specializations[1].i = 0; // out_embed_dim
        specializations[2].i = 0; // dst_seqlen
        specializations[3].i = 0; // num_heads
        specializations[4].i = 0; // num_heads / num_group
        specializations[5].i = 0; // qk_cstep
        specializations[6].i = 0; // v_cstep
        specializations[7].i = 0; // out_cstep

        {
            pipeline_sdpa_qkv_cross = new Pipeline(vkdev);
            pipeline_sdpa_qkv_cross->set_local_size_xyz(8, 8, 1);
            pipeline_sdpa_qkv_cross->create(LayerShaderType::sdpa_qkv_cross, opt, specializations);
        }
    }

    {
        qk_softmax = ncnn::create_layer_vulkan(ncnn::LayerType::Softmax);
        qk_softmax->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(0, -1);
        pd.set(1, 1);
        qk_softmax->load_param(pd);
        qk_softmax->load_model(ModelBinFromMatArray(0));
        qk_softmax->create_pipeline(opt);
    }

    {
        kvcache_concat = ncnn::create_layer_vulkan(ncnn::LayerType::Concat);
        kvcache_concat->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(0, 1); // axis
        kvcache_concat->load_param(pd);
        kvcache_concat->load_model(ModelBinFromMatArray(0));
        kvcache_concat->create_pipeline(opt);
    }

    return 0;
}

int SDPA_vulkan::destroy_pipeline(const Option& opt)
{
    delete pipeline_sdpa_qk_cross;
    pipeline_sdpa_qk_cross = 0;

    delete pipeline_sdpa_qkv_cross;
    pipeline_sdpa_qkv_cross = 0;

    if (qk_softmax)
    {
        qk_softmax->destroy_pipeline(opt);
        delete qk_softmax;
        qk_softmax = 0;
    }

    if (kvcache_concat)
    {
        kvcache_concat->destroy_pipeline(opt);
        delete kvcache_concat;
        kvcache_concat = 0;
    }

    return 0;
}

int SDPA_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkMat& query = bottom_blobs[0];
    const VkMat& cur_key = bottom_blobs[1];
    const VkMat& cur_value = bottom_blobs[2];
    const VkMat& attn_mask_blob = attn_mask ? bottom_blobs[3] : VkMat();
    const VkMat& past_key = kv_cache ? bottom_blobs[attn_mask ? 4 : 3] : VkMat();
    const VkMat& past_value = kv_cache ? bottom_blobs[attn_mask ? 5 : 4] : VkMat();

    const int embed_dim = query.w;
    const int src_seqlen = query.h;
    const int num_heads = query.c;
    const int cur_seqlen = cur_key.h;
    const int num_group = cur_key.c;
    const int out_embed_dim = cur_value.w;
    const int past_seqlen = kv_cache ? past_key.h : 0;
    const int dst_seqlen = past_seqlen + cur_seqlen;

    const float _scale = scale == 0.f ? 1.f / sqrt(embed_dim) : scale;

    const size_t elemsize = query.elemsize;

    VkMat key;
    if (past_seqlen > 0)
    {
        key.create(embed_dim, dst_seqlen, num_group, elemsize, opt.blob_vkallocator);
        if (key.empty())
            return -100;

        std::vector<VkMat> inputs(2);
        inputs[0] = past_key;
        inputs[1] = cur_key;
        std::vector<VkMat> outputs(1);
        kvcache_concat->forward(inputs, outputs, cmd, opt);
        key = outputs[0];
    }
    else
    {
        key = cur_key;
    }

    const int num_heads_per_group = num_heads / num_group;

    VkMat qk_cross(dst_seqlen, src_seqlen, num_heads, elemsize, opt.workspace_vkallocator);
    if (qk_cross.empty())
        return -100;

    // qk_cross;
    {
        int M = src_seqlen;
        int N = dst_seqlen;
        int K = embed_dim;
        int B = num_heads;

        std::vector<VkMat> bindings(4);
        bindings[0] = query;
        bindings[1] = key;
        bindings[2] = qk_cross;
        bindings[3] = attn_mask_blob;

        std::vector<vk_constant_type> constants(11);
        constants[0].f = _scale;
        constants[1].i = M;
        constants[2].i = N;
        constants[3].i = K;
        constants[4].i = B;
        constants[5].i = attn_mask_blob.dims;
        constants[6].i = num_heads_per_group;
        constants[7].i = query.cstep;
        constants[8].i = key.cstep;
        constants[9].i = qk_cross.cstep;
        constants[10].i = attn_mask_blob.cstep;

        VkMat dispatcher;
        dispatcher.w = (N + 1) / 2;
        dispatcher.h = (M + 1) / 2;
        dispatcher.c = B;

        cmd.record_pipeline(pipeline_sdpa_qk_cross, bindings, constants, dispatcher);
    }

    qk_softmax->forward_inplace(qk_cross, cmd, opt);

    VkMat value;
    if (past_seqlen > 0)
    {
        value.create(out_embed_dim, dst_seqlen, num_group, elemsize, opt.blob_vkallocator);
        if (value.empty())
            return -100;

        std::vector<VkMat> inputs(2);
        inputs[0] = past_value;
        inputs[1] = cur_value;
        std::vector<VkMat> outputs(1);
        kvcache_concat->forward(inputs, outputs, cmd, opt);
        value = outputs[0];
    }
    else
    {
        value = cur_value;
    }

    VkMat& top_blob = top_blobs[0];
    top_blob.create(out_embed_dim, src_seqlen, num_heads, elemsize, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    // qkv_cross;
    {
        int M = src_seqlen;
        int N = out_embed_dim;
        int K = dst_seqlen;
        int B = num_heads;

        std::vector<VkMat> bindings(3);
        bindings[0] = qk_cross;
        bindings[1] = value;
        bindings[2] = top_blob;

        std::vector<vk_constant_type> constants(8);
        constants[0].i = M;
        constants[1].i = N;
        constants[2].i = K;
        constants[3].i = B;
        constants[4].i = num_heads_per_group;
        constants[5].i = qk_cross.cstep;
        constants[6].i = value.cstep;
        constants[7].i = top_blob.cstep;

        VkMat dispatcher;
        dispatcher.w = (N + 1) / 2;
        dispatcher.h = (M + 1) / 2;
        dispatcher.c = B;

        cmd.record_pipeline(pipeline_sdpa_qkv_cross, bindings, constants, dispatcher);
    }

    if (kv_cache)
    {
        top_blobs[1] = key;
        top_blobs[2] = value;
    }

    return 0;
}

} // namespace ncnn
