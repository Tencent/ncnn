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
    pipeline_sdpa_qk_cross = 0;
    pipeline_sdpa_qkv_cross = 0;
    pipeline_kvcache_copy = 0;
    pipeline_kvcache_append = 0;

    for (int i = 0; i < 8; i++)
    {
        pipeline_sdpa_fa[i] = 0;
    }
    use_flash_attention = false;
    FA_coopmat_M = 0;
    FA_coopmat_N = 0;
    FA_coopmat_K = 0;
    FA_coopmat_subgroup_size = 0;
    FA_UNROLL_SG_M = 1;
    FA_UNROLL_WG_M = 1;

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;
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
    use_cooperative_matrix = vkdev->info.support_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_fp16_storage || opt.use_fp16_packed);

    bool use_bf16_cooperative_matrix = false;
    if (vkdev->info.support_bf16_cooperative_matrix() && opt.use_cooperative_matrix && opt.use_bf16_storage)
    {
        use_cooperative_matrix = true;
        use_bf16_cooperative_matrix = true;
    }

    use_flash_attention = (opt.use_fp16_storage || opt.use_fp16_packed || opt.use_bf16_storage || opt.use_bf16_packed);
    if (use_flash_attention && use_cooperative_matrix)
    {
        const uint32_t support_subgroup_ops = vkdev->info.support_subgroup_ops();
        const uint32_t required_subgroup_ops = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        use_flash_attention = ((support_subgroup_ops & required_subgroup_ops) == required_subgroup_ops);
    }

    if (use_flash_attention)
    {
        if (use_cooperative_matrix)
        {
            int M = 1024;
            int N = 1024;
            int K = 1024;

            if (use_bf16_cooperative_matrix)
            {
                vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_BFLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, FA_coopmat_M, FA_coopmat_N, FA_coopmat_K, FA_coopmat_subgroup_size);
            }
            else
            {
                vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, FA_coopmat_M, FA_coopmat_N, FA_coopmat_K, FA_coopmat_subgroup_size);
            }

            // assert FA_coopmat_M != 0 && FA_coopmat_N != 0 && FA_coopmat_K != 0

            if (FA_coopmat_N != FA_coopmat_K || FA_coopmat_subgroup_size < FA_coopmat_N)
            {
                // not implemented yet
                use_flash_attention = false;
            }
            else
            {
                // fa
                FA_UNROLL_SG_M = 2;

                FA_UNROLL_WG_M = 2;

                std::vector<vk_specialization_type> specializations(1 + 8);
                specializations[0].i = attn_mask;

                specializations[1 + 0].u32 = FA_coopmat_M;
                specializations[1 + 1].u32 = FA_coopmat_N;
                specializations[1 + 2].u32 = FA_coopmat_K;
                specializations[1 + 3].u32 = FA_coopmat_subgroup_size;
                specializations[1 + 4].u32 = FA_UNROLL_SG_M;
                specializations[1 + 5].u32 = FA_UNROLL_WG_M;

                for (int i = 0; i < 8; i++)
                {
                    int MAX_OUT_CHUNKS = i + 1;
                    int UNROLL_P_N = std::min(4, FA_coopmat_subgroup_size / FA_coopmat_N);

                    specializations[1 + 6].u32 = MAX_OUT_CHUNKS;
                    specializations[1 + 7].u32 = UNROLL_P_N;

                    pipeline_sdpa_fa[i] = new Pipeline(vkdev);
                    pipeline_sdpa_fa[i]->set_subgroup_size(FA_coopmat_subgroup_size);
                    pipeline_sdpa_fa[i]->set_local_size_xyz(FA_coopmat_subgroup_size * FA_UNROLL_WG_M, 1, 1);
                    pipeline_sdpa_fa[i]->create(LayerShaderType::sdpa_fa_cm, opt, specializations);
                }
            }
        }
        else
        {
            FA_coopmat_M = 4;
            FA_coopmat_N = 32;
            FA_coopmat_K = 32;
            FA_UNROLL_WG_M = 4;
            const int subgroup_size = vkdev->info.subgroup_size();

            // assert FA_coopmat_N == FA_coopmat_K
            // assert local_size % FA_coopmat_M == 0

            // fa
            std::vector<vk_specialization_type> specializations(1 + 6);
            specializations[0].i = attn_mask;

            specializations[1 + 0].u32 = FA_coopmat_M;
            specializations[1 + 1].u32 = FA_coopmat_N;
            specializations[1 + 2].u32 = FA_coopmat_K;
            specializations[1 + 3].u32 = subgroup_size;
            specializations[1 + 4].u32 = FA_UNROLL_WG_M;

            for (int i = 0; i < 8; i++)
            {
                int MAX_OUT_CHUNKS = i + 1;

                specializations[1 + 5].u32 = MAX_OUT_CHUNKS;

                pipeline_sdpa_fa[i] = new Pipeline(vkdev);
                pipeline_sdpa_fa[i]->set_subgroup_size(subgroup_size);
                pipeline_sdpa_fa[i]->set_local_size_xyz(subgroup_size * FA_UNROLL_WG_M, 1, 1);
                pipeline_sdpa_fa[i]->create(LayerShaderType::sdpa_fa, opt, specializations);
            }
        }
    }

    if (use_cooperative_matrix)
    {
        int M = 1024;
        int N = 1024;
        int K = 1024;

        if (use_bf16_cooperative_matrix)
        {
            vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_BFLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);
        }
        else
        {
            vkdev->info.get_optimal_cooperative_matrix_mnk(M, N, K, VK_COMPONENT_TYPE_FLOAT16_KHR, opt.use_fp16_arithmetic ? VK_COMPONENT_TYPE_FLOAT16_KHR : VK_COMPONENT_TYPE_FLOAT32_KHR, VK_SCOPE_SUBGROUP_KHR, coopmat_M, coopmat_N, coopmat_K, coopmat_subgroup_size);
        }

        // assert coopmat_M != 0 && coopmat_N != 0 && coopmat_K != 0

        UNROLL_SG_M = std::min((M + coopmat_M - 1) / coopmat_M, 2);
        UNROLL_SG_N = std::min((N + coopmat_N - 1) / coopmat_N, 2);
        UNROLL_SG_K = std::min((K + coopmat_K - 1) / coopmat_K, 2);

        UNROLL_WG_M = std::min((M + coopmat_M * UNROLL_SG_M - 1) / (coopmat_M * UNROLL_SG_M), 2);
        UNROLL_WG_N = std::min((N + coopmat_N * UNROLL_SG_N - 1) / (coopmat_N * UNROLL_SG_N), 2);

        // qk cross
        {
            std::vector<vk_specialization_type> specializations(13 + 9);
            specializations[0].i = attn_mask;
            specializations[1].f = 0.f; // scale
            specializations[2].i = 0;   // M
            specializations[3].i = 0;   // N
            specializations[4].i = 0;   // K
            specializations[5].i = 0;   // B
            specializations[6].i = 1;   // transB
            specializations[7].i = 0;   // attn_mask.dims
            specializations[8].i = 0;   // num_heads_per_group
            specializations[9].i = 0;   // A_cstep
            specializations[10].i = 0;  // B_cstep
            specializations[11].i = 0;  // out_cstep
            specializations[12].i = 0;  // mask_cstep

            specializations[13 + 0].u32 = coopmat_M;
            specializations[13 + 1].u32 = coopmat_N;
            specializations[13 + 2].u32 = coopmat_K;
            specializations[13 + 3].u32 = coopmat_subgroup_size;
            specializations[13 + 4].u32 = UNROLL_SG_M;
            specializations[13 + 5].u32 = UNROLL_SG_N;
            specializations[13 + 6].u32 = UNROLL_SG_K;
            specializations[13 + 7].u32 = UNROLL_WG_M;
            specializations[13 + 8].u32 = UNROLL_WG_N;

            pipeline_sdpa_qk_cross = new Pipeline(vkdev);
            pipeline_sdpa_qk_cross->set_subgroup_size(coopmat_subgroup_size);
            pipeline_sdpa_qk_cross->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_sdpa_qk_cross->create(LayerShaderType::sdpa_cross_cm, opt, specializations);
        }

        // qkv cross
        {
            std::vector<vk_specialization_type> specializations(13 + 9);
            specializations[0].i = 0;   // attn_mask;
            specializations[1].f = 1.f; // scale
            specializations[2].i = 0;   // M
            specializations[3].i = 0;   // N
            specializations[4].i = 0;   // K
            specializations[5].i = 0;   // B
            specializations[6].i = 0;   // transB
            specializations[7].i = 0;   // attn_mask.dims
            specializations[8].i = 0;   // num_heads_per_group
            specializations[9].i = 0;   // A_cstep
            specializations[10].i = 0;  // B_cstep
            specializations[11].i = 0;  // out_cstep
            specializations[12].i = 0;  // mask_cstep

            specializations[13 + 0].u32 = coopmat_M;
            specializations[13 + 1].u32 = coopmat_N;
            specializations[13 + 2].u32 = coopmat_K;
            specializations[13 + 3].u32 = coopmat_subgroup_size;
            specializations[13 + 4].u32 = UNROLL_SG_M;
            specializations[13 + 5].u32 = UNROLL_SG_N;
            specializations[13 + 6].u32 = UNROLL_SG_K;
            specializations[13 + 7].u32 = UNROLL_WG_M;
            specializations[13 + 8].u32 = UNROLL_WG_N;

            pipeline_sdpa_qkv_cross = new Pipeline(vkdev);
            pipeline_sdpa_qkv_cross->set_subgroup_size(coopmat_subgroup_size);
            pipeline_sdpa_qkv_cross->set_local_size_xyz(coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N, 1, 1);
            pipeline_sdpa_qkv_cross->create(LayerShaderType::sdpa_cross_cm, opt, specializations);
        }
    }
    else
    {
        // qk cross
        {
            std::vector<vk_specialization_type> specializations(13);
            specializations[0].i = attn_mask;
            specializations[1].f = 0.f; // scale
            specializations[2].i = 0;   // M
            specializations[3].i = 0;   // N
            specializations[4].i = 0;   // K
            specializations[5].i = 0;   // B
            specializations[6].i = 1;   // transB
            specializations[7].i = 0;   // attn_mask.dims
            specializations[8].i = 0;   // num_heads_per_group
            specializations[9].i = 0;   // A_cstep
            specializations[10].i = 0;  // B_cstep
            specializations[11].i = 0;  // out_cstep
            specializations[12].i = 0;  // mask_cstep

            pipeline_sdpa_qk_cross = new Pipeline(vkdev);
            pipeline_sdpa_qk_cross->set_local_size_xyz(8, 8, 1);
            pipeline_sdpa_qk_cross->create(LayerShaderType::sdpa_cross, opt, specializations);
        }

        // qkv cross
        {
            std::vector<vk_specialization_type> specializations(13);
            specializations[0].i = 0;   // attn_mask;
            specializations[1].f = 1.f; // scale
            specializations[2].i = 0;   // M
            specializations[3].i = 0;   // N
            specializations[4].i = 0;   // K
            specializations[5].i = 0;   // B
            specializations[6].i = 0;   // transB
            specializations[7].i = 0;   // attn_mask.dims
            specializations[8].i = 0;   // num_heads_per_group
            specializations[9].i = 0;   // A_cstep
            specializations[10].i = 0;  // B_cstep
            specializations[11].i = 0;  // out_cstep
            specializations[12].i = 0;  // mask_cstep

            pipeline_sdpa_qkv_cross = new Pipeline(vkdev);
            pipeline_sdpa_qkv_cross->set_local_size_xyz(8, 8, 1);
            pipeline_sdpa_qkv_cross->create(LayerShaderType::sdpa_cross, opt, specializations);
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

    if (kv_cache)
    {
        std::vector<vk_specialization_type> specializations;
        pipeline_kvcache_copy = new Pipeline(vkdev);
        pipeline_kvcache_copy->set_local_size_xyz(8, 8, 1);
        pipeline_kvcache_copy->create(LayerShaderType::sdpa_kvcache_copy, opt, specializations);

        pipeline_kvcache_append = new Pipeline(vkdev);
        pipeline_kvcache_append->set_local_size_xyz(8, 8, 1);
        pipeline_kvcache_append->create(LayerShaderType::sdpa_kvcache_append, opt, specializations);
    }

    return 0;
}

int SDPA_vulkan::destroy_pipeline(const Option& opt)
{
    delete pipeline_sdpa_qk_cross;
    pipeline_sdpa_qk_cross = 0;

    delete pipeline_sdpa_qkv_cross;
    pipeline_sdpa_qkv_cross = 0;

    delete pipeline_kvcache_copy;
    pipeline_kvcache_copy = 0;

    delete pipeline_kvcache_append;
    pipeline_kvcache_append = 0;

    for (int i = 0; i < 8; i++)
    {
        delete pipeline_sdpa_fa[i];
        pipeline_sdpa_fa[i] = 0;
    }

    if (qk_softmax)
    {
        qk_softmax->destroy_pipeline(opt);
        delete qk_softmax;
        qk_softmax = 0;
    }

    use_flash_attention = false;
    FA_coopmat_M = 0;
    FA_coopmat_N = 0;
    FA_coopmat_K = 0;
    FA_coopmat_subgroup_size = 0;
    FA_UNROLL_SG_M = 1;
    FA_UNROLL_WG_M = 1;

    use_cooperative_matrix = false;
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;
    UNROLL_SG_M = 1;
    UNROLL_SG_N = 1;
    UNROLL_SG_K = 1;
    UNROLL_WG_M = 1;
    UNROLL_WG_N = 1;

    return 0;
}

int SDPA_vulkan::create_or_grow_kvcache(const VkMat& cache, VkMat& new_cache, int new_seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack, VkCompute& cmd, const Option& opt) const
{
    if (!cache.empty() && new_seqlen <= cache.h)
    {
        new_cache = cache;
        new_cache.h = new_seqlen;
        return 0;
    }

    VkAllocator* allocator = opt.kvcache_vkallocator ? opt.kvcache_vkallocator : opt.blob_vkallocator;
    if (opt.kvcache_vkallocator && !cache.empty() && cache.allocator == allocator)
    {
        const int capacity = (int)(cache.cstep / cache.w);
        if (new_seqlen <= capacity)
        {
            new_cache = cache;
            new_cache.h = new_seqlen;
            return 0;
        }
    }

    int capacity = new_seqlen > 0 ? new_seqlen : 1;
    if (opt.kvcache_vkallocator)
    {
        const int current_capacity = cache.empty() ? 0 : (int)(cache.cstep / cache.w);
        capacity = kvcache_capacity(current_capacity, new_seqlen, opt.kvcache_max_seqlen_hint);
    }

    VkMat m;
    m.create(head_dim, capacity, num_kv_head, elemsize, elempack, allocator);
    if (m.empty())
        return -100;

    if (!cache.empty())
    {
        std::vector<VkMat> bindings(2);
        bindings[0] = cache;
        bindings[1] = m;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = cache.w;
        constants[1].i = cache.h;
        constants[2].i = cache.cstep;
        constants[3].i = m.cstep;

        cmd.record_pipeline(pipeline_kvcache_copy, bindings, constants, cache);
    }

    m.h = new_seqlen;
    new_cache = m;

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

    VkMat key = cur_key;
    VkMat value = cur_value;
    if (kv_cache)
    {
        VkMat& cached_key = top_blobs[1];
        VkMat& cached_value = top_blobs[2];

        int retk = create_or_grow_kvcache(past_key, cached_key, dst_seqlen, num_group, embed_dim, cur_key.elemsize, cur_key.elempack, cmd, opt);
        if (retk != 0)
            return retk;

        int retv = create_or_grow_kvcache(past_value, cached_value, dst_seqlen, num_group, out_embed_dim, cur_value.elemsize, cur_value.elempack, cmd, opt);
        if (retv != 0)
            return retv;

        std::vector<VkMat> key_bindings(2);
        key_bindings[0] = cur_key;
        key_bindings[1] = cached_key;
        std::vector<vk_constant_type> key_constants(6);
        key_constants[0].i = embed_dim;
        key_constants[1].i = cur_key.h;
        key_constants[2].i = cur_key.cstep;
        key_constants[3].i = cached_key.w;
        key_constants[4].i = cached_key.cstep;
        key_constants[5].i = past_seqlen;
        cmd.record_pipeline(pipeline_kvcache_append, key_bindings, key_constants, cur_key);

        std::vector<VkMat> value_bindings(2);
        value_bindings[0] = cur_value;
        value_bindings[1] = cached_value;
        std::vector<vk_constant_type> value_constants(6);
        value_constants[0].i = out_embed_dim;
        value_constants[1].i = cur_value.h;
        value_constants[2].i = cur_value.cstep;
        value_constants[3].i = cached_value.w;
        value_constants[4].i = cached_value.cstep;
        value_constants[5].i = past_seqlen;
        cmd.record_pipeline(pipeline_kvcache_append, value_bindings, value_constants, cur_value);

        key = cached_key;
        value = cached_value;
    }
    const int num_heads_per_group = num_heads / num_group;

    if (use_flash_attention && embed_dim % 8 == 0 && out_embed_dim % 8 == 0 && out_embed_dim <= FA_coopmat_N * 8)
    {
        VkMat& top_blob = top_blobs[0];
        top_blob.create(out_embed_dim, src_seqlen, num_heads, elemsize, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        if (use_cooperative_matrix)
        {
            std::vector<VkMat> bindings(5);
            bindings[0] = query;
            bindings[1] = key;
            bindings[2] = value;
            bindings[3] = top_blob;
            bindings[4] = attn_mask_blob;

            std::vector<vk_constant_type> constants(13);
            constants[0].f = _scale;
            constants[1].i = src_seqlen;
            constants[2].i = dst_seqlen;
            constants[3].i = embed_dim;
            constants[4].i = out_embed_dim;
            constants[5].i = num_heads;
            constants[6].i = attn_mask_blob.dims && attn_mask_blob.c > 1 ? 3 : attn_mask_blob.dims;
            constants[7].i = num_heads_per_group;
            constants[8].i = query.cstep;
            constants[9].i = key.cstep;
            constants[10].i = value.cstep;
            constants[11].i = top_blob.cstep;
            constants[12].i = attn_mask_blob.cstep;

            const int blocks_x = 1;
            const int blocks_y = (src_seqlen + FA_coopmat_M * FA_UNROLL_SG_M * FA_UNROLL_WG_M - 1) / (FA_coopmat_M * FA_UNROLL_SG_M * FA_UNROLL_WG_M);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (FA_coopmat_subgroup_size * FA_UNROLL_WG_M);
            dispatcher.h = 1;
            dispatcher.c = num_heads;

            const int MAX_OUT_CHUNKS = (out_embed_dim + FA_coopmat_N - 1) / FA_coopmat_N;

            const Pipeline* pipeline = pipeline_sdpa_fa[MAX_OUT_CHUNKS - 1];

            cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
        }
        else
        {
            std::vector<VkMat> bindings(5);
            bindings[0] = query;
            bindings[1] = key;
            bindings[2] = value;
            bindings[3] = top_blob;
            bindings[4] = attn_mask_blob;

            std::vector<vk_constant_type> constants(13);
            constants[0].f = _scale;
            constants[1].i = src_seqlen;
            constants[2].i = dst_seqlen;
            constants[3].i = embed_dim;
            constants[4].i = out_embed_dim;
            constants[5].i = num_heads;
            constants[6].i = attn_mask_blob.dims && attn_mask_blob.c > 1 ? 3 : attn_mask_blob.dims;
            constants[7].i = num_heads_per_group;
            constants[8].i = query.cstep;
            constants[9].i = key.cstep;
            constants[10].i = value.cstep;
            constants[11].i = top_blob.cstep;
            constants[12].i = attn_mask_blob.cstep;

            const int subgroup_size = vkdev->info.subgroup_size();

            const int blocks_x = 1;
            const int blocks_y = (src_seqlen + FA_coopmat_M - 1) / FA_coopmat_M;

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (subgroup_size * FA_UNROLL_WG_M);
            dispatcher.h = 1;
            dispatcher.c = num_heads;

            const int MAX_OUT_CHUNKS = (out_embed_dim + FA_coopmat_N - 1) / FA_coopmat_N;

            const Pipeline* pipeline = pipeline_sdpa_fa[MAX_OUT_CHUNKS - 1];

            cmd.record_pipeline(pipeline, bindings, constants, dispatcher);
        }

        return 0;
    }

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

        if (use_cooperative_matrix)
        {
            const int blocks_x = (M + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (N + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = B;

            cmd.record_pipeline(pipeline_sdpa_qk_cross, bindings, constants, dispatcher);
        }
        else
        {
            VkMat dispatcher;
            dispatcher.w = (N + 3) / 4;
            dispatcher.h = (M + 3) / 4;
            dispatcher.c = B;

            cmd.record_pipeline(pipeline_sdpa_qk_cross, bindings, constants, dispatcher);
        }
    }

    qk_softmax->forward_inplace(qk_cross, cmd, opt);

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

        std::vector<VkMat> bindings(4);
        bindings[0] = qk_cross;
        bindings[1] = value;
        bindings[2] = top_blob;
        bindings[3] = VkMat();

        std::vector<vk_constant_type> constants(11);
        constants[0].f = 1.f; // scale
        constants[1].i = M;
        constants[2].i = N;
        constants[3].i = K;
        constants[4].i = B;
        constants[5].i = 0; // attn_mask_dims
        constants[6].i = num_heads_per_group;
        constants[7].i = qk_cross.cstep;
        constants[8].i = value.cstep;
        constants[9].i = top_blob.cstep;
        constants[10].i = 0; // mask_cstep

        if (use_cooperative_matrix)
        {
            const int blocks_x = (M + coopmat_M * UNROLL_SG_M * UNROLL_WG_M - 1) / (coopmat_M * UNROLL_SG_M * UNROLL_WG_M);
            const int blocks_y = (N + coopmat_N * UNROLL_SG_N * UNROLL_WG_N - 1) / (coopmat_N * UNROLL_SG_N * UNROLL_WG_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size * UNROLL_WG_M * UNROLL_WG_N);
            dispatcher.h = 1;
            dispatcher.c = B;

            cmd.record_pipeline(pipeline_sdpa_qkv_cross, bindings, constants, dispatcher);
        }
        else
        {
            VkMat dispatcher;
            dispatcher.w = (N + 3) / 4;
            dispatcher.h = (M + 3) / 4;
            dispatcher.c = B;

            cmd.record_pipeline(pipeline_sdpa_qkv_cross, bindings, constants, dispatcher);
        }
    }

    return 0;
}

} // namespace ncnn
