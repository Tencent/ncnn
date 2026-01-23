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

    pipeline_sdpa_fa = 0;
    use_flash_attention = false;

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
    if (vkdev->info.support_bf16_cooperative_matrix() && opt.use_cooperative_matrix && (opt.use_bf16_storage || opt.use_bf16_packed))
    {
        use_cooperative_matrix = true;
        use_bf16_cooperative_matrix = true;
    }

    if (use_flash_attention)
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

        // fa
        {
            std::vector<vk_specialization_type> specializations(1 + 4);
            specializations[0].i = attn_mask;

            specializations[1 + 0].u32 = coopmat_M;
            specializations[1 + 1].u32 = coopmat_N;
            specializations[1 + 2].u32 = coopmat_K;
            specializations[1 + 3].u32 = coopmat_subgroup_size;

            pipeline_sdpa_fa = new Pipeline(vkdev);
            pipeline_sdpa_fa->set_subgroup_size(coopmat_subgroup_size);
            pipeline_sdpa_fa->set_local_size_xyz(coopmat_subgroup_size, 1, 1);
            pipeline_sdpa_fa->create(LayerShaderType::sdpa_fa_cm, opt, specializations);
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

    delete pipeline_sdpa_fa;
    pipeline_sdpa_fa = 0;

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

    use_flash_attention = false;

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



    if (use_flash_attention && embed_dim % 16 == 0 && out_embed_dim % 16 == 0)
    {
        NCNN_LOGE("flash attention enabled");
        NCNN_LOGE("num_heads_per_group %d", num_heads_per_group);
        NCNN_LOGE("embed_dim %d", embed_dim);
        NCNN_LOGE("out_embed_dim %d", out_embed_dim);

        VkMat& top_blob = top_blobs[0];
        top_blob.create(out_embed_dim, src_seqlen, num_heads, elemsize, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        // fa
        {
            std::vector<VkMat> bindings(4);
            bindings[0] = query;
            bindings[1] = key;
            bindings[2] = value;
            bindings[3] = top_blob;
            bindings[4] = attn_mask_blob;

            std::vector<vk_constant_type> constants(13);
            constants[0].f = 1.f; // scale
            constants[1].i = src_seqlen;
            constants[2].i = dst_seqlen;
            constants[3].i = embed_dim;
            constants[4].i = out_embed_dim;
            constants[5].i = num_heads;
            constants[6].i = 0; // attn_mask_dims
            constants[7].i = num_heads_per_group;
            constants[8].i = query.cstep;
            constants[9].i = key.cstep;
            constants[10].i = value.cstep;
            constants[11].i = top_blob.cstep;
            constants[12].i = attn_mask_blob.cstep;

            const int blocks_x = (src_seqlen + coopmat_M - 1) / (coopmat_M);
            const int blocks_y = 1;//(out_embed_dim + coopmat_N - 1) / (coopmat_N);

            VkMat dispatcher;
            dispatcher.w = (blocks_x * blocks_y) * (coopmat_subgroup_size);
            dispatcher.h = 1;
            dispatcher.c = B;

            cmd.record_pipeline(pipeline_sdpa_qkv_cross, bindings, constants, dispatcher);
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
            dispatcher.w = (N + 1) / 2;
            dispatcher.h = (M + 1) / 2;
            dispatcher.c = B;

            cmd.record_pipeline(pipeline_sdpa_qk_cross, bindings, constants, dispatcher);
        }
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
            dispatcher.w = (N + 1) / 2;
            dispatcher.h = (M + 1) / 2;
            dispatcher.c = B;

            cmd.record_pipeline(pipeline_sdpa_qkv_cross, bindings, constants, dispatcher);
        }
    }

    if (kv_cache)
    {
        top_blobs[1] = key;
        top_blobs[2] = value;
    }

    return 0;
}

} // namespace ncnn
