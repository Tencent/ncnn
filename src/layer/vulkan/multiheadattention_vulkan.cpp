// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "multiheadattention_vulkan.h"

#include "layer_shader_type.h"
#include "layer_type.h"

namespace ncnn {

MultiHeadAttention_vulkan::MultiHeadAttention_vulkan()
{
    support_vulkan = true;
    support_vulkan_packing = true;

    q_gemm = 0;
    k_gemm = 0;
    v_gemm = 0;

    qk_softmax = 0;

    o_gemm = 0;

    pipeline_multiheadattention_qk_cross = 0;
    pipeline_multiheadattention_qk_cross_pack4 = 0;
    pipeline_multiheadattention_qk_cross_pack1to4 = 0;
    pipeline_multiheadattention_qk_cross_pack4to1 = 0;

    pipeline_multiheadattention_qkv_cross = 0;
    pipeline_multiheadattention_qkv_cross_pack4 = 0;
    pipeline_multiheadattention_qkv_cross_pack1to4 = 0;
    pipeline_multiheadattention_qkv_cross_pack4to1 = 0;

    pipeline_kvcache_copy = 0;
    pipeline_kvcache_append = 0;
    pipeline_kvcache_qk = 0;
    pipeline_kvcache_qkv = 0;
}

int MultiHeadAttention_vulkan::load_param(const ParamDict& pd)
{
    int ret = MultiHeadAttention::load_param(pd);

    if (int8_scale_term)
    {
        support_vulkan = false;
    }

    return ret;
}

int MultiHeadAttention_vulkan::create_pipeline(const Option& opt)
{
    if (weight_block_quantize)
    {
        NCNN_LOGE("MultiHeadAttention weight block quantization is not supported by Vulkan");
        return -1;
    }

    // const int embed_dim_per_head = embed_dim / num_heads;
    const int qdim = weight_data_size / embed_dim;
    {
        q_gemm = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
        q_gemm->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(0, scale);
        pd.set(1, 1.f);
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 1);         // constantA
        pd.set(5, 0);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, embed_dim); // M
        pd.set(8, 0);         // N
        pd.set(9, qdim);      // K
        pd.set(10, 1);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        if (kv_cache)
            pd.set(12, 1); // output_elempack
        pd.set(14, 0);     // output_transpose
        q_gemm->load_param(pd);
        Mat weights[2];
        weights[0] = q_weight_data;
        weights[1] = q_bias_data;
        q_gemm->load_model(ModelBinFromMatArray(weights));
        q_gemm->create_pipeline(opt);

        if (opt.lightmode)
        {
            q_weight_data.release();
            q_bias_data.release();
        }
    }

    {
        k_gemm = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
        k_gemm->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 1);         // constantA
        pd.set(5, 0);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, embed_dim); // M
        pd.set(8, 0);         // N
        pd.set(9, kdim);      // K
        pd.set(10, 1);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        if (kv_cache)
            pd.set(12, 1); // output_elempack
        pd.set(14, 0);     // output_transpose
        k_gemm->load_param(pd);
        Mat weights[2];
        weights[0] = k_weight_data;
        weights[1] = k_bias_data;
        k_gemm->load_model(ModelBinFromMatArray(weights));
        k_gemm->create_pipeline(opt);

        if (opt.lightmode)
        {
            k_weight_data.release();
            k_bias_data.release();
        }
    }

    {
        v_gemm = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
        v_gemm->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(2, 0);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 1);         // constantA
        pd.set(5, 0);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, embed_dim); // M
        pd.set(8, 0);         // N
        pd.set(9, vdim);      // K
        pd.set(10, 1);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        if (kv_cache)
            pd.set(12, 1); // output_elempack
        pd.set(14, 0);     // output_transpose
        v_gemm->load_param(pd);
        Mat weights[2];
        weights[0] = v_weight_data;
        weights[1] = v_bias_data;
        v_gemm->load_model(ModelBinFromMatArray(weights));
        v_gemm->create_pipeline(opt);

        if (opt.lightmode)
        {
            v_weight_data.release();
            v_bias_data.release();
        }
    }

    if (!kv_cache)
    {
        std::vector<vk_specialization_type> specializations(6);
        specializations[0].i = attn_mask;
        specializations[1].i = 0; //constantM;
        specializations[2].i = 0; //constantN;
        specializations[3].i = 0; //embed_dim_per_head;//constantK;
        specializations[4].i = num_heads;
        specializations[5].i = 0; //attn_mask.dims;

        {
            pipeline_multiheadattention_qk_cross = new Pipeline(vkdev);
            pipeline_multiheadattention_qk_cross->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qk_cross->create(LayerShaderType::multiheadattention_qk_cross, opt, specializations);
        }
        {
            pipeline_multiheadattention_qk_cross_pack4 = new Pipeline(vkdev);
            pipeline_multiheadattention_qk_cross_pack4->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qk_cross_pack4->create(LayerShaderType::multiheadattention_qk_cross_pack4, opt, specializations);
        }
        {
            pipeline_multiheadattention_qk_cross_pack1to4 = new Pipeline(vkdev);
            pipeline_multiheadattention_qk_cross_pack1to4->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qk_cross_pack1to4->create(LayerShaderType::multiheadattention_qk_cross_pack1to4, opt, specializations);
        }
        {
            pipeline_multiheadattention_qk_cross_pack4to1 = new Pipeline(vkdev);
            pipeline_multiheadattention_qk_cross_pack4to1->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qk_cross_pack4to1->create(LayerShaderType::multiheadattention_qk_cross_pack4to1, opt, specializations);
        }
    }
    if (!kv_cache)
    {
        std::vector<vk_specialization_type> specializations(4);
        specializations[0].i = 0; //constantM;
        specializations[1].i = 0; //embed_dim_per_head;//constantN;
        specializations[2].i = 0; //constantK;
        specializations[3].i = num_heads;

        {
            pipeline_multiheadattention_qkv_cross = new Pipeline(vkdev);
            pipeline_multiheadattention_qkv_cross->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qkv_cross->create(LayerShaderType::multiheadattention_qkv_cross, opt, specializations);
        }
        {
            pipeline_multiheadattention_qkv_cross_pack4 = new Pipeline(vkdev);
            pipeline_multiheadattention_qkv_cross_pack4->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qkv_cross_pack4->create(LayerShaderType::multiheadattention_qkv_cross_pack4, opt, specializations);
        }
        {
            pipeline_multiheadattention_qkv_cross_pack1to4 = new Pipeline(vkdev);
            pipeline_multiheadattention_qkv_cross_pack1to4->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qkv_cross_pack1to4->create(LayerShaderType::multiheadattention_qkv_cross_pack1to4, opt, specializations);
        }
        {
            pipeline_multiheadattention_qkv_cross_pack4to1 = new Pipeline(vkdev);
            pipeline_multiheadattention_qkv_cross_pack4to1->set_local_size_xyz(8, 8, 1);
            pipeline_multiheadattention_qkv_cross_pack4to1->create(LayerShaderType::multiheadattention_qkv_cross_pack4to1, opt, specializations);
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
        o_gemm = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
        o_gemm->vkdev = vkdev;
        ncnn::ParamDict pd;
        pd.set(2, 1);         // transA
        pd.set(3, 1);         // transB
        pd.set(4, 0);         // constantA
        pd.set(5, 1);         // constantB
        pd.set(6, 1);         // constantC
        pd.set(7, 0);         // M = outch
        pd.set(8, qdim);      // N = size
        pd.set(9, embed_dim); // K = maxk*inch
        pd.set(10, 4);        // constant_broadcast_type_C
        pd.set(11, 0);        // output_N1M
        o_gemm->load_param(pd);
        Mat weights[2];
        weights[0] = out_weight_data;
        weights[1] = out_bias_data;
        o_gemm->load_model(ModelBinFromMatArray(weights));
        o_gemm->create_pipeline(opt);

        if (opt.lightmode)
        {
            out_weight_data.release();
            out_bias_data.release();
        }
    }

    if (kv_cache)
    {
        std::vector<vk_specialization_type> specializations;

        pipeline_kvcache_copy = new Pipeline(vkdev);
        pipeline_kvcache_copy->set_local_size_xyz(8, 8, 1);
        pipeline_kvcache_copy->create(LayerShaderType::multiheadattention_kvcache_copy, opt, specializations);

        pipeline_kvcache_append = new Pipeline(vkdev);
        pipeline_kvcache_append->set_local_size_xyz(8, 8, 1);
        pipeline_kvcache_append->create(LayerShaderType::multiheadattention_kvcache_append, opt, specializations);

        std::vector<vk_specialization_type> qk_specializations(1);
        qk_specializations[0].i = attn_mask;

        pipeline_kvcache_qk = new Pipeline(vkdev);
        pipeline_kvcache_qk->set_local_size_xyz(8, 8, 1);
        pipeline_kvcache_qk->create(LayerShaderType::multiheadattention_kvcache_qk, opt, qk_specializations);

        std::vector<vk_specialization_type> qkv_specializations;

        pipeline_kvcache_qkv = new Pipeline(vkdev);
        pipeline_kvcache_qkv->set_local_size_xyz(8, 8, 1);
        pipeline_kvcache_qkv->create(LayerShaderType::multiheadattention_kvcache_qkv, opt, qkv_specializations);
    }

    return 0;
}

int MultiHeadAttention_vulkan::destroy_pipeline(const Option& opt)
{
    if (weight_block_quantize)
        return 0;

    if (q_gemm)
    {
        q_gemm->destroy_pipeline(opt);
        delete q_gemm;
        q_gemm = 0;
    }

    if (k_gemm)
    {
        k_gemm->destroy_pipeline(opt);
        delete k_gemm;
        k_gemm = 0;
    }

    if (v_gemm)
    {
        v_gemm->destroy_pipeline(opt);
        delete v_gemm;
        v_gemm = 0;
    }

    delete pipeline_multiheadattention_qk_cross;
    pipeline_multiheadattention_qk_cross = 0;

    delete pipeline_multiheadattention_qk_cross_pack4;
    pipeline_multiheadattention_qk_cross_pack4 = 0;

    delete pipeline_multiheadattention_qk_cross_pack1to4;
    pipeline_multiheadattention_qk_cross_pack1to4 = 0;

    delete pipeline_multiheadattention_qk_cross_pack4to1;
    pipeline_multiheadattention_qk_cross_pack4to1 = 0;

    delete pipeline_multiheadattention_qkv_cross;
    pipeline_multiheadattention_qkv_cross = 0;

    delete pipeline_multiheadattention_qkv_cross_pack4;
    pipeline_multiheadattention_qkv_cross_pack4 = 0;

    delete pipeline_multiheadattention_qkv_cross_pack1to4;
    pipeline_multiheadattention_qkv_cross_pack1to4 = 0;

    delete pipeline_multiheadattention_qkv_cross_pack4to1;
    pipeline_multiheadattention_qkv_cross_pack4to1 = 0;

    if (qk_softmax)
    {
        qk_softmax->destroy_pipeline(opt);
        delete qk_softmax;
        qk_softmax = 0;
    }

    if (o_gemm)
    {
        o_gemm->destroy_pipeline(opt);
        delete o_gemm;
        o_gemm = 0;
    }

    delete pipeline_kvcache_copy;
    pipeline_kvcache_copy = 0;

    delete pipeline_kvcache_append;
    pipeline_kvcache_append = 0;

    delete pipeline_kvcache_qk;
    pipeline_kvcache_qk = 0;

    delete pipeline_kvcache_qkv;
    pipeline_kvcache_qkv = 0;

    return 0;
}

int MultiHeadAttention_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    if (weight_block_quantize)
    {
        NCNN_LOGE("MultiHeadAttention weight block quantization is not supported by Vulkan");
        return -1;
    }

    if (q_gemm)
    {
        q_gemm->upload_model(cmd, opt);
    }

    if (k_gemm)
    {
        k_gemm->upload_model(cmd, opt);
    }

    if (v_gemm)
    {
        v_gemm->upload_model(cmd, opt);
    }

    if (o_gemm)
    {
        o_gemm->upload_model(cmd, opt);
    }

    return 0;
}

int MultiHeadAttention_vulkan::create_or_grow_kvcache(const VkMat& cache, VkMat& new_cache, int new_seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack, VkCompute& cmd, const Option& opt) const
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

int MultiHeadAttention_vulkan::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    if (weight_block_quantize)
    {
        NCNN_LOGE("MultiHeadAttention weight block quantization is not supported by Vulkan");
        return -1;
    }

    int q_blob_i = 0;
    int k_blob_i = 0;
    int v_blob_i = 0;
    int attn_mask_i = 0;
    int cached_xk_i = 0;
    int cached_xv_i = 0;
    resolve_bottom_blob_index((int)bottom_blobs.size(), q_blob_i, k_blob_i, v_blob_i, attn_mask_i, cached_xk_i, cached_xv_i);

    const VkMat& q_blob = bottom_blobs[q_blob_i];
    const VkMat& k_blob = bottom_blobs[k_blob_i];
    const VkMat& v_blob = bottom_blobs[v_blob_i];
    const VkMat& attn_mask_blob = attn_mask ? bottom_blobs[attn_mask_i] : VkMat();
    VkMat empty_cache;
    const VkMat& past_xk_blob = kv_cache ? bottom_blobs[cached_xk_i] : empty_cache;
    const VkMat& past_xv_blob = kv_cache ? bottom_blobs[cached_xv_i] : empty_cache;
    VkMat& cached_xk_blob = kv_cache ? top_blobs[1] : empty_cache;
    VkMat& cached_xv_blob = kv_cache ? top_blobs[2] : empty_cache;

    VkMat past_xk_blob_unpacked;
    if (kv_cache && !past_xk_blob.empty() && past_xk_blob.elempack != 1)
    {
        vkdev->convert_packing(past_xk_blob, past_xk_blob_unpacked, 1, cmd, opt);
        if (past_xk_blob_unpacked.empty())
            return -100;
    }
    else
    {
        past_xk_blob_unpacked = past_xk_blob;
    }

    VkMat past_xv_blob_unpacked;
    if (kv_cache && !past_xv_blob.empty() && past_xv_blob.elempack != 1)
    {
        vkdev->convert_packing(past_xv_blob, past_xv_blob_unpacked, 1, cmd, opt);
        if (past_xv_blob_unpacked.empty())
            return -100;
    }
    else
    {
        past_xv_blob_unpacked = past_xv_blob;
    }

    const int embed_dim_per_head = embed_dim / num_heads;
    const int src_seqlen = q_blob.h * q_blob.elempack;
    const int cur_seqlen = k_blob.h * k_blob.elempack;
    const int past_seqlen = kv_cache && !past_xk_blob_unpacked.empty() ? past_xk_blob_unpacked.h : 0;
    const bool append_kv = past_seqlen == 0 || q_blob_i == k_blob_i;
    const int append_seqlen = append_kv ? cur_seqlen : 0;
    const int dst_seqlen = past_seqlen + append_seqlen;

    VkMat q_affine;
    q_gemm->forward(q_blob, q_affine, cmd, opt);

    VkMat k_affine;
    VkMat current_key;
    VkMat current_value;
    if (kv_cache)
    {
        if (append_seqlen > 0)
        {
            int retk = k_gemm->forward(k_blob, current_key, cmd, opt);
            if (retk != 0)
                return retk;
            int retv = v_gemm->forward(v_blob, current_value, cmd, opt);
            if (retv != 0)
                return retv;
        }

        int retk = create_or_grow_kvcache(past_xk_blob_unpacked, cached_xk_blob, dst_seqlen, num_heads, embed_dim_per_head, current_key.elemsize, 1, cmd, opt);
        if (retk != 0)
            return retk;

        int retv = create_or_grow_kvcache(past_xv_blob_unpacked, cached_xv_blob, dst_seqlen, num_heads, embed_dim_per_head, current_value.elemsize, 1, cmd, opt);
        if (retv != 0)
            return retv;

        if (append_seqlen > 0)
        {
            VkMat dispatcher;
            dispatcher.w = embed_dim_per_head;
            dispatcher.h = append_seqlen;
            dispatcher.c = num_heads;

            std::vector<vk_constant_type> constants(4);
            constants[0].i = append_seqlen;
            constants[1].i = embed_dim_per_head;
            constants[2].i = cached_xk_blob.cstep;
            constants[3].i = past_seqlen;

            std::vector<VkMat> bindings(2);
            bindings[0] = current_key;
            bindings[1] = cached_xk_blob;
            cmd.record_pipeline(pipeline_kvcache_append, bindings, constants, dispatcher);

            constants[2].i = cached_xv_blob.cstep;
            bindings[0] = current_value;
            bindings[1] = cached_xv_blob;
            cmd.record_pipeline(pipeline_kvcache_append, bindings, constants, dispatcher);
        }
    }
    else
    {
        k_gemm->forward(k_blob, k_affine, cmd, opt);
    }

    VkMat qk_cross;
    if (kv_cache)
    {
        VkMat attn_mask_blob_unpacked;
        if (attn_mask && attn_mask_blob.elempack != 1)
        {
            vkdev->convert_packing(attn_mask_blob, attn_mask_blob_unpacked, 1, cmd, opt);
            if (attn_mask_blob_unpacked.empty())
                return -100;
        }
        else
        {
            attn_mask_blob_unpacked = attn_mask_blob;
        }

        qk_cross.create(dst_seqlen, src_seqlen * num_heads, q_affine.elemsize, 1, opt.workspace_vkallocator);
        if (qk_cross.empty())
            return -100;

        std::vector<VkMat> bindings(4);
        bindings[0] = q_affine;
        bindings[1] = cached_xk_blob;
        bindings[2] = qk_cross;
        bindings[3] = attn_mask_blob_unpacked;

        std::vector<vk_constant_type> constants(7);
        constants[0].i = src_seqlen;
        constants[1].i = dst_seqlen;
        constants[2].i = embed_dim_per_head;
        constants[3].i = num_heads;
        constants[4].i = cached_xk_blob.cstep;
        constants[5].i = attn_mask_blob_unpacked.dims;
        constants[6].i = attn_mask_blob_unpacked.cstep;

        VkMat dispatcher;
        dispatcher.w = dst_seqlen;
        dispatcher.h = src_seqlen;
        dispatcher.c = num_heads;
        cmd.record_pipeline(pipeline_kvcache_qk, bindings, constants, dispatcher);
    }
    else
    {
        int M = q_affine.w;
        int N = k_affine.w;
        int K = q_affine.h * q_affine.elempack / num_heads;
        int B = num_heads;

        int K_elempack = K % 4 == 0 ? 4 : 1;
        int M_elempack = M % 4 == 0 ? 4 : 1;
        int MB_elempack = (M * B) % 4 == 0 ? 4 : 1;
        size_t M_elemsize = q_affine.elemsize / q_affine.elempack * M_elempack;

        if (K_elempack < q_affine.elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(q_affine, tmp, K_elempack, cmd, opt);
            q_affine = tmp;
        }
        if (K_elempack < k_affine.elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(k_affine, tmp, K_elempack, cmd, opt);
            k_affine = tmp;
        }
        VkMat attn_mask_blob_unpacked = attn_mask_blob;
        if (M_elempack < attn_mask_blob.elempack)
        {
            vkdev->convert_packing(attn_mask_blob, attn_mask_blob_unpacked, M_elempack, cmd, opt);
        }

        qk_cross.create(N, M / M_elempack * B, M_elemsize, M_elempack, opt.blob_vkallocator);
        if (qk_cross.empty())
            return -100;

        std::vector<VkMat> bindings(4);
        bindings[0] = q_affine;
        bindings[1] = k_affine;
        bindings[2] = qk_cross;
        bindings[3] = attn_mask_blob_unpacked;

        std::vector<vk_constant_type> constants(5);
        constants[0].i = M / M_elempack;
        constants[1].i = N;
        constants[2].i = K / K_elempack;
        constants[3].i = B;
        constants[4].i = attn_mask_blob_unpacked.dims;

        VkMat dispatcher;
        dispatcher.w = N;
        dispatcher.h = M / M_elempack;
        dispatcher.c = B;

        const Pipeline* pipeline = 0;
        if (K_elempack == 1 && M_elempack == 1)
        {
            pipeline = pipeline_multiheadattention_qk_cross;
        }
        if (K_elempack == 1 && M_elempack == 4)
        {
            pipeline = pipeline_multiheadattention_qk_cross_pack1to4;
        }
        if (K_elempack == 4 && M_elempack == 1)
        {
            pipeline = pipeline_multiheadattention_qk_cross_pack4to1;
        }
        if (K_elempack == 4 && M_elempack == 4)
        {
            pipeline = pipeline_multiheadattention_qk_cross_pack4;
        }

        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

        if (MB_elempack > M_elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(qk_cross, tmp, MB_elempack, cmd, opt);
            qk_cross = tmp;
        }
    }

    q_affine.release();

    if (!kv_cache)
    {
        k_affine.release();
    }

    qk_softmax->forward_inplace(qk_cross, cmd, opt);

    if (vkdev->info.vendor_id() == 0x10de)
    {
        // FIXME softmax produces nan result on nvidia (about 20% chance)
        // memory barrier seems to be not enough here
        // device copy-to and copy-back is better than queue submit anyway  --- nihui

        cmd.submit_and_wait();
        cmd.reset();

        // VkImageMat qk_cross2;
        // cmd.record_buffer_to_image(qk_cross, qk_cross2, opt);
        // cmd.record_image_to_buffer(qk_cross2, qk_cross, opt);
    }

    VkMat v_affine;
    if (!kv_cache)
    {
        v_gemm->forward(v_blob, v_affine, cmd, opt);
    }

    VkMat qkv_cross;
    if (kv_cache)
    {
        qkv_cross.create(src_seqlen, embed_dim_per_head * num_heads, cached_xv_blob.elemsize, 1, opt.workspace_vkallocator);
        if (qkv_cross.empty())
            return -100;

        std::vector<VkMat> bindings(3);
        bindings[0] = qk_cross;
        bindings[1] = cached_xv_blob;
        bindings[2] = qkv_cross;

        std::vector<vk_constant_type> constants(5);
        constants[0].i = src_seqlen;
        constants[1].i = dst_seqlen;
        constants[2].i = embed_dim_per_head;
        constants[3].i = num_heads;
        constants[4].i = cached_xv_blob.cstep;

        VkMat dispatcher;
        dispatcher.w = embed_dim_per_head;
        dispatcher.h = src_seqlen;
        dispatcher.c = num_heads;
        cmd.record_pipeline(pipeline_kvcache_qkv, bindings, constants, dispatcher);
    }
    else
    {
        int M = qk_cross.h * qk_cross.elempack / num_heads;
        int N = v_affine.h * v_affine.elempack / num_heads;
        int K = v_affine.w;
        int B = num_heads;

        int M_elempack = M % 4 == 0 ? 4 : 1;
        int N_elempack = N % 4 == 0 ? 4 : 1;
        int NB_elempack = (N * B) % 4 == 0 ? 4 : 1;
        size_t N_elemsize = v_affine.elemsize / v_affine.elempack * N_elempack;

        if (M_elempack < qk_cross.elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(qk_cross, tmp, M_elempack, cmd, opt);
            qk_cross = tmp;
        }

        if (N_elempack < v_affine.elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(v_affine, tmp, N_elempack, cmd, opt);
            v_affine = tmp;
        }

        qkv_cross.create(M, N / N_elempack * B, N_elemsize, N_elempack, opt.blob_vkallocator);
        if (qkv_cross.empty())
            return -100;

        std::vector<VkMat> bindings(3);
        bindings[0] = qk_cross;
        bindings[1] = v_affine;
        bindings[2] = qkv_cross;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = M / M_elempack;
        constants[1].i = N / N_elempack;
        constants[2].i = K;
        constants[3].i = B;

        VkMat dispatcher;
        dispatcher.w = N / N_elempack;
        dispatcher.h = M / M_elempack;
        dispatcher.c = B;

        const Pipeline* pipeline = 0;
        if (M_elempack == 1 && N_elempack == 1)
        {
            pipeline = pipeline_multiheadattention_qkv_cross;
        }
        if (M_elempack == 1 && N_elempack == 4)
        {
            pipeline = pipeline_multiheadattention_qkv_cross_pack1to4;
        }
        if (M_elempack == 4 && N_elempack == 1)
        {
            pipeline = pipeline_multiheadattention_qkv_cross_pack4to1;
        }
        if (M_elempack == 4 && N_elempack == 4)
        {
            pipeline = pipeline_multiheadattention_qkv_cross_pack4;
        }

        cmd.record_pipeline(pipeline, bindings, constants, dispatcher);

        if (NB_elempack > N_elempack)
        {
            VkMat tmp;
            vkdev->convert_packing(qkv_cross, tmp, NB_elempack, cmd, opt);
            qkv_cross = tmp;
        }
    }

    qk_cross.release();

    if (!kv_cache)
    {
        v_affine.release();
    }

    o_gemm->forward(qkv_cross, top_blobs[0], cmd, opt);

    return 0;
}

} // namespace ncnn
