// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_sdpa_kvcache(int head_dim, int value_dim, int num_heads, int num_kv_heads, int mask_type, int storage_type, int num_threads = 1)
{
    ncnn::Layer* reference = ncnn::create_layer_naive("SDPA");
    ncnn::Layer* op = ncnn::create_layer_cpu("SDPA");
    if (!reference || !op)
    {
        delete reference;
        delete op;
        return -1;
    }

    if (storage_type == 1 && !op->support_bf16_storage)
    {
        delete reference;
        delete op;
        return 0;
    }

    ncnn::ParamDict pd;
    pd.set(5, mask_type != 0);
    pd.set(7, 1); // kv_cache
#if NCNN_INT8
    pd.set(18, storage_type == 2 ? 2 : 0); // int8_scale_term
#endif
    reference->load_param(pd);
    reference->load_model(ncnn::ModelBinFromMatArray(0));
    op->load_param(pd);
    op->load_model(ncnn::ModelBinFromMatArray(0));

    ncnn::Option reference_opt;
    reference_opt.num_threads = 1;
    ncnn::Option opt;
    opt.num_threads = num_threads;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = storage_type == 1;
    opt.kvcache_max_seqlen_hint = 32;

    int ret = reference->create_pipeline(reference_opt);
    if (ret == 0)
        ret = op->create_pipeline(opt);

    ncnn::Mat reference_key;
    ncnn::Mat reference_value;
    ncnn::Mat key_cache;
    ncnn::Mat value_cache;
    const int append_lengths[] = {13, 1, 1, 5, 17};

    for (int i = 0; ret == 0 && i < 5; i++)
    {
        const int cur_seqlen = append_lengths[i];
        const int dst_seqlen = reference_key.h + cur_seqlen;
        ncnn::Mat reference_query = RandomMat(head_dim, cur_seqlen, num_heads);
        ncnn::Mat reference_key_current = RandomMat(head_dim, cur_seqlen, num_kv_heads);
        ncnn::Mat reference_value_current = RandomMat(value_dim, cur_seqlen, num_kv_heads);
        ncnn::Mat reference_mask;
        if (mask_type == 1)
            reference_mask = RandomMat(dst_seqlen, cur_seqlen);
        if (mask_type == 3)
            reference_mask = RandomMat(dst_seqlen, cur_seqlen, num_heads);

        ncnn::Mat query = reference_query;
        ncnn::Mat key = reference_key_current;
        ncnn::Mat value = reference_value_current;
        ncnn::Mat mask = reference_mask;
#if NCNN_BF16
        if (storage_type == 1)
        {
            ncnn::Mat query_bf16;
            ncnn::Mat key_bf16;
            ncnn::Mat value_bf16;
            ncnn::Mat mask_bf16;
            ncnn::cast_float32_to_bfloat16(reference_query, query_bf16, reference_opt);
            ncnn::cast_float32_to_bfloat16(reference_key_current, key_bf16, reference_opt);
            ncnn::cast_float32_to_bfloat16(reference_value_current, value_bf16, reference_opt);
            if (mask_type)
                ncnn::cast_float32_to_bfloat16(reference_mask, mask_bf16, reference_opt);
            ncnn::cast_bfloat16_to_float32(query_bf16, reference_query, reference_opt);
            ncnn::cast_bfloat16_to_float32(key_bf16, reference_key_current, reference_opt);
            ncnn::cast_bfloat16_to_float32(value_bf16, reference_value_current, reference_opt);
            if (mask_type)
                ncnn::cast_bfloat16_to_float32(mask_bf16, reference_mask, reference_opt);
            query = query_bf16;
            key = key_bf16;
            value = value_bf16;
            mask = mask_bf16;
        }
#endif // NCNN_BF16

        std::vector<ncnn::Mat> reference_bottoms;
        reference_bottoms.push_back(reference_query);
        reference_bottoms.push_back(reference_key_current);
        reference_bottoms.push_back(reference_value_current);
        if (mask_type)
            reference_bottoms.push_back(reference_mask);
        reference_bottoms.push_back(reference_key);
        reference_bottoms.push_back(reference_value);
        reference_key.release();
        reference_value.release();

        std::vector<ncnn::Mat> reference_tops(3);
        ret = reference->forward(reference_bottoms, reference_tops, reference_opt);
        if (ret != 0)
            break;

        std::vector<ncnn::Mat> bottoms;
        bottoms.push_back(query);
        bottoms.push_back(key);
        bottoms.push_back(value);
        if (mask_type)
            bottoms.push_back(mask);
        bottoms.push_back(key_cache);
        bottoms.push_back(value_cache);
        key_cache.release();
        value_cache.release();

        std::vector<ncnn::Mat> tops(3);
        ret = op->forward(bottoms, tops, opt);
        if (ret != 0)
            break;

        if (CompareMat(reference_tops[0], tops[0], storage_type == 0 ? 0.001f : 0.01f) != 0)
            ret = -1;
        if (tops[1].empty() || tops[2].empty())
            ret = -1;

        reference_key = reference_tops[1];
        reference_value = reference_tops[2];
        reference_tops[1].release();
        reference_tops[2].release();

        key_cache = tops[1];
        value_cache = tops[2];
        tops[1].release();
        tops[2].release();
    }

    reference->destroy_pipeline(reference_opt);
    op->destroy_pipeline(opt);
    delete reference;
    delete op;

    if (ret != 0)
        fprintf(stderr, "test_sdpa_kvcache failed head_dim=%d value_dim=%d num_heads=%d num_kv_heads=%d mask_type=%d storage_type=%d\n", head_dim, value_dim, num_heads, num_kv_heads, mask_type, storage_type);

    return ret;
}

static int test_sdpa_0()
{
    return 0
           || test_sdpa_kvcache(32, 20, 8, 8, 0, 0)
           || test_sdpa_kvcache(37, 29, 15, 3, 1, 0, 4)
           || test_sdpa_kvcache(63, 47, 31, 1, 3, 0, 4)
           || test_sdpa_kvcache(64, 64, 16, 1, 0, 0)
#if NCNN_BF16
           || test_sdpa_kvcache(37, 29, 15, 3, 1, 1)
           || test_sdpa_kvcache(63, 47, 31, 1, 3, 1, 4)
#endif // NCNN_BF16
#if NCNN_INT8
           || test_sdpa_kvcache(32, 20, 8, 8, 0, 2)
           || test_sdpa_kvcache(37, 29, 15, 3, 1, 2)
#endif // NCNN_INT8
           ;
}

int main()
{
    SRAND(7767517);

    return test_sdpa_0();
}
