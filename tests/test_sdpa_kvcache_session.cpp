// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "net.h"
#include "testutil.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif // NCNN_VULKAN

#include <stdio.h>

static const char sdpa_param[] = "7767517\n"
                                 "5 8\n"
                                 "Input q_input 0 1 q\n"
                                 "Input k_input 0 1 k\n"
                                 "Input v_input 0 1 v\n"
                                 "Input cache_input 0 2 past_k past_v\n"
                                 "SDPA sdpa 5 3 q k v past_k past_v out out_k out_v 7=1\n";

static const char sdpa_mask_param[] = "7767517\n"
                                      "6 9\n"
                                      "Input q_input 0 1 q\n"
                                      "Input k_input 0 1 k\n"
                                      "Input v_input 0 1 v\n"
                                      "Input mask_input 0 1 mask\n"
                                      "Input cache_input 0 2 past_k past_v\n"
                                      "SDPA sdpa 6 3 q k v mask past_k past_v out out_k out_v 5=1 7=1\n";

static const unsigned int empty_model[1] = {0};

static void fill_sdpa_input(ncnn::Mat& m, float base)
{
    for (int q = 0; q < m.c; q++)
    {
        ncnn::Mat channel = m.channel(q);
        for (int y = 0; y < m.h; y++)
        {
            float* ptr = channel.row(y);
            for (int x = 0; x < m.w; x++)
                ptr[x] = base + q * 0.03f + y * 0.005f + x * 0.001f;
        }
    }
}

static int run_sdpa_step(ncnn::Net& net, ncnn::Mat& output, ncnn::Mat& key_cache, ncnn::Mat& value_cache, int cur_seqlen, int step, ncnn::Allocator* kvcache_allocator, int cache_extract_type, int use_bf16_storage = 0, int num_heads = 4, int num_kv_heads = 2, int head_dim = 8, int value_dim = 6, int max_seqlen_hint = 32, int mask_type = 0)
{
    ncnn::Mat query(head_dim, cur_seqlen, num_heads);
    ncnn::Mat key(head_dim, cur_seqlen, num_kv_heads);
    ncnn::Mat value(value_dim, cur_seqlen, num_kv_heads);
    fill_sdpa_input(query, 0.1f + step * 0.17f);
    fill_sdpa_input(key, -0.2f + step * 0.13f);
    fill_sdpa_input(value, 0.3f - step * 0.11f);

#if NCNN_BF16
    if (use_bf16_storage)
    {
        ncnn::Option opt;
        ncnn::Mat query_bf16;
        ncnn::Mat key_bf16;
        ncnn::Mat value_bf16;
        ncnn::cast_float32_to_bfloat16(query, query_bf16, opt);
        ncnn::cast_float32_to_bfloat16(key, key_bf16, opt);
        ncnn::cast_float32_to_bfloat16(value, value_bf16, opt);
        query = query_bf16;
        key = key_bf16;
        value = value_bf16;
    }
#else
    (void)use_bf16_storage;
#endif // NCNN_BF16

    ncnn::Extractor ex = net.create_extractor();
    if (kvcache_allocator)
        ex.set_kvcache_allocator(kvcache_allocator);
    ex.set_kvcache_max_seqlen_hint(max_seqlen_hint);

    ex.input("q", query);
    ex.input("k", key);
    ex.input("v", value);
    ncnn::Mat mask;
    if (mask_type)
    {
        const int dst_seqlen = key_cache.h + cur_seqlen;
        if (mask_type == 1)
            mask.create(dst_seqlen, cur_seqlen);
        else
            mask.create(dst_seqlen, cur_seqlen, num_heads);
        fill_sdpa_input(mask, -0.1f + step * 0.02f);
#if NCNN_BF16
        if (use_bf16_storage)
        {
            ncnn::Option opt;
            ncnn::Mat mask_bf16;
            ncnn::cast_float32_to_bfloat16(mask, mask_bf16, opt);
            mask = mask_bf16;
        }
#endif // NCNN_BF16
        ex.input("mask", mask);
    }
    if (!key_cache.empty())
    {
        ex.input("past_k", key_cache);
        ex.input("past_v", value_cache);
    }

    int ret = ex.extract("out", output);
    if (ret != 0)
        return ret;
    ret = ex.extract("out_k", key_cache, cache_extract_type);
    if (ret != 0)
        return ret;

    return ex.extract("out_v", value_cache, cache_extract_type);
}

static int run_extractor_kvcache(ncnn::Allocator* kvcache_allocator, int cache_extract_type, std::vector<ncnn::Mat>& outputs, int use_bf16_storage = 0, int num_heads = 4, int num_kv_heads = 2, int head_dim = 8, int value_dim = 6, int max_seqlen_hint = 32, int mask_type = 0, int num_threads = 1, int kvcache_allocator_step = 0, int first_prefill_seqlen = 15)
{
    ncnn::Net net;
    net.opt.lightmode = false;
    net.opt.use_vulkan_compute = false;
    net.opt.use_bf16_storage = use_bf16_storage;
    net.opt.num_threads = num_threads;

    if (net.load_param_mem(mask_type ? sdpa_mask_param : sdpa_param) != 0)
        return -1;
    net.load_model((const unsigned char*)empty_model);

    ncnn::Mat key_cache;
    ncnn::Mat value_cache;
    outputs.resize(5);
    const int append_lengths[] = {first_prefill_seqlen, 2, 1, 18, 1};
    int ret = 0;
    for (int i = 0; ret == 0 && i < 5; i++)
    {
        const void* old_key_data = key_cache.data;
        const void* old_value_data = value_cache.data;
        const int old_key_capacity = key_cache.empty() ? 0 : (int)(key_cache.cstep / key_cache.w);
        const int old_value_capacity = value_cache.empty() ? 0 : (int)(value_cache.cstep / value_cache.w);
        const int new_seqlen = key_cache.h + append_lengths[i];

        ncnn::Allocator* step_kvcache_allocator = i >= kvcache_allocator_step ? kvcache_allocator : 0;
        const bool old_key_reusable = old_key_data && key_cache.allocator == step_kvcache_allocator;
        const bool old_value_reusable = old_value_data && value_cache.allocator == step_kvcache_allocator;
        ret = run_sdpa_step(net, outputs[i], key_cache, value_cache, append_lengths[i], i, step_kvcache_allocator, cache_extract_type, use_bf16_storage, num_heads, num_kv_heads, head_dim, value_dim, max_seqlen_hint, mask_type);
        if (ret == 0 && (key_cache.empty() || value_cache.empty()))
            ret = -1;
        if (ret == 0)
        {
            if (step_kvcache_allocator && (key_cache.allocator != step_kvcache_allocator || value_cache.allocator != step_kvcache_allocator))
                ret = -1;
            if (key_cache.w != head_dim || key_cache.h != new_seqlen || key_cache.c != num_kv_heads || key_cache.elempack != 1)
                ret = -1;
            if (value_cache.w != value_dim || value_cache.h != new_seqlen || value_cache.c != num_kv_heads || value_cache.elempack != 1)
                ret = -1;
            if (key_cache.elembits() != (use_bf16_storage ? 16 : 32) || value_cache.elembits() != (use_bf16_storage ? 16 : 32))
                ret = -1;
            if (key_cache.cstep < (size_t)key_cache.w * key_cache.h || value_cache.cstep < (size_t)value_cache.w * value_cache.h)
                ret = -1;
            if (!old_key_data && max_seqlen_hint >= new_seqlen && key_cache.cstep < (size_t)key_cache.w * max_seqlen_hint)
                ret = -1;
            if (!old_value_data && max_seqlen_hint >= new_seqlen && value_cache.cstep < (size_t)value_cache.w * max_seqlen_hint)
                ret = -1;
            if (old_key_reusable && (new_seqlen <= old_key_capacity) != (key_cache.data == old_key_data))
                ret = -1;
            if (old_value_reusable && (new_seqlen <= old_value_capacity) != (value_cache.data == old_value_data))
                ret = -1;
        }
    }

    key_cache.release();
    value_cache.release();

    return ret;
}

static int test_extractor_kvcache(int use_bf16_storage, int num_heads, int num_kv_heads, int head_dim, int value_dim, int max_seqlen_hint, int mask_type, int num_threads, int kvcache_allocator_step, int first_prefill_seqlen)
{
    std::vector<ncnn::Mat> reference_outputs;
    int ret = run_extractor_kvcache(0, 1, reference_outputs, use_bf16_storage, num_heads, num_kv_heads, head_dim, value_dim, max_seqlen_hint, mask_type, num_threads, 0, first_prefill_seqlen);

    ncnn::UnlockedPoolAllocator kvcache_allocator;
    std::vector<ncnn::Mat> outputs;
    if (ret == 0)
        ret = run_extractor_kvcache(&kvcache_allocator, 1, outputs, use_bf16_storage, num_heads, num_kv_heads, head_dim, value_dim, max_seqlen_hint, mask_type, num_threads, kvcache_allocator_step, first_prefill_seqlen);

    const float epsilon = use_bf16_storage ? 0.01f : 0.001f;
    for (int i = 0; ret == 0 && i < (int)outputs.size(); i++)
    {
        if (CompareMat(reference_outputs[i], outputs[i], epsilon) != 0)
        {
            fprintf(stderr, "test_extractor_kvcache failed storage=%d qhead=%d kvhead=%d head_dim=%d value_dim=%d step=%d\n", use_bf16_storage, num_heads, num_kv_heads, head_dim, value_dim, i);
            ret = -1;
        }
    }

    return ret;
}

static int test_extractor_kvcache_extract_type()
{
    std::vector<ncnn::Mat> reference_outputs;
    int ret = run_extractor_kvcache(0, 1, reference_outputs);

    ncnn::UnlockedPoolAllocator kvcache_allocator;
    std::vector<ncnn::Mat> outputs;
    if (ret == 0)
        ret = run_extractor_kvcache(&kvcache_allocator, 0, outputs);

    for (int i = 0; ret == 0 && i < (int)outputs.size(); i++)
    {
        if (CompareMat(reference_outputs[i], outputs[i], 0.001f) != 0)
            ret = -1;
    }

    if (ret != 0)
        fprintf(stderr, "test_extractor_kvcache_extract_type failed ret=%d\n", ret);

    return ret;
}

static int test_extractor_kvcache()
{
    return 0
           || test_extractor_kvcache(0, 4, 4, 7, 5, 0, 0, 1, 0, 15)
           || test_extractor_kvcache(0, 8, 2, 15, 13, 64, 1, 4, 0, 15)
           || test_extractor_kvcache(0, 16, 1, 17, 19, 8, 3, 4, 0, 15)
           || test_extractor_kvcache(0, 1, 1, 9, 11, 32, 1, 4, 0, 3)
           || test_extractor_kvcache(0, 12, 1, 17, 19, 32, 3, 1, 0, 15)
           || test_extractor_kvcache(0, 20, 1, 17, 19, 32, 3, 1, 0, 15)
           || test_extractor_kvcache(0, 8, 2, 9, 11, 32, 0, 2, 1, 15)
#if NCNN_BF16
           || test_extractor_kvcache(1, 4, 4, 7, 5, 0, 0, 1, 0, 15)
           || test_extractor_kvcache(1, 8, 2, 15, 13, 64, 1, 4, 0, 15)
           || test_extractor_kvcache(1, 16, 1, 17, 19, 8, 3, 4, 0, 15)
           || test_extractor_kvcache(1, 1, 1, 9, 11, 32, 1, 4, 0, 3)
           || test_extractor_kvcache(1, 12, 1, 17, 19, 32, 3, 1, 0, 15)
           || test_extractor_kvcache(1, 20, 1, 17, 19, 32, 3, 1, 0, 15)
#endif // NCNN_BF16
           || test_extractor_kvcache_extract_type();
}

static int test_kvcache_allocator_alias()
{
    ncnn::UnlockedPoolAllocator allocator;

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;

    if (net.load_param_mem(sdpa_param) != 0)
        return -1;
    net.load_model((const unsigned char*)empty_model);

    ncnn::Mat query(8, 1, 4);
    ncnn::Mat key(8, 1, 2);
    ncnn::Mat value(6, 1, 2);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_blob_allocator(&allocator);
    ex.set_kvcache_allocator(&allocator);
    ex.input("q", query);
    ex.input("k", key);
    ex.input("v", value);

    ncnn::Mat output;
    int ret = ex.extract("out", output);
    if (ret != -1)
    {
        fprintf(stderr, "test_kvcache_allocator_alias failed ret=%d\n", ret);
        return -1;
    }

    return 0;
}

#if NCNN_BATCH
static int test_kvcache_batch_rejected(int use_kvcache_allocator)
{
    ncnn::Net net;
    net.opt.lightmode = false;
    net.opt.use_vulkan_compute = false;

    if (net.load_param_mem(sdpa_param) != 0)
        return -1;
    net.load_model((const unsigned char*)empty_model);

    ncnn::Mat query;
    ncnn::Mat key;
    ncnn::Mat value;
    query.create(8, 1, 4, 4u, 1, 2);
    key.create(8, 1, 2, 4u, 1, 2);
    value.create(6, 1, 2, 4u, 1, 2);
    query.fill(0.1f);
    key.fill(0.2f);
    value.fill(0.3f);

    ncnn::UnlockedPoolAllocator kvcache_allocator;
    ncnn::Extractor ex = net.create_extractor();
    if (use_kvcache_allocator)
        ex.set_kvcache_allocator(&kvcache_allocator);
    ex.set_kvcache_max_seqlen_hint(16);
    ex.input("q", query);
    ex.input("k", key);
    ex.input("v", value);

    ncnn::Mat output;
    int ret = ex.extract("out", output);
    if (ret != -1)
    {
        fprintf(stderr, "test_kvcache_batch_rejected failed allocator=%d ret=%d\n", use_kvcache_allocator, ret);
        return -1;
    }

    return 0;
}

static int test_kvcache_batch_rejected()
{
    return 0
           || test_kvcache_batch_rejected(0)
           || test_kvcache_batch_rejected(1);
}
#endif // NCNN_BATCH

#if NCNN_VULKAN
static int test_legacy_vulkan_extractor_kvcache()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 0;

    ncnn::Net net;
    net.opt.lightmode = false;
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.set_vulkan_device(vkdev);

    if (net.load_param_mem(sdpa_param) != 0)
        return -1;
    net.load_model((const unsigned char*)empty_model);

    ncnn::Mat key_cache;
    ncnn::Mat value_cache;
    std::vector<ncnn::Mat> reference_outputs;
    int ret = run_extractor_kvcache(0, 1, reference_outputs);

    const int append_lengths[] = {15, 2, 1};
    for (int i = 0; ret == 0 && i < 3; i++)
    {
        ncnn::Mat output;
        ret = run_sdpa_step(net, output, key_cache, value_cache, append_lengths[i], i, 0, 1);
        if (ret == 0 && CompareMat(reference_outputs[i], output, 0.001f) != 0)
            ret = -1;
    }

    if (ret != 0)
        fprintf(stderr, "test_legacy_vulkan_extractor_kvcache failed ret=%d\n", ret);

    return ret;
}

static int run_sdpa_vulkan_step(ncnn::Net& net, ncnn::Mat& output, ncnn::VkMat& key_cache, ncnn::VkMat& value_cache, int cur_seqlen, int step, ncnn::VkAllocator* kvcache_vkallocator, ncnn::VkAllocator* blob_vkallocator, ncnn::VkAllocator* staging_vkallocator)
{
    const ncnn::VulkanDevice* vkdev = net.vulkan_device();

    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    ncnn::Mat query(8, cur_seqlen, 4);
    ncnn::Mat key(8, cur_seqlen, 2);
    ncnn::Mat value(6, cur_seqlen, 2);
    fill_sdpa_input(query, 0.1f + step * 0.17f);
    fill_sdpa_input(key, -0.2f + step * 0.13f);
    fill_sdpa_input(value, 0.3f - step * 0.11f);

    ncnn::VkCompute cmd(vkdev);
    ncnn::VkMat query_gpu;
    ncnn::VkMat key_gpu;
    ncnn::VkMat value_gpu;
    cmd.record_clone(query, query_gpu, opt);
    cmd.record_clone(key, key_gpu, opt);
    cmd.record_clone(value, value_gpu, opt);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_kvcache_vkallocator(kvcache_vkallocator);
    ex.set_kvcache_max_seqlen_hint(0);
    ex.set_blob_vkallocator(blob_vkallocator);
    ex.set_workspace_vkallocator(blob_vkallocator);
    ex.set_staging_vkallocator(staging_vkallocator);
    ex.input("q", query_gpu);
    ex.input("k", key_gpu);
    ex.input("v", value_gpu);
    if (!key_cache.empty())
    {
        ex.input("past_k", key_cache);
        ex.input("past_v", value_cache);
        key_cache.release();
        value_cache.release();
    }

    ncnn::VkMat output_gpu;
    ncnn::VkMat next_key_cache;
    ncnn::VkMat next_value_cache;
    int ret = ex.extract("out", output_gpu, cmd);
    if (ret == 0)
        ret = ex.extract("out_k", next_key_cache, cmd);
    if (ret == 0)
        ret = ex.extract("out_v", next_value_cache, cmd);

    if (ret == 0)
        cmd.record_clone(output_gpu, output, opt);
    if (ret == 0)
        ret = cmd.submit_and_wait();
    if (ret != 0)
        return ret;
    key_cache = next_key_cache;
    value_cache = next_value_cache;

    return 0;
}

static int test_external_vulkan_kvcache()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 0;

    ncnn::VkBlobAllocator blob_vkallocator(vkdev);
    ncnn::VkBlobAllocator kvcache_vkallocator(vkdev);
    ncnn::VkStagingAllocator staging_vkallocator(vkdev);

    ncnn::Net net;
    net.opt.lightmode = false;
    net.opt.use_vulkan_compute = true;
    net.opt.use_packing_layout = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.set_vulkan_device(vkdev);

    int result = net.load_param_mem(sdpa_param);
    if (result == 0)
        net.load_model((const unsigned char*)empty_model);

    if (result == 0)
    {
        ncnn::Mat output;
        ncnn::VkMat key_cache;
        ncnn::VkMat value_cache;
        int ret = run_sdpa_vulkan_step(net, output, key_cache, value_cache, 1, 0, &blob_vkallocator, &blob_vkallocator, &staging_vkallocator);
        if (ret != -1)
        {
            fprintf(stderr, "test_vulkan_kvcache_allocator_alias failed ret=%d\n", ret);
            result = -1;
        }
    }

    std::vector<ncnn::Mat> reference_outputs;
    if (result == 0)
        result = run_extractor_kvcache(0, 1, reference_outputs);

    ncnn::VkMat key_cache;
    ncnn::VkMat value_cache;
    const int append_lengths[] = {15, 2, 1};
    for (int i = 0; result == 0 && i < 3; i++)
    {
        ncnn::Mat output;
        result = run_sdpa_vulkan_step(net, output, key_cache, value_cache, append_lengths[i], i, &kvcache_vkallocator, &blob_vkallocator, &staging_vkallocator);
        if (result == 0 && CompareMat(reference_outputs[i], output, 0.001f) != 0)
            result = -1;
        if (result == 0 && (key_cache.allocator != &kvcache_vkallocator || value_cache.allocator != &kvcache_vkallocator))
            result = -1;
    }

    key_cache.release();
    value_cache.release();

    if (result != 0)
        fprintf(stderr, "test_external_vulkan_kvcache failed ret=%d\n", result);

    return result;
}

#endif // NCNN_VULKAN

int main()
{
    return 0
           || test_extractor_kvcache()
           || test_kvcache_allocator_alias()
#if NCNN_BATCH
           || test_kvcache_batch_rejected()
#endif
#if NCNN_VULKAN
           || test_legacy_vulkan_extractor_kvcache()
           || test_external_vulkan_kvcache()
#endif
           ;
}
