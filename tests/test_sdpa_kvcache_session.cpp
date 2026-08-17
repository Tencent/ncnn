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

static int run_sdpa_step(ncnn::Net& net, ncnn::Mat& output, ncnn::Mat& key_cache, ncnn::Mat& value_cache, int cur_seqlen, int step, ncnn::Allocator* kvcache_allocator, int cache_extract_type)
{
    ncnn::Mat query(8, cur_seqlen, 4);
    ncnn::Mat key(8, cur_seqlen, 2);
    ncnn::Mat value(6, cur_seqlen, 2);
    fill_sdpa_input(query, 0.1f + step * 0.17f);
    fill_sdpa_input(key, -0.2f + step * 0.13f);
    fill_sdpa_input(value, 0.3f - step * 0.11f);

    ncnn::Extractor ex = net.create_extractor();
    if (kvcache_allocator)
    {
        ex.set_kvcache_allocator(kvcache_allocator);
        ex.set_kvcache_max_seqlen_hint(32);
    }

    ex.input("q", query);
    ex.input("k", key);
    ex.input("v", value);
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

static int run_extractor_kvcache(ncnn::Allocator* kvcache_allocator, int cache_extract_type, std::vector<ncnn::Mat>& outputs)
{
    ncnn::Net net;
    net.opt.lightmode = false;
    net.opt.use_vulkan_compute = false;

    if (net.load_param_mem(sdpa_param) != 0)
        return -1;
    net.load_model((const unsigned char*)empty_model);

    ncnn::Mat key_cache;
    ncnn::Mat value_cache;
    outputs.resize(3);
    const int append_lengths[] = {15, 2, 1};
    int ret = 0;
    for (int i = 0; ret == 0 && i < 3; i++)
    {
        ret = run_sdpa_step(net, outputs[i], key_cache, value_cache, append_lengths[i], i, kvcache_allocator, cache_extract_type);
        if (ret == 0 && (key_cache.empty() || value_cache.empty()))
            ret = -1;
        if (ret == 0 && kvcache_allocator)
        {
            if (key_cache.allocator != kvcache_allocator || value_cache.allocator != kvcache_allocator)
                ret = -1;
            if (key_cache.cstep < (size_t)key_cache.w * 32 || value_cache.cstep < (size_t)value_cache.w * 32)
                ret = -1;
        }
    }

    key_cache.release();
    value_cache.release();

    return ret;
}

static int test_extractor_kvcache()
{
    std::vector<ncnn::Mat> reference_outputs;
    int ret = run_extractor_kvcache(0, 1, reference_outputs);

    ncnn::UnlockedPoolAllocator kvcache_allocator;
    std::vector<ncnn::Mat> outputs;
    if (ret == 0)
        ret = run_extractor_kvcache(&kvcache_allocator, 1, outputs);

    for (int i = 0; ret == 0 && i < 3; i++)
    {
        if (CompareMat(reference_outputs[i], outputs[i], 0.001f) != 0)
            ret = -1;
    }

    std::vector<ncnn::Mat> default_type_outputs;
    if (ret == 0)
        ret = run_extractor_kvcache(&kvcache_allocator, 0, default_type_outputs);

    for (int i = 0; ret == 0 && i < 3; i++)
    {
        if (CompareMat(reference_outputs[i], default_type_outputs[i], 0.001f) != 0)
            ret = -1;
    }

    if (ret != 0)
        fprintf(stderr, "test_extractor_kvcache failed ret=%d\n", ret);

    return ret;
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
static int test_kvcache_batch_rejected()
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
    ex.set_kvcache_allocator(&kvcache_allocator);
    ex.set_kvcache_max_seqlen_hint(16);
    ex.input("q", query);
    ex.input("k", key);
    ex.input("v", value);

    ncnn::Mat output;
    int ret = ex.extract("out", output);
    if (ret != -1)
    {
        fprintf(stderr, "test_kvcache_batch_rejected failed ret=%d\n", ret);
        return -1;
    }

    return 0;
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
