// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "kvcache_storage.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif // NCNN_VULKAN

#include <stdio.h>

class TestAllocator : public ncnn::Allocator
{
public:
    TestAllocator()
        : fail(false), allocation_count(0), free_count(0)
    {
    }

    virtual void* fastMalloc(size_t size)
    {
        if (fail)
            return 0;

        allocation_count++;
        return ncnn::fastMalloc(size);
    }

    virtual void fastFree(void* ptr)
    {
        free_count++;
        ncnn::fastFree(ptr);
    }

    bool fail;
    int allocation_count;
    int free_count;
};

static void fill_cache(ncnn::Mat& cache)
{
    for (int q = 0; q < cache.c; q++)
    {
        ncnn::Mat head = cache.channel(q);
        for (int y = 0; y < cache.h; y++)
        {
            float* ptr = head.row(y);
            for (int x = 0; x < cache.w; x++)
            {
                ptr[x] = q * 1000.f + y * 100.f + x;
            }
        }
    }
}

static int check_cache(const ncnn::Mat& cache, int seqlen)
{
    for (int q = 0; q < cache.c; q++)
    {
        const ncnn::Mat head = cache.channel(q);
        for (int y = 0; y < seqlen; y++)
        {
            const float* ptr = head.row(y);
            for (int x = 0; x < cache.w; x++)
            {
                const float expected = q * 1000.f + y * 100.f + x;
                if (ptr[x] != expected)
                    return -1;
            }
        }
    }

    return 0;
}

static int test_create_expand_destroy()
{
    TestAllocator allocator;
    ncnn::CPUKVCacheStorage storage(0, &allocator);

    ncnn::Mat cache;
    int ret = storage.create(cache, 2, 3, 5, 4u, 1);
    int result = ret != 0 || cache.empty() || !storage.owns(cache) ? -1 : 0;

    if (result == 0)
        fill_cache(cache);

    void* data = cache.data;
    ncnn::Mat expanded_cache;
    if (result == 0)
    {
        ret = storage.expand(cache, expanded_cache, 15);
        if (ret != 0 || expanded_cache.data != data || expanded_cache.h != 15 || check_cache(expanded_cache, 2) != 0)
            result = -1;
        else
        {
            cache = expanded_cache;
            expanded_cache.release();
        }
    }

    if (result == 0)
    {
        ret = storage.expand(cache, expanded_cache, 17);
        if (ret != 0 || expanded_cache.data == data || expanded_cache.h != 17 || check_cache(expanded_cache, 2) != 0)
            result = -1;
        else
        {
            cache = expanded_cache;
            expanded_cache.release();
        }
    }

    storage.destroy(expanded_cache);
    storage.destroy(cache);
    if (!cache.empty() || allocator.allocation_count != allocator.free_count)
        result = -1;

    if (result != 0)
        fprintf(stderr, "test_create_expand_destroy failed\n");

    return result;
}

static int test_max_seqlen_hint()
{
    ncnn::CPUKVCacheStorage storage(8);

    ncnn::Mat cache;
    int ret = storage.create(cache, 2, 2, 4, 4u, 1);
    int result = ret;

    void* data = cache.data;
    ncnn::Mat expanded_cache;
    if (result == 0)
    {
        ret = storage.expand(cache, expanded_cache, 8);
        if (ret != 0 || expanded_cache.data != data)
            result = -1;
        else
        {
            cache = expanded_cache;
            expanded_cache.release();
        }
    }

    if (result == 0)
    {
        ret = storage.expand(cache, expanded_cache, 9);
        if (ret != 0 || expanded_cache.data == data)
            result = -1;
        else
        {
            cache = expanded_cache;
            expanded_cache.release();
        }
    }

    storage.destroy(expanded_cache);
    storage.destroy(cache);
    if (result != 0)
        fprintf(stderr, "test_max_seqlen_hint failed ret=%d\n", result);
    return result;
}

static int test_moderate_prefill_reserve()
{
    ncnn::CPUKVCacheStorage storage;

    ncnn::Mat cache;
    int ret = storage.create(cache, 4096, 1, 1, 4u, 1);
    int result = ret;

    void* data = cache.data;
    ncnn::Mat expanded_cache;
    if (result == 0)
    {
        ret = storage.expand(cache, expanded_cache, 4097);
        if (ret != 0 || expanded_cache.data != data)
            result = -1;
        else
        {
            cache = expanded_cache;
            expanded_cache.release();
        }
    }

    if (result == 0)
    {
        ret = storage.expand(cache, expanded_cache, 10000);
        if (ret != 0 || expanded_cache.data == data)
            result = -1;
        else
        {
            cache = expanded_cache;
            expanded_cache.release();
        }
    }

    storage.destroy(expanded_cache);
    storage.destroy(cache);
    if (result != 0)
        fprintf(stderr, "test_moderate_prefill_reserve failed ret=%d\n", result);
    return result;
}

static int test_empty_create()
{
    ncnn::CPUKVCacheStorage storage;

    ncnn::Mat cache;
    int ret = storage.create(cache, 0, 2, 8, 4u, 1);
    int result = ret != 0 || cache.empty() || cache.h != 0 ? -1 : 0;

    ncnn::Mat expanded_cache;
    if (result == 0)
    {
        ret = storage.expand(cache, expanded_cache, 1);
        if (ret != 0 || expanded_cache.h != 1)
            result = -1;
        else
        {
            cache = expanded_cache;
            expanded_cache.release();
        }
    }

    storage.destroy(expanded_cache);
    storage.destroy(cache);
    if (result != 0)
        fprintf(stderr, "test_empty_create failed\n");
    return result;
}

static int test_expand_oom()
{
    TestAllocator allocator;
    ncnn::CPUKVCacheStorage storage(0, &allocator);

    ncnn::Mat cache;
    int ret = storage.create(cache, 2, 2, 4, 4u, 1);
    int result = ret;

    if (result == 0)
        fill_cache(cache);

    void* data = cache.data;
    allocator.fail = true;

    ncnn::Mat expanded_cache;
    if (result == 0)
    {
        ret = storage.expand(cache, expanded_cache, 17);
        if (ret != -100 || !expanded_cache.empty() || cache.data != data || cache.h != 2 || check_cache(cache, 2) != 0)
            result = -1;
    }

    storage.destroy(expanded_cache);
    storage.destroy(cache);
    if (result != 0)
        fprintf(stderr, "test_expand_oom failed ret=%d\n", result);
    return result;
}

#if NCNN_VULKAN
static int test_vulkan_expand()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 0;

    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    int result = 0;
    {
        ncnn::VkKVCacheStorage storage(vkdev);
        ncnn::VkCompute cmd(vkdev);

        ncnn::VkMat cache;
        int ret = storage.create(cache, 2, 2, 4, 4u, 1, cmd);
        if (ret != 0)
            result = ret;

        ncnn::Mat cache_cpu(4, 2, 2);
        fill_cache(cache_cpu);

        ncnn::Option opt;
        opt.blob_vkallocator = cache.allocator;
        opt.staging_vkallocator = staging_vkallocator;

        if (result == 0)
        {
            cmd.record_clone(cache_cpu, cache, opt);
            ret = cmd.submit_and_wait();
            if (ret != 0)
                result = ret;
        }

        ncnn::VkBufferMemory* data = cache.data;
        ncnn::VkMat expanded_cache;
        if (result == 0)
        {
            ncnn::VkCompute expand_cmd(vkdev);
            ret = storage.expand(cache, expanded_cache, 15, expand_cmd);
            if (ret != 0 || expanded_cache.data != data)
                result = -1;
            cache = expanded_cache;
            expanded_cache.release();
        }

        if (result == 0)
        {
            ncnn::VkCompute expand_cmd(vkdev);
            ret = storage.expand(cache, expanded_cache, 17, expand_cmd);
            if (ret != 0 || expanded_cache.data == data)
                result = -1;

            cache = expanded_cache;
            expanded_cache.release();

            ncnn::Mat expanded_cache_cpu;
            opt.blob_vkallocator = cache.allocator;
            expand_cmd.record_clone(cache, expanded_cache_cpu, opt);
            ret = expand_cmd.submit_and_wait();
            if (ret != 0 || check_cache(expanded_cache_cpu, 2) != 0)
                result = -1;
        }

        storage.destroy(cache);
    }

    vkdev->reclaim_staging_allocator(staging_vkallocator);
    if (result != 0)
        fprintf(stderr, "test_vulkan_expand failed ret=%d\n", result);
    return result;
}

static int test_vulkan_empty_create()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 0;

    ncnn::VkKVCacheStorage storage(vkdev);
    ncnn::VkCompute cmd(vkdev);

    ncnn::VkMat cache;
    int ret = storage.create(cache, 0, 2, 8, 4u, 1, cmd);
    int result = ret != 0 || cache.empty() || cache.h != 0 ? -1 : 0;

    ncnn::VkMat expanded_cache;
    if (result == 0)
    {
        ret = storage.expand(cache, expanded_cache, 1, cmd);
        if (ret != 0 || expanded_cache.h != 1)
            result = -1;
        else
        {
            cache = expanded_cache;
            expanded_cache.release();
        }
    }

    storage.destroy(expanded_cache);
    storage.destroy(cache);
    if (result != 0)
        fprintf(stderr, "test_vulkan_empty_create failed\n");
    return result;
}
#endif // NCNN_VULKAN

int main()
{
    return 0
           || test_create_expand_destroy()
           || test_max_seqlen_hint()
           || test_moderate_prefill_reserve()
           || test_empty_create()
           || test_expand_oom()
#if NCNN_VULKAN
           || test_vulkan_expand()
           || test_vulkan_empty_create()
#endif
        ;
}
