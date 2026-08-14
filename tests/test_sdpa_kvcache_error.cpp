// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "kvcache_storage.h"
#include "testutil.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif // NCNN_VULKAN

#include <stdio.h>

static int check_cache(const ncnn::Mat& cache, float value, float epsilon = 0.f)
{
    for (int q = 0; q < cache.c; q++)
    {
        const ncnn::Mat head = cache.channel(q);
        for (int y = 0; y < cache.h; y++)
        {
            const float* ptr = head.row(y);
            for (int x = 0; x < cache.w; x++)
            {
                if (ptr[x] < value - epsilon || ptr[x] > value + epsilon)
                    return -1;
            }
        }
    }

    return 0;
}

class FailAllocator : public ncnn::Allocator
{
public:
    FailAllocator()
        : fail_after(-1), allocation_count(0)
    {
    }

    virtual void* fastMalloc(size_t size)
    {
        if (fail_after >= 0 && allocation_count >= fail_after)
            return 0;

        allocation_count++;
        return ncnn::fastMalloc(size);
    }

    virtual void fastFree(void* ptr)
    {
        ncnn::fastFree(ptr);
    }

    int fail_after;
    int allocation_count;
};

static int test_pair_expand_oom()
{
    FailAllocator allocator;
    ncnn::CPUKVCacheStorage storage(0, &allocator);

    ncnn::Mat key_cache;
    ncnn::Mat value_cache;
    int ret = storage.create(key_cache, 2, 2, 4, 4u, 1);
    if (ret == 0)
        ret = storage.create(value_cache, 2, 2, 4, 4u, 1);
    int result = ret;

    if (result == 0)
    {
        key_cache.fill(1.f);
        value_cache.fill(2.f);
    }
    void* key_data = key_cache.data;
    void* value_data = value_cache.data;

    ncnn::Layer* op = ncnn::create_layer_naive("SDPA");
    if (!op)
        result = -1;

    ncnn::ParamDict pd;
    pd.set(7, 1);
    op->load_param(pd);

    ncnn::Option opt;
    opt.kvcache_storage = &storage;
    if (op && result == 0)
        ret = op->create_pipeline(opt);
    if (ret != 0)
        result = ret;

    std::vector<ncnn::Mat> bottom_blobs(5);
    bottom_blobs[0] = ncnn::Mat(4, 3, 4);
    bottom_blobs[1] = ncnn::Mat(4, 15, 2);
    bottom_blobs[2] = ncnn::Mat(4, 15, 2);
    bottom_blobs[3] = key_cache;
    bottom_blobs[4] = value_cache;

    allocator.fail_after = allocator.allocation_count + 1;

    std::vector<ncnn::Mat> top_blobs(3);
    if (result == 0)
        ret = op->forward(bottom_blobs, top_blobs, opt);

    if (op)
    {
        op->destroy_pipeline(opt);
        delete op;
    }

    if (result == 0 && (ret != -100 || !top_blobs[1].empty() || !top_blobs[2].empty()))
        result = -1;
    if (result == 0 && (key_cache.data != key_data || value_cache.data != value_data || check_cache(key_cache, 1.f) != 0 || check_cache(value_cache, 2.f) != 0))
        result = -1;

    bottom_blobs[3].release();
    bottom_blobs[4].release();
    storage.destroy(key_cache);
    storage.destroy(value_cache);

    if (result != 0)
        fprintf(stderr, "test_pair_expand_oom failed ret=%d\n", result);

    return result;
}

static int test_kvcache_storage_mismatch()
{
    ncnn::CPUKVCacheStorage storage_a;
    ncnn::CPUKVCacheStorage storage_b;

    ncnn::Mat key_cache;
    ncnn::Mat value_cache;
    int ret = storage_a.create(key_cache, 2, 2, 4, 4u, 1);
    if (ret == 0)
        ret = storage_a.create(value_cache, 2, 2, 4, 4u, 1);

    ncnn::Layer* op = ncnn::create_layer_naive("SDPA");
    if (!op)
        ret = -1;

    ncnn::Option opt;
    opt.kvcache_storage = &storage_b;
    if (op)
    {
        ncnn::ParamDict pd;
        pd.set(7, 1);
        op->load_param(pd);
        if (ret == 0)
            ret = op->create_pipeline(opt);
    }

    if (ret == 0)
    {
        std::vector<ncnn::Mat> bottom_blobs(5);
        bottom_blobs[0] = ncnn::Mat(4, 1, 4);
        bottom_blobs[1] = ncnn::Mat(4, 1, 2);
        bottom_blobs[2] = ncnn::Mat(4, 1, 2);
        bottom_blobs[3] = key_cache;
        bottom_blobs[4] = value_cache;

        std::vector<ncnn::Mat> top_blobs(3);
        ret = op->forward(bottom_blobs, top_blobs, opt);
        if (ret == -1 && top_blobs[1].empty() && top_blobs[2].empty() && storage_a.owns(key_cache) && storage_a.owns(value_cache))
            ret = 0;
        else
            ret = -1;
    }

    if (op)
    {
        op->destroy_pipeline(opt);
        delete op;
    }

    storage_a.destroy(key_cache);
    storage_a.destroy(value_cache);

    if (ret != 0)
        fprintf(stderr, "test_kvcache_storage_mismatch failed ret=%d\n", ret);

    return ret;
}

#if NCNN_VULKAN

static int test_vulkan_kvcache_storage_mismatch()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 0;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    int result = 0;
    {
        ncnn::VkKVCacheStorage storage_a(vkdev);
        ncnn::VkKVCacheStorage storage_b(vkdev);
        ncnn::VkCompute create_cmd(vkdev);
        ncnn::VkMat key_cache;
        ncnn::VkMat value_cache;
        int ret = storage_a.create(key_cache, 2, 2, 4, 4u, 1, create_cmd);
        if (ret == 0)
            ret = storage_a.create(value_cache, 2, 2, 4, 4u, 1, create_cmd);
        if (ret != 0)
            result = ret;

        ncnn::Layer* op = ncnn::create_layer_vulkan("SDPA");
        if (!op)
            result = -1;

        ncnn::Option opt;
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = false;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.blob_vkallocator = blob_vkallocator;
        opt.workspace_vkallocator = blob_vkallocator;
        opt.staging_vkallocator = staging_vkallocator;
        opt.kvcache_storage = &storage_b;

        if (op)
        {
            op->vkdev = vkdev;
            ncnn::ParamDict pd;
            pd.set(7, 1);
            op->load_param(pd);
            ret = op->create_pipeline(opt);
            if (ret != 0)
                result = ret;
        }

        if (result == 0)
        {
            std::vector<ncnn::VkMat> bottom_blobs(5);
            bottom_blobs[0].create(4, 1, 4, 4u, 1, blob_vkallocator);
            bottom_blobs[1].create(4, 1, 2, 4u, 1, blob_vkallocator);
            bottom_blobs[2].create(4, 1, 2, 4u, 1, blob_vkallocator);
            bottom_blobs[3] = key_cache;
            bottom_blobs[4] = value_cache;

            std::vector<ncnn::VkMat> top_blobs(3);
            ncnn::VkCompute cmd(vkdev);
            ret = op->forward(bottom_blobs, top_blobs, cmd, opt);
            if (ret != -1 || !top_blobs[1].empty() || !top_blobs[2].empty() || !storage_a.owns(key_cache) || !storage_a.owns(value_cache))
                result = -1;
        }

        if (op)
        {
            op->destroy_pipeline(opt);
            delete op;
        }

        storage_a.destroy(key_cache);
        storage_a.destroy(value_cache);
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    if (result != 0)
        fprintf(stderr, "test_vulkan_kvcache_storage_mismatch failed ret=%d\n", result);

    return result;
}

class FailSecondExpandVkKVCacheStorage : public ncnn::KVCacheStorage
{
public:
    FailSecondExpandVkKVCacheStorage(const ncnn::VulkanDevice* vkdev)
        : storage(vkdev), expand_count(0)
    {
    }

    virtual int create(ncnn::Mat&, int, int, int, size_t, int)
    {
        return -1;
    }

    virtual int expand(const ncnn::Mat&, ncnn::Mat&, int)
    {
        return -1;
    }

    virtual void destroy(ncnn::Mat&)
    {
    }

    virtual int create(ncnn::VkMat& cache, int seqlen, int num_kv_head, int head_dim, size_t elemsize, int elempack, ncnn::VkCompute& cmd)
    {
        return storage.create(cache, seqlen, num_kv_head, head_dim, elemsize, elempack, cmd);
    }

    virtual int expand(const ncnn::VkMat& cache, ncnn::VkMat& expanded_cache, int new_seqlen, ncnn::VkCompute& cmd)
    {
        expand_count++;
        if (expand_count == 2)
            return -100;

        return storage.expand(cache, expanded_cache, new_seqlen, cmd);
    }

    virtual void destroy(ncnn::VkMat& cache)
    {
        storage.destroy(cache);
    }

    virtual bool owns(const ncnn::VkMat& cache) const
    {
        return storage.owns(cache);
    }

private:
    ncnn::VkKVCacheStorage storage;
    int expand_count;
};

static int test_vulkan_pair_expand_error()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 0;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    int result = 0;
    {
        FailSecondExpandVkKVCacheStorage storage(vkdev);
        ncnn::VkCompute create_cmd(vkdev);
        ncnn::VkMat key_cache;
        ncnn::VkMat value_cache;
        int ret = storage.create(key_cache, 2, 2, 4, 4u, 1, create_cmd);
        if (ret == 0)
            ret = storage.create(value_cache, 2, 2, 4, 4u, 1, create_cmd);
        if (ret != 0)
            result = ret;

        ncnn::Layer* op = ncnn::create_layer_vulkan("SDPA");
        if (!op)
            result = -1;

        ncnn::Option opt;
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = false;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.blob_vkallocator = blob_vkallocator;
        opt.workspace_vkallocator = blob_vkallocator;
        opt.staging_vkallocator = staging_vkallocator;
        opt.kvcache_storage = &storage;

        if (op)
        {
            op->vkdev = vkdev;
            ncnn::ParamDict pd;
            pd.set(7, 1);
            op->load_param(pd);
            ret = op->create_pipeline(opt);
            if (ret != 0)
                result = ret;
        }

        ncnn::VkBufferMemory* key_data = key_cache.data;
        ncnn::VkBufferMemory* value_data = value_cache.data;

        if (result == 0)
        {
            std::vector<ncnn::VkMat> bottom_blobs(5);
            bottom_blobs[0].create(4, 1, 4, 4u, 1, blob_vkallocator);
            bottom_blobs[1].create(4, 15, 2, 4u, 1, blob_vkallocator);
            bottom_blobs[2].create(4, 15, 2, 4u, 1, blob_vkallocator);
            bottom_blobs[3] = key_cache;
            bottom_blobs[4] = value_cache;

            std::vector<ncnn::VkMat> top_blobs(3);
            ncnn::VkCompute cmd(vkdev);
            ret = op->forward(bottom_blobs, top_blobs, cmd, opt);

            if (ret != -100 || !top_blobs[1].empty() || !top_blobs[2].empty())
                result = -1;
            if (key_cache.data != key_data || value_cache.data != value_data)
                result = -1;
        }

        if (op)
        {
            op->destroy_pipeline(opt);
            delete op;
        }

        storage.destroy(key_cache);
        storage.destroy(value_cache);
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return result;
}
#endif // NCNN_VULKAN

int main()
{
    return 0
           || test_pair_expand_oom()
           || test_kvcache_storage_mismatch()
#if NCNN_VULKAN
           || test_vulkan_kvcache_storage_mismatch()
           || test_vulkan_pair_expand_error()
#endif
           ;
}
