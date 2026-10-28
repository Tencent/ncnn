// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu.h"
#include "mat.h"

#include <stdio.h>

static PFN_vkAllocateMemory original_allocate_memory = 0;
static const ncnn::VulkanDevice* allocation_device = 0;
static uint64_t allocated_bytes[2] = {0, 0};

static VKAPI_ATTR VkResult VKAPI_CALL count_allocate_memory(VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory)
{
    const VkResult ret = original_allocate_memory(device, info, callbacks, memory);
    if (ret == VK_SUCCESS)
        allocated_bytes[allocation_device->is_device_local(info->memoryTypeIndex) ? 0 : 1] += info->allocationSize;
    return ret;
}

static int test_weight_memory_usage(bool host)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkWeightAllocator weights(vkdev, host, 4096);
    uint64_t device_bytes, host_bytes;
    if (weights.get_weight_memory_usage(device_bytes, host_bytes) || device_bytes || host_bytes)
        return -1;

    allocated_bytes[0] = allocated_bytes[1] = 0;
    allocation_device = vkdev;
    original_allocate_memory = ncnn::vkAllocateMemory;
    ncnn::vkAllocateMemory = count_allocate_memory;
    ncnn::VkMat first(17, (size_t)4u, &weights);
    ncnn::VkMat second(19, (size_t)4u, &weights);
    ncnn::VkImageMat image(16, 16, 1, (size_t)4u, &weights);
    ncnn::vkAllocateMemory = original_allocate_memory;
    if (first.empty() || second.empty() || image.empty())
        return -1;
    if (weights.get_weight_memory_usage(device_bytes, host_bytes)
            || device_bytes != allocated_bytes[0] || host_bytes != allocated_bytes[1])
    {
        fprintf(stderr, "weight memory usage differs from actual allocations host=%d\n", host);
        return -1;
    }
    // Dropping views does not release their pooled backing allocations.
    first.release();
    second.release();
    image.release();
    if (weights.get_weight_memory_usage(device_bytes, host_bytes)
            || device_bytes != allocated_bytes[0] || host_bytes != allocated_bytes[1])
        return -1;
    weights.clear();
    if (weights.get_weight_memory_usage(device_bytes, host_bytes) || device_bytes || host_bytes || weights.needs_weight_cache())
        return -1;
    return 0;
}

static int test_weight_file_memory_usage()
{
#if NCNN_STDIO && (!defined(__ANDROID__) || __ANDROID_API__ >= 21) && ((defined(_WIN32) && !NCNN_WINXP) || defined(__ANDROID__) || defined(__OHOS__) || defined(__linux__) || __APPLE__)
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkWeightAllocator weights(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(".");
    ncnn::VkMat weight(64, (size_t)4u, &weights);
    if (weight.empty())
        return -1;
    uint64_t device_bytes, host_bytes;
    if (weights.get_weight_memory_usage(device_bytes, host_bytes) || device_bytes || host_bytes)
    {
        fprintf(stderr, "file mappings counted as resident Vulkan allocations\n");
        return -1;
    }
    if (weights.finalize_weights() || weights.get_weight_memory_usage(device_bytes, host_bytes)
            || device_bytes || host_bytes || !weights.needs_weight_cache() || !weights.has_disk_weights())
    {
        fprintf(stderr, "file mappings must not count as resident Vulkan allocations\n");
        return -1;
    }
    weight.release();
    weights.clear();
    weights.set_weight_memory_limit(8192, 8192, 4096, 4096);
    weight.create(64, (size_t)4u, &weights);
    if (weight.empty() || weights.finalize_weights())
        return -1;
    // Residents are allocated directly while loading, with space reserved for exchange.
    if (weights.get_weight_memory_usage(device_bytes, host_bytes) || !device_bytes || device_bytes > 4096 || host_bytes)
    {
        fprintf(stderr, "resident weights must report actual allocation bytes within the reserved budget\n");
        return -1;
    }
#endif
    return 0;
}

int main()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;
    return test_weight_memory_usage(false) || test_weight_memory_usage(true) || test_weight_file_memory_usage();
}
