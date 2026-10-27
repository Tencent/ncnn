// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TEST_WEIGHT_STORAGE_FAILURE_H
#define TEST_WEIGHT_STORAGE_FAILURE_H

#include "gpu.h"

// used only by the static weight storage failure test
class VulkanFailure;
static VulkanFailure* g_vulkan_failure = 0;

class VulkanFailure
{
public:
    enum Failure
    {
        ALLOCATE_MEMORY,
        BIND_MEMORY,
        MAP_MEMORY,
        SUBMIT,
        SECOND_SUBMIT
    };

    explicit VulkanFailure(Failure failure, int fail_at = 1)
        : triggered(false), failure(failure), fail_at(failure == SECOND_SUBMIT ? 2 : fail_at), submissions(0), allocate(ncnn::vkAllocateMemory), bind(ncnn::vkBindBufferMemory), map(ncnn::vkMapMemory), submit(ncnn::vkQueueSubmit)
    {
        g_vulkan_failure = this;
        ncnn::vkAllocateMemory = allocate_memory;
        ncnn::vkBindBufferMemory = bind_memory;
        ncnn::vkMapMemory = map_memory;
        ncnn::vkQueueSubmit = queue_submit;
    }

    ~VulkanFailure()
    {
        ncnn::vkAllocateMemory = allocate;
        ncnn::vkBindBufferMemory = bind;
        ncnn::vkMapMemory = map;
        ncnn::vkQueueSubmit = submit;
        g_vulkan_failure = 0;
    }

    int submission_count() const
    {
        return submissions;
    }

    bool triggered;

private:
    VulkanFailure(const VulkanFailure&);
    VulkanFailure& operator=(const VulkanFailure&);

    static VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory)
    {
        VulkanFailure* f = g_vulkan_failure;
        if (f->failure == ALLOCATE_MEMORY)
        {
            f->triggered = true;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        return f->allocate(device, info, callbacks, memory);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL bind_memory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset)
    {
        VulkanFailure* f = g_vulkan_failure;
        if (f->failure == BIND_MEMORY)
        {
            f->triggered = true;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        return f->bind(device, buffer, memory, offset);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL map_memory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** data)
    {
        VulkanFailure* f = g_vulkan_failure;
        if (f->failure == MAP_MEMORY)
        {
            f->triggered = true;
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        return f->map(device, memory, offset, size, flags, data);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, uint32_t count, const VkSubmitInfo* submits, VkFence fence)
    {
        VulkanFailure* f = g_vulkan_failure;
        f->submissions++;
        if ((f->failure == SUBMIT || f->failure == SECOND_SUBMIT) && f->submissions == f->fail_at)
        {
            f->triggered = true;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        return f->submit(queue, count, submits, fence);
    }

    Failure failure;
    int fail_at;
    int submissions;
    PFN_vkAllocateMemory allocate;
    PFN_vkBindBufferMemory bind;
    PFN_vkMapMemory map;
    PFN_vkQueueSubmit submit;
};

#endif // TEST_WEIGHT_STORAGE_FAILURE_H
