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
        SECOND_SUBMIT,
        FENCE_WAIT,
        FENCE_STATUS,
        ALLOCATE_COMMAND,
        CREATE_FENCE,
        CREATE_SEMAPHORE,
        TRACK_COMMANDS
    };

    explicit VulkanFailure(Failure failure, int fail_at = 1)
        : triggered(false), failure(failure), fail_at(failure == SECOND_SUBMIT ? 2 : fail_at), submissions(0), allocate(ncnn::vkAllocateMemory), bind(ncnn::vkBindBufferMemory), map(ncnn::vkMapMemory),
          submit(ncnn::vkQueueSubmit), wait(ncnn::vkWaitForFences), allocate_command(ncnn::vkAllocateCommandBuffers), create_fence(ncnn::vkCreateFence), create_semaphore(ncnn::vkCreateSemaphore),
          reset_command(ncnn::vkResetCommandBuffer), reset_fences(ncnn::vkResetFences), fence_status(ncnn::vkGetFenceStatus)
    {
        calls = 0;
        reused_command = false;
        resets = 0;
        status_errors = 0;
        g_vulkan_failure = this;
        ncnn::vkAllocateMemory = allocate_memory;
        ncnn::vkBindBufferMemory = bind_memory;
        ncnn::vkMapMemory = map_memory;
        ncnn::vkQueueSubmit = queue_submit;
        ncnn::vkWaitForFences = wait_for_fences;
        ncnn::vkAllocateCommandBuffers = allocate_commands;
        ncnn::vkCreateFence = make_fence;
        ncnn::vkCreateSemaphore = make_semaphore;
        ncnn::vkResetCommandBuffer = reset_commands;
        ncnn::vkResetFences = reset_task_fences;
        ncnn::vkGetFenceStatus = get_fence_status;
    }

    ~VulkanFailure()
    {
        ncnn::vkAllocateMemory = allocate;
        ncnn::vkBindBufferMemory = bind;
        ncnn::vkMapMemory = map;
        ncnn::vkQueueSubmit = submit;
        ncnn::vkWaitForFences = wait;
        ncnn::vkAllocateCommandBuffers = allocate_command;
        ncnn::vkCreateFence = create_fence;
        ncnn::vkCreateSemaphore = create_semaphore;
        ncnn::vkResetCommandBuffer = reset_command;
        ncnn::vkResetFences = reset_fences;
        ncnn::vkGetFenceStatus = fence_status;
        g_vulkan_failure = 0;
    }

    int submission_count() const
    {
        return submissions;
    }

    bool triggered;
    bool reused_command;
    int resets;
    int status_errors;
    std::vector<VkCommandBuffer> submitted_commands;

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
        if (f->failure == TRACK_COMMANDS)
        {
            for (uint32_t i = 0; i < count; i++)
                for (uint32_t j = 0; j < submits[i].commandBufferCount; j++)
                {
                    const VkCommandBuffer command = submits[i].pCommandBuffers[j];
                    for (size_t k = 0; k < f->submitted_commands.size(); k++)
                        if (f->submitted_commands[k] == command)
                            f->reused_command = true;
                    f->submitted_commands.push_back(command);
                }
        }
        if ((f->failure == SUBMIT || f->failure == SECOND_SUBMIT) && f->submissions == f->fail_at)
        {
            f->triggered = true;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        return f->submit(queue, count, submits, fence);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL wait_for_fences(VkDevice device, uint32_t count, const VkFence* fences, VkBool32 all, uint64_t timeout)
    {
        VulkanFailure* f = g_vulkan_failure;
        if (f->failure == FENCE_WAIT && f->calls++ < f->fail_at)
        {
            f->triggered = true;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        return f->wait(device, count, fences, all, timeout);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL allocate_commands(VkDevice device, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* commands)
    {
        VulkanFailure* f = g_vulkan_failure;
        if (f->failure == ALLOCATE_COMMAND && ++f->calls == f->fail_at)
        {
            f->triggered = true;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        return f->allocate_command(device, info, commands);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL make_fence(VkDevice device, const VkFenceCreateInfo* info, const VkAllocationCallbacks* allocator, VkFence* fence)
    {
        VulkanFailure* f = g_vulkan_failure;
        if (f->failure == CREATE_FENCE && ++f->calls == f->fail_at)
        {
            f->triggered = true;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        return f->create_fence(device, info, allocator, fence);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL make_semaphore(VkDevice device, const VkSemaphoreCreateInfo* info, const VkAllocationCallbacks* allocator, VkSemaphore* semaphore)
    {
        VulkanFailure* f = g_vulkan_failure;
        if (f->failure == CREATE_SEMAPHORE && ++f->calls == f->fail_at)
        {
            f->triggered = true;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        return f->create_semaphore(device, info, allocator, semaphore);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL reset_commands(VkCommandBuffer command, VkCommandBufferResetFlags flags)
    {
        VulkanFailure* f = g_vulkan_failure;
        f->resets++;
        if (f->failure == TRACK_COMMANDS)
            for (size_t i = 0; i < f->submitted_commands.size(); i++)
                if (f->submitted_commands[i] == command)
                    f->reused_command = true;
        return f->reset_command(command, flags);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL reset_task_fences(VkDevice device, uint32_t count, const VkFence* fences)
    {
        VulkanFailure* f = g_vulkan_failure;
        f->resets++;
        return f->reset_fences(device, count, fences);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL get_fence_status(VkDevice device, VkFence fence)
    {
        VulkanFailure* f = g_vulkan_failure;
        if (f->failure == FENCE_STATUS && f->calls++ < f->fail_at)
        {
            f->triggered = true;
            f->status_errors++;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        return f->fence_status(device, fence);
    }

    int calls;
    PFN_vkAllocateCommandBuffers allocate_command;
    PFN_vkCreateFence create_fence;
    PFN_vkCreateSemaphore create_semaphore;
    PFN_vkResetCommandBuffer reset_command;
    PFN_vkResetFences reset_fences;
    PFN_vkGetFenceStatus fence_status;
    Failure failure;
    int fail_at;
    int submissions;
    PFN_vkAllocateMemory allocate;
    PFN_vkBindBufferMemory bind;
    PFN_vkMapMemory map;
    PFN_vkQueueSubmit submit;
    PFN_vkWaitForFences wait;
};

#endif // TEST_WEIGHT_STORAGE_FAILURE_H
