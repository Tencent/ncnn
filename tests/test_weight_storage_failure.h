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
        CREATE_BUFFER,
        MAP_MEMORY,
        MISSING_MAP,
        SUBMIT,
        SECOND_SUBMIT,
        FENCE_WAIT,
        UPLOAD_FENCE_WAIT,
        ALLOCATE_COMMAND,
        RESET_COMMAND,
        CREATE_FENCE,
        RESET_FENCE,
        CREATE_SEMAPHORE,
        CREATE_DESCRIPTOR_POOL,
        ALLOCATE_DESCRIPTOR_SET,
        TRACK_COMMANDS
    };

    explicit VulkanFailure(Failure failure, int fail_at = 1)
        : triggered(false), failure(failure), fail_at(failure == SECOND_SUBMIT ? 2 : fail_at), submissions(0), allocate(ncnn::vkAllocateMemory), bind(ncnn::vkBindBufferMemory), create_buffer(ncnn::vkCreateBuffer), map(ncnn::vkMapMemory), submit(ncnn::vkQueueSubmit), wait(ncnn::vkWaitForFences), allocate_command(ncnn::vkAllocateCommandBuffers), create_fence(ncnn::vkCreateFence), create_semaphore(ncnn::vkCreateSemaphore), reset_command(ncnn::vkResetCommandBuffer), reset_fences(ncnn::vkResetFences), create_descriptor_pool(ncnn::vkCreateDescriptorPool), allocate_descriptor_sets(ncnn::vkAllocateDescriptorSets)
    {
        calls = 0;
        reused_command = false;
        unsafe_command_reset = false;
        command_allocations = 0;
        resets = 0;
        wait_errors = 0;
        upload_fence = 0;
        g_vulkan_failure = this;
        ncnn::vkAllocateMemory = allocate_memory;
        ncnn::vkBindBufferMemory = bind_memory;
        ncnn::vkCreateBuffer = make_buffer;
        ncnn::vkMapMemory = map_memory;
        ncnn::vkQueueSubmit = queue_submit;
        ncnn::vkWaitForFences = wait_for_fences;
        ncnn::vkAllocateCommandBuffers = allocate_commands;
        ncnn::vkCreateFence = make_fence;
        ncnn::vkCreateSemaphore = make_semaphore;
        ncnn::vkResetCommandBuffer = reset_commands;
        ncnn::vkResetFences = reset_task_fences;
        ncnn::vkCreateDescriptorPool = make_descriptor_pool;
        ncnn::vkAllocateDescriptorSets = allocate_descriptors;
    }

    ~VulkanFailure()
    {
        ncnn::vkAllocateMemory = allocate;
        ncnn::vkBindBufferMemory = bind;
        ncnn::vkCreateBuffer = create_buffer;
        ncnn::vkMapMemory = map;
        ncnn::vkQueueSubmit = submit;
        ncnn::vkWaitForFences = wait;
        ncnn::vkAllocateCommandBuffers = allocate_command;
        ncnn::vkCreateFence = create_fence;
        ncnn::vkCreateSemaphore = create_semaphore;
        ncnn::vkResetCommandBuffer = reset_command;
        ncnn::vkResetFences = reset_fences;
        ncnn::vkCreateDescriptorPool = create_descriptor_pool;
        ncnn::vkAllocateDescriptorSets = allocate_descriptor_sets;
        g_vulkan_failure = 0;
    }

    int submission_count() const
    {
        return submissions;
    }

    bool triggered;
    bool reused_command;
    bool unsafe_command_reset;
    int command_allocations;
    int resets;
    int wait_errors;
    std::vector<VkCommandBuffer> submitted_commands;
    std::vector<VkFence> submitted_fences;

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

    static VKAPI_ATTR VkResult VKAPI_CALL make_buffer(VkDevice device, const VkBufferCreateInfo* info, const VkAllocationCallbacks* callbacks, VkBuffer* buffer)
    {
        VulkanFailure* f = g_vulkan_failure;
        if (f->failure == CREATE_BUFFER)
        {
            f->triggered = true;
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
        return f->create_buffer(device, info, callbacks, buffer);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL map_memory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** data)
    {
        VulkanFailure* f = g_vulkan_failure;
        if (f->failure == MAP_MEMORY)
        {
            f->triggered = true;
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        const VkResult ret = f->map(device, memory, offset, size, flags, data);
        if (f->failure == MISSING_MAP && ret == VK_SUCCESS)
        {
            // Keep the native mapping valid for allocator cleanup.
            *data = 0;
            f->triggered = true;
        }
        return ret;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, uint32_t count, const VkSubmitInfo* submits, VkFence fence)
    {
        VulkanFailure* f = g_vulkan_failure;
        f->submissions++;
        if (f->failure == UPLOAD_FENCE_WAIT && count == 2 && submits[1].commandBufferCount == 0)
            f->upload_fence = fence;
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
                    f->submitted_fences.push_back(fence);
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
        if ((f->failure == FENCE_WAIT || (f->failure == UPLOAD_FENCE_WAIT && count == 1 && fences[0] == f->upload_fence)) && f->calls++ < f->fail_at)
        {
            f->triggered = true;
            f->wait_errors++;
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
        const VkResult ret = f->allocate_command(device, info, commands);
        if (ret == VK_SUCCESS)
            f->command_allocations += info->commandBufferCount;
        return ret;
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
    static VKAPI_ATTR VkResult VKAPI_CALL make_descriptor_pool(VkDevice device, const VkDescriptorPoolCreateInfo* info, const VkAllocationCallbacks* allocator, VkDescriptorPool* pool)
    {
        VulkanFailure* f = g_vulkan_failure;
        if (f->failure == CREATE_DESCRIPTOR_POOL && ++f->calls == f->fail_at)
        {
            f->triggered = true;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        return f->create_descriptor_pool(device, info, allocator, pool);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL allocate_descriptors(VkDevice device, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* sets)
    {
        VulkanFailure* f = g_vulkan_failure;
        if (f->failure == ALLOCATE_DESCRIPTOR_SET && ++f->calls == f->fail_at)
        {
            f->triggered = true;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        return f->allocate_descriptor_sets(device, info, sets);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL reset_commands(VkCommandBuffer command, VkCommandBufferResetFlags flags)
    {
        VulkanFailure* f = g_vulkan_failure;
        f->resets++;
        if (f->failure == RESET_COMMAND && ++f->calls == f->fail_at)
        {
            f->triggered = true;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        if (f->failure == TRACK_COMMANDS)
            for (int i = (int)f->submitted_commands.size() - 1; i >= 0; i--)
                if (f->submitted_commands[i] == command)
                {
                    if (ncnn::vkGetFenceStatus(ncnn::get_gpu_device()->vkdevice(), f->submitted_fences[i]) != VK_SUCCESS)
                        f->unsafe_command_reset = true;
                    break;
                }
        return f->reset_command(command, flags);
    }

    static VKAPI_ATTR VkResult VKAPI_CALL reset_task_fences(VkDevice device, uint32_t count, const VkFence* fences)
    {
        VulkanFailure* f = g_vulkan_failure;
        f->resets++;
        if (f->failure == RESET_FENCE && ++f->calls == f->fail_at)
        {
            f->triggered = true;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        return f->reset_fences(device, count, fences);
    }

    int calls;
    VkFence upload_fence;
    PFN_vkAllocateCommandBuffers allocate_command;
    PFN_vkCreateFence create_fence;
    PFN_vkCreateSemaphore create_semaphore;
    PFN_vkResetCommandBuffer reset_command;
    PFN_vkResetFences reset_fences;
    PFN_vkCreateDescriptorPool create_descriptor_pool;
    PFN_vkAllocateDescriptorSets allocate_descriptor_sets;
    Failure failure;
    int fail_at;
    int submissions;
    PFN_vkAllocateMemory allocate;
    PFN_vkBindBufferMemory bind;
    PFN_vkCreateBuffer create_buffer;
    PFN_vkMapMemory map;
    PFN_vkQueueSubmit submit;
    PFN_vkWaitForFences wait;
};

#endif // TEST_WEIGHT_STORAGE_FAILURE_H
