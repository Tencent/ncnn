// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "allocator.h"

#include <stdio.h>
#include <algorithm>
#include "gpu.h"

namespace ncnn {

Allocator::~Allocator() 
{

}

PoolAllocator::PoolAllocator()
{
    size_compare_ratio = 192;// 0.75f * 256
}

PoolAllocator::~PoolAllocator()
{
    clear();

    if (!payouts.empty())
    {
        fprintf(stderr, "FATAL ERROR! pool allocator destroyed too early\n");
        std::list< std::pair<size_t, void*> >::iterator it = payouts.begin();
        for (; it != payouts.end(); it++)
        {
            void* ptr = it->second;
            fprintf(stderr, "%p still in use\n", ptr);
        }
    }
}

void PoolAllocator::clear()
{
    budgets_lock.lock();

    std::list< std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); it++)
    {
        void* ptr = it->second;
        ncnn::fastFree(ptr);
    }
    budgets.clear();

    budgets_lock.unlock();
}

void PoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        fprintf(stderr, "invalid size compare ratio %f\n", scr);
        return;
    }

    size_compare_ratio = (unsigned int)(scr * 256);
}

void* PoolAllocator::fastMalloc(size_t size)
{
    budgets_lock.lock();

    // find free budget
    std::list< std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); it++)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            budgets.erase(it);

            budgets_lock.unlock();

            payouts_lock.lock();

            payouts.push_back(std::make_pair(bs, ptr));

            payouts_lock.unlock();

            return ptr;
        }
    }

    budgets_lock.unlock();

    // new
    void* ptr = ncnn::fastMalloc(size);

    payouts_lock.lock();

    payouts.push_back(std::make_pair(size, ptr));

    payouts_lock.unlock();

    return ptr;
}

void PoolAllocator::fastFree(void* ptr)
{
    payouts_lock.lock();

    // return to budgets
    std::list< std::pair<size_t, void*> >::iterator it = payouts.begin();
    for (; it != payouts.end(); it++)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            payouts.erase(it);

            payouts_lock.unlock();

            budgets_lock.lock();

            budgets.push_back(std::make_pair(size, ptr));

            budgets_lock.unlock();

            return;
        }
    }

    payouts_lock.unlock();

    fprintf(stderr, "FATAL ERROR! pool allocator get wild %p\n", ptr);
    ncnn::fastFree(ptr);
}

UnlockedPoolAllocator::UnlockedPoolAllocator()
{
    size_compare_ratio = 192;// 0.75f * 256
}

UnlockedPoolAllocator::~UnlockedPoolAllocator()
{
    clear();

    if (!payouts.empty())
    {
        fprintf(stderr, "FATAL ERROR! unlocked pool allocator destroyed too early\n");
        std::list< std::pair<size_t, void*> >::iterator it = payouts.begin();
        for (; it != payouts.end(); it++)
        {
            void* ptr = it->second;
            fprintf(stderr, "%p still in use\n", ptr);
        }
    }
}

void UnlockedPoolAllocator::clear()
{
    std::list< std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); it++)
    {
        void* ptr = it->second;
        ncnn::fastFree(ptr);
    }
    budgets.clear();
}

void UnlockedPoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        fprintf(stderr, "invalid size compare ratio %f\n", scr);
        return;
    }

    size_compare_ratio = (unsigned int)(scr * 256);
}

void* UnlockedPoolAllocator::fastMalloc(size_t size)
{
    // find free budget
    std::list< std::pair<size_t, void*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); it++)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            budgets.erase(it);

            payouts.push_back(std::make_pair(bs, ptr));

            return ptr;
        }
    }

    // new
    void* ptr = ncnn::fastMalloc(size);

    payouts.push_back(std::make_pair(size, ptr));

    return ptr;
}

void UnlockedPoolAllocator::fastFree(void* ptr)
{
    // return to budgets
    std::list< std::pair<size_t, void*> >::iterator it = payouts.begin();
    for (; it != payouts.end(); it++)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            payouts.erase(it);

            budgets.push_back(std::make_pair(size, ptr));

            return;
        }
    }

    fprintf(stderr, "FATAL ERROR! unlocked pool allocator get wild %p\n", ptr);
    ncnn::fastFree(ptr);
}

#if NCNN_VULKAN
VkAllocator::VkAllocator(VulkanDevice* _vkdev) : vkdev(_vkdev)
{
    mappable = false;
}

VkBuffer VkAllocator::create_buffer(size_t size, VkBufferUsageFlags usage)
{
    VkBufferCreateInfo bufferCreateInfo;
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.pNext = 0;
    bufferCreateInfo.flags = 0;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = usage;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 0;// TODO respect transfer queue
    bufferCreateInfo.pQueueFamilyIndices = 0;

    VkBuffer buffer;
    VkResult ret = vkCreateBuffer(vkdev->vkdevice(), &bufferCreateInfo, 0, &buffer);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateBuffer failed %d\n", ret);
        return 0;
    }

    return buffer;
}

VkDeviceMemory VkAllocator::allocate_memory(size_t size, uint32_t memory_type_index)
{
    VkMemoryAllocateInfo memoryAllocateInfo;
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = 0;
    memoryAllocateInfo.allocationSize = size;
    memoryAllocateInfo.memoryTypeIndex = memory_type_index;

    VkDeviceMemory memory = 0;
    VkResult ret = vkAllocateMemory(vkdev->vkdevice(), &memoryAllocateInfo, 0, &memory);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkAllocateMemory failed %d\n", ret);
    }

    return memory;
}

VkDeviceMemory VkAllocator::allocate_dedicated_memory(size_t size, uint32_t memory_type_index, VkBuffer buffer)
{
    VkMemoryAllocateInfo memoryAllocateInfo;
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = 0;
    memoryAllocateInfo.allocationSize = size;
    memoryAllocateInfo.memoryTypeIndex = memory_type_index;

    VkMemoryDedicatedAllocateInfoKHR memoryDedicatedAllocateInfo;
    memoryDedicatedAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR;
    memoryDedicatedAllocateInfo.pNext = 0;
    memoryDedicatedAllocateInfo.buffer = buffer;
    memoryAllocateInfo.pNext = &memoryDedicatedAllocateInfo;

    VkDeviceMemory memory = 0;
    VkResult ret = vkAllocateMemory(vkdev->vkdevice(), &memoryAllocateInfo, nullptr, &memory);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkAllocateMemory failed %d\n", ret);
    }

    return memory;
}

VkBufferAllocator::VkBufferAllocator(VulkanDevice* _vkdev) : VkAllocator(_vkdev)
{
    mappable = vkdev->info.device_local_memory_index == vkdev->info.unified_memory_index;

    size_compare_ratio = 192;// 0.75f * 256
}

VkBufferAllocator::~VkBufferAllocator()
{
    clear();
}

void VkBufferAllocator::clear()
{
    fprintf(stderr, "VkBufferAllocator %lu\n", budgets.size());

    std::list< std::pair<size_t, VkBufferMemory*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); it++)
    {
        VkBufferMemory* ptr = it->second;

//         fprintf(stderr, "VkBufferAllocator F %p\n", ptr->buffer);

        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }
    budgets.clear();
}

void VkBufferAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        fprintf(stderr, "invalid size compare ratio %f\n", scr);
        return;
    }

    size_compare_ratio = (unsigned int)(scr * 256);
}

VkBufferMemory* VkBufferAllocator::fastMalloc(size_t size)
{
    // find free budget
    std::list< std::pair<size_t, VkBufferMemory*> >::iterator it = budgets.begin();
    for (; it != budgets.end(); it++)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * size_compare_ratio) >> 8) <= size)
        {
            VkBufferMemory* ptr = it->second;

            budgets.erase(it);

            payouts.push_back(std::make_pair(bs, ptr));

//             fprintf(stderr, "VkBufferAllocator M %p %lu reused %lu\n", ptr->buffer, size, bs);

            return ptr;
        }
    }

    // create new
    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    ptr->offset = 0;

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), ptr->buffer, &memoryRequirements);

    ptr->memory = allocate_memory(memoryRequirements.size, vkdev->info.device_local_memory_index);

    vkBindBufferMemory(vkdev->vkdevice(), ptr->buffer, ptr->memory, 0);

    payouts.push_back(std::make_pair(size, ptr));

//     fprintf(stderr, "VkBufferAllocator M %p %lu\n", ptr->buffer, size);

    return ptr;
}

void VkBufferAllocator::fastFree(VkBufferMemory* ptr)
{
//     fprintf(stderr, "VkBufferAllocator F %p\n", ptr->buffer);

    // return to budgets
    std::list< std::pair<size_t, VkBufferMemory*> >::iterator it = payouts.begin();
    for (; it != payouts.end(); it++)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            payouts.erase(it);

            budgets.push_back(std::make_pair(size, ptr));

            return;
        }
    }

    fprintf(stderr, "FATAL ERROR! unlocked vulkan pool allocator get wild %p\n", ptr->buffer);

    vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
    vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

    delete ptr;
}

static inline size_t least_common_multiple(size_t a, size_t b)
{
    if (a == b)
        return a;

    if (a > b)
        return least_common_multiple(b, a);

    size_t lcm = b;
    while (lcm % a != 0)
    {
        lcm += b;
    }

    return lcm;
}

VkWeightBufferAllocator::VkWeightBufferAllocator(VulkanDevice* _vkdev) : VkAllocator(_vkdev)
{
    mappable = vkdev->info.device_local_memory_index == vkdev->info.unified_memory_index;

    block_size = 8 * 1024 * 1024;// 8M
    buffer_offset_alignment = vkdev->info.buffer_offset_alignment;

    if (mappable)
    {
        // least common multiple for memory_map_alignment and buffer_offset_alignment
        size_t memory_map_alignment = vkdev->info.memory_map_alignment;
        buffer_offset_alignment = least_common_multiple(buffer_offset_alignment, memory_map_alignment);
    }
}

VkWeightBufferAllocator::~VkWeightBufferAllocator()
{
    clear();
}

void VkWeightBufferAllocator::clear()
{
    fprintf(stderr, "VkWeightBufferAllocator %lu %lu\n", buffer_blocks.size(), dedicated_buffer_blocks.size());

    buffer_block_free_spaces.clear();

    for (size_t i=0; i<buffer_blocks.size(); i++)
    {
        VkBufferMemory* ptr = buffer_blocks[i];

        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }
    buffer_blocks.clear();

    for (size_t i=0; i<dedicated_buffer_blocks.size(); i++)
    {
        VkBufferMemory* ptr = dedicated_buffer_blocks[i];

        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }
    dedicated_buffer_blocks.clear();
}

VkBufferMemory* VkWeightBufferAllocator::fastMalloc(size_t size)
{
//     fprintf(stderr, "VkWeightBufferAllocator fastMalloc %lu\n", size);

    size_t aligned_size = alignSize(size, buffer_offset_alignment);

    const int buffer_block_count = buffer_blocks.size();

    // find first spare space in buffer_blocks
    int block_index = -1;
    size_t block_offset = 0;
    for (int i=0; i<buffer_block_count; i++)
    {
        size_t free_size = buffer_block_free_spaces[i];
        if (free_size >= aligned_size)
        {
            block_index = i;
            block_offset = block_size - free_size;
            break;
        }
    }

    if (block_index != -1)
    {
        // return sub buffer
        VkBufferMemory* ptr = new VkBufferMemory;

        ptr->buffer = buffer_blocks[block_index]->buffer;
        ptr->offset = block_offset;
        ptr->memory = buffer_blocks[block_index]->memory;

        buffer_block_free_spaces[block_index] -= aligned_size;

        return ptr;
    }

    size_t new_block_size = std::max(block_size, aligned_size);

    // create new block
    VkBufferMemory* block = new VkBufferMemory;

    block->buffer = create_buffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    block->offset = 0;

    if (vkdev->info.support_VK_KHR_get_memory_requirements2 && vkdev->info.support_VK_KHR_dedicated_allocation)
    {
        VkBufferMemoryRequirementsInfo2KHR bufferMemoryRequirementsInfo2;
        bufferMemoryRequirementsInfo2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR;
        bufferMemoryRequirementsInfo2.pNext = 0;
        bufferMemoryRequirementsInfo2.buffer = block->buffer;

        VkMemoryRequirements2KHR memoryRequirements2;
        memoryRequirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR;
        memoryRequirements2.pNext = 0;

        VkMemoryDedicatedRequirementsKHR memoryDedicatedRequirements;
        memoryDedicatedRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR;
        memoryDedicatedRequirements.pNext = 0;
        memoryRequirements2.pNext = &memoryDedicatedRequirements;

        vkdev->vkGetBufferMemoryRequirements2KHR(vkdev->vkdevice(), &bufferMemoryRequirementsInfo2, &memoryRequirements2);

        bool dedicatedAllocation = memoryDedicatedRequirements.requiresDedicatedAllocation || memoryDedicatedRequirements.prefersDedicatedAllocation;

        if (dedicatedAllocation)
        {
            block->memory = allocate_dedicated_memory(memoryRequirements2.memoryRequirements.size, vkdev->info.device_local_memory_index, block->buffer);

            vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0);

            dedicated_buffer_blocks.push_back(block);

            // return sub buffer
            VkBufferMemory* ptr = new VkBufferMemory;

            ptr->buffer = block->buffer;
            ptr->offset = 0;
            ptr->memory = block->memory;

            return ptr;
        }
    }

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), block->buffer, &memoryRequirements);

    block->memory = allocate_memory(memoryRequirements.size, vkdev->info.device_local_memory_index);

    vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0);

//     fprintf(stderr, "VkWeightBufferAllocator M %p\n", block->buffer);

    buffer_blocks.push_back(block);

    buffer_block_free_spaces.push_back(new_block_size - aligned_size);

    // return sub buffer
    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = block->buffer;
    ptr->offset = 0;
    ptr->memory = block->memory;

    return ptr;
}

void VkWeightBufferAllocator::fastFree(VkBufferMemory* ptr)
{
//     fprintf(stderr, "VkWeightBufferAllocator F %p\n", ptr->buffer);

    delete ptr;
}

VkStagingBufferAllocator::VkStagingBufferAllocator(VulkanDevice* _vkdev) : VkAllocator(_vkdev)
{
    mappable = true;

    memory_type_index = vkdev->info.unified_memory_index;

    if (memory_type_index == -1)
        memory_type_index = vkdev->info.host_visible_memory_index;

}

VkStagingBufferAllocator::~VkStagingBufferAllocator()
{
    clear();
}

void VkStagingBufferAllocator::clear()
{
    fprintf(stderr, "VkStagingBufferAllocator %lu\n", staging_buffers.size());

    for (size_t i=0; i<staging_buffers.size(); i++)
    {
        VkBufferMemory* ptr = staging_buffers[i];

//         fprintf(stderr, "VkStagingBufferAllocator F %p\n", ptr->buffer);

        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }

    staging_buffers.clear();
}

VkBufferMemory* VkStagingBufferAllocator::fastMalloc(size_t size)
{
    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    ptr->offset = 0;

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), ptr->buffer, &memoryRequirements);

    ptr->memory = allocate_memory(memoryRequirements.size, memory_type_index);

    vkBindBufferMemory(vkdev->vkdevice(), ptr->buffer, ptr->memory, 0);

//     fprintf(stderr, "VkStagingBufferAllocator M %p %lu\n", ptr->buffer, size);

    return ptr;
}

void VkStagingBufferAllocator::fastFree(VkBufferMemory* ptr)
{
//     fprintf(stderr, "VkStagingBufferAllocator F %p\n", ptr->buffer);

    staging_buffers.push_back(ptr);
}

VkWeightStagingBufferAllocator::VkWeightStagingBufferAllocator(VulkanDevice* _vkdev) : VkAllocator(_vkdev)
{
    mappable = true;

    memory_type_index = vkdev->info.host_visible_memory_index;
}

VkWeightStagingBufferAllocator::~VkWeightStagingBufferAllocator()
{
}

VkBufferMemory* VkWeightStagingBufferAllocator::fastMalloc(size_t size)
{
    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    ptr->offset = 0;

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), ptr->buffer, &memoryRequirements);

    ptr->memory = allocate_memory(memoryRequirements.size, memory_type_index);

    vkBindBufferMemory(vkdev->vkdevice(), ptr->buffer, ptr->memory, 0);

//     fprintf(stderr, "VkWeightStagingBufferAllocator M %p %lu\n", ptr->buffer, size);

    return ptr;
}

void VkWeightStagingBufferAllocator::fastFree(VkBufferMemory* ptr)
{
//     fprintf(stderr, "VkWeightStagingBufferAllocator F %p\n", ptr->buffer);

    vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
    vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

    delete ptr;
}

#endif // NCNN_VULKAN

} // namespace ncnn
