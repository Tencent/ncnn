// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// file mappings must use 64-bit offsets on 32-bit posix builds too
#ifndef _WIN32
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
#endif

#include "allocator.h"

#include "gpu.h"
#include "pipeline.h"

#if __ANDROID_API__ >= 26
#include <android/hardware_buffer.h>
#endif // __ANDROID_API__ >= 26

#if NCNN_VULKAN
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#if NCNN_BENCHMARK
#include "benchmark.h"
#endif
#endif // NCNN_VULKAN

namespace ncnn {

Allocator::~Allocator()
{
}

class PoolAllocatorPrivate
{
public:
    Mutex budgets_lock;
    Mutex payouts_lock;
    unsigned int size_compare_ratio; // 0~256
    size_t size_drop_threshold;
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};

PoolAllocator::PoolAllocator()
    : Allocator(), d(new PoolAllocatorPrivate)
{
    d->size_compare_ratio = 0;
    d->size_drop_threshold = 10;
}

PoolAllocator::~PoolAllocator()
{
    clear();

    if (!d->payouts.empty())
    {
        NCNN_LOGE("FATAL ERROR! pool allocator destroyed too early");
#if NCNN_STDIO
        std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
        for (; it != d->payouts.end(); ++it)
        {
            void* ptr = it->second;
            NCNN_LOGE("%p still in use", ptr);
        }
#endif
    }

    delete d;
}

PoolAllocator::PoolAllocator(const PoolAllocator&)
    : d(0)
{
}

PoolAllocator& PoolAllocator::operator=(const PoolAllocator&)
{
    return *this;
}

void PoolAllocator::clear()
{
    d->budgets_lock.lock();

    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        void* ptr = it->second;
        ncnn::fastFree(ptr);
    }
    d->budgets.clear();

    d->budgets_lock.unlock();
}

void PoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        NCNN_LOGE("invalid size compare ratio %f", scr);
        return;
    }

    d->size_compare_ratio = (unsigned int)(scr * 256);
}

void PoolAllocator::set_size_drop_threshold(size_t threshold)
{
    d->size_drop_threshold = threshold;
}

void* PoolAllocator::fastMalloc(size_t size)
{
    d->budgets_lock.lock();

    // find free budget
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin(), it_max = d->budgets.begin(), it_min = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * d->size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            d->budgets.erase(it);

            d->budgets_lock.unlock();

            d->payouts_lock.lock();

            d->payouts.push_back(std::make_pair(bs, ptr));

            d->payouts_lock.unlock();

            return ptr;
        }

        if (bs < it_min->first)
        {
            it_min = it;
        }
        if (bs > it_max->first)
        {
            it_max = it;
        }
    }

    if (d->budgets.size() >= d->size_drop_threshold)
    {
        // All chunks in pool are not chosen. Then try to drop some outdated
        // chunks and return them to OS.
        if (it_max->first < size)
        {
            // Current query is asking for a chunk larger than any cached chunks.
            // Then remove the smallest one.
            ncnn::fastFree(it_min->second);
            d->budgets.erase(it_min);
        }
        else if (it_min->first > size)
        {
            // Current query is asking for a chunk smaller than any cached chunks.
            // Then remove the largest one.
            ncnn::fastFree(it_max->second);
            d->budgets.erase(it_max);
        }
    }

    d->budgets_lock.unlock();

    // new
    void* ptr = ncnn::fastMalloc(size);

    d->payouts_lock.lock();

    d->payouts.push_back(std::make_pair(size, ptr));

    d->payouts_lock.unlock();

    return ptr;
}

void PoolAllocator::fastFree(void* ptr)
{
    d->payouts_lock.lock();

    // return to budgets
    std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
    for (; it != d->payouts.end(); ++it)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            d->payouts.erase(it);

            d->payouts_lock.unlock();

            d->budgets_lock.lock();

            d->budgets.push_back(std::make_pair(size, ptr));

            d->budgets_lock.unlock();

            return;
        }
    }

    d->payouts_lock.unlock();

    NCNN_LOGE("FATAL ERROR! pool allocator get wild %p", ptr);
    ncnn::fastFree(ptr);
}

class UnlockedPoolAllocatorPrivate
{
public:
    unsigned int size_compare_ratio; // 0~256
    size_t size_drop_threshold;
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};

UnlockedPoolAllocator::UnlockedPoolAllocator()
    : Allocator(), d(new UnlockedPoolAllocatorPrivate)
{
    d->size_compare_ratio = 0;
    d->size_drop_threshold = 10;
}

UnlockedPoolAllocator::~UnlockedPoolAllocator()
{
    clear();

    if (!d->payouts.empty())
    {
        NCNN_LOGE("FATAL ERROR! unlocked pool allocator destroyed too early");
#if NCNN_STDIO
        std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
        for (; it != d->payouts.end(); ++it)
        {
            void* ptr = it->second;
            NCNN_LOGE("%p still in use", ptr);
        }
#endif
    }

    delete d;
}

UnlockedPoolAllocator::UnlockedPoolAllocator(const UnlockedPoolAllocator&)
    : d(0)
{
}

UnlockedPoolAllocator& UnlockedPoolAllocator::operator=(const UnlockedPoolAllocator&)
{
    return *this;
}

void UnlockedPoolAllocator::clear()
{
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        void* ptr = it->second;
        ncnn::fastFree(ptr);
    }
    d->budgets.clear();
}

void UnlockedPoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        NCNN_LOGE("invalid size compare ratio %f", scr);
        return;
    }

    d->size_compare_ratio = (unsigned int)(scr * 256);
}

void UnlockedPoolAllocator::set_size_drop_threshold(size_t threshold)
{
    d->size_drop_threshold = threshold;
}

void* UnlockedPoolAllocator::fastMalloc(size_t size)
{
    // find free budget
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin(), it_max = d->budgets.begin(), it_min = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * d->size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            d->budgets.erase(it);

            d->payouts.push_back(std::make_pair(bs, ptr));

            return ptr;
        }

        if (bs > it_max->first)
        {
            it_max = it;
        }
        if (bs < it_min->first)
        {
            it_min = it;
        }
    }

    if (d->budgets.size() >= d->size_drop_threshold)
    {
        if (it_max->first < size)
        {
            ncnn::fastFree(it_min->second);
            d->budgets.erase(it_min);
        }
        else if (it_min->first > size)
        {
            ncnn::fastFree(it_max->second);
            d->budgets.erase(it_max);
        }
    }

    // new
    void* ptr = ncnn::fastMalloc(size);

    d->payouts.push_back(std::make_pair(size, ptr));

    return ptr;
}

void UnlockedPoolAllocator::fastFree(void* ptr)
{
    // return to budgets
    std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
    for (; it != d->payouts.end(); ++it)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            d->payouts.erase(it);

            d->budgets.push_back(std::make_pair(size, ptr));

            return;
        }
    }

    NCNN_LOGE("FATAL ERROR! unlocked pool allocator get wild %p", ptr);
    ncnn::fastFree(ptr);
}

#if NCNN_VULKAN
VkAllocator::VkAllocator(const VulkanDevice* _vkdev)
    : vkdev(_vkdev)
{
    buffer_memory_type_index = (uint32_t)-1;
    image_memory_type_index = (uint32_t)-1;
    reserved_type_index = (uint32_t)-1;
    mappable = false;
    coherent = false;
    cached = false;
}

VkAllocator::~VkAllocator()
{
    clear();
}

void VkAllocator::clear()
{
}

static inline size_t round_up(size_t n, size_t multiple)
{
    return (n + multiple - 1) / multiple * multiple;
}

static inline size_t round_down(size_t n, size_t multiple)
{
    return n / multiple * multiple;
}

static int flush_buffer_memory(const VulkanDevice* vkdev, const VkBufferMemory* ptr, size_t offset, size_t size)
{
    VkMappedMemoryRange mappedMemoryRange;
    mappedMemoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mappedMemoryRange.pNext = 0;
    mappedMemoryRange.memory = ptr->memory;
    mappedMemoryRange.offset = round_down(ptr->offset + offset, vkdev->info.non_coherent_atom_size());
    mappedMemoryRange.size = round_up(ptr->offset + offset + size, vkdev->info.non_coherent_atom_size()) - mappedMemoryRange.offset;

    VkResult ret = vkFlushMappedMemoryRanges(vkdev->vkdevice(), 1, &mappedMemoryRange);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkFlushMappedMemoryRanges failed %d", ret);
        return -1;
    }

    return 0;
}

int VkAllocator::flush(VkBufferMemory* ptr)
{
    if (coherent)
        return 0;

    return flush_buffer_memory(vkdev, ptr, 0, ptr->capacity);
}

int VkAllocator::invalidate(VkBufferMemory* ptr)
{
    if (coherent)
        return 0;

    VkMappedMemoryRange mappedMemoryRange;
    mappedMemoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mappedMemoryRange.pNext = 0;
    mappedMemoryRange.memory = ptr->memory;
    mappedMemoryRange.offset = round_down(ptr->offset, vkdev->info.non_coherent_atom_size());
    mappedMemoryRange.size = round_up(ptr->offset + ptr->capacity, vkdev->info.non_coherent_atom_size()) - mappedMemoryRange.offset;

    VkResult ret = vkInvalidateMappedMemoryRanges(vkdev->vkdevice(), 1, &mappedMemoryRange);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkInvalidateMappedMemoryRanges failed %d", ret);
        return -1;
    }

    return 0;
}

static VkBuffer create_buffer(const VulkanDevice* vkdev, size_t size, VkBufferUsageFlags usage, VkExternalMemoryHandleTypeFlags handle_types, bool concurrent)
{
    VkBufferCreateInfo bufferCreateInfo;
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.pNext = 0;
    bufferCreateInfo.flags = 0;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = usage;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferCreateInfo.queueFamilyIndexCount = 0;
    bufferCreateInfo.pQueueFamilyIndices = 0;
    const uint32_t families[2] = {vkdev->info.compute_queue_family_index(), vkdev->info.transfer_queue_family_index()};
    if (concurrent && families[0] != families[1])
    {
        bufferCreateInfo.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bufferCreateInfo.queueFamilyIndexCount = 2;
        bufferCreateInfo.pQueueFamilyIndices = families;
    }

    VkExternalMemoryBufferCreateInfo external_info;
    if (handle_types)
    {
        external_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        external_info.pNext = 0;
        external_info.handleTypes = handle_types;
        bufferCreateInfo.pNext = &external_info;
    }

    VkBuffer buffer = 0;
    VkResult ret = vkCreateBuffer(vkdev->vkdevice(), &bufferCreateInfo, 0, &buffer);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateBuffer failed %d", ret);
        return 0;
    }

    return buffer;
}

static VkDeviceMemory allocate_memory(const VulkanDevice* vkdev, size_t size, uint32_t memory_type_index)
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
        NCNN_LOGE("vkAllocateMemory failed %d", ret);
        return 0;
    }

    return memory;
}

VkBuffer VkAllocator::create_buffer(size_t size, VkBufferUsageFlags usage)
{
    return ncnn::create_buffer(vkdev, size, usage, 0, false);
}

VkDeviceMemory VkAllocator::allocate_memory(size_t size, uint32_t memory_type_index)
{
    return ncnn::allocate_memory(vkdev, size, memory_type_index);
}

static VkDeviceMemory allocate_dedicated_memory(const VulkanDevice* vkdev, size_t size, uint32_t memory_type_index, VkImage image, VkBuffer buffer)
{
    VkMemoryAllocateInfo memoryAllocateInfo;
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = 0;
    memoryAllocateInfo.allocationSize = size;
    memoryAllocateInfo.memoryTypeIndex = memory_type_index;

    VkMemoryDedicatedAllocateInfoKHR memoryDedicatedAllocateInfo;
    memoryDedicatedAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR;
    memoryDedicatedAllocateInfo.pNext = 0;
    memoryDedicatedAllocateInfo.image = image;
    memoryDedicatedAllocateInfo.buffer = buffer;
    memoryAllocateInfo.pNext = &memoryDedicatedAllocateInfo;

    VkDeviceMemory memory = 0;
    VkResult ret = vkAllocateMemory(vkdev->vkdevice(), &memoryAllocateInfo, 0, &memory);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkAllocateMemory failed %d", ret);
        return 0;
    }

    return memory;
}

VkDeviceMemory VkAllocator::allocate_dedicated_memory(size_t size, uint32_t memory_type_index, VkImage image, VkBuffer buffer)
{
    return ncnn::allocate_dedicated_memory(vkdev, size, memory_type_index, image, buffer);
}

static VkDeviceMemory allocate_import_host_memory(const VulkanDevice* vkdev, size_t size, uint32_t memory_type_index, void* host_ptr)
{
    VkMemoryAllocateInfo memoryAllocateInfo;
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = 0;
    memoryAllocateInfo.allocationSize = size;
    memoryAllocateInfo.memoryTypeIndex = memory_type_index;

    VkImportMemoryHostPointerInfoEXT importMemoryHostPointerInfo;
    importMemoryHostPointerInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    importMemoryHostPointerInfo.pNext = 0;
    importMemoryHostPointerInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    importMemoryHostPointerInfo.pHostPointer = host_ptr;
    memoryAllocateInfo.pNext = &importMemoryHostPointerInfo;

    VkDeviceMemory memory = 0;
    VkResult ret = vkAllocateMemory(vkdev->vkdevice(), &memoryAllocateInfo, 0, &memory);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkAllocateMemory failed %d", ret);
        return 0;
    }

    return memory;
}

VkDeviceMemory VkAllocator::allocate_import_host_memory(size_t size, uint32_t memory_type_index, void* host_ptr)
{
    return ncnn::allocate_import_host_memory(vkdev, size, memory_type_index, host_ptr);
}

VkImage VkAllocator::create_image(int width, int height, int depth, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage)
{
    VkImageCreateInfo imageCreateInfo;
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    imageCreateInfo.pNext = 0;
    imageCreateInfo.flags = 0;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_3D;
    imageCreateInfo.format = format;
    imageCreateInfo.extent.width = width;
    imageCreateInfo.extent.height = height;
    imageCreateInfo.extent.depth = depth;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = tiling;
    imageCreateInfo.usage = usage;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.queueFamilyIndexCount = 0;
    imageCreateInfo.pQueueFamilyIndices = 0;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image;
    VkResult ret = vkCreateImage(vkdev->vkdevice(), &imageCreateInfo, 0, &image);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateImage failed %d %d %d %d %d %d %d", ret, width, height, depth, format, tiling, usage);
        return 0;
    }

    return image;
}

VkImageView VkAllocator::create_imageview(VkImage image, VkFormat format)
{
    VkImageViewCreateInfo imageViewCreateInfo;
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.pNext = 0;
    imageViewCreateInfo.flags = 0;
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    imageViewCreateInfo.format = format;
    imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    VkImageView imageview;
    VkResult ret = vkCreateImageView(vkdev->vkdevice(), &imageViewCreateInfo, 0, &imageview);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateImageView failed %d", ret);
        return 0;
    }

    return imageview;
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

class VkBlobAllocatorPrivate
{
public:
    size_t block_size;
    size_t buffer_offset_alignment;
    size_t bind_memory_offset_alignment;
    std::vector<std::list<std::pair<size_t, size_t> > > buffer_budgets;
    std::vector<VkBufferMemory*> buffer_blocks;
    std::vector<std::list<std::pair<size_t, size_t> > > image_memory_budgets;
    std::vector<VkDeviceMemory> image_memory_blocks;
};

VkBlobAllocator::VkBlobAllocator(const VulkanDevice* _vkdev, size_t preferred_block_size)
    : VkAllocator(_vkdev), d(new VkBlobAllocatorPrivate)
{
    d->buffer_offset_alignment = vkdev->info.buffer_offset_alignment();
    d->bind_memory_offset_alignment = vkdev->info.buffer_image_granularity();

    if (vkdev->info.type() == 1)
    {
        // on integrated gpu, there may be device local only memory too, eg. AMD APU
        // assuming larger alignment always keeps us safe :)

        // least common multiple for memory_map_alignment and buffer_offset_alignment and non_coherent_atom_size
        d->buffer_offset_alignment = least_common_multiple(d->buffer_offset_alignment, vkdev->info.memory_map_alignment());
        d->buffer_offset_alignment = least_common_multiple(d->buffer_offset_alignment, vkdev->info.non_coherent_atom_size());
    }

    if (vkdev->info.support_VK_KHR_robustness2() || vkdev->info.support_VK_EXT_robustness2())
    {
        size_t robust_storage_buffer_access_size_alignment = vkdev->info.queryRobustness2Properties().robustStorageBufferAccessSizeAlignment;
        d->buffer_offset_alignment = least_common_multiple(d->buffer_offset_alignment, robust_storage_buffer_access_size_alignment);
    }

    d->block_size = alignSize(preferred_block_size, d->buffer_offset_alignment);
}

VkBlobAllocator::~VkBlobAllocator()
{
    clear();

    delete d;
}

VkBlobAllocator::VkBlobAllocator(const VkBlobAllocator&)
    : VkAllocator(0), d(0)
{
}

VkBlobAllocator& VkBlobAllocator::operator=(const VkBlobAllocator&)
{
    return *this;
}

void VkBlobAllocator::clear()
{
    //     NCNN_LOGE("VkBlobAllocator %lu", buffer_blocks.size());

    for (size_t i = 0; i < d->buffer_blocks.size(); i++)
    {
        VkBufferMemory* ptr = d->buffer_blocks[i];

        //         std::list< std::pair<size_t, size_t> >::iterator it = buffer_budgets[i].begin();
        //         while (it != buffer_budgets[i].end())
        //         {
        //             NCNN_LOGE("VkBlobAllocator budget %p %lu %lu", ptr->buffer, it->first, it->second);
        //             it++;
        //         }

        if (mappable)
            vkUnmapMemory(vkdev->vkdevice(), ptr->memory);

        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }
    d->buffer_blocks.clear();

    d->buffer_budgets.clear();

    for (size_t i = 0; i < d->image_memory_blocks.size(); i++)
    {
        VkDeviceMemory memory = d->image_memory_blocks[i];

        //         std::list< std::pair<size_t, size_t> >::iterator it = d->image_memory_budgets[i].begin();
        //         while (it != d->image_memory_budgets[i].end())
        //         {
        //             NCNN_LOGE("VkBlobAllocator budget %p %lu %lu", memory, it->first, it->second);
        //             it++;
        //         }

        vkFreeMemory(vkdev->vkdevice(), memory, 0);
    }
    d->image_memory_blocks.clear();

    d->image_memory_budgets.clear();
}

VkBufferMemory* VkBlobAllocator::fastMalloc(size_t size)
{
    size_t aligned_size = alignSize(size, d->buffer_offset_alignment);

    const int buffer_block_count = d->buffer_blocks.size();

    // find first spare space in buffer_blocks
    for (int i = 0; i < buffer_block_count; i++)
    {
        std::list<std::pair<size_t, size_t> >::iterator it = d->buffer_budgets[i].begin();
        while (it != d->buffer_budgets[i].end())
        {
            size_t budget_size = it->second;
            if (budget_size < aligned_size)
            {
                it++;
                continue;
            }

            // return sub buffer
            VkBufferMemory* ptr = new VkBufferMemory;

            ptr->buffer = d->buffer_blocks[i]->buffer;
            ptr->offset = it->first;
            ptr->memory = d->buffer_blocks[i]->memory;
            ptr->capacity = aligned_size;
            ptr->mapped_ptr = d->buffer_blocks[i]->mapped_ptr;
            ptr->memory_type_index = d->buffer_blocks[i]->memory_type_index;
            ptr->storage_type = vkdev->is_device_local(ptr->memory_type_index) ? 0 : 1;
            ptr->buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            ptr->access_flags = 0;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

            // adjust buffer_budgets
            if (budget_size == aligned_size)
            {
                d->buffer_budgets[i].erase(it);
            }
            else
            {
                it->first += aligned_size;
                it->second -= aligned_size;
            }

            //             NCNN_LOGE("VkBlobAllocator M %p +%lu %lu", ptr->buffer, ptr->offset, ptr->capacity);

            return ptr;
        }
    }

    size_t new_block_size = std::max(d->block_size, aligned_size);

    // create new block
    VkBufferMemory* block = new VkBufferMemory;

    block->buffer = create_buffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    block->offset = 0;

    // TODO respect VK_KHR_dedicated_allocation ?

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), block->buffer, &memoryRequirements);

    // setup memory type and alignment
    if (buffer_memory_type_index == (uint32_t)-1)
    {
        if (vkdev->info.type() == 1)
        {
            // integrated gpu, prefer unified memory
            buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);

            // on amd integrated gpu, there is a faster and larger device-only heap
            uint32_t device_local_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            const VkPhysicalDeviceMemoryProperties& memory_properties = vkdev->info.physicalDeviceMemoryProperties();
            uint32_t buffer_heap_index = memory_properties.memoryTypes[buffer_memory_type_index].heapIndex;
            uint32_t device_local_heap_index = memory_properties.memoryTypes[device_local_memory_type_index].heapIndex;
            if (device_local_heap_index < buffer_heap_index && memory_properties.memoryHeaps[device_local_heap_index].size > memory_properties.memoryHeaps[buffer_heap_index].size)
            {
                buffer_memory_type_index = device_local_memory_type_index;
            }
        }
        else
        {
            // discrete gpu, device local
            buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        }

        mappable = vkdev->is_mappable(buffer_memory_type_index);
        coherent = vkdev->is_coherent(buffer_memory_type_index);
        cached = vkdev->is_cached(buffer_memory_type_index);
    }

    block->memory = allocate_memory(memoryRequirements.size, buffer_memory_type_index);
    if (!block->memory)
    {
        vkDestroyBuffer(vkdev->vkdevice(), block->buffer, 0);
        delete block;
        return 0;
    }

    // ignore memoryRequirements.alignment as we always bind at zero offset
    vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0);

    block->mapped_ptr = 0;
    if (mappable)
    {
        vkMapMemory(vkdev->vkdevice(), block->memory, 0, new_block_size, 0, &block->mapped_ptr);
    }

    block->memory_type_index = buffer_memory_type_index;
    block->storage_type = vkdev->is_device_local(block->memory_type_index) ? 0 : 1;
    block->buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    d->buffer_blocks.push_back(block);

    // return sub buffer
    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = block->buffer;
    ptr->offset = 0;
    ptr->memory = block->memory;
    ptr->capacity = aligned_size;
    ptr->mapped_ptr = block->mapped_ptr;
    ptr->memory_type_index = block->memory_type_index;
    ptr->storage_type = vkdev->is_device_local(ptr->memory_type_index) ? 0 : 1;
    ptr->buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    // adjust buffer_budgets
    std::list<std::pair<size_t, size_t> > budget;
    if (new_block_size > aligned_size)
    {
        budget.push_back(std::make_pair(aligned_size, new_block_size - aligned_size));
    }
    d->buffer_budgets.push_back(budget);

    //     NCNN_LOGE("VkBlobAllocator M %p +%lu %lu", ptr->buffer, ptr->offset, ptr->capacity);

    return ptr;
}

void VkBlobAllocator::fastFree(VkBufferMemory* ptr)
{
    //     NCNN_LOGE("VkBlobAllocator F %p +%lu %lu", ptr->buffer, ptr->offset, ptr->capacity);

    const int buffer_block_count = d->buffer_blocks.size();

    int block_index = -1;
    for (int i = 0; i < buffer_block_count; i++)
    {
        if (d->buffer_blocks[i]->buffer == ptr->buffer && d->buffer_blocks[i]->memory == ptr->memory)
        {
            block_index = i;
            break;
        }
    }

    if (block_index == -1)
    {
        NCNN_LOGE("FATAL ERROR! unlocked VkBlobAllocator get wild %p", ptr->buffer);

        delete ptr;

        return;
    }

    // merge
    std::list<std::pair<size_t, size_t> >::iterator it_merge_left = d->buffer_budgets[block_index].end();
    std::list<std::pair<size_t, size_t> >::iterator it_merge_right = d->buffer_budgets[block_index].end();
    std::list<std::pair<size_t, size_t> >::iterator it = d->buffer_budgets[block_index].begin();
    for (; it != d->buffer_budgets[block_index].end(); it++)
    {
        if (it->first + it->second == ptr->offset)
        {
            it_merge_left = it;
        }
        else if (ptr->offset + ptr->capacity == it->first)
        {
            it_merge_right = it;
        }
    }

    if (it_merge_left != d->buffer_budgets[block_index].end() && it_merge_right != d->buffer_budgets[block_index].end())
    {
        it_merge_left->second = it_merge_right->first + it_merge_right->second - it_merge_left->first;
        d->buffer_budgets[block_index].erase(it_merge_right);
    }
    else if (it_merge_left != d->buffer_budgets[block_index].end())
    {
        it_merge_left->second = ptr->offset + ptr->capacity - it_merge_left->first;
    }
    else if (it_merge_right != d->buffer_budgets[block_index].end())
    {
        it_merge_right->second = it_merge_right->first + it_merge_right->second - ptr->offset;
        it_merge_right->first = ptr->offset;
    }
    else
    {
        if (ptr->offset == 0)
        {
            // chain leading block
            d->buffer_budgets[block_index].push_front(std::make_pair(ptr->offset, ptr->capacity));
        }
        else
        {
            d->buffer_budgets[block_index].push_back(std::make_pair(ptr->offset, ptr->capacity));
        }
    }

    delete ptr;
}

VkImageMemory* VkBlobAllocator::fastMalloc(int w, int h, int c, size_t elemsize, int elempack)
{
    if (elempack != 1 && elempack != 4 && elempack != 8)
    {
        NCNN_LOGE("elempack must be 1 4 8");
        return 0;
    }

    // resolve format
    VkFormat format = VK_FORMAT_UNDEFINED;

    if (elemsize / elempack == 4)
    {
        // fp32
        if (elempack == 1) format = VK_FORMAT_R32_SFLOAT;
        if (elempack == 4) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 8) format = VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    if (elemsize / elempack == 2)
    {
        // fp16
        if (elempack == 1) format = VK_FORMAT_R16_SFLOAT;
        if (elempack == 4) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 8) format = VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    if (elemsize / elempack == 1)
    {
        // int8
        if (elempack == 1) format = VK_FORMAT_R8_SINT;
        if (elempack == 4) format = VK_FORMAT_R8G8B8A8_SINT;
        if (elempack == 8) format = VK_FORMAT_R8G8B8A8_SINT;
    }

    // resolve image width height depth
    int width = w;
    int height = h;
    int depth = c;

    // large elempack spills on image w
    if (elempack == 8) width *= 2;

    if (width > (int)vkdev->info.max_image_dimension_3d() || height > (int)vkdev->info.max_image_dimension_3d() || depth > (int)vkdev->info.max_image_dimension_3d())
    {
        NCNN_LOGE("image dimension too large %d %d %d > %d", width, height, depth, (int)vkdev->info.max_image_dimension_3d());
        return 0;
    }

    VkImageMemory* ptr = new VkImageMemory;

    ptr->image = create_image(width, height, depth, format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    ptr->width = width;
    ptr->height = height;
    ptr->depth = depth;
    ptr->format = format;

    // TODO respect VK_KHR_dedicated_allocation ?
    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(vkdev->vkdevice(), ptr->image, &memoryRequirements);

    const size_t size = memoryRequirements.size;
    const size_t alignment = std::max((size_t)memoryRequirements.alignment, d->bind_memory_offset_alignment);

    size_t aligned_size = alignSize(size, alignment);

    const int image_memory_block_count = d->image_memory_blocks.size();

    // find first spare space in image_memory_blocks
    for (int i = 0; i < image_memory_block_count; i++)
    {
#if __APPLE__
        // HACK moltenvk v1.2.3 is unhappy for image binding with offset  :(
        break;
#endif

        std::list<std::pair<size_t, size_t> >::iterator it = d->image_memory_budgets[i].begin();
        while (it != d->image_memory_budgets[i].end())
        {
            // we cannot use it->first directly for base offset alignment
            size_t bind_base_offset = it->first;
            size_t bind_offset = alignSize(bind_base_offset, alignment);
            size_t budget_size = it->second;
            if (budget_size < aligned_size + (bind_offset - bind_base_offset))
            {
                it++;
                continue;
            }

            // bind at memory offset
            ptr->memory = d->image_memory_blocks[i];
            ptr->bind_offset = bind_offset;
            ptr->bind_capacity = aligned_size;

            vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

            // do not allow host access to optimal tiling image
            ptr->mapped_ptr = 0;
            ptr->memory_type_index = image_memory_type_index;

            ptr->imageview = create_imageview(ptr->image, format);

            ptr->access_flags = 0;
            ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            ptr->command_refcount = 0;

            if (bind_base_offset != bind_offset)
            {
                // NOTE there is small offset inside bind_base_offset and bind_offset
                // adjust ptr->bind_offset and ptr->bind_capacity after vkBindImageMemory
                // so that memory management could be easier
                aligned_size += (bind_offset - bind_base_offset);

                ptr->bind_offset = bind_base_offset;
                ptr->bind_capacity = aligned_size;
            }

            // adjust image_memory_budgets
            if (budget_size == aligned_size)
            {
                d->image_memory_budgets[i].erase(it);
            }
            else
            {
                it->first += aligned_size;
                it->second -= aligned_size;
            }

            //             NCNN_LOGE("VkBlobAllocator M %p +%lu %lu", ptr->memory, ptr->bind_offset, ptr->bind_capacity);

            return ptr;
        }
    }

    // setup memory type and alignment
    if (image_memory_type_index == (uint32_t)-1)
    {
        if (vkdev->info.type() == 1)
        {
            // integrated gpu, prefer unified memory
            image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);

            // on amd integrated gpu, there is a faster and larger device-only heap
            uint32_t device_local_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            const VkPhysicalDeviceMemoryProperties& memory_properties = vkdev->info.physicalDeviceMemoryProperties();
            uint32_t buffer_heap_index = memory_properties.memoryTypes[image_memory_type_index].heapIndex;
            uint32_t device_local_heap_index = memory_properties.memoryTypes[device_local_memory_type_index].heapIndex;
            if (device_local_heap_index < buffer_heap_index && memory_properties.memoryHeaps[device_local_heap_index].size > memory_properties.memoryHeaps[buffer_heap_index].size)
            {
                image_memory_type_index = device_local_memory_type_index;
            }
        }
        else
        {
            // discrete gpu, device local
            image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        }

        mappable = vkdev->is_mappable(image_memory_type_index);
        coherent = vkdev->is_coherent(image_memory_type_index);
        cached = vkdev->is_cached(image_memory_type_index);
    }

    // create new block
    size_t new_block_size = std::max(d->block_size, aligned_size);

#if __APPLE__
    // HACK moltenvk v1.2.3 is unhappy for image binding with offset
    // always ignore block size for smaller memory footprint :(
    new_block_size = aligned_size;
#endif

    // bind at memory offset
    ptr->memory = allocate_memory(new_block_size, image_memory_type_index);
    if (!ptr->memory)
    {
        vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);
        delete ptr;
        return 0;
    }
    ptr->bind_offset = 0;
    ptr->bind_capacity = aligned_size;

    // ignore memoryRequirements2.memoryRequirements.alignment as we always bind at zero offset
    vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

    // do not allow host access to optimal tiling image
    ptr->mapped_ptr = 0;
    ptr->memory_type_index = image_memory_type_index;

    ptr->imageview = create_imageview(ptr->image, format);

    ptr->access_flags = 0;
    ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    ptr->command_refcount = 0;

    // adjust image_memory_budgets
    d->image_memory_blocks.push_back(ptr->memory);

    std::list<std::pair<size_t, size_t> > budget;
    if (new_block_size > aligned_size)
    {
        budget.push_back(std::make_pair(aligned_size, new_block_size - aligned_size));
    }
    d->image_memory_budgets.push_back(budget);

    //     NCNN_LOGE("VkBlobAllocator M %p +%lu %lu", ptr->memory, ptr->bind_offset, ptr->bind_capacity);

    return ptr;
}

void VkBlobAllocator::fastFree(VkImageMemory* ptr)
{
    //     NCNN_LOGE("VkBlobAllocator F %p +%lu %lu", ptr->memory, ptr->bind_offset, ptr->bind_capacity);

    const int image_memory_block_count = d->image_memory_blocks.size();

    int block_index = -1;
    for (int i = 0; i < image_memory_block_count; i++)
    {
        if (d->image_memory_blocks[i] == ptr->memory)
        {
            block_index = i;
            break;
        }
    }

    if (block_index == -1)
    {
        NCNN_LOGE("FATAL ERROR! unlocked VkBlobAllocator get wild %p", ptr->memory);

        if (!ptr->command_refcount)
        {
            vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
            vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

            delete ptr;
        }

        return;
    }

    // merge
    std::list<std::pair<size_t, size_t> >::iterator it_merge_left = d->image_memory_budgets[block_index].end();
    std::list<std::pair<size_t, size_t> >::iterator it_merge_right = d->image_memory_budgets[block_index].end();
    std::list<std::pair<size_t, size_t> >::iterator it = d->image_memory_budgets[block_index].begin();
    for (; it != d->image_memory_budgets[block_index].end(); it++)
    {
        if (it->first + it->second == ptr->bind_offset)
        {
            it_merge_left = it;
        }
        else if (ptr->bind_offset + ptr->bind_capacity == it->first)
        {
            it_merge_right = it;
        }
    }

    if (it_merge_left != d->image_memory_budgets[block_index].end() && it_merge_right != d->image_memory_budgets[block_index].end())
    {
        it_merge_left->second = it_merge_right->first + it_merge_right->second - it_merge_left->first;
        d->image_memory_budgets[block_index].erase(it_merge_right);
    }
    else if (it_merge_left != d->image_memory_budgets[block_index].end())
    {
        it_merge_left->second = ptr->bind_offset + ptr->bind_capacity - it_merge_left->first;
    }
    else if (it_merge_right != d->image_memory_budgets[block_index].end())
    {
        it_merge_right->second = it_merge_right->first + it_merge_right->second - ptr->bind_offset;
        it_merge_right->first = ptr->bind_offset;
    }
    else
    {
        if (ptr->bind_offset == 0)
        {
            // chain leading block
            d->image_memory_budgets[block_index].push_front(std::make_pair(ptr->bind_offset, ptr->bind_capacity));
        }
        else
        {
            d->image_memory_budgets[block_index].push_back(std::make_pair(ptr->bind_offset, ptr->bind_capacity));
        }
    }

    if (!ptr->command_refcount)
    {
        vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
        vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

        delete ptr;
    }
}

int upload_weight_buffers(const VulkanDevice* vkdev, const std::vector<VkBufferMemory>& buffers, size_t host_budget);

size_t weight_buffer_alignment(const VulkanDevice* vkdev);
int weight_buffer_memory_size(const VulkanDevice* vkdev, size_t size, size_t& allocation_size, bool concurrent = false);
int create_weight_buffer(const VulkanDevice* vkdev, size_t size, int storage_type, bool unified, size_t budget, VkBufferMemory*& buffer, size_t& allocation_size, bool allocate = true);
void destroy_weight_buffer(const VulkanDevice* vkdev, VkBufferMemory* buffer);
int flush_weight_buffer(const VulkanDevice* vkdev, const VkBufferMemory* buffer, size_t offset, size_t size);

size_t weight_buffer_alignment(const VulkanDevice* vkdev)
{
    size_t alignment = std::max((size_t)4, vkdev->info.buffer_offset_alignment());
    alignment = std::max(alignment, vkdev->info.non_coherent_atom_size());
    if (vkdev->info.support_VK_KHR_robustness2() || vkdev->info.support_VK_EXT_robustness2())
        alignment = std::max(alignment, (size_t)vkdev->info.queryRobustness2Properties().robustStorageBufferAccessSizeAlignment);
    return alignment;
}

static bool unified_weight_memory(const VulkanDevice* vkdev)
{
    // validate mapped device-local memory against the actual buffer requirements when allocating
    return vkdev->info.type() == 1 || vkdev->info.type() == 3;
}

static size_t weight_buffer_size_limit(const VulkanDevice* vkdev, bool host = false)
{
    VkDeviceSize limit = (size_t)-1;
    if (vkdev->info.max_buffer_size())
        limit = std::min(limit, vkdev->info.max_buffer_size());
    if (vkdev->info.max_memory_allocation_size())
        limit = std::min(limit, vkdev->info.max_memory_allocation_size());
    VkMemoryPropertyFlags required = host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (!host && unified_weight_memory(vkdev))
        required |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    VkDeviceSize heap_size = 0;
    const VkPhysicalDeviceMemoryProperties& properties = vkdev->info.physicalDeviceMemoryProperties();
    for (uint32_t i = 0; i < properties.memoryTypeCount; i++)
    {
        const VkMemoryType& type = properties.memoryTypes[i];
        const VkMemoryHeap& heap = properties.memoryHeaps[type.heapIndex];
        if ((type.propertyFlags & required) != required || (host && vkdev->info.type() == 0 && (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)))
            continue;
        heap_size = std::max(heap_size, heap.size);
    }
    limit = std::min(limit, heap_size);
    const size_t alignment = weight_buffer_alignment(vkdev);
    return (size_t)limit / alignment * alignment;
}

static int create_weight_buffer_handle(const VulkanDevice* vkdev, size_t size, bool concurrent, VkBuffer& buffer, VkMemoryRequirements& requirements)
{
    const size_t alignment = weight_buffer_alignment(vkdev);
    if (!size || size > (size_t)-1 - alignment || alignSize(size, alignment) > weight_buffer_size_limit(vkdev, concurrent))
        return -1;
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer = create_buffer(vkdev, alignSize(size, alignment), usage, 0, concurrent);
    if (!buffer)
        return -100;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), buffer, &requirements);
    if (requirements.size > (size_t)-1)
    {
        vkDestroyBuffer(vkdev->vkdevice(), buffer, 0);
        buffer = VK_NULL_HANDLE;
        return -1;
    }
    return 0;
}

int weight_buffer_memory_size(const VulkanDevice* vkdev, size_t size, size_t& allocation_size, bool concurrent)
{
    allocation_size = 0;
    if (!size)
        return 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkMemoryRequirements requirements;
    const int ret = create_weight_buffer_handle(vkdev, size, concurrent, buffer, requirements);
    if (ret)
        return ret;
    allocation_size = (size_t)requirements.size;
    vkDestroyBuffer(vkdev->vkdevice(), buffer, 0);
    return 0;
}

// Fit automatic exchanges using actual memory requirements, including padding.
int fit_weight_capacity(const VulkanDevice* vkdev, size_t budget, bool automatic, bool concurrent, size_t& capacity)
{
    if (!capacity)
        return 0;
    size_t limit = weight_buffer_size_limit(vkdev, concurrent);
    if (budget)
        limit = std::min(limit, budget);
    const size_t alignment = weight_buffer_alignment(vkdev);
    limit = limit / alignment * alignment;
    if (capacity > limit)
    {
        if (!automatic)
            return -1;
        capacity = limit;
    }
    while (capacity)
    {
        size_t allocation;
        if (weight_buffer_memory_size(vkdev, capacity, allocation, concurrent))
            return -1;
        if (allocation <= limit)
            return 0;
        if (!automatic)
            return -1;
        const size_t excess = allocation - limit;
        if (excess > (size_t)-1 - alignment)
            return -1;
        const size_t shrink = alignSize(excess, alignment);
        if (shrink >= capacity)
            return -1;
        capacity -= shrink;
    }
    return -1;
}

void destroy_weight_buffer(const VulkanDevice* vkdev, VkBufferMemory* buffer)
{
    if (!buffer)
        return;
    if (buffer->mapped_ptr)
        vkUnmapMemory(vkdev->vkdevice(), buffer->memory);
    vkDestroyBuffer(vkdev->vkdevice(), buffer->buffer, 0);
    vkFreeMemory(vkdev->vkdevice(), buffer->memory, 0);
    delete buffer;
}

static int allocate_weight_buffer_memory(const VulkanDevice* vkdev, VkBufferMemory* buffer)
{
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), buffer->buffer, &requirements);
    VkDeviceMemory memory = allocate_memory(vkdev, (size_t)requirements.size, buffer->memory_type_index);
    if (!memory)
        return -100;
    buffer->memory = memory;
    VkResult ret = vkBindBufferMemory(vkdev->vkdevice(), buffer->buffer, memory, 0);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("bind weight memory failed %d", ret);
        return -100;
    }
    if (vkdev->is_mappable(buffer->memory_type_index))
    {
        void* mapped_ptr = 0;
        ret = vkMapMemory(vkdev->vkdevice(), buffer->memory, 0, VK_WHOLE_SIZE, 0, &mapped_ptr);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("map weight memory failed %d", ret);
            return -100;
        }
        buffer->mapped_ptr = mapped_ptr;
    }
    return 0;
}

int create_weight_buffer(const VulkanDevice* vkdev, size_t size, int storage_type, bool unified, size_t budget, VkBufferMemory*& buffer, size_t& allocation_size, bool allocate)
{
    buffer = 0;
    allocation_size = 0;
    VkBufferMemory* ptr = new VkBufferMemory;
    ptr->buffer = VK_NULL_HANDLE;
    ptr->memory = VK_NULL_HANDLE;
    ptr->mapped_ptr = 0;
    VkMemoryRequirements requirements;
    int ret = create_weight_buffer_handle(vkdev, size, !unified && storage_type == 1, ptr->buffer, requirements);
    if (ret)
    {
        delete ptr;
        return ret;
    }
    if ((budget && requirements.size > budget) || (vkdev->info.max_memory_allocation_size() && requirements.size > vkdev->info.max_memory_allocation_size()))
    {
        NCNN_LOGE("weight allocation exceeds budget or device limit: needs %llu, budget %llu, limit %llu", (unsigned long long)requirements.size, (unsigned long long)budget,
                  (unsigned long long)vkdev->info.max_memory_allocation_size());
        destroy_weight_buffer(vkdev, ptr);
        return -100;
    }
    VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkMemoryPropertyFlags preferred = 0;
    VkMemoryPropertyFlags avoided = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    if (unified)
    {
        required |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        preferred = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        avoided = 0;
    }
    else if (storage_type == 1)
    {
        required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        preferred = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        avoided = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
    const uint32_t memory_type = vkdev->find_memory_index(requirements.memoryTypeBits, required, preferred, avoided);
    const VkPhysicalDeviceMemoryProperties& properties = vkdev->info.physicalDeviceMemoryProperties();
    if (memory_type == (uint32_t)-1 || (!unified && (budget || vkdev->info.type() == 0) && storage_type == 1 && (properties.memoryHeaps[properties.memoryTypes[memory_type].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)))
    {
        NCNN_LOGE("no suitable weight memory type for storage %d", storage_type);
        destroy_weight_buffer(vkdev, ptr);
        return -1;
    }
    if (requirements.size > properties.memoryHeaps[properties.memoryTypes[memory_type].heapIndex].size)
    {
        NCNN_LOGE("weight allocation exceeds memory heap size");
        destroy_weight_buffer(vkdev, ptr);
        return -100;
    }
    ptr->offset = 0;
    ptr->capacity = alignSize(size, weight_buffer_alignment(vkdev));
    ptr->memory_type_index = memory_type;
    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    ptr->refcount = 0;
    ptr->storage_type = unified ? 0 : storage_type;
    ptr->buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (allocate && allocate_weight_buffer_memory(vkdev, ptr))
    {
        destroy_weight_buffer(vkdev, ptr);
        return -100;
    }
    buffer = ptr;
    allocation_size = (size_t)requirements.size;
    return 0;
}

int flush_weight_buffer(const VulkanDevice* vkdev, const VkBufferMemory* buffer, size_t offset, size_t size)
{
    if (!buffer || !buffer->mapped_ptr || offset > buffer->capacity || size > buffer->capacity - offset)
        return -1;
    if (!size || vkdev->is_coherent(buffer->memory_type_index))
        return 0;
    return flush_buffer_memory(vkdev, buffer, offset, size);
}

class WeightFile
{
public:
    WeightFile();
    ~WeightFile();

    int set_directory(const char* path);
    int open();
    void clear();
    int reserve(size_t size, size_t alignment, void*& data);
    int finalize();
#if NCNN_BENCHMARK
    int discard_page_cache() const;
#endif

private:
    WeightFile(const WeightFile&);
    WeightFile& operator=(const WeightFile&);

    struct Mapping
    {
        void* base;
        size_t mapped_size;
    };

    std::vector<char> directory;
    FILE* file;
    uint64_t file_size;
#if defined(__APPLE__)
    uint64_t allocated_size;
#endif
    std::vector<Mapping> mappings;
    bool readonly;
    int error;
};

WeightFile::WeightFile()
    : file(0), file_size(0), readonly(false), error(0)
{
#if defined(__APPLE__)
    allocated_size = 0;
#endif
}

WeightFile::~WeightFile()
{
    clear();
}

int WeightFile::set_directory(const char* path)
{
    if (file)
    {
        NCNN_LOGE("weight file directory must be configured before opening the backing file");
        return -1;
    }
    directory.clear();
    if (path && path[0])
    {
        const size_t length = strlen(path) + 1;
        directory.resize(length);
        memcpy(directory.data(), path, length);
    }
    return 0;
}

int WeightFile::open()
{
    if (file)
        return error;

#if defined(_WIN32)
    std::vector<wchar_t> path;
    if (directory.empty())
    {
        const DWORD length = GetTempPathW(0, 0);
        if (!length)
            return -1;
        path.resize(length + 64);
        if (!GetTempPathW(length, path.data()))
            return -1;
    }
    else
    {
        const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, directory.data(), -1, 0, 0);
        if (!length)
            return -1;
        path.resize(length + 64);
        if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, directory.data(), -1, path.data(), length))
            return -1;
    }
    const size_t prefix = wcslen(path.data());
    static volatile LONG sequence = 0;
    HANDLE handle = INVALID_HANDLE_VALUE;
    for (int i = 0; i < 64; i++)
    {
        swprintf(path.data() + prefix, 64, L"/ncnn-weights-%lu-%lu.tmp", GetCurrentProcessId(), (unsigned long)InterlockedIncrement(&sequence));
        handle = CreateFileW(path.data(), GENERIC_READ | GENERIC_WRITE, 0, 0, CREATE_NEW, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, 0);
        if (handle != INVALID_HANDLE_VALUE || (GetLastError() != ERROR_FILE_EXISTS && GetLastError() != ERROR_ALREADY_EXISTS))
            break;
    }
    if (handle == INVALID_HANDLE_VALUE)
    {
        NCNN_LOGE("create prepared weight backing file failed %lu", GetLastError());
        return -1;
    }
    const int fd = _open_osfhandle((intptr_t)handle, _O_RDWR | _O_BINARY);
    if (fd < 0)
        CloseHandle(handle);
    else
    {
        file = _fdopen(fd, "w+b");
        if (!file)
            _close(fd);
    }
#else
    const char* temporary_directory = directory.empty() ? getenv("TMPDIR") : directory.data();
    if (temporary_directory && temporary_directory[0])
    {
        std::string path = temporary_directory;
        path += "/ncnn-weights-XXXXXX";
        const int fd = mkstemp(&path[0]);
        if (fd >= 0)
        {
            // unlink immediately so interruption cannot leave prepared files behind
            if (unlink(path.c_str()) == 0)
                file = fdopen(fd, "w+b");
            if (!file)
                close(fd);
        }
    }
    else
    {
        file = tmpfile();
    }
#endif
    if (!file)
    {
        NCNN_LOGE("create prepared weight backing file failed %d", errno);
        return -1;
    }
#if !defined(_WIN32)
    if (fcntl(fileno(file), F_SETFD, FD_CLOEXEC))
    {
        NCNN_LOGE("set weight file close-on-exec failed %d", errno);
        fclose(file);
        file = 0;
        return -1;
    }
#endif
    return 0;
}

void WeightFile::clear()
{
    for (size_t i = 0; i < mappings.size(); i++)
    {
#if defined(_WIN32)
        UnmapViewOfFile(mappings[i].base);
#else
        munmap(mappings[i].base, mappings[i].mapped_size);
#endif
    }
    mappings.clear();
    if (file)
        fclose(file);
    file = 0;
    file_size = 0;
#if defined(__APPLE__)
    allocated_size = 0;
#endif
    readonly = false;
    error = 0;
}

int WeightFile::reserve(size_t size, size_t alignment, void*& data)
{
    data = 0;
    if (!file || error || readonly || !size || !alignment)
        return -1;

    const uint64_t padding = (alignment - file_size % alignment) % alignment;
    if (padding > (uint64_t)INT64_MAX - file_size || size > (uint64_t)INT64_MAX - file_size - padding)
    {
        NCNN_LOGE("prepared weight backing file range overflow");
        return -1;
    }
    const uint64_t offset = file_size + padding;
    const uint64_t end = offset + size;
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    const size_t granularity = info.dwAllocationGranularity;
    FILE_ALLOCATION_INFO allocation;
    allocation.AllocationSize.QuadPart = end;
    HANDLE file_handle = (HANDLE)_get_osfhandle(_fileno(file));
    if (!SetFileInformationByHandle(file_handle, FileAllocationInfo, &allocation, sizeof(allocation)))
    {
        NCNN_LOGE("reserve prepared weight file space failed %lu", GetLastError());
        error = -1;
        return error;
    }
#else
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
        return -1;
    const size_t granularity = (size_t)page_size;
    if (sizeof(off_t) < 8 && end > INT32_MAX)
        return -1;
    // Extending a sparse file alone does not reserve space for mapped writes.
    int ret;
#if defined(__APPLE__)
    ret = 0;
    if (end > allocated_size)
    {
        fstore_t allocation = {};
        allocation.fst_flags = F_ALLOCATEALL;
        allocation.fst_posmode = F_PEOFPOSMODE;
        allocation.fst_length = (off_t)(end - allocated_size);
        do
        {
            ret = fcntl(fileno(file), F_PREALLOCATE, &allocation);
        } while (ret && errno == EINTR);
        if (!ret)
            allocated_size += allocation.fst_bytesalloc;
    }
    if (!ret)
        ret = ftruncate(fileno(file), (off_t)end);
    if (ret)
        ret = errno;
#else
    do
    {
        ret = posix_fallocate(fileno(file), (off_t)file_size, (off_t)(end - file_size));
    } while (ret == EINTR);
#endif
    if (ret)
    {
        NCNN_LOGE("reserve prepared weight file space failed %d", ret);
        error = -1;
        return error;
    }
#endif
    const uint64_t map_offset = offset / granularity * granularity;
    const size_t prefix = (size_t)(offset - map_offset);
    if (size > (size_t)-1 - prefix)
    {
        error = -100;
        return error;
    }
    Mapping m;
    m.mapped_size = prefix + size;
#if defined(_WIN32)
    HANDLE handle = CreateFileMapping((HANDLE)_get_osfhandle(_fileno(file)), 0, PAGE_READWRITE, (DWORD)(end >> 32), (DWORD)end, 0);
    if (!handle)
    {
        NCNN_LOGE("create prepared weight mapping failed %lu", GetLastError());
        error = -100;
        return error;
    }
    m.base = MapViewOfFile(handle, FILE_MAP_READ | FILE_MAP_WRITE, (DWORD)(map_offset >> 32), (DWORD)map_offset, m.mapped_size);
    const DWORD map_error = m.base ? 0 : GetLastError();
    CloseHandle(handle);
    if (!m.base)
    {
        NCNN_LOGE("map prepared weight file failed %lu", map_error);
        error = -100;
        return error;
    }
#else
    m.base = mmap(0, m.mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(file), (off_t)map_offset);
    if (m.base == MAP_FAILED)
    {
        NCNN_LOGE("map prepared weight file failed %d", errno);
        error = -100;
        return error;
    }
#endif
    // each view remains at the same address while later allocations grow the file
    mappings.push_back(m);
    file_size = end;
    data = (unsigned char*)m.base + prefix;
    return 0;
}

int WeightFile::finalize()
{
    if (error || readonly)
        return error;
    for (size_t i = 0; i < mappings.size(); i++)
    {
        const Mapping& m = mappings[i];
#if defined(_WIN32)
        DWORD old_protect;
        if (!FlushViewOfFile(m.base, m.mapped_size) || !VirtualProtect(m.base, m.mapped_size, PAGE_READONLY, &old_protect))
#else
        if (msync(m.base, m.mapped_size, MS_SYNC) || mprotect(m.base, m.mapped_size, PROT_READ))
#endif
        {
            NCNN_LOGE("finalize prepared weight mapping failed");
            error = -1;
            break;
        }
    }
    readonly = true;
    return error;
}

#if NCNN_BENCHMARK
int WeightFile::discard_page_cache() const
{
#if defined(__linux__)
    if (!file || error || !readonly || fdatasync(fileno(file)) != 0)
        return -1;
    for (size_t i = 0; i < mappings.size(); i++)
        if (madvise(mappings[i].base, mappings[i].mapped_size, MADV_DONTNEED))
            return -1;
    return posix_fadvise(fileno(file), 0, 0, POSIX_FADV_DONTNEED);
#else
    return -1;
#endif
}

#endif

struct VkWeightRecord
{
    size_t data_size;
    size_t allocation_size;
    const void* source;
    const VkBufferMemory* memory;
};

class VkWeightStore
{
public:
    explicit VkWeightStore(const VulkanDevice* _vkdev);
    void clear();
    int set_weights_in_host_disk(bool enable, bool prefer_host_memory);
    int set_weight_budgets(size_t device_budget, size_t host_budget);
    int prepare_residency(bool& prepared);
    int allocate_residents(int storage_type);

    const VulkanDevice* vkdev;
    bool prefer_host_disk;
    WeightFile file;
    std::vector<VkWeightRecord> weights;
    size_t managed_weight_count;
    size_t max_source_size;
    size_t max_external_size;
    size_t group_size;
    size_t max_group_size;
    bool unified;
    bool finalized;
    int finalization_error;
    size_t weight_budgets[2];
    size_t cache_sizes[2];
    size_t resident_sizes[2];
    uint32_t weight_group;
    std::vector<uint32_t> weight_groups;
    std::vector<VkBufferMemory*> resident_blocks;
    std::vector<int> resident_block_for_weight;
    std::vector<size_t> resident_offsets;

private:
    int weight_layout_size(const std::vector<uint32_t>& ids, uint64_t& total, bool concurrent = false) const;
    int place_resident_weights(const std::vector<uint32_t>& ids, int storage_type, size_t budget);
    int plan_weights();
};

// clear() frees the backing storage; old handles can only release their metadata.
struct VkWeightBufferMemory : public VkBufferMemory
{
    uint64_t allocation_generation;
};

class VkWeightAllocatorPrivate
{
public:
    explicit VkWeightAllocatorPrivate(const VulkanDevice* vkdev)
        : allocator(0), generation(0), allocation_generation(0), store(vkdev)
    {
    }

    static VkWeightStore& store_for(const VkWeightAllocator* allocator)
    {
        return allocator->d->store;
    }

    VkBufferMemory* fastMalloc_resident(VkWeightAllocator* allocator, size_t size);

    const VkAllocator* allocator;
    uint64_t generation;
    uint64_t allocation_generation;
    size_t block_size;
    size_t buffer_offset_alignment;
    size_t bind_memory_offset_alignment;
    std::vector<size_t> buffer_block_free_spaces;
    std::vector<VkBufferMemory*> buffer_blocks;
    std::vector<VkBufferMemory*> dedicated_buffer_blocks;
    std::vector<size_t> image_memory_block_free_spaces;
    std::vector<VkDeviceMemory> image_memory_blocks;
    std::vector<VkDeviceMemory> dedicated_image_memory_blocks;

    bool prefer_host_memory;
    VkWeightStore store;
#if !defined(_WIN32)
    std::vector<void*> host_ptrs;
#endif
};

// Recognize live weight allocators without changing VkAllocator's public API.
static Mutex weight_allocators_lock;
static std::vector<VkWeightAllocatorPrivate*> weight_allocators;
static uint64_t weight_allocator_generation = 0;

static void update_weight_generation(VkWeightAllocatorPrivate* allocator)
{
    MutexLockGuard lock(weight_allocators_lock);
    allocator->generation = ++weight_allocator_generation;
}

VkWeightStore::VkWeightStore(const VulkanDevice* _vkdev)
    : vkdev(_vkdev)
{
    prefer_host_disk = false;
    managed_weight_count = 0;
    max_source_size = 0;
    max_external_size = 0;
    group_size = 0;
    max_group_size = 0;
    unified = unified_weight_memory(vkdev);
    finalized = false;
    finalization_error = 0;
    weight_group = 0;
    for (int i = 0; i < 2; i++)
    {
        weight_budgets[i] = 0;
        cache_sizes[i] = 0;
        resident_sizes[i] = 0;
    }
}

VkWeightAllocator::VkWeightAllocator(const VulkanDevice* _vkdev, bool _prefer_host_memory, size_t preferred_block_size)
    : VkAllocator(_vkdev), d(new VkWeightAllocatorPrivate(_vkdev))
{
    d->buffer_offset_alignment = vkdev->info.buffer_offset_alignment();
    d->bind_memory_offset_alignment = vkdev->info.buffer_image_granularity();

    if (vkdev->info.type() == 1)
    {
        // on integrated gpu, there may be device local only memory too, eg. AMD APU
        // assuming larger alignment always keeps us safe :)

        // least common multiple for memory_map_alignment and buffer_offset_alignment and non_coherent_atom_size
        d->buffer_offset_alignment = least_common_multiple(d->buffer_offset_alignment, vkdev->info.memory_map_alignment());
        d->buffer_offset_alignment = least_common_multiple(d->buffer_offset_alignment, vkdev->info.non_coherent_atom_size());
    }

    if (vkdev->info.support_VK_KHR_robustness2() || vkdev->info.support_VK_EXT_robustness2())
    {
        size_t robust_storage_buffer_access_size_alignment = vkdev->info.queryRobustness2Properties().robustStorageBufferAccessSizeAlignment;
        d->buffer_offset_alignment = least_common_multiple(d->buffer_offset_alignment, robust_storage_buffer_access_size_alignment);
    }

    if (_prefer_host_memory && vkdev->info.support_VK_EXT_external_memory_host())
    {
        size_t min_imported_host_pointer_alignment = vkdev->info.queryExternalMemoryHostProperties().minImportedHostPointerAlignment;
        d->buffer_offset_alignment = least_common_multiple(d->buffer_offset_alignment, min_imported_host_pointer_alignment);
    }

    d->block_size = alignSize(preferred_block_size, d->buffer_offset_alignment);

    d->prefer_host_memory = _prefer_host_memory;
    {
        MutexLockGuard lock(weight_allocators_lock);
        d->allocator = this;
        d->generation = ++weight_allocator_generation;
        weight_allocators.push_back(d);
    }
}

VkWeightAllocator::~VkWeightAllocator()
{
    clear();
    {
        MutexLockGuard lock(weight_allocators_lock);
        for (size_t i = 0; i < weight_allocators.size(); i++)
        {
            if (weight_allocators[i] != d)
                continue;
            weight_allocators.erase(weight_allocators.begin() + i);
            break;
        }
    }

    delete d;
}

VkWeightAllocator::VkWeightAllocator(const VkWeightAllocator&)
    : VkAllocator(0), d(0)
{
}

VkWeightAllocator& VkWeightAllocator::operator=(const VkWeightAllocator&)
{
    return *this;
}

void VkWeightStore::clear()
{
    file.clear();
    managed_weight_count = 0;
    max_source_size = 0;
    max_external_size = 0;
    group_size = 0;
    max_group_size = 0;
    finalized = false;
    finalization_error = 0;
    weights.clear();
    weight_groups.clear();
    weight_group = 0;

    for (size_t i = 0; i < resident_blocks.size(); i++)
        destroy_weight_buffer(vkdev, resident_blocks[i]);
    resident_blocks.clear();
    resident_block_for_weight.clear();
    resident_offsets.clear();
    resident_sizes[0] = 0;
    resident_sizes[1] = 0;
}

void VkWeightAllocator::clear()
{
    update_weight_generation(d);
    d->allocation_generation++;
    d->store.clear();

    //     NCNN_LOGE("VkWeightAllocator %lu %lu", d->buffer_blocks.size(), d->dedicated_buffer_blocks.size());

    d->buffer_block_free_spaces.clear();

    for (size_t i = 0; i < d->buffer_blocks.size(); i++)
    {
        VkBufferMemory* ptr = d->buffer_blocks[i];

        if (mappable)
            vkUnmapMemory(vkdev->vkdevice(), ptr->memory);

        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }
    d->buffer_blocks.clear();

    for (size_t i = 0; i < d->dedicated_buffer_blocks.size(); i++)
    {
        VkBufferMemory* ptr = d->dedicated_buffer_blocks[i];

        if (mappable)
            vkUnmapMemory(vkdev->vkdevice(), ptr->memory);

        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }
    d->dedicated_buffer_blocks.clear();

    d->image_memory_block_free_spaces.clear();

    for (size_t i = 0; i < d->image_memory_blocks.size(); i++)
    {
        VkDeviceMemory memory = d->image_memory_blocks[i];

        vkFreeMemory(vkdev->vkdevice(), memory, 0);
    }
    d->image_memory_blocks.clear();

    for (size_t i = 0; i < d->dedicated_image_memory_blocks.size(); i++)
    {
        VkDeviceMemory memory = d->dedicated_image_memory_blocks[i];

        vkFreeMemory(vkdev->vkdevice(), memory, 0);
    }
    d->dedicated_image_memory_blocks.clear();

#if !defined(_WIN32)
    for (size_t i = 0; i < d->host_ptrs.size(); i++)
    {
        void* host_ptr = d->host_ptrs[i];

        ncnn::fastFree(host_ptr);
    }
    d->host_ptrs.clear();
#endif
}

// fastMalloc() with alignment parameter and no malloc overread
static void* fastMalloc_with_alignment(size_t size, size_t alignment)
{
#if _MSC_VER
    return _aligned_malloc(size, alignment);
#elif ((defined(__unix__) || defined(__APPLE__)) && !defined(__ANDROID__)) || (__ANDROID__ && __ANDROID_API__ >= 17)
    void* ptr = 0;
    if (posix_memalign(&ptr, alignment, size))
        ptr = 0;
    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(alignment, size);
#else
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + alignment);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, alignment);
    adata[-1] = udata;
    return adata;
#endif
}

int VkWeightStore::set_weights_in_host_disk(bool enable, bool prefer_host_memory)
{
    if (!enable && weight_budgets[0])
    {
        NCNN_LOGE("cannot disable the backing file while weight residency budgets are active");
        return -1;
    }
    if (enable && file.open() != 0)
    {
        NCNN_LOGE("disk weight storage unavailable, falling back to %s", prefer_host_memory ? "host memory" : "device memory");
        prefer_host_disk = false;
        return -1;
    }
    prefer_host_disk = enable;
    return 0;
}

int VkWeightAllocator::set_weight_file_directory(const char* path)
{
    if (!d->buffer_blocks.empty() || !d->dedicated_buffer_blocks.empty()
            || !d->image_memory_blocks.empty() || !d->dedicated_image_memory_blocks.empty()
            || !d->store.weights.empty())
    {
        NCNN_LOGE("weight file directory must be configured before allocating weights");
        return -1;
    }
    return d->store.file.set_directory(path);
}

int VkWeightAllocator::set_weights_in_host_disk(bool enable)
{
    return d->store.set_weights_in_host_disk(enable, d->prefer_host_memory);
}

int VkWeightStore::set_weight_budgets(size_t device_budget, size_t host_budget)
{
    for (size_t i = 0; i < weights.size(); i++)
        if (weights[i].memory)
            return -1;
    if (unified)
    {
        device_budget = device_budget && host_budget ? std::min(device_budget, host_budget) : std::max(device_budget, host_budget);
        host_budget = 0;
    }
    else if ((device_budget == 0) != (host_budget == 0))
    {
        NCNN_LOGE("disk weight residency requires both host and device budgets");
        return -1;
    }
    weight_budgets[0] = device_budget;
    weight_budgets[1] = host_budget;
    cache_sizes[0] = 0;
    cache_sizes[1] = 0;
    return 0;
}

int VkWeightAllocator::set_weight_budgets(size_t device_budget, size_t host_budget)
{
    if (!d->buffer_blocks.empty() || !d->dedicated_buffer_blocks.empty()
            || !d->image_memory_blocks.empty() || !d->dedicated_image_memory_blocks.empty())
    {
        NCNN_LOGE("weight residency budgets must be configured before allocating native weights");
        return -1;
    }
    return d->store.set_weight_budgets(device_budget, host_budget);
}

// Loading boundaries and cache limits are internal to Net and command execution.
void begin_weight_upload(VkWeightAllocator* allocator)
{
    VkWeightStore& store = VkWeightAllocatorPrivate::store_for(allocator);
    store.weight_group++;
    store.group_size = 0;
}

int get_weight_cache_limits(const VkWeightAllocator* allocator, size_t& device_budget, size_t& host_budget, size_t& device_size, size_t& host_size)
{
    const VkWeightStore& store = VkWeightAllocatorPrivate::store_for(allocator);
    device_budget = host_budget = device_size = host_size = 0;
    if (store.finalization_error || ((store.prefer_host_disk || !store.weights.empty()) && !store.finalized))
        return -1;
    if (!store.managed_weight_count)
        return 0;
    const size_t alignment = weight_buffer_alignment(allocator->vkdev);
    const size_t source_size = alignSize(store.max_source_size, alignment);
    const size_t external_size = alignSize(store.max_external_size, alignment);
    device_budget = store.weight_budgets[0] - store.resident_sizes[0];
    host_budget = store.weight_budgets[1] - store.resident_sizes[1];
    device_size = store.weight_budgets[0] ? store.cache_sizes[0] : (source_size > (size_t)-1 / 2 ? source_size : source_size * 2);
    if (!store.weight_budgets[0] && store.weight_group)
        device_size = std::max(device_size, store.max_group_size);
    host_size = store.weight_budgets[0] ? store.cache_sizes[1] : (external_size > (size_t)-1 / 2 ? external_size : external_size * 2);
    if (store.unified || !store.max_external_size)
        host_size = 0;
    return 1;
}

int get_weight_source_generation(const VkAllocator* allocator, uint64_t& generation)
{
    MutexLockGuard lock(weight_allocators_lock);
    generation = 0;
    for (size_t i = 0; i < weight_allocators.size(); i++)
    {
        const VkWeightAllocatorPrivate* p = weight_allocators[i];
        if (p->allocator != allocator)
            continue;
        generation = p->generation;
        const VkWeightStore& store = p->store;
        if (store.finalization_error || ((store.prefer_host_disk || !store.weights.empty()) && !store.finalized))
            return -1;
        return 1;
    }
    return 0;
}

#if NCNN_BENCHMARK
int discard_weight_file_cache(const VkWeightAllocator* allocator)
{
    return VkWeightAllocatorPrivate::store_for(allocator).file.discard_page_cache();
}
#endif

struct WeightSizeCompare
{
    explicit WeightSizeCompare(const std::vector<VkWeightRecord>& _weights)
        : weights(_weights)
    {
    }
    bool operator()(uint32_t a, uint32_t b) const
    {
        return weights[a].data_size != weights[b].data_size ? weights[a].data_size < weights[b].data_size : a < b;
    }
    const std::vector<VkWeightRecord>& weights;
};

static size_t weight_block_size(const std::vector<VkWeightRecord>& weights, const std::vector<uint32_t>& ids, size_t begin, size_t alignment, size_t limit)
{
    size_t size = 0;
    for (size_t i = begin; i < ids.size(); i++)
    {
        const size_t next = alignSize(weights[ids[i]].data_size, alignment);
        if (next > limit - size)
            break;
        size += next;
    }
    return size;
}

int VkWeightStore::weight_layout_size(const std::vector<uint32_t>& ids, uint64_t& total, bool concurrent) const
{
    const size_t alignment = weight_buffer_alignment(vkdev);
    const size_t limit = weight_buffer_size_limit(vkdev, concurrent);
    total = 0;
    size_t free_size = 0;
    for (size_t i = 0; i < ids.size(); i++)
    {
        const size_t size = alignSize(weights[ids[i]].data_size, alignment);
        if (size > free_size)
        {
            const size_t block_size = weight_block_size(weights, ids, i, alignment, std::min(limit, std::max(size, (size_t)8 * 1024 * 1024)));
            size_t allocation_size;
            if (!block_size || weight_buffer_memory_size(vkdev, block_size, allocation_size, concurrent) || allocation_size > (uint64_t)-1 - total)
                return -1;
            total += allocation_size;
            free_size = block_size;
        }
        free_size -= size;
    }
    return 0;
}

int VkWeightStore::place_resident_weights(const std::vector<uint32_t>& ids, int storage_type, size_t budget)
{
    const size_t alignment = weight_buffer_alignment(vkdev);
    int block = -1;
    size_t offset = 0;
    for (size_t i = 0; i < ids.size(); i++)
    {
        const uint32_t id = ids[i];
        const size_t size = alignSize(weights[id].data_size, alignment);
        const size_t available = budget - resident_sizes[storage_type];
        if (block < 0 || size > resident_blocks[block]->capacity - offset)
        {
            if (size > available)
                continue;
            size_t limit = std::min(available, std::max(size, (size_t)8 * 1024 * 1024));
            const bool concurrent = !unified && storage_type == 1;
            if (fit_weight_capacity(vkdev, available, true, concurrent, limit) || limit < size)
                continue;
            const size_t block_size = weight_block_size(weights, ids, i, alignment, limit);
            size_t allocation_size;
            VkBufferMemory* backing = 0;
            // Fill device weights before committing the host resident allocation.
            if (create_weight_buffer(vkdev, block_size, storage_type, unified, available, backing, allocation_size, storage_type != 1))
                return -100;
            resident_sizes[storage_type] += allocation_size;
            block = (int)resident_blocks.size();
            resident_blocks.push_back(backing);
            offset = 0;
        }
        resident_block_for_weight[id] = block;
        resident_offsets[id] = offset;
        weights[id].allocation_size = size;
        offset += size;
    }
    return 0;
}

int VkWeightStore::plan_weights()
{
    const size_t alignment = weight_buffer_alignment(vkdev);
    const size_t device_budget = weight_budgets[0];
    const size_t host_budget = weight_budgets[1];
    std::vector<uint32_t> ids;
    std::vector<size_t> group_sizes((size_t)weight_group + 1, 0);
    size_t max_size = 0;
    size_t max_group_size = 0;
    for (size_t i = 0; i < weights.size(); i++)
    {
        const VkWeightRecord& record = weights[i];
        if (!record.memory)
            continue;
        if (record.memory->storage_type != 2 || record.data_size > (size_t)-1 - alignment)
            return -1;
        const size_t size = alignSize(record.data_size, alignment);
        max_size = std::max(max_size, size);
        size_t& group = group_sizes[weight_groups[i]];
        group = size > (size_t)-1 - group ? (size_t)-1 : group + size;
        max_group_size = std::max(max_group_size, group);
        ids.push_back((uint32_t)i);
    }
#if NCNN_SIMPLESTL
    std::partial_sort(ids.begin(), ids.end(), ids.end(), WeightSizeCompare(weights));
#else
    std::sort(ids.begin(), ids.end(), WeightSizeCompare(weights));
#endif
    resident_block_for_weight.resize(weights.size(), -1);
    resident_offsets.resize(weights.size(), 0);
    uint64_t total;
    if (weight_layout_size(ids, total))
        return -1;
    const bool automatic_device_cache = !cache_sizes[0];
    size_t device_cache = cache_sizes[0];
    if (total <= device_budget)
        device_cache = 0;
    else if (automatic_device_cache)
    {
        // Reserve the working set first, then retain small weights permanently.
        device_cache = std::min(device_budget, max_group_size) / alignment * alignment;
    }
    size_t device_exchange_allocation = 0;
    if (fit_weight_capacity(vkdev, device_budget, automatic_device_cache, false, device_cache)
            || weight_buffer_memory_size(vkdev, device_cache, device_exchange_allocation)
            || (total > device_budget && device_cache < max_size))
    {
        NCNN_LOGE("weight device budget %llu cannot hold exchange %llu (largest weight %llu)", (unsigned long long)device_budget, (unsigned long long)device_cache, (unsigned long long)max_size);
        return -1;
    }
    if (place_resident_weights(ids, 0, device_budget - device_exchange_allocation))
        return -100;

    std::vector<uint32_t> remaining;
    for (size_t i = 0; i < group_sizes.size(); i++)
        group_sizes[i] = 0;
    max_size = 0;
    max_group_size = 0;
    for (size_t i = 0; i < ids.size(); i++)
    {
        if (resident_block_for_weight[ids[i]] >= 0)
            continue;
        remaining.push_back(ids[i]);
        const size_t size = alignSize(weights[ids[i]].data_size, alignment);
        max_size = std::max(max_size, size);
        size_t& group = group_sizes[weight_groups[ids[i]]];
        group = size > (size_t)-1 - group ? (size_t)-1 : group + size;
        max_group_size = std::max(max_group_size, group);
    }
    if (remaining.empty())
    {
        cache_sizes[0] = 0;
        cache_sizes[1] = 0;
        return 0;
    }
    if (automatic_device_cache)
    {
        const size_t doubled = max_size > (size_t)-1 / 2 ? (size_t)-1 : max_size * 2;
        device_cache = std::min(device_budget - resident_sizes[0], std::max(doubled, max_group_size)) / alignment * alignment;
        if (fit_weight_capacity(vkdev, device_budget - resident_sizes[0], true, false, device_cache) || device_cache < max_size)
            return -1;
    }
    if (!device_cache)
        return -1;
    cache_sizes[0] = device_cache;
    if (unified)
        return 0;

    if (weight_layout_size(remaining, total, true))
        return -1;
    const bool automatic_host_cache = !cache_sizes[1];
    size_t host_cache = cache_sizes[1];
    if (total <= host_budget)
        host_cache = 0;
    else if (!host_cache)
    {
        const size_t doubled = max_size > (size_t)-1 / 2 ? (size_t)-1 : max_size * 2;
        host_cache = std::min(doubled, std::max(alignment, host_budget / 4 / alignment * alignment));
    }
    size_t host_exchange_allocation = 0;
    if (fit_weight_capacity(vkdev, host_budget, automatic_host_cache, true, host_cache)
            || weight_buffer_memory_size(vkdev, host_cache, host_exchange_allocation, true) || host_exchange_allocation > host_budget)
    {
        NCNN_LOGE("weight host budget %llu cannot hold exchange %llu (allocation %llu)", (unsigned long long)host_budget, (unsigned long long)host_cache, (unsigned long long)host_exchange_allocation);
        return -1;
    }
    cache_sizes[1] = host_cache;
    if (place_resident_weights(remaining, 1, host_budget - host_exchange_allocation))
        return -100;
    bool has_file_weights = false;
    for (size_t i = 0; i < remaining.size(); i++)
        has_file_weights |= resident_block_for_weight[remaining[i]] < 0;
    if (!has_file_weights)
        cache_sizes[1] = 0;
    else if (!host_cache)
        return -1;
    return 0;
}

int VkWeightStore::prepare_residency(bool& prepared)
{
    prepared = false;
    if (finalization_error)
        return finalization_error;
    int ret = file.finalize();
    for (size_t i = 0; i < weights.size(); i++)
    {
        if (!weights[i].memory || weights[i].memory->storage_type != 2)
            continue;
        weights[i].memory->access_flags = VK_ACCESS_HOST_READ_BIT;
        weights[i].memory->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    }
    if (ret || finalized || !weight_budgets[0])
        return ret;
    if (!prefer_host_disk)
    {
        NCNN_LOGE("disk weight residency requires a prepared backing file");
        return -1;
    }
    prepared = true;
    return plan_weights();
}

int VkWeightStore::allocate_residents(int storage_type)
{
    for (size_t i = 0; i < resident_blocks.size(); i++)
    {
        VkBufferMemory* backing = resident_blocks[i];
        if (backing->storage_type == storage_type && !backing->memory && allocate_weight_buffer_memory(vkdev, backing))
            return -100;
    }
    for (size_t i = 0; i < weights.size(); i++)
    {
        const int block = resident_block_for_weight[i];
        if (block < 0)
            continue;
        const VkBufferMemory* backing = resident_blocks[block];
        if (backing->storage_type != storage_type)
            continue;
        VkWeightRecord& record = weights[i];
        // preserve the logical object and its mat references
        VkBufferMemory* memory = const_cast<VkBufferMemory*>(record.memory);
        memory->buffer = backing->buffer;
        memory->memory = backing->memory;
        memory->offset = resident_offsets[i];
        memory->capacity = record.allocation_size;
        memory->mapped_ptr = backing->mapped_ptr;
        memory->memory_type_index = backing->memory_type_index;
        memory->buffer_usage = backing->buffer_usage;
        memory->access_flags = backing->mapped_ptr ? VK_ACCESS_HOST_WRITE_BIT : VK_ACCESS_TRANSFER_WRITE_BIT;
        memory->stage_flags = backing->mapped_ptr ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT;
        memory->storage_type = storage_type;
        memory->external_ptr = storage_type == 1 ? (unsigned char*)backing->mapped_ptr + memory->offset : 0;
        memory->external_size = storage_type == 1 ? record.data_size : 0;
        if (storage_type == 0)
            managed_weight_count--;
    }
    return 0;
}

int VkWeightAllocator::finalize_weights(size_t device_cache_size, size_t host_cache_size)
{
    if (!d->store.finalized && !d->store.finalization_error)
    {
        d->store.cache_sizes[0] = device_cache_size;
        d->store.cache_sizes[1] = d->store.unified ? 0 : host_cache_size;
    }
    if (d->store.weights.empty() && !d->store.prefer_host_disk)
        return 0;
    bool prepared;
    int ret = d->store.prepare_residency(prepared);
    for (int storage_type = 0; prepared && !ret && storage_type <= 1; storage_type++)
    {
        ret = d->store.allocate_residents(storage_type);
        if (ret)
            break;
        std::vector<VkBufferMemory> uploads;
        for (size_t i = 0; i < d->store.weights.size(); i++)
        {
            const VkWeightRecord& record = d->store.weights[i];
            if (!record.memory || record.memory->storage_type != storage_type)
                continue;
            VkBufferMemory buffer = *record.memory;
            buffer.external_ptr = const_cast<void*>(record.source);
            buffer.external_size = record.data_size;
            uploads.push_back(buffer);
        }
        // loading device staging is released before committing host resident buffers
        ret = upload_weight_buffers(vkdev, uploads, d->store.weight_budgets[1]);
    }
    if (!ret)
    {
        // Every Extractor shares these model-owned buffers. Finish their state
        // once, after all uploads, so inference only reads their metadata too.
        for (size_t i = 0; i < d->store.weights.size(); i++)
        {
            VkBufferMemory* memory = const_cast<VkBufferMemory*>(d->store.weights[i].memory);
            if (memory && memory->storage_type == 0)
            {
                memory->access_flags = VK_ACCESS_SHADER_READ_BIT;
                memory->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            }
        }
        d->store.finalized = true;
    }
    d->store.finalization_error = ret;
    return ret;
}

VkBufferMemory* VkWeightAllocator::fastMalloc(size_t size)
{
    if (d->store.finalized || d->store.finalization_error)
    {
        NCNN_LOGE("cannot allocate weights after residency finalization");
        return 0;
    }
    if (d->store.weight_budgets[0] && !d->store.prefer_host_disk)
    {
        NCNN_LOGE("weight residency budgets require a prepared backing file");
        return 0;
    }
    if (d->store.weights.size() >= 0xffffffffu || size > (size_t)-1 - d->buffer_offset_alignment)
    {
        NCNN_LOGE("prepared weight allocation overflow");
        return 0;
    }

    VkWeightRecord record;
    record.source = 0;
    record.data_size = size;
    record.allocation_size = alignSize(size, d->buffer_offset_alignment);

    VkBufferMemory* ptr;
    if (d->store.prefer_host_disk)
    {
        void* external_ptr = 0;
        if (d->store.file.open() != 0 || d->store.file.reserve(record.data_size, d->buffer_offset_alignment, external_ptr) != 0)
            return 0;

        ptr = new VkWeightBufferMemory;
        ptr->external_ptr = external_ptr;
        record.source = external_ptr;
        ptr->buffer = VK_NULL_HANDLE;
        ptr->offset = 0;
        ptr->capacity = record.allocation_size;
        ptr->memory = VK_NULL_HANDLE;
        ptr->mapped_ptr = 0;
        ptr->memory_type_index = (uint32_t)-1;
        ptr->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        ptr->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
        ptr->storage_type = 2;
    }
    else
    {
        ptr = d->fastMalloc_resident(this, size);
        if (!ptr)
            return 0;
    }

    if (ptr->storage_type != 0)
    {
        if (ptr->storage_type == 1 && ptr->mapped_ptr)
            ptr->external_ptr = (unsigned char*)ptr->mapped_ptr + ptr->offset;
        ptr->external_size = record.data_size;
    }
    // native weights need no prepared directory or pointer lookup at release
    if (d->store.prefer_host_disk)
    {
        record.memory = ptr;
        d->store.weights.push_back(record);
        d->store.weight_groups.push_back(d->store.weight_group);
        d->store.max_external_size = std::max(d->store.max_external_size, size);
    }
    if (ptr->storage_type != 0)
    {
        d->store.managed_weight_count++;
        d->store.max_source_size = std::max(d->store.max_source_size, size);
        const size_t aligned_size = alignSize(size, weight_buffer_alignment(vkdev));
        d->store.group_size = aligned_size > (size_t)-1 - d->store.group_size ? (size_t)-1 : d->store.group_size + aligned_size;
        d->store.max_group_size = std::max(d->store.max_group_size, d->store.group_size);
    }
    static_cast<VkWeightBufferMemory*>(ptr)->allocation_generation = d->allocation_generation;
    update_weight_generation(d);
    return ptr;
}

VkBufferMemory* VkWeightAllocatorPrivate::fastMalloc_resident(VkWeightAllocator* allocator, size_t size)
{
    const VulkanDevice* vkdev = allocator->vkdev;

    //     NCNN_LOGE("VkWeightAllocator fastMalloc %lu", size);

    size_t aligned_size = alignSize(size, buffer_offset_alignment);

    const int buffer_block_count = buffer_blocks.size();

    // find first spare space in buffer_blocks
    for (int i = 0; i < buffer_block_count; i++)
    {
        size_t free_size = buffer_block_free_spaces[i];
        if (free_size >= aligned_size)
        {
            size_t block_offset = block_size - free_size;

            // return sub buffer
            VkBufferMemory* ptr = new VkWeightBufferMemory;

            ptr->buffer = buffer_blocks[i]->buffer;
            ptr->offset = block_offset;
            ptr->memory = buffer_blocks[i]->memory;
            ptr->capacity = aligned_size;
            ptr->mapped_ptr = buffer_blocks[i]->mapped_ptr;
            ptr->memory_type_index = buffer_blocks[i]->memory_type_index;
            ptr->storage_type = vkdev->is_device_local(ptr->memory_type_index) ? 0 : 1;
            ptr->buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            ptr->access_flags = 0;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

            buffer_block_free_spaces[i] -= aligned_size;

            return ptr;
        }
    }

    size_t new_block_size = std::max(block_size, aligned_size);

    // create new block
    VkBufferMemory* block = new VkBufferMemory;

    VkExternalMemoryHandleTypeFlags handle_types = 0;
#if !defined(_WIN32)
    if (prefer_host_memory && vkdev->info.support_VK_EXT_external_memory_host())
        handle_types = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
#endif
    // host weights may be initialized on compute and read by transfer in several contexts
    block->buffer = ncnn::create_buffer(vkdev, new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, handle_types, prefer_host_memory);
    if (!block->buffer)
    {
        delete block;
        return 0;
    }
    block->offset = 0;

    if (vkdev->info.support_VK_KHR_get_memory_requirements2() && vkdev->info.support_VK_KHR_dedicated_allocation() && !prefer_host_memory)
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
            // setup memory type and alignment
            if (allocator->buffer_memory_type_index == (uint32_t)-1)
            {
                if (vkdev->info.type() == 1)
                {
                    // integrated gpu, prefer unified memory
                    allocator->buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);

                    // on amd integrated gpu, there is a faster and larger device-only heap
                    uint32_t device_local_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                    const VkPhysicalDeviceMemoryProperties& memory_properties = vkdev->info.physicalDeviceMemoryProperties();
                    uint32_t buffer_heap_index = memory_properties.memoryTypes[allocator->buffer_memory_type_index].heapIndex;
                    uint32_t device_local_heap_index = memory_properties.memoryTypes[device_local_memory_type_index].heapIndex;
                    if (device_local_heap_index < buffer_heap_index && memory_properties.memoryHeaps[device_local_heap_index].size > memory_properties.memoryHeaps[buffer_heap_index].size)
                    {
                        allocator->buffer_memory_type_index = device_local_memory_type_index;
                    }
                }
                else
                {
                    // discrete gpu, device local
                    if (vkdev->info.resizable_bar_enabled())
                    {
                        allocator->buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                    }
                    else
                    {
                        allocator->buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                    }
                }

                allocator->mappable = vkdev->is_mappable(allocator->buffer_memory_type_index);
                allocator->coherent = vkdev->is_coherent(allocator->buffer_memory_type_index);
                allocator->cached = vkdev->is_cached(allocator->buffer_memory_type_index);
            }

            block->memory = ncnn::allocate_dedicated_memory(vkdev, memoryRequirements2.memoryRequirements.size, allocator->buffer_memory_type_index, 0, block->buffer);
            if (!block->memory)
            {
                vkDestroyBuffer(vkdev->vkdevice(), block->buffer, 0);
                delete block;
                return 0;
            }

            // ignore memoryRequirements2.memoryRequirements.alignment as we always bind at zero offset
            vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0);

            block->mapped_ptr = 0;
            if (allocator->mappable)
            {
                vkMapMemory(vkdev->vkdevice(), block->memory, 0, new_block_size, 0, &block->mapped_ptr);
            }

            block->memory_type_index = allocator->buffer_memory_type_index;

            block->storage_type = vkdev->is_device_local(block->memory_type_index) ? 0 : 1;

            block->buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

            dedicated_buffer_blocks.push_back(block);

            // return sub buffer
            VkBufferMemory* ptr = new VkWeightBufferMemory;

            ptr->buffer = block->buffer;
            ptr->offset = 0;
            ptr->memory = block->memory;
            ptr->capacity = new_block_size;
            ptr->mapped_ptr = block->mapped_ptr;
            ptr->memory_type_index = block->memory_type_index;
            ptr->storage_type = vkdev->is_device_local(ptr->memory_type_index) ? 0 : 1;
            ptr->buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            ptr->access_flags = 0;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

            return ptr;
        }
    }

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), block->buffer, &memoryRequirements);

    if (prefer_host_memory)
    {
#if !defined(_WIN32)
        if (vkdev->info.support_VK_EXT_external_memory_host())
        {
            void* host_ptr = fastMalloc_with_alignment(memoryRequirements.size, buffer_offset_alignment);

            if (host_ptr)
            {
                VkMemoryHostPointerPropertiesEXT pointerProperties;
                pointerProperties.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
                pointerProperties.pNext = 0;
                VkResult ret = vkdev->vkGetMemoryHostPointerPropertiesEXT(vkdev->vkdevice(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host_ptr, &pointerProperties);
                if (ret != VK_SUCCESS)
                {
                    NCNN_LOGE("vkGetMemoryHostPointerPropertiesEXT failed %d", ret);
                    ncnn::fastFree(host_ptr);
                    vkDestroyBuffer(vkdev->vkdevice(), block->buffer, 0);
                    delete block;
                    return 0;
                }

                // setup memory type and alignment
                if (allocator->buffer_memory_type_index == (uint32_t)-1)
                {
                    allocator->buffer_memory_type_index = vkdev->find_memory_index(pointerProperties.memoryTypeBits & memoryRequirements.memoryTypeBits, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

                    allocator->mappable = vkdev->is_mappable(allocator->buffer_memory_type_index);
                    allocator->coherent = vkdev->is_coherent(allocator->buffer_memory_type_index);
                    allocator->cached = vkdev->is_cached(allocator->buffer_memory_type_index);
                }

                block->memory = ncnn::allocate_import_host_memory(vkdev, memoryRequirements.size, allocator->buffer_memory_type_index, host_ptr);
                if (!block->memory)
                {
                    // oom
                    ncnn::fastFree(host_ptr);
                    prefer_host_memory = false;
                }
                else
                {
                    host_ptrs.push_back(host_ptr);
                }
            }
            else
            {
                // oom
                prefer_host_memory = false;
            }
        }
        else
#endif // !defined(_WIN32)
        {
            // setup memory type and alignment
            if (allocator->buffer_memory_type_index == (uint32_t)-1)
            {
                allocator->buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

                allocator->mappable = vkdev->is_mappable(allocator->buffer_memory_type_index);
                allocator->coherent = vkdev->is_coherent(allocator->buffer_memory_type_index);
                allocator->cached = vkdev->is_cached(allocator->buffer_memory_type_index);
            }

            block->memory = ncnn::allocate_memory(vkdev, memoryRequirements.size, allocator->buffer_memory_type_index);
            if (!block->memory)
            {
                // oom
                prefer_host_memory = false;
            }
        }

        if (!prefer_host_memory)
        {
            NCNN_LOGE("weight allocator fallback to device memory");
            allocator->buffer_memory_type_index = (uint32_t)-1;
            allocator->image_memory_type_index = (uint32_t)-1;
        }
    }
    if (!prefer_host_memory)
    {
        // setup memory type and alignment
        if (allocator->buffer_memory_type_index == (uint32_t)-1)
        {
            if (vkdev->info.type() == 1)
            {
                // integrated gpu, prefer unified memory
                allocator->buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);

                // on amd integrated gpu, there is a faster and larger device-only heap
                uint32_t device_local_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                const VkPhysicalDeviceMemoryProperties& memory_properties = vkdev->info.physicalDeviceMemoryProperties();
                uint32_t buffer_heap_index = memory_properties.memoryTypes[allocator->buffer_memory_type_index].heapIndex;
                uint32_t device_local_heap_index = memory_properties.memoryTypes[device_local_memory_type_index].heapIndex;
                if (device_local_heap_index < buffer_heap_index && memory_properties.memoryHeaps[device_local_heap_index].size > memory_properties.memoryHeaps[buffer_heap_index].size)
                {
                    allocator->buffer_memory_type_index = device_local_memory_type_index;
                }
            }
            else
            {
                // discrete gpu, device local
                if (vkdev->info.resizable_bar_enabled())
                {
                    allocator->buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                }
                else
                {
                    allocator->buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                }
            }

            allocator->mappable = vkdev->is_mappable(allocator->buffer_memory_type_index);
            allocator->coherent = vkdev->is_coherent(allocator->buffer_memory_type_index);
            allocator->cached = vkdev->is_cached(allocator->buffer_memory_type_index);
        }

        block->memory = ncnn::allocate_memory(vkdev, memoryRequirements.size, allocator->buffer_memory_type_index);
    }
    if (!block->memory)
    {
        vkDestroyBuffer(vkdev->vkdevice(), block->buffer, 0);
        delete block;
        return 0;
    }

    // ignore memoryRequirements.alignment as we always bind at zero offset
    vkBindBufferMemory(vkdev->vkdevice(), block->buffer, block->memory, 0);

    //     NCNN_LOGE("VkWeightAllocator M %p", block->buffer);

    block->mapped_ptr = 0;
    if (allocator->mappable)
    {
        vkMapMemory(vkdev->vkdevice(), block->memory, 0, new_block_size, 0, &block->mapped_ptr);
    }

    block->memory_type_index = allocator->buffer_memory_type_index;
    block->storage_type = vkdev->is_device_local(block->memory_type_index) ? 0 : 1;
    block->buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    buffer_blocks.push_back(block);

    buffer_block_free_spaces.push_back(new_block_size - aligned_size);

    // return sub buffer
    VkBufferMemory* ptr = new VkWeightBufferMemory;

    ptr->buffer = block->buffer;
    ptr->offset = 0;
    ptr->memory = block->memory;
    ptr->capacity = aligned_size;
    ptr->mapped_ptr = block->mapped_ptr;
    ptr->memory_type_index = block->memory_type_index;
    ptr->storage_type = vkdev->is_device_local(ptr->memory_type_index) ? 0 : 1;
    ptr->buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    return ptr;
}

void VkWeightAllocator::fastFree(VkBufferMemory* ptr)
{
    VkWeightBufferMemory* memory = static_cast<VkWeightBufferMemory*>(ptr);
    if (memory->allocation_generation != d->allocation_generation)
    {
        delete memory;
        return;
    }
    update_weight_generation(d);
    //     NCNN_LOGE("VkWeightAllocator F %p", ptr->buffer);

    if (ptr->storage_type == 2 || !d->store.resident_blocks.empty())
    {
        for (size_t i = 0; i < d->store.weights.size(); i++)
        {
            if (d->store.weights[i].memory != ptr)
                continue;
            d->store.weights[i].memory = 0;
            break;
        }
    }
    if (ptr->external_size && ptr->storage_type != 0)
        d->store.managed_weight_count--;
    delete memory;
}

int VkWeightAllocator::flush(VkBufferMemory* ptr)
{
    return flush_weight_buffer(vkdev, ptr, 0, ptr->capacity);
}

int VkWeightAllocator::invalidate(VkBufferMemory* ptr)
{
    if (!ptr->mapped_ptr || ptr->memory_type_index >= vkdev->info.physicalDeviceMemoryProperties().memoryTypeCount)
        return -1;
    if (vkdev->is_coherent(ptr->memory_type_index))
        return 0;
    VkMappedMemoryRange range = {};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = ptr->memory;
    const size_t atom = vkdev->info.non_coherent_atom_size();
    range.offset = round_down(ptr->offset, atom);
    range.size = round_up(ptr->offset + ptr->capacity, atom) - range.offset;
    VkResult ret = vkInvalidateMappedMemoryRanges(vkdev->vkdevice(), 1, &range);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkInvalidateMappedMemoryRanges failed %d", ret);
        return -1;
    }
    return 0;
}

VkImageMemory* VkWeightAllocator::fastMalloc(int w, int h, int c, size_t elemsize, int elempack)
{
    if (d->store.weight_budgets[0])
    {
        NCNN_LOGE("image weights are not supported with a disk residency budget");
        return 0;
    }
    if (elempack != 1 && elempack != 4 && elempack != 8 && elempack != 16 && elempack != 32 && elempack != 64)
    {
        NCNN_LOGE("elempack must be 1 4 8 16 32 64");
        return 0;
    }

    // resolve format
    VkFormat format = VK_FORMAT_UNDEFINED;

    if (elemsize / elempack == 4)
    {
        // fp32
        if (elempack == 1) format = VK_FORMAT_R32_SFLOAT;
        if (elempack == 4) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 8) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 16) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 32) format = VK_FORMAT_R32G32B32A32_SFLOAT;
        if (elempack == 64) format = VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    if (elemsize / elempack == 2)
    {
        // fp16
        if (elempack == 1) format = VK_FORMAT_R16_SFLOAT;
        if (elempack == 4) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 8) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 16) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 32) format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (elempack == 64) format = VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    if (elemsize / elempack == 1)
    {
        // int8
        if (elempack == 1) format = VK_FORMAT_R8_SINT;
        if (elempack == 4) format = VK_FORMAT_R8G8B8A8_SINT;
        if (elempack == 8) format = VK_FORMAT_R8G8B8A8_SINT;
        if (elempack == 16) format = VK_FORMAT_R8G8B8A8_SINT;
        if (elempack == 32) format = VK_FORMAT_R8G8B8A8_SINT;
        if (elempack == 64) format = VK_FORMAT_R8G8B8A8_SINT;
    }

    // resolve image width height depth
    int width = w;
    int height = h;
    int depth = c;

    // large elempack spills on image w
    if (elempack == 8) width *= 2;
    if (elempack == 16) width *= 4;
    if (elempack == 32) width *= 8;
    if (elempack == 64) width *= 16;

    if (width > (int)vkdev->info.max_image_dimension_3d() || height > (int)vkdev->info.max_image_dimension_3d() || depth > (int)vkdev->info.max_image_dimension_3d())
    {
        NCNN_LOGE("image dimension too large %d %d %d > %d", width, height, depth, (int)vkdev->info.max_image_dimension_3d());
        return 0;
    }

    VkImageMemory* ptr = new VkImageMemory;

    ptr->image = create_image(width, height, depth, format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    ptr->width = width;
    ptr->height = height;
    ptr->depth = depth;
    ptr->format = format;

    if (vkdev->info.support_VK_KHR_get_memory_requirements2() && vkdev->info.support_VK_KHR_dedicated_allocation() && !d->prefer_host_memory)
    {
        VkImageMemoryRequirementsInfo2KHR imageMemoryRequirementsInfo2;
        imageMemoryRequirementsInfo2.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR;
        imageMemoryRequirementsInfo2.pNext = 0;
        imageMemoryRequirementsInfo2.image = ptr->image;

        VkMemoryRequirements2KHR memoryRequirements2;
        memoryRequirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR;
        memoryRequirements2.pNext = 0;

        VkMemoryDedicatedRequirementsKHR memoryDedicatedRequirements;
        memoryDedicatedRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR;
        memoryDedicatedRequirements.pNext = 0;
        memoryRequirements2.pNext = &memoryDedicatedRequirements;

        vkdev->vkGetImageMemoryRequirements2KHR(vkdev->vkdevice(), &imageMemoryRequirementsInfo2, &memoryRequirements2);

        bool dedicatedAllocation = memoryDedicatedRequirements.requiresDedicatedAllocation || memoryDedicatedRequirements.prefersDedicatedAllocation;

        if (dedicatedAllocation)
        {
            // setup memory type and alignment
            if (image_memory_type_index == (uint32_t)-1)
            {
                if (vkdev->info.type() == 1)
                {
                    // integrated gpu, prefer unified memory
                    image_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);

                    // on amd integrated gpu, there is a faster and larger device-only heap
                    uint32_t device_local_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                    const VkPhysicalDeviceMemoryProperties& memory_properties = vkdev->info.physicalDeviceMemoryProperties();
                    uint32_t buffer_heap_index = memory_properties.memoryTypes[image_memory_type_index].heapIndex;
                    uint32_t device_local_heap_index = memory_properties.memoryTypes[device_local_memory_type_index].heapIndex;
                    if (device_local_heap_index < buffer_heap_index && memory_properties.memoryHeaps[device_local_heap_index].size > memory_properties.memoryHeaps[buffer_heap_index].size)
                    {
                        image_memory_type_index = device_local_memory_type_index;
                    }
                }
                else
                {
                    // discrete gpu, device local
                    if (vkdev->info.resizable_bar_enabled())
                    {
                        image_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                    }
                    else
                    {
                        image_memory_type_index = vkdev->find_memory_index(memoryRequirements2.memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                    }
                }

                mappable = vkdev->is_mappable(image_memory_type_index);
                coherent = vkdev->is_coherent(image_memory_type_index);
                cached = vkdev->is_cached(image_memory_type_index);
            }

            // bind memory
            ptr->memory = allocate_dedicated_memory(memoryRequirements2.memoryRequirements.size, image_memory_type_index, ptr->image, 0);
            if (!ptr->memory)
            {
                vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);
                delete ptr;
                return 0;
            }
            ptr->bind_offset = 0;
            ptr->bind_capacity = memoryRequirements2.memoryRequirements.size;

            // ignore memoryRequirements2.memoryRequirements.alignment as we always bind at zero offset
            vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

            // do not allow host access to optimal tiling image
            ptr->mapped_ptr = 0;
            ptr->memory_type_index = image_memory_type_index;

            ptr->imageview = create_imageview(ptr->image, format);

            ptr->access_flags = 0;
            ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            ptr->command_refcount = 0;

            d->dedicated_image_memory_blocks.push_back(ptr->memory);

            return ptr;
        }
    }

    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(vkdev->vkdevice(), ptr->image, &memoryRequirements);

    const size_t size = memoryRequirements.size;
    const size_t alignment = std::max((size_t)memoryRequirements.alignment, d->bind_memory_offset_alignment);

    size_t aligned_size = alignSize(size, alignment);

    const int image_memory_block_count = d->image_memory_blocks.size();

    // find first spare space in buffer_blocks
    for (int i = 0; i < image_memory_block_count; i++)
    {
        // we cannot use image_memory_block_free_spaces[i] directly for base offset alignment
        size_t bind_base_offset = d->block_size - d->image_memory_block_free_spaces[i];
        size_t bind_offset = alignSize(bind_base_offset, alignment);
        if (d->image_memory_block_free_spaces[i] >= aligned_size + (bind_offset - bind_base_offset))
        {
            // bind at memory offset
            ptr->memory = d->image_memory_blocks[i];
            ptr->bind_offset = bind_offset;
            ptr->bind_capacity = aligned_size;

            vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

            // do not allow host access to optimal tiling image
            ptr->mapped_ptr = 0;
            ptr->memory_type_index = image_memory_type_index;

            ptr->imageview = create_imageview(ptr->image, format);

            ptr->access_flags = 0;
            ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
            ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            ptr->command_refcount = 0;

            if (bind_base_offset != bind_offset)
            {
                // NOTE there is small offset inside bind_base_offset and bind_offset
                // adjust ptr->bind_offset and ptr->bind_capacity after vkBindImageMemory
                // so that memory management could be easier
                aligned_size += (bind_offset - bind_base_offset);

                ptr->bind_offset = bind_base_offset;
                ptr->bind_capacity = aligned_size;
            }

            d->image_memory_block_free_spaces[i] -= aligned_size;

            return ptr;
        }
    }

    // create new block
    size_t new_block_size = std::max(d->block_size, aligned_size);

    if (d->prefer_host_memory)
    {
#if !defined(_WIN32)
        if (vkdev->info.support_VK_EXT_external_memory_host())
        {
            void* host_ptr = fastMalloc_with_alignment(new_block_size, d->buffer_offset_alignment);

            if (host_ptr)
            {
                VkMemoryHostPointerPropertiesEXT pointerProperties;
                pointerProperties.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
                pointerProperties.pNext = 0;
                VkResult ret = vkdev->vkGetMemoryHostPointerPropertiesEXT(vkdev->vkdevice(), VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host_ptr, &pointerProperties);
                if (ret != VK_SUCCESS)
                {
                    NCNN_LOGE("vkGetMemoryHostPointerPropertiesEXT failed %d", ret);
                    ncnn::fastFree(host_ptr);
                    vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);
                    delete ptr;
                    return 0;
                }

                // setup memory type and alignment
                if (image_memory_type_index == (uint32_t)-1)
                {
                    image_memory_type_index = vkdev->find_memory_index(pointerProperties.memoryTypeBits, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

                    mappable = vkdev->is_mappable(image_memory_type_index);
                    coherent = vkdev->is_coherent(image_memory_type_index);
                    cached = vkdev->is_cached(image_memory_type_index);
                }

                ptr->memory = allocate_import_host_memory(new_block_size, image_memory_type_index, host_ptr);
                if (!ptr->memory)
                {
                    // oom
                    ncnn::fastFree(host_ptr);
                    d->prefer_host_memory = false;
                }
                else
                {
                    d->host_ptrs.push_back(host_ptr);
                }
            }
            else
            {
                // oom
                d->prefer_host_memory = false;
            }
        }
        else
#endif // !defined(_WIN32)
        {
            // setup memory type and alignment
            if (image_memory_type_index == (uint32_t)-1)
            {
                image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

                mappable = vkdev->is_mappable(image_memory_type_index);
                coherent = vkdev->is_coherent(image_memory_type_index);
                cached = vkdev->is_cached(image_memory_type_index);
            }

            // bind at memory offset
            ptr->memory = allocate_memory(new_block_size, image_memory_type_index);
            if (!ptr->memory)
            {
                // oom
                d->prefer_host_memory = false;
            }
        }

        if (!d->prefer_host_memory)
        {
            NCNN_LOGE("weight allocator fallback to device memory");
            buffer_memory_type_index = (uint32_t)-1;
            image_memory_type_index = (uint32_t)-1;
        }
    }
    if (!d->prefer_host_memory)
    {
        // setup memory type and alignment
        if (image_memory_type_index == (uint32_t)-1)
        {
            if (vkdev->info.type() == 1)
            {
                // integrated gpu, prefer unified memory
                image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0);

                // on amd integrated gpu, there is a faster and larger device-only heap
                uint32_t device_local_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                const VkPhysicalDeviceMemoryProperties& memory_properties = vkdev->info.physicalDeviceMemoryProperties();
                uint32_t buffer_heap_index = memory_properties.memoryTypes[image_memory_type_index].heapIndex;
                uint32_t device_local_heap_index = memory_properties.memoryTypes[device_local_memory_type_index].heapIndex;
                if (device_local_heap_index < buffer_heap_index && memory_properties.memoryHeaps[device_local_heap_index].size > memory_properties.memoryHeaps[buffer_heap_index].size)
                {
                    image_memory_type_index = device_local_memory_type_index;
                }
            }
            else
            {
                // discrete gpu, device local
                if (vkdev->info.resizable_bar_enabled())
                {
                    image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
                }
                else
                {
                    image_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                }
            }

            mappable = vkdev->is_mappable(image_memory_type_index);
            coherent = vkdev->is_coherent(image_memory_type_index);
            cached = vkdev->is_cached(image_memory_type_index);
        }

        // bind at memory offset
        ptr->memory = allocate_memory(new_block_size, image_memory_type_index);
    }
    if (!ptr->memory)
    {
        vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);
        delete ptr;
        return 0;
    }
    ptr->bind_offset = 0;
    ptr->bind_capacity = aligned_size;

    // ignore memoryRequirements2.memoryRequirements.alignment as we always bind at zero offset
    vkBindImageMemory(vkdev->vkdevice(), ptr->image, ptr->memory, ptr->bind_offset);

    // do not allow host access to optimal tiling image
    ptr->mapped_ptr = 0;
    ptr->memory_type_index = image_memory_type_index;

    ptr->imageview = create_imageview(ptr->image, format);

    ptr->access_flags = 0;
    ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    ptr->command_refcount = 0;

    d->image_memory_blocks.push_back(ptr->memory);
    d->image_memory_block_free_spaces.push_back(new_block_size - aligned_size);

    return ptr;
}

void VkWeightAllocator::fastFree(VkImageMemory* ptr)
{
    //     NCNN_LOGE("VkWeightAllocator F %p", ptr->memory);

    if (!ptr->command_refcount)
    {
        vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
        vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);

        delete ptr;
    }
}

class VkStagingAllocatorPrivate
{
public:
    unsigned int size_compare_ratio; // 0~256
    std::list<VkBufferMemory*> buffer_budgets;
};

VkStagingAllocator::VkStagingAllocator(const VulkanDevice* _vkdev)
    : VkAllocator(_vkdev), d(new VkStagingAllocatorPrivate)
{
    mappable = true;
    coherent = true;

    d->size_compare_ratio = 192; // 0.75f * 256
}

VkStagingAllocator::~VkStagingAllocator()
{
    clear();

    delete d;
}

VkStagingAllocator::VkStagingAllocator(const VkStagingAllocator&)
    : VkAllocator(0), d(0)
{
}

VkStagingAllocator& VkStagingAllocator::operator=(const VkStagingAllocator&)
{
    return *this;
}

void VkStagingAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        NCNN_LOGE("invalid size compare ratio %f", scr);
        return;
    }

    d->size_compare_ratio = (unsigned int)(scr * 256);
}

void VkStagingAllocator::clear()
{
    //     NCNN_LOGE("VkStagingAllocator %lu", buffer_budgets.size());

    for (std::list<VkBufferMemory*>::iterator it = d->buffer_budgets.begin(); it != d->buffer_budgets.end(); it++)
    {
        VkBufferMemory* ptr = *it;

        //         NCNN_LOGE("VkStagingAllocator F %p", ptr->buffer);

        vkUnmapMemory(vkdev->vkdevice(), ptr->memory);
        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

        delete ptr;
    }
    d->buffer_budgets.clear();
}

VkBufferMemory* VkStagingAllocator::fastMalloc(size_t size)
{
    // find free budget
    std::list<VkBufferMemory*>::iterator it = d->buffer_budgets.begin();
    for (; it != d->buffer_budgets.end(); it++)
    {
        VkBufferMemory* ptr = *it;

        size_t capacity = ptr->capacity;

        // size_compare_ratio ~ 100%
        if (capacity >= size && ((capacity * d->size_compare_ratio) >> 8) <= size)
        {
            d->buffer_budgets.erase(it);

            //             NCNN_LOGE("VkStagingAllocator M %p %lu reused %lu", ptr->buffer, size, capacity);

            return ptr;
        }
    }

    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!ptr->buffer)
    {
        delete ptr;
        return 0;
    }
    ptr->offset = 0;

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), ptr->buffer, &memoryRequirements);

    // setup memory type
    if (buffer_memory_type_index == (uint32_t)-1)
    {
        buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    if (buffer_memory_type_index == (uint32_t)-1)
    {
        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        delete ptr;
        return 0;
    }
    ptr->memory = allocate_memory(memoryRequirements.size, buffer_memory_type_index);
    if (!ptr->memory)
    {
        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        delete ptr;
        return 0;
    }

    // ignore memoryRequirements.alignment as we always bind at zero offset
    ptr->mapped_ptr = 0;
    VkResult ret = vkBindBufferMemory(vkdev->vkdevice(), ptr->buffer, ptr->memory, 0);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkBindBufferMemory failed %d", ret);
        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);
        delete ptr;
        return 0;
    }
    ret = vkMapMemory(vkdev->vkdevice(), ptr->memory, 0, size, 0, &ptr->mapped_ptr);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkMapMemory failed %d", ret);
        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);
        delete ptr;
        return 0;
    }
    ptr->capacity = size;

    ptr->memory_type_index = buffer_memory_type_index;

    ptr->storage_type = vkdev->is_device_local(ptr->memory_type_index) ? 0 : 1;

    ptr->buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    //     NCNN_LOGE("VkStagingAllocator M %p %lu", ptr->buffer, size);

    return ptr;
}

void VkStagingAllocator::fastFree(VkBufferMemory* ptr)
{
    //     NCNN_LOGE("VkStagingAllocator F %p", ptr->buffer);

    // return to buffer_budgets
    d->buffer_budgets.push_back(ptr);
}

VkImageMemory* VkStagingAllocator::fastMalloc(int w, int h, int c, size_t elemsize, int /* elempack */)
{
    // staging image is mainly used for storing small piece of dynamic parameters
    // we allocate host memory as a fake image, it's simple and good

    const size_t size = w * h * c * elemsize;

    VkImageMemory* ptr = new VkImageMemory;

    ptr->image = 0;
    ptr->width = w;
    ptr->height = h;
    ptr->depth = c;
    ptr->format = VK_FORMAT_UNDEFINED;
    ptr->memory = 0;
    ptr->bind_offset = 0;
    ptr->bind_capacity = size;

    ptr->mapped_ptr = malloc(size);
    ptr->memory_type_index = (uint32_t)-1;

    ptr->imageview = 0;

    ptr->access_flags = 0;
    ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ptr->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
    ptr->command_refcount = 0;

    //     NCNN_LOGE("VkStagingAllocator M %p %d %d %d %d %d", ptr->image, dims, width, height, depth, format);

    return ptr;
}

void VkStagingAllocator::fastFree(VkImageMemory* ptr)
{
    //     NCNN_LOGE("VkStagingAllocator F %p", ptr->image);

    free(ptr->mapped_ptr);

    delete ptr;
}

class VkWeightStagingAllocatorPrivate
{
public:
};

VkWeightStagingAllocator::VkWeightStagingAllocator(const VulkanDevice* _vkdev)
    : VkAllocator(_vkdev), d(new VkWeightStagingAllocatorPrivate)
{
    mappable = true;
    coherent = true;
}

VkWeightStagingAllocator::~VkWeightStagingAllocator()
{
    delete d;
}

VkWeightStagingAllocator::VkWeightStagingAllocator(const VkWeightStagingAllocator&)
    : VkAllocator(0), d(0)
{
}

VkWeightStagingAllocator& VkWeightStagingAllocator::operator=(const VkWeightStagingAllocator&)
{
    return *this;
}

VkBufferMemory* VkWeightStagingAllocator::fastMalloc(size_t size)
{
    VkBufferMemory* ptr = new VkBufferMemory;

    ptr->buffer = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    ptr->offset = 0;

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(vkdev->vkdevice(), ptr->buffer, &memoryRequirements);

    // setup memory type
    if (buffer_memory_type_index == (uint32_t)-1)
    {
        buffer_memory_type_index = vkdev->find_memory_index(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    ptr->memory = allocate_memory(memoryRequirements.size, buffer_memory_type_index);
    if (!ptr->memory)
    {
        vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
        delete ptr;
        return 0;
    }

    // ignore memoryRequirements.alignment as we always bind at zero offset
    vkBindBufferMemory(vkdev->vkdevice(), ptr->buffer, ptr->memory, 0);

    ptr->capacity = size;

    vkMapMemory(vkdev->vkdevice(), ptr->memory, 0, size, 0, &ptr->mapped_ptr);

    ptr->memory_type_index = buffer_memory_type_index;

    ptr->storage_type = vkdev->is_device_local(ptr->memory_type_index) ? 0 : 1;

    ptr->buffer_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    ptr->access_flags = 0;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    //     NCNN_LOGE("VkWeightStagingAllocator M %p %lu", ptr->buffer, size);

    return ptr;
}

void VkWeightStagingAllocator::fastFree(VkBufferMemory* ptr)
{
    //     NCNN_LOGE("VkWeightStagingAllocator F %p", ptr->buffer);

    vkUnmapMemory(vkdev->vkdevice(), ptr->memory);
    vkDestroyBuffer(vkdev->vkdevice(), ptr->buffer, 0);
    vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

    delete ptr;
}

VkImageMemory* VkWeightStagingAllocator::fastMalloc(int /*w*/, int /*h*/, int /*c*/, size_t /*elemsize*/, int /*elempack*/)
{
    return 0;
}

void VkWeightStagingAllocator::fastFree(VkImageMemory* /*ptr*/)
{
}

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
VkAndroidHardwareBufferImageAllocator::VkAndroidHardwareBufferImageAllocator(const VulkanDevice* _vkdev, AHardwareBuffer* _hb)
    : VkAllocator(_vkdev), hb(_hb)
{
    samplerYcbcrConversion = 0;

    init();
}

VkAndroidHardwareBufferImageAllocator::~VkAndroidHardwareBufferImageAllocator()
{
    if (samplerYcbcrConversion)
    {
        vkdev->vkDestroySamplerYcbcrConversionKHR(vkdev->vkdevice(), samplerYcbcrConversion, 0);
        samplerYcbcrConversion = 0;
    }
}

VkAndroidHardwareBufferImageAllocator::VkAndroidHardwareBufferImageAllocator(const VkAndroidHardwareBufferImageAllocator&)
    : VkAllocator(0)
{
}

VkAndroidHardwareBufferImageAllocator& VkAndroidHardwareBufferImageAllocator::operator=(const VkAndroidHardwareBufferImageAllocator&)
{
    return *this;
}

VkBufferMemory* VkAndroidHardwareBufferImageAllocator::fastMalloc(size_t /*size*/)
{
    return 0;
}

void VkAndroidHardwareBufferImageAllocator::fastFree(VkBufferMemory* /*ptr*/)
{
}

VkImageMemory* VkAndroidHardwareBufferImageAllocator::fastMalloc(int /*w*/, int /*h*/, int /*c*/, size_t /*elemsize*/, int /*elempack*/)
{
    VkResult ret;

    VkExternalFormatANDROID externalFormat;
    externalFormat.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID;
    externalFormat.pNext = 0;
    externalFormat.externalFormat = bufferFormatProperties.externalFormat;

    VkExternalMemoryImageCreateInfo externalMemoryImageCreateInfo;
    externalMemoryImageCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
    externalMemoryImageCreateInfo.pNext = &externalFormat,
    externalMemoryImageCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

    VkImageCreateInfo imageCreateInfo;
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
    imageCreateInfo.pNext = &externalMemoryImageCreateInfo;
    imageCreateInfo.flags = 0;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = VK_FORMAT_UNDEFINED;
    imageCreateInfo.extent.width = bufferDesc.width;
    imageCreateInfo.extent.height = bufferDesc.height;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.queueFamilyIndexCount = 0;
    imageCreateInfo.pQueueFamilyIndices = 0;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = 0;
    ret = vkCreateImage(vkdev->vkdevice(), &imageCreateInfo, 0, &image);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateImage failed %d", ret);
        return 0;
    }

    // setup memory type
    if (image_memory_type_index == (uint32_t)-1)
    {
        image_memory_type_index = vkdev->find_memory_index(bufferProperties.memoryTypeBits, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }

    VkImportAndroidHardwareBufferInfoANDROID importAndroidHardwareBufferInfo;
    importAndroidHardwareBufferInfo.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
    importAndroidHardwareBufferInfo.pNext = 0;
    importAndroidHardwareBufferInfo.buffer = hb;

    VkMemoryDedicatedAllocateInfo memoryDedicatedAllocateInfo;
    memoryDedicatedAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    memoryDedicatedAllocateInfo.pNext = &importAndroidHardwareBufferInfo;
    memoryDedicatedAllocateInfo.image = image;
    memoryDedicatedAllocateInfo.buffer = VK_NULL_HANDLE;

    VkMemoryAllocateInfo memoryAllocateInfo;
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = &memoryDedicatedAllocateInfo;
    memoryAllocateInfo.allocationSize = bufferProperties.allocationSize;
    memoryAllocateInfo.memoryTypeIndex = image_memory_type_index;

    VkDeviceMemory memory = 0;
    ret = vkAllocateMemory(vkdev->vkdevice(), &memoryAllocateInfo, 0, &memory);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkAllocateMemory failed %d", ret);
        return 0;
    }

    VkBindImageMemoryInfo bindImageMemoryInfo;
    bindImageMemoryInfo.sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO;
    bindImageMemoryInfo.pNext = 0;
    bindImageMemoryInfo.image = image;
    bindImageMemoryInfo.memory = memory;
    bindImageMemoryInfo.memoryOffset = 0;
    ret = vkdev->vkBindImageMemory2KHR(vkdev->vkdevice(), 1, &bindImageMemoryInfo);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkBindImageMemory2KHR failed %d", ret);
        vkDestroyImage(vkdev->vkdevice(), image, 0);
        return 0;
    }

    VkSamplerYcbcrConversionInfoKHR samplerYcbcrConversionInfo;
    samplerYcbcrConversionInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO_KHR;
    samplerYcbcrConversionInfo.pNext = &externalFormat;
    samplerYcbcrConversionInfo.conversion = samplerYcbcrConversion;

    VkImageViewCreateInfo imageViewCreateInfo;
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.pNext = &samplerYcbcrConversionInfo;
    imageViewCreateInfo.flags = 0;
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewCreateInfo.format = VK_FORMAT_UNDEFINED;
    imageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    VkImageView imageview = 0;
    ret = vkCreateImageView(vkdev->vkdevice(), &imageViewCreateInfo, 0, &imageview);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateImageView failed %d", ret);
        vkDestroyImage(vkdev->vkdevice(), image, 0);
        vkFreeMemory(vkdev->vkdevice(), memory, 0);
        return 0;
    }

    VkImageMemory* ptr = new VkImageMemory;
    ptr->image = image;
    ptr->memory = memory;
    ptr->imageview = imageview;
    ptr->mapped_ptr = 0;
    ptr->memory_type_index = (uint32_t)-1;
    ptr->access_flags = 0;
    ptr->image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    ptr->stage_flags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    return ptr;
}

void VkAndroidHardwareBufferImageAllocator::fastFree(VkImageMemory* ptr)
{
    vkDestroyImageView(vkdev->vkdevice(), ptr->imageview, 0);
    vkDestroyImage(vkdev->vkdevice(), ptr->image, 0);
    vkFreeMemory(vkdev->vkdevice(), ptr->memory, 0);

    delete ptr;
}

int VkAndroidHardwareBufferImageAllocator::init()
{
    AHardwareBuffer_describe(hb, &bufferDesc);

    VkResult ret;

    // resolve externalFormat
    bufferFormatProperties.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID;
    bufferFormatProperties.pNext = 0;

    bufferProperties.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;
    bufferProperties.pNext = &bufferFormatProperties;

    ret = vkdev->vkGetAndroidHardwareBufferPropertiesANDROID(vkdev->vkdevice(), hb, &bufferProperties);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkGetAndroidHardwareBufferPropertiesANDROID failed %d", ret);
        return -1;
    }

    // setup samplerYcbcrConversion
    VkExternalFormatANDROID externalFormat;
    externalFormat.sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID;
    externalFormat.pNext = 0;
    externalFormat.externalFormat = bufferFormatProperties.externalFormat;

    VkSamplerYcbcrConversionCreateInfoKHR samplerYcbcrConversionCreateInfo;
    samplerYcbcrConversionCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO_KHR;
    samplerYcbcrConversionCreateInfo.pNext = &externalFormat;
    samplerYcbcrConversionCreateInfo.format = VK_FORMAT_UNDEFINED;
    samplerYcbcrConversionCreateInfo.ycbcrModel = bufferFormatProperties.suggestedYcbcrModel;
    samplerYcbcrConversionCreateInfo.ycbcrRange = bufferFormatProperties.suggestedYcbcrRange;
    samplerYcbcrConversionCreateInfo.components = bufferFormatProperties.samplerYcbcrConversionComponents;
    samplerYcbcrConversionCreateInfo.xChromaOffset = bufferFormatProperties.suggestedXChromaOffset;
    samplerYcbcrConversionCreateInfo.yChromaOffset = bufferFormatProperties.suggestedYChromaOffset;
    samplerYcbcrConversionCreateInfo.chromaFilter = VK_FILTER_NEAREST;
    samplerYcbcrConversionCreateInfo.forceExplicitReconstruction = VK_FALSE;

    ret = vkdev->vkCreateSamplerYcbcrConversionKHR(vkdev->vkdevice(), &samplerYcbcrConversionCreateInfo, 0, &samplerYcbcrConversion);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateSamplerYcbcrConversionKHR failed %d", ret);
        return -1;
    }

    return 0;
}

int VkAndroidHardwareBufferImageAllocator::width() const
{
    return bufferDesc.width;
}

int VkAndroidHardwareBufferImageAllocator::height() const
{
    return bufferDesc.height;
}

uint64_t VkAndroidHardwareBufferImageAllocator::external_format() const
{
    return bufferFormatProperties.externalFormat;
}
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API

#endif // NCNN_VULKAN

} // namespace ncnn
