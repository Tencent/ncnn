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
VkAllocator::VkAllocator(VulkanDevice* _vkdev, int _type)
    : vkdev(_vkdev), type(_type),
      compute_queue_index(_vkdev->info.compute_queue_index),
      memory_type_index(_type == 0 ? _vkdev->info.device_local_memory_index : _vkdev->info.host_visible_memory_index)
{
    device = *_vkdev;

    size_compare_ratio = 192;// 0.75f * 256
}

VkAllocator::~VkAllocator()
{
    for (int i=0; i<(int)images_to_destroy.size(); i++)
    {
        vkDestroyImage(device, images_to_destroy[i], 0);
    }
    images_to_destroy.clear();
    for (int i=0; i<(int)imageviews_to_destroy.size(); i++)
    {
        vkDestroyImageView(device, imageviews_to_destroy[i], 0);
    }
    imageviews_to_destroy.clear();
    for (int i=0; i<(int)buffers_to_destroy.size(); i++)
    {
        vkDestroyBuffer(device, buffers_to_destroy[i], 0);
    }
    buffers_to_destroy.clear();

    clear();

    if (!payouts.empty())
    {
        fprintf(stderr, "FATAL ERROR! unlocked pool vulkan allocator destroyed too early\n");
        std::list< std::pair<size_t, VkDeviceMemory> >::iterator it = payouts.begin();
        for (; it != payouts.end(); it++)
        {
            VkDeviceMemory ptr = it->second;
            fprintf(stderr, "%p still in use\n", ptr);
        }
    }
}

VkImage VkAllocator::create_image(VkImageType imageType, int w, int h, int c)
{
    VkImageCreateInfo imageCreateInfo;
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.pNext = 0;
    imageCreateInfo.flags = 0;
    imageCreateInfo.imageType = imageType;
    imageCreateInfo.format = VK_FORMAT_R32_SFLOAT;
    imageCreateInfo.extent.width = w;
    imageCreateInfo.extent.height = h;
    imageCreateInfo.extent.depth = c;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageCreateInfo.queueFamilyIndexCount = 1;
    imageCreateInfo.pQueueFamilyIndices = &compute_queue_index;
    imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image;
    VkResult ret = vkCreateImage(device, &imageCreateInfo, 0, &image);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateImage failed %d\n", ret);
        return 0;
    }

    return image;
}

VkImageView VkAllocator::create_imageview(VkImageViewType viewType, VkImage image)
{
    VkComponentMapping componentMapping;
    componentMapping.r = VK_COMPONENT_SWIZZLE_R;
    componentMapping.g = VK_COMPONENT_SWIZZLE_G;
    componentMapping.b = VK_COMPONENT_SWIZZLE_B;
    componentMapping.a = VK_COMPONENT_SWIZZLE_A;

    VkImageSubresourceRange subresourceRange;
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.baseArrayLayer = 0;
    subresourceRange.layerCount = 1;

    VkImageViewCreateInfo imageViewCreateInfo;
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.pNext = 0;
    imageViewCreateInfo.flags = 0;
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = viewType;
    imageViewCreateInfo.format = VK_FORMAT_R32_SFLOAT;
    imageViewCreateInfo.components = componentMapping;
    imageViewCreateInfo.subresourceRange = subresourceRange;

    VkImageView imageView;
    VkResult ret = vkCreateImageView(device, &imageViewCreateInfo, 0, &imageView);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateImageView failed %d\n", ret);
        return 0;
    }

    return imageView;
}

VkBuffer VkAllocator::create_buffer(VkBufferUsageFlags usage, int size)
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

    VkBuffer buffer;
    VkResult ret = vkCreateBuffer(device, &bufferCreateInfo, 0, &buffer);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateBuffer failed %d\n", ret);
        return 0;
    }

    return buffer;
}

void VkAllocator::destroy_image(VkImage image)
{
    images_to_destroy.push_back(image);
}

void VkAllocator::destroy_imageview(VkImageView imageview)
{
    imageviews_to_destroy.push_back(imageview);
}

void VkAllocator::destroy_buffer(VkBuffer buffer)
{
    buffers_to_destroy.push_back(buffer);
}

void VkAllocator::clear()
{
    std::list< std::pair<size_t, VkDeviceMemory> >::iterator it = budgets.begin();
    for (; it != budgets.end(); it++)
    {
        VkDeviceMemory ptr = it->second;

        vkFreeMemory(device, ptr, 0);
    }
    budgets.clear();
}

void VkAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        fprintf(stderr, "invalid size compare ratio %f\n", scr);
        return;
    }

    size_compare_ratio = (unsigned int)(scr * 256);
}

VkDeviceMemory VkAllocator::fastMalloc(size_t size)
{
    fprintf(stderr, "VkAllocator fastMalloc %lu\n", size);

    // find free budget
    std::list< std::pair<size_t, VkDeviceMemory> >::iterator it = budgets.begin();
    for (; it != budgets.end(); it++)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * size_compare_ratio) >> 8) <= size)
        {
            VkDeviceMemory ptr = it->second;

            budgets.erase(it);

            payouts.push_back(std::make_pair(bs, ptr));

            return ptr;
        }
    }

    // new
    VkMemoryAllocateInfo memoryAllocateInfo;
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = 0;
    memoryAllocateInfo.allocationSize = size;
    memoryAllocateInfo.memoryTypeIndex = memory_type_index;

    VkDeviceMemory ptr = 0;
    VkResult ret = vkAllocateMemory(device, &memoryAllocateInfo, 0, &ptr);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkAllocateMemory failed %d\n", ret);
    }

    payouts.push_back(std::make_pair(size, ptr));

    return ptr;
}

void VkAllocator::fastFree(VkDeviceMemory ptr)
{
    fprintf(stderr, "VkAllocator fastFree %p\n", ptr);

    // return to budgets
    std::list< std::pair<size_t, VkDeviceMemory> >::iterator it = payouts.begin();
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

    fprintf(stderr, "FATAL ERROR! unlocked vulkan pool allocator get wild %p\n", ptr);

    vkFreeMemory(device, ptr, 0);
}
#endif // NCNN_VULKAN

} // namespace ncnn
