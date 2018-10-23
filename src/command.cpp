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

#include "command.h"

#if NCNN_VULKAN

#include <stdio.h>

#include "layer.h"

namespace ncnn {

Command::Command(VulkanDevice* _vkdev, VkAllocator* _staging_allocator) : vkdev(_vkdev), staging_allocator(_staging_allocator)
{
    device = *vkdev;

    // get queue
    vkGetDeviceQueue(device, vkdev->info.compute_queue_index, 0, &queue);

    create_command_pool();

    create_command_buffer();
}

Command::~Command()
{
    vkFreeCommandBuffers(device, command_pool, 1, &command_buffer);

    vkDestroyCommandPool(device, command_pool, 0);
}

int Command::begin()
{
    VkCommandBufferBeginInfo commandBufferBeginInfo;
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.pNext = 0;
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    commandBufferBeginInfo.pInheritanceInfo = 0;

    VkResult ret = vkBeginCommandBuffer(command_buffer, &commandBufferBeginInfo);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkBeginCommandBuffer failed %d\n", ret);
        return -1;
    }

    return 0;
}

UploadEvent Command::record_upload(const Mat& src, const VkMat& dst)
{
    // image layout transition, undefined to transfer-dst-optimal
    {
    VkImageMemoryBarrier imageBarrier;
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.pNext = 0;
    imageBarrier.srcAccessMask = 0;
    imageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = dst.image;
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(command_buffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
        0, 0, 0, 0, 1, &imageBarrier);
    }

    // alloc staging buffer
    // upload to gpu via staging buffer
    VkBuffer staging_buffer = staging_allocator->create_buffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, src.w * src.h * src.c * sizeof(float));

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, staging_buffer, &memoryRequirements);

    // memoryRequirements.size
    // memoryRequirements.alignment
    // memoryRequirements.memoryTypeBits

    VkDeviceMemory staging_memory = staging_allocator->fastMalloc(memoryRequirements.size);

    VkResult ret = vkBindBufferMemory(device, staging_buffer, staging_memory, 0);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkBindBufferMemory failed %d\n", ret);
    }

    // copy
    copy_mat_to_staging(src, staging_memory);

    // staging buffer to image
    VkBufferImageCopy region;
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset.x = 0;
    region.imageOffset.y = 0;
    region.imageOffset.z = 0;
    region.imageExtent.width = dst.w;
    region.imageExtent.height = dst.h;
    region.imageExtent.depth = dst.c;

    vkCmdCopyBufferToImage(command_buffer, staging_buffer, dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VkEvent event;
    {
    VkEventCreateInfo eventCreateInfo;
    eventCreateInfo.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    eventCreateInfo.pNext = 0;
    eventCreateInfo.flags = 0;

    VkResult ret = vkCreateEvent(device, &eventCreateInfo, 0, &event);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateEvent failed %d\n", ret);
    }
    }

    vkCmdSetEvent(command_buffer, event, VK_PIPELINE_STAGE_TRANSFER_BIT);

    UploadEvent ue;
    ue.event = event;
    ue.src = src;
    ue.dst = dst;
    ue.staging_buffer = staging_buffer;
    ue.staging_memory = staging_memory;

    return ue;
}

void Command::wait_upload(UploadEvent& event)
{
    VkImageMemoryBarrier imageBarrier;
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.pNext = 0;
    imageBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = event.dst.image;
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;

    vkCmdWaitEvents(command_buffer, 1, &event.event,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, 0, 0, 1, &imageBarrier);
}

void Command::wait_upload(std::vector<UploadEvent>& events)
{
    const int count = events.size();

    std::vector<VkImageMemoryBarrier> imageBarriers;
    imageBarriers.resize(count);
    std::vector<VkEvent> vkevents;
    vkevents.resize(count);

    for (int i=0; i<count; i++)
    {
        imageBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageBarriers[i].pNext = 0;
        imageBarriers[i].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        imageBarriers[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        imageBarriers[i].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imageBarriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarriers[i].image = events[i].dst.image;
        imageBarriers[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBarriers[i].subresourceRange.baseMipLevel = 0;
        imageBarriers[i].subresourceRange.levelCount = 1;
        imageBarriers[i].subresourceRange.baseArrayLayer = 0;
        imageBarriers[i].subresourceRange.layerCount = 1;

        vkevents[i] = events[i].event;
    }

    vkCmdWaitEvents(command_buffer, count, vkevents.data(),
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, 0, 0, count, imageBarriers.data());
}

DownloadEvent Command::record_download(const VkMat& src, const Mat& dst)
{
    // image layout transition, general to transfer-src-optimal
    {
    VkImageMemoryBarrier imageBarrier;
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.pNext = 0;
    imageBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = src.image;
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(command_buffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
        0, 0, 0, 0, 1, &imageBarrier);
    }

    // download from gpu via staging buffer
    VkBuffer staging_buffer = staging_allocator->create_buffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, src.w * src.h * src.c * sizeof(float));

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(device, staging_buffer, &memoryRequirements);

    // memoryRequirements.size
    // memoryRequirements.alignment
    // memoryRequirements.memoryTypeBits

    VkDeviceMemory staging_memory = staging_allocator->fastMalloc(memoryRequirements.size);

    fprintf(stderr, "top_blob_staging %lu\n", memoryRequirements.size);

    VkResult ret = vkBindBufferMemory(device, staging_buffer, staging_memory, 0);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkBindBufferMemory failed %d\n", ret);
    }

    // image to staging buffer
    VkBufferImageCopy region;
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset.x = 0;
    region.imageOffset.y = 0;
    region.imageOffset.z = 0;
    region.imageExtent.width = src.w;
    region.imageExtent.height = src.h;
    region.imageExtent.depth = src.c;

//     fprintf(stderr, "top_blob_gpu = %d %d %d %p\n", src.w, src.h, src.c, src.image);

    vkCmdCopyImageToBuffer(command_buffer, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buffer, 1, &region);

    VkEvent event;
    {
    VkEventCreateInfo eventCreateInfo;
    eventCreateInfo.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    eventCreateInfo.pNext = 0;
    eventCreateInfo.flags = 0;

    VkResult ret = vkCreateEvent(device, &eventCreateInfo, 0, &event);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateEvent failed %d\n", ret);
    }
    }

    vkCmdSetEvent(command_buffer, event, VK_PIPELINE_STAGE_TRANSFER_BIT);

    DownloadEvent de;
    de.event = event;
    de.src = src;
    de.dst = dst;
    de.staging_buffer = staging_buffer;
    de.staging_memory = staging_memory;

    return de;
}

void Command::wait_download(DownloadEvent& event)
{
    VkImageMemoryBarrier imageBarrier;
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.pNext = 0;
    imageBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    imageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = event.src.image;
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;

    vkCmdWaitEvents(command_buffer, 1, &event.event,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, 0, 0, 1, &imageBarrier);

    // copy
//     copy_staging_to_mat(staging_memory, event.dst);
}

void Command::wait_download(std::vector<DownloadEvent>& events)
{
    const int count = events.size();

    std::vector<VkImageMemoryBarrier> imageBarriers;
    imageBarriers.resize(count);
    std::vector<VkEvent> vkevents;
    vkevents.resize(count);

    for (int i=0; i<count; i++)
    {
        imageBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageBarriers[i].pNext = 0;
        imageBarriers[i].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        imageBarriers[i].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        imageBarriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageBarriers[i].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        imageBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        imageBarriers[i].image = events[i].src.image;
        imageBarriers[i].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBarriers[i].subresourceRange.baseMipLevel = 0;
        imageBarriers[i].subresourceRange.levelCount = 1;
        imageBarriers[i].subresourceRange.baseArrayLayer = 0;
        imageBarriers[i].subresourceRange.layerCount = 1;

        vkevents[i] = events[i].event;
    }

    vkCmdWaitEvents(command_buffer, count, vkevents.data(),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, 0, 0, count, imageBarriers.data());

    // copy
//     for (int i=0; i<count; i++)
//     {
//         copy_staging_to_mat(events[i].staging_memory, events[i].dst);
//     }
}

LayerEvent Command::record_layer(const Layer* layer, uint32_t group_count_xyz[3])
{
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layer->pipeline);

    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layer->pipeline_layout, 0, 1, &layer->descriptorset, 0, 0);

//     vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_data), &pcdata);

    vkCmdDispatch(command_buffer, group_count_xyz[0], group_count_xyz[1], group_count_xyz[2]);

//     VkDispatchIndirectCommand dispatch_param;
//     dispatch_param.x = group_x;
//     dispatch_param.y = group_y;
//     dispatch_param.z = group_z;

//     vkCmdDispatchIndirect(commandBuffer, buffer, offset);

    VkEvent event;
    {
    VkEventCreateInfo eventCreateInfo;
    eventCreateInfo.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
    eventCreateInfo.pNext = 0;
    eventCreateInfo.flags = 0;

    VkResult ret = vkCreateEvent(device, &eventCreateInfo, 0, &event);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateEvent failed %d\n", ret);
    }
    }

    vkCmdSetEvent(command_buffer, event, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    LayerEvent le;
    le.event = event;
    le.layer = layer;

    return le;
}

void Command::wait_layer(LayerEvent& event)
{
    VkMemoryBarrier memoryBarrier;
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.pNext = 0;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdWaitEvents(command_buffer, 1, &event.event,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        1, &memoryBarrier, 0, 0, 0, 0);
}

int Command::end()
{
    VkResult ret = vkEndCommandBuffer(command_buffer);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkEndCommandBuffer failed %d\n", ret);
        return -1;
    }

    return 0;
}

int Command::submit()
{
    VkSubmitInfo submitInfo;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = 0;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = 0;
    submitInfo.pWaitDstStageMask = 0;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &command_buffer;
    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = 0;

    VkResult ret = vkQueueSubmit(queue, 1, &submitInfo, 0);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkQueueSubmit failed %d\n", ret);
        return -1;
    }

    return 0;
}

int Command::copy_mat_to_staging(const Mat& src, VkDeviceMemory staging_memory)
{
    void* mapped_ptr = 0;
    VkResult ret = vkMapMemory(device, staging_memory, 0, VK_WHOLE_SIZE, 0, &mapped_ptr);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkMapMemory failed %d\n", ret);
        return -1;
    }

    int w = src.w;
    int h = src.h;
    int channels = src.c;
    int size = w * h;

    for (int p=0; p<channels; p++)
    {
        const float* ptr = src.channel(p);
        float* outptr = (float*)mapped_ptr + size * p;

        memcpy(outptr, ptr, size * sizeof(float));
    }

    // TODO hold mapped ptr ?
    vkUnmapMemory(device, staging_memory);

    return 0;
}

int Command::copy_staging_to_mat(VkDeviceMemory staging_memory, Mat& dst)
{
    void* mapped_ptr = 0;
    VkResult ret = vkMapMemory(device, staging_memory, 0, VK_WHOLE_SIZE, 0, &mapped_ptr);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkMapMemory failed %d\n", ret);
        return -1;
    }

    int w = dst.w;
    int h = dst.h;
    int channels = dst.c;
    int size = w * h;

    for (int p=0; p<channels; p++)
    {
        const float* ptr = (const float*)mapped_ptr + size * p;
        float* outptr = dst.channel(p);

        memcpy(outptr, ptr, size * sizeof(float));
    }

    // TODO hold mapped ptr ?
    vkUnmapMemory(device, staging_memory);

    return 0;
}

int Command::create_command_pool()
{
    VkCommandPoolCreateInfo commandPoolCreateInfo;
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.pNext = 0;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = vkdev->info.compute_queue_index;

    VkResult ret = vkCreateCommandPool(device, &commandPoolCreateInfo, 0, &command_pool);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateCommandPool failed %d\n", ret);
        return -1;
    }

    return 0;
}

int Command::create_command_buffer()
{
    VkCommandBufferAllocateInfo commandBufferAllocateInfo;
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.pNext = 0;
    commandBufferAllocateInfo.commandPool = command_pool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;

    VkResult ret = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &command_buffer);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkAllocateCommandBuffers failed %d\n", ret);
        return -1;
    }

    return 0;
}

} // namespace ncnn

#endif // NCNN_VULKAN
