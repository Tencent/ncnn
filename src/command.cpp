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

    // create fence
    VkFenceCreateInfo fenceCreateInfo;
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.pNext = 0;
    fenceCreateInfo.flags = 0;

    VkResult ret = vkCreateFence(device, &fenceCreateInfo, 0, &fence);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateFence failed %d\n", ret);
    }
}

Command::~Command()
{
    vkDestroyFence(device, fence, 0);

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

void Command::record_imagelayout_barrier(VkMat& image, int type)
{
    VkAccessFlags srcAccessMask;
    VkAccessFlags dstAccessMask;
    VkImageLayout oldLayout;
    VkImageLayout newLayout;

    VkPipelineStageFlags srcStageMask;
    VkPipelineStageFlags dstStageMask;

    if (type == 0)
    {
        // prepare for blob upload, undefined to transfer-dst-optimal
        srcAccessMask = 0;
        dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (type == 1)
    {
        // prepare for weight blob compute, transfer-dst-optimal to general
        srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        newLayout = VK_IMAGE_LAYOUT_GENERAL;

        srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else if (type == 2)
    {
        // prepare for output blob compute, undefined to general
        srcAccessMask = 0;
        dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        newLayout = VK_IMAGE_LAYOUT_GENERAL;

        srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    else if (type == 3)
    {
        // prepare for blob download, general to transfer-src-optimal
        srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    VkImageMemoryBarrier imageBarrier;
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.pNext = 0;
    imageBarrier.srcAccessMask = srcAccessMask;
    imageBarrier.dstAccessMask = dstAccessMask;
    imageBarrier.oldLayout = oldLayout;
    imageBarrier.newLayout = newLayout;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = image.image;
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 0, 0, 1, &imageBarrier);
}

void Command::record_upload(const Mat& src, VkMat& dst)
{
    dst.prepare_staging_buffer(staging_allocator);

    dst.staging_buffer_upload(src);

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

    vkCmdCopyBufferToImage(command_buffer, dst.staging_buffer, dst.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

void Command::record_download(VkMat& src, const Mat& dst)
{
    src.prepare_staging_buffer(staging_allocator);

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

    vkCmdCopyImageToBuffer(command_buffer, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, src.staging_buffer, 1, &region);
}

void Command::record_layer(const Layer* layer, uint32_t group_count_xyz[3])
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
}

void Command::record_compute_barrier()
{
    VkMemoryBarrier memoryBarrier;
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.pNext = 0;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 1, &memoryBarrier, 0, 0, 0, 0);
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

    VkResult ret = vkQueueSubmit(queue, 1, &submitInfo, fence);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkQueueSubmit failed %d\n", ret);
        return -1;
    }

    return 0;
}

int Command::wait()
{
    VkResult ret = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkWaitForFences failed %d\n", ret);
        return -1;
    }

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
