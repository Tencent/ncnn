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

namespace ncnn {

Command::Command(VulkanDevice* _vkdev) : vkdev(_vkdev)
{
    device = vkdev->vkdevice();

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
    fprintf(stderr, "==================== begin\n");

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

void Command::record_upload(const VkMat& m)
{
    fprintf(stderr, "record_upload %p to %p\n", m.staging_buffer, m.buffer);

    VkBufferCopy region;
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = m.total() * m.elemsize;

    vkCmdCopyBuffer(command_buffer, m.staging_buffer, m.buffer, 1, &region);
}

void Command::record_download(const VkMat& m)
{
    fprintf(stderr, "record_download %p to %p\n", m.buffer, m.staging_buffer);

    VkBufferCopy region;
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = m.total() * m.elemsize;

    vkCmdCopyBuffer(command_buffer, m.buffer, m.staging_buffer, 1, &region);
}

void Command::record_clone(const VkMat& src, const VkMat& dst)
{
    fprintf(stderr, "record_clone %p %p\n", src.buffer, dst.buffer);

    VkBufferCopy region;
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = src.total() * src.elemsize;

    vkCmdCopyBuffer(command_buffer, src.buffer, dst.buffer, 1, &region);
}

void Command::record_bind_pipeline(VkPipeline pipeline)
{
//     fprintf(stderr, "record_bind_pipeline %p\n", pipeline);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
}

void Command::record_update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorUpdateTemplate descriptor_update_template, const std::vector<VkMat>& bindings)
{
//     fprintf(stderr, "record_update_bindings %p %p\n", pipeline_layout, descriptor_update_template);

    const int binding_count = bindings.size();

    std::vector<VkDescriptorBufferInfo> descriptorBufferInfos;
    descriptorBufferInfos.resize(binding_count);

    for (int i=0; i<binding_count; i++)
    {
        descriptorBufferInfos[i].buffer = bindings[i].buffer;
        descriptorBufferInfos[i].offset = 0;
        descriptorBufferInfos[i].range = VK_WHOLE_SIZE;
    }

    vkdev->vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, descriptor_update_template, pipeline_layout, 0, descriptorBufferInfos.data());
}

void Command::record_push_constants(VkPipelineLayout pipeline_layout, const std::vector<int>& constants)
{
//     fprintf(stderr, "record_push_constants %p\n", pipeline_layout);

    vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constants.size() * sizeof(int), constants.data());
}

void Command::record_dispatch(uint32_t* group_count_xyz)
{
//     fprintf(stderr, "record_dispatch %d %d %d\n", group_count_xyz[0], group_count_xyz[1], group_count_xyz[2]);

    vkCmdDispatch(command_buffer, group_count_xyz[0], group_count_xyz[1], group_count_xyz[2]);
}

void Command::record_upload_compute_barrier(const VkMat& m)
{
//     fprintf(stderr, "record_upload_compute_barrier %p\n", m.buffer);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = m.buffer;
    bufferBarrier.offset = 0;
    bufferBarrier.size = VK_WHOLE_SIZE;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

void Command::record_compute_download_barrier(const VkMat& m)
{
//     fprintf(stderr, "record_compute_download_barrier %p\n", m.buffer);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = m.buffer;
    bufferBarrier.offset = 0;
    bufferBarrier.size = VK_WHOLE_SIZE;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

void Command::record_compute_compute_barrier(const VkMat& m)
{
//     fprintf(stderr, "record_compute_compute_barrier %p\n", m.buffer);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = m.buffer;
    bufferBarrier.offset = 0;
    bufferBarrier.size = VK_WHOLE_SIZE;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

int Command::end()
{
    fprintf(stderr, "==================== end\n");

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
    fprintf(stderr, "==================== submit\n");

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
    fprintf(stderr, "==================== wait\n");

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
