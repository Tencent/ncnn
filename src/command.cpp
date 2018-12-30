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
    // get queue
    vkGetDeviceQueue(vkdev->vkdevice(), vkdev->info.compute_queue_index, 0, &queue);

    create_command_pool();

    create_command_buffer();

    // create fence
    VkFenceCreateInfo fenceCreateInfo;
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.pNext = 0;
    fenceCreateInfo.flags = 0;

    VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &fence);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateFence failed %d\n", ret);
    }
}

Command::~Command()
{
    if (!vkdev->info.support_VK_KHR_push_descriptor)
    {
        for (size_t i=0; i<descriptorsets.size(); i++)
        {
            vkFreeDescriptorSets(vkdev->vkdevice(), descriptor_pools[i], 1, &descriptorsets[i]);
            vkDestroyDescriptorPool(vkdev->vkdevice(), descriptor_pools[i], 0);
        }
    }

    vkDestroyFence(vkdev->vkdevice(), fence, 0);

    vkFreeCommandBuffers(vkdev->vkdevice(), command_pool, 1, &command_buffer);

    vkDestroyCommandPool(vkdev->vkdevice(), command_pool, 0);
}

int Command::begin()
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return begin_command_buffer();

    record_type r;
    r.type = 0;
    delayed_records.push_back(r);

    return 0;
}

void Command::record_upload(const VkMat& m)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return copy_buffer(m.staging_buffer, m.buffer, m.total() * m.elemsize);

    record_type r;
    r.type = 1;
    r.copy = { m.staging_buffer, m.buffer, m.total() * m.elemsize };
    delayed_records.push_back(r);
}

void Command::record_download(const VkMat& m)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return copy_buffer(m.buffer, m.staging_buffer, m.total() * m.elemsize);

    record_type r;
    r.type = 1;
    r.copy = { m.buffer, m.staging_buffer, m.total() * m.elemsize };
    delayed_records.push_back(r);
}

void Command::record_clone(const VkMat& src, const VkMat& dst)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return copy_buffer(src.buffer, dst.buffer, src.total() * src.elemsize);

    record_type r;
    r.type = 1;
    r.copy = { src.buffer, dst.buffer, src.total() * src.elemsize };
    delayed_records.push_back(r);
}

void Command::record_bind_pipeline(VkPipeline pipeline)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return bind_pipeline(pipeline);

    record_type r;
    r.type = 2;
    r.bind_pipeline = { pipeline };
    delayed_records.push_back(r);
}

void Command::record_update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplate descriptor_update_template, const std::vector<VkMat>& bindings)
{
    const int binding_count = bindings.size();

    std::vector<VkDescriptorBufferInfo> descriptorBufferInfos(binding_count);
    for (int i=0; i<binding_count; i++)
    {
        descriptorBufferInfos[i].buffer = bindings[i].buffer;
        descriptorBufferInfos[i].offset = 0;
        descriptorBufferInfos[i].range = VK_WHOLE_SIZE;
    }

    if (vkdev->info.support_VK_KHR_push_descriptor)
        return update_bindings(pipeline_layout, descriptor_update_template, descriptorBufferInfos);

    // create new descriptor_pool and descriptorset
    VkDescriptorPool descriptor_pool;
    {
        VkDescriptorPoolSize poolSize;
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = binding_count;

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.pNext = 0;
        descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        descriptorPoolCreateInfo.maxSets = 1;
        descriptorPoolCreateInfo.poolSizeCount = 1;
        descriptorPoolCreateInfo.pPoolSizes = &poolSize;

        VkResult ret = vkCreateDescriptorPool(vkdev->vkdevice(), &descriptorPoolCreateInfo, 0, &descriptor_pool);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkCreateDescriptorPool failed %d\n", ret);
            return;
        }
    }
    descriptor_pools.push_back(descriptor_pool);

    VkDescriptorSet descriptorset;
    {
        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.pNext = 0;
        descriptorSetAllocateInfo.descriptorPool = descriptor_pool;
        descriptorSetAllocateInfo.descriptorSetCount = 1;
        descriptorSetAllocateInfo.pSetLayouts = &descriptorset_layout;

        VkResult ret = vkAllocateDescriptorSets(vkdev->vkdevice(), &descriptorSetAllocateInfo, &descriptorset);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkAllocateDescriptorSets failed %d\n", ret);
            return;
        }
    }
    descriptorsets.push_back(descriptorset);

    fprintf(stderr, "update descriptorset %p\n", descriptorset);

    std::vector<VkWriteDescriptorSet> writeDescriptorSets(binding_count);
    for (int i=0; i<binding_count; i++)
    {
        writeDescriptorSets[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[i].pNext = 0;
        writeDescriptorSets[i].dstSet = descriptorset;
        writeDescriptorSets[i].dstBinding = i;
        writeDescriptorSets[i].dstArrayElement = 0;
        writeDescriptorSets[i].descriptorCount = 1;
        writeDescriptorSets[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSets[i].pImageInfo = 0;
        writeDescriptorSets[i].pBufferInfo = &descriptorBufferInfos[i];
        writeDescriptorSets[i].pTexelBufferView = 0;
    }

    vkUpdateDescriptorSets(vkdev->vkdevice(), binding_count, writeDescriptorSets.data(), 0, 0);

    record_type r;
    r.type = 3;
    r.bind_descriptorset = { pipeline_layout, descriptorset };
    delayed_records.push_back(r);
}

void Command::record_push_constants(VkPipelineLayout pipeline_layout, const std::vector<vk_constant_type>& constants)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return push_constants(pipeline_layout, constants);

    record_type r;
    r.type = 4;
    r.push_constants = { pipeline_layout };
    r.constants = constants;
    delayed_records.push_back(r);
}

void Command::record_dispatch(const uint32_t* group_count_xyz)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return dispatch(group_count_xyz);

    record_type r;
    r.type = 5;
    r.dispatch.group_count_xyz[0] = group_count_xyz[0];
    r.dispatch.group_count_xyz[1] = group_count_xyz[1];
    r.dispatch.group_count_xyz[2] = group_count_xyz[2];
    delayed_records.push_back(r);
}

void Command::record_upload_compute_barrier(const VkMat& m)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return upload_compute_barrier(m.buffer);

    record_type r;
    r.type = 6;
    r.upload_compute_barrier.buffer = m.buffer;
    delayed_records.push_back(r);
}

void Command::record_compute_download_barrier(const VkMat& m)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return compute_download_barrier(m.buffer);

    record_type r;
    r.type = 7;
    r.compute_download_barrier.buffer = m.buffer;
    delayed_records.push_back(r);
}

void Command::record_compute_compute_barrier(const VkMat& m)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return compute_compute_barrier(m.buffer);

    record_type r;
    r.type = 8;
    r.compute_compute_barrier.buffer = m.buffer;
    delayed_records.push_back(r);
}

int Command::end()
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return end_command_buffer();

    record_type r;
    r.type = 9;
    delayed_records.push_back(r);

    return 0;
}

int Command::submit()
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return queue_submit();

    // handle delayed records
    for (size_t i=0; i<delayed_records.size(); i++)
    {
        const record_type& r = delayed_records[i];

        switch (r.type)
        {
        case 0:
            begin_command_buffer();
            break;
        case 1:
            copy_buffer(r.copy.src, r.copy.dst, r.copy.size);
            break;
        case 2:
            bind_pipeline(r.bind_pipeline.pipeline);
            break;
        case 3:
            bind_descriptorset(r.bind_descriptorset.pipeline_layout, r.bind_descriptorset.descriptorset);
            break;
        case 4:
            push_constants(r.push_constants.pipeline_layout, r.constants);
            break;
        case 5:
            dispatch(r.dispatch.group_count_xyz);
            break;
        case 6:
            upload_compute_barrier(r.upload_compute_barrier.buffer);
            break;
        case 7:
            compute_download_barrier(r.compute_download_barrier.buffer);
            break;
        case 8:
            compute_compute_barrier(r.compute_compute_barrier.buffer);
            break;
        case 9:
            end_command_buffer();
            break;
        }
    }

    return queue_submit();
}

int Command::begin_command_buffer()
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

void Command::copy_buffer(VkBuffer src, VkBuffer dst, int size)
{
    fprintf(stderr, "cmd copy %p to %p\n", src, dst);

    VkBufferCopy region;
    region.srcOffset = 0;
    region.dstOffset = 0;
    region.size = size;

    vkCmdCopyBuffer(command_buffer, src, dst, 1, &region);
}

void Command::bind_pipeline(VkPipeline pipeline)
{
    fprintf(stderr, "cmd bind_pipeline %p\n", pipeline);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
}

void Command::bind_descriptorset(VkPipelineLayout pipeline_layout, VkDescriptorSet descriptorset)
{
    fprintf(stderr, "cmd bind_descriptorset %p %p\n", pipeline_layout, descriptorset);

    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptorset, 0, 0);
}

void Command::update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorUpdateTemplate descriptor_update_template, const std::vector<VkDescriptorBufferInfo>& descriptorBufferInfos)
{
    fprintf(stderr, "cmd update_bindings %p %p\n", pipeline_layout, descriptor_update_template);

    vkdev->vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, descriptor_update_template, pipeline_layout, 0, descriptorBufferInfos.data());
}

void Command::push_constants(VkPipelineLayout pipeline_layout, const std::vector<vk_constant_type>& constants)
{
    fprintf(stderr, "cmd push_constants %p\n", pipeline_layout);

    vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constants.size() * sizeof(vk_constant_type), constants.data());
}

void Command::dispatch(const uint32_t* group_count_xyz)
{
    fprintf(stderr, "cmd dispatch %d %d %d\n", group_count_xyz[0], group_count_xyz[1], group_count_xyz[2]);

    vkCmdDispatch(command_buffer, group_count_xyz[0], group_count_xyz[1], group_count_xyz[2]);
}

void Command::upload_compute_barrier(VkBuffer buffer)
{
    fprintf(stderr, "cmd upload_compute_barrier %p\n", buffer);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = 0;
    bufferBarrier.size = VK_WHOLE_SIZE;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

void Command::compute_download_barrier(VkBuffer buffer)
{
    fprintf(stderr, "cmd compute_download_barrier %p\n", buffer);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = 0;
    bufferBarrier.size = VK_WHOLE_SIZE;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

void Command::compute_compute_barrier(VkBuffer buffer)
{
    fprintf(stderr, "cmd compute_compute_barrier %p\n", buffer);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = 0;
    bufferBarrier.size = VK_WHOLE_SIZE;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

int Command::end_command_buffer()
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

int Command::queue_submit()
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

    VkResult ret = vkWaitForFences(vkdev->vkdevice(), 1, &fence, VK_TRUE, UINT64_MAX);
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

    VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &command_pool);
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

    VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &command_buffer);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkAllocateCommandBuffers failed %d\n", ret);
        return -1;
    }

    return 0;
}

} // namespace ncnn

#endif // NCNN_VULKAN
