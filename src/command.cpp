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
#include "option.h"

namespace ncnn {

Command::Command(const VulkanDevice* _vkdev, uint32_t _queue_family_index) : vkdev(_vkdev), queue_family_index(_queue_family_index)
{
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
    vkDestroyFence(vkdev->vkdevice(), fence, 0);

    vkFreeCommandBuffers(vkdev->vkdevice(), command_pool, 1, &command_buffer);

    vkDestroyCommandPool(vkdev->vkdevice(), command_pool, 0);
}

int Command::create_command_pool()
{
    VkCommandPoolCreateInfo commandPoolCreateInfo;
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.pNext = 0;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex = queue_family_index;

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

int Command::begin_command_buffer()
{
//     fprintf(stderr, "==================== begin\n");

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

int Command::end_command_buffer()
{
//     fprintf(stderr, "==================== end\n");

    VkResult ret = vkEndCommandBuffer(command_buffer);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkEndCommandBuffer failed %d\n", ret);
        return -1;
    }

    return 0;
}

int Command::queue_submit_and_wait_fence()
{
    // acquire queue and reclaim on return
    VkQueue queue = vkdev->acquire_queue(queue_family_index);
    if (queue == 0)
    {
        fprintf(stderr, "out of compute queue\n");
        return -1;
    }

//     fprintf(stderr, "==================== submit\n");
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
            vkdev->reclaim_queue(queue_family_index, queue);
            return -1;
        }
    }

//     fprintf(stderr, "==================== wait\n");
    {
        VkResult ret = vkWaitForFences(vkdev->vkdevice(), 1, &fence, VK_TRUE, UINT64_MAX);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkWaitForFences failed %d\n", ret);
            vkdev->reclaim_queue(queue_family_index, queue);
            return -1;
        }
    }

    vkdev->reclaim_queue(queue_family_index, queue);
    return 0;
}

VkCompute::VkCompute(const VulkanDevice* _vkdev) : Command(_vkdev, _vkdev->info.compute_queue_family_index)
{
#if NCNN_BENCHMARK
    query_count = 0;
    query_pool = 0;
#endif // NCNN_BENCHMARK

    if (vkdev->info.support_VK_KHR_push_descriptor)
    {
        begin_command_buffer();
    }
}

VkCompute::~VkCompute()
{
    if (!vkdev->info.support_VK_KHR_push_descriptor)
    {
        for (size_t i=0; i<descriptorsets.size(); i++)
        {
            vkFreeDescriptorSets(vkdev->vkdevice(), descriptor_pools[i], 1, &descriptorsets[i]);
            vkDestroyDescriptorPool(vkdev->vkdevice(), descriptor_pools[i], 0);
        }
    }

#if NCNN_BENCHMARK
    if (query_pool)
    {
        // all submitted commands that refer to queryPool must have completed execution
        vkResetCommandBuffer(command_buffer, 0);

        vkDestroyQueryPool(vkdev->vkdevice(), query_pool, 0);
    }
#endif // NCNN_BENCHMARK
}

void VkCompute::record_upload(const VkMat& m)
{
    if (m.allocator->mappable)
        return;

    record_prepare_transfer_barrier(m);

    if (vkdev->info.support_VK_KHR_push_descriptor)
        return copy_buffer(m.staging_buffer(), m.staging_buffer_offset(), m.buffer(), m.buffer_offset(), m.total() * m.elemsize);

    record_type r;
    r.type = 0;
    r.copy.src = m.staging_buffer();
    r.copy.src_offset = m.staging_buffer_offset();
    r.copy.dst = m.buffer();
    r.copy.dst_offset = m.buffer_offset();
    r.copy.size = m.total() * m.elemsize;
    delayed_records.push_back(r);
}

void VkCompute::record_download(const VkMat& m)
{
    if (m.allocator->mappable)
    {
        record_prepare_host_barrier(m);
        return;
    }

    record_prepare_transfer_barrier(m);

    if (vkdev->info.support_VK_KHR_push_descriptor)
    {
        copy_buffer(m.buffer(), m.buffer_offset(), m.staging_buffer(), m.staging_buffer_offset(), m.total() * m.elemsize);
        record_prepare_host_barrier(m);
        return;
    }

    record_type r;
    r.type = 0;
    r.copy.src = m.buffer();
    r.copy.src_offset = m.buffer_offset();
    r.copy.dst = m.staging_buffer();
    r.copy.dst_offset = m.staging_buffer_offset();
    r.copy.size = m.total() * m.elemsize;
    delayed_records.push_back(r);

    record_prepare_host_barrier(m);
}

void VkCompute::record_clone(const VkMat& src, const VkMat& dst)
{
    record_prepare_transfer_barrier(src);
    record_prepare_transfer_barrier(dst);

    if (vkdev->info.support_VK_KHR_push_descriptor)
        return copy_buffer(src.buffer(), src.buffer_offset(), dst.buffer(), dst.buffer_offset(), src.total() * src.elemsize);

    record_type r;
    r.type = 0;
    r.copy.src = src.buffer();
    r.copy.src_offset = src.buffer_offset();
    r.copy.dst = dst.buffer();
    r.copy.dst_offset = dst.buffer_offset();
    r.copy.size = src.total() * src.elemsize;
    delayed_records.push_back(r);
}

void VkCompute::record_copy_region(const VkMat& src, const VkMat& dst, const VkBufferCopy& region)
{
    std::vector<VkBufferCopy> regions(1);
    regions[0] = region;

    record_copy_regions(src, dst, regions);
}

void VkCompute::record_copy_regions(const VkMat& src, const VkMat& dst, const std::vector<VkBufferCopy>& regions)
{
    record_prepare_transfer_barrier(src);
    record_prepare_transfer_barrier(dst);

    if (vkdev->info.support_VK_KHR_push_descriptor)
        return copy_buffer_regions(src.buffer(), dst.buffer(), regions);

    record_type r;
    r.type = 1;
    r.copy_regions.src = src.buffer();
    r.copy_regions.dst = dst.buffer();
    r.regions = regions;
    delayed_records.push_back(r);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& m)
{
    const int binding_count = bindings.size();
    for (int i=0; i<binding_count; i++)
    {
        // skip readonly weight blob
        if (bindings[i].data->state == 4)
            continue;

        record_prepare_compute_barrier(bindings[i]);
    }

    record_bind_pipeline(pipeline->pipeline);

    record_update_bindings(pipeline->pipeline_layout, pipeline->descriptorset_layout, pipeline->descriptor_update_template, bindings);

    record_push_constants(pipeline->pipeline_layout, constants);

    uint32_t group_count_xyz[3];
    group_count_xyz[0] = (m.w + pipeline->local_size_x - 1) / pipeline->local_size_x;
    group_count_xyz[1] = (m.h + pipeline->local_size_y - 1) / pipeline->local_size_y;
    group_count_xyz[2] = (m.c + pipeline->local_size_z - 1) / pipeline->local_size_z;

    record_dispatch(group_count_xyz);
}

#if NCNN_BENCHMARK
void VkCompute::record_write_timestamp(uint32_t query)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return write_timestamp(query);

    record_type r;
    r.type = 10;
    r.write_timestamp.query = query;
    delayed_records.push_back(r);
}
#endif // NCNN_BENCHMARK

void VkCompute::record_queue_transfer_acquire(const VkMat& m, uint32_t src_queue_family_index)
{
    if (queue_family_index == src_queue_family_index)
        return;

    if (vkdev->info.support_VK_KHR_push_descriptor)
        return queue_transfer_acquire_barrier(m.buffer(), m.buffer_offset(), m.total() * m.elemsize, src_queue_family_index);

    record_type r;
    r.type = 16;
    r.queue_transfer_acquire_barrier.buffer = m.buffer();
    r.queue_transfer_acquire_barrier.offset = m.buffer_offset();
    r.queue_transfer_acquire_barrier.size = m.total() * m.elemsize;
    r.queue_transfer_acquire_barrier.src_queue_family_index = src_queue_family_index;
    delayed_records.push_back(r);
}

#if __ANDROID_API__ >= 26
void VkCompute::record_import_android_hardware_buffer(const ImportAndroidHardwareBufferPipeline* pipeline, const VkImageMat& im, const VkMat& m)
{
    record_initial_image_compute_barrier(im);

    record_bind_pipeline(pipeline->pipeline);

    record_update_import_android_hardware_buffer_bindings(pipeline->pipeline_layout, pipeline->descriptorset_layout, pipeline->descriptor_update_template, pipeline->sampler, im, m);

    uint32_t group_count_xyz[3];
    group_count_xyz[0] = (m.w + 7) / 8;
    group_count_xyz[1] = (m.h + 7) / 8;
    group_count_xyz[2] = 1;

    record_dispatch(group_count_xyz);
}
#endif // __ANDROID_API__ >= 26

void VkCompute::record_bind_pipeline(VkPipeline pipeline)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return bind_pipeline(pipeline);

    record_type r;
    r.type = 2;
    r.bind_pipeline.pipeline = pipeline;
    delayed_records.push_back(r);
}

void VkCompute::record_update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const std::vector<VkMat>& bindings)
{
    const int binding_count = bindings.size();

    if (binding_count == 0)
        return;

    std::vector<VkDescriptorBufferInfo> descriptorBufferInfos(binding_count);
    for (int i=0; i<binding_count; i++)
    {
        descriptorBufferInfos[i].buffer = bindings[i].buffer();
        descriptorBufferInfos[i].offset = bindings[i].buffer_offset();
        descriptorBufferInfos[i].range = bindings[i].total() * bindings[i].elemsize;
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

//     fprintf(stderr, "update descriptorset %p\n", descriptorset);

    if (vkdev->info.support_VK_KHR_descriptor_update_template)
    {
        vkdev->vkUpdateDescriptorSetWithTemplateKHR(vkdev->vkdevice(), descriptorset, descriptor_update_template, descriptorBufferInfos.data());
    }
    else
    {
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
    }

    record_type r;
    r.type = 3;
    r.bind_descriptorset.pipeline_layout = pipeline_layout;
    r.bind_descriptorset.descriptorset = descriptorset;
    delayed_records.push_back(r);
}

void VkCompute::record_push_constants(VkPipelineLayout pipeline_layout, const std::vector<vk_constant_type>& constants)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return push_constants(pipeline_layout, constants);

    record_type r;
    r.type = 4;
    r.push_constants.pipeline_layout = pipeline_layout;
    r.constants = constants;
    delayed_records.push_back(r);
}

void VkCompute::record_dispatch(const uint32_t* group_count_xyz)
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

void VkCompute::record_transfer_compute_barrier(const VkMat& m)
{
    m.data->state = 3;

    if (vkdev->info.support_VK_KHR_push_descriptor)
        return transfer_compute_barrier(m.buffer(), m.buffer_offset(), m.total() * m.elemsize);

    record_type r;
    r.type = 6;
    r.transfer_compute_barrier.buffer = m.buffer();
    r.transfer_compute_barrier.offset = m.buffer_offset();
    r.transfer_compute_barrier.size = m.total() * m.elemsize;
    delayed_records.push_back(r);
}

void VkCompute::record_compute_transfer_barrier(const VkMat& m)
{
    m.data->state = 2;

    if (vkdev->info.support_VK_KHR_push_descriptor)
        return compute_transfer_barrier(m.buffer(), m.buffer_offset(), m.total() * m.elemsize);

    record_type r;
    r.type = 7;
    r.compute_transfer_barrier.buffer = m.buffer();
    r.compute_transfer_barrier.offset = m.buffer_offset();
    r.compute_transfer_barrier.size = m.total() * m.elemsize;
    delayed_records.push_back(r);
}

void VkCompute::record_compute_compute_barrier(const VkMat& m)
{
    m.data->state = 3;

    if (vkdev->info.support_VK_KHR_push_descriptor)
        return compute_compute_barrier(m.buffer(), m.buffer_offset(), m.total() * m.elemsize);

    record_type r;
    r.type = 8;
    r.compute_compute_barrier.buffer = m.buffer();
    r.compute_compute_barrier.offset = m.buffer_offset();
    r.compute_compute_barrier.size = m.total() * m.elemsize;
    delayed_records.push_back(r);
}

void VkCompute::record_transfer_transfer_barrier(const VkMat& m)
{
    m.data->state = 2;

    if (vkdev->info.support_VK_KHR_push_descriptor)
        return transfer_transfer_barrier(m.buffer(), m.buffer_offset(), m.total() * m.elemsize);

    record_type r;
    r.type = 9;
    r.transfer_transfer_barrier.buffer = m.buffer();
    r.transfer_transfer_barrier.offset = m.buffer_offset();
    r.transfer_transfer_barrier.size = m.total() * m.elemsize;
    delayed_records.push_back(r);
}

void VkCompute::record_host_transfer_barrier(const VkMat& m)
{
    m.data->state = 2;

    if (!m.allocator->mappable && !m.staging_data)
        return;

    VkBuffer buffer = m.allocator->mappable ? m.buffer() : m.staging_buffer();
    size_t buffer_offset = m.allocator->mappable ? m.buffer_offset() : m.staging_buffer_offset();

    if (vkdev->info.support_VK_KHR_push_descriptor)
        return host_transfer_barrier(buffer, buffer_offset, m.total() * m.elemsize);

    record_type r;
    r.type = 12;
    r.host_transfer_barrier.buffer = buffer;
    r.host_transfer_barrier.offset = buffer_offset;
    r.host_transfer_barrier.size = m.total() * m.elemsize;
    delayed_records.push_back(r);
}

void VkCompute::record_transfer_host_barrier(const VkMat& m)
{
    m.data->state = 1;

    if (!m.allocator->mappable && !m.staging_data)
        return;

    VkBuffer buffer = m.allocator->mappable ? m.buffer() : m.staging_buffer();
    size_t buffer_offset = m.allocator->mappable ? m.buffer_offset() : m.staging_buffer_offset();

    if (vkdev->info.support_VK_KHR_push_descriptor)
        return transfer_host_barrier(buffer, buffer_offset, m.total() * m.elemsize);

    record_type r;
    r.type = 13;
    r.transfer_host_barrier.buffer = buffer;
    r.transfer_host_barrier.offset = buffer_offset;
    r.transfer_host_barrier.size = m.total() * m.elemsize;
    delayed_records.push_back(r);
}

void VkCompute::record_host_compute_barrier(const VkMat& m)
{
    m.data->state = 3;

    if (!m.allocator->mappable && !m.staging_data)
        return;

    VkBuffer buffer = m.allocator->mappable ? m.buffer() : m.staging_buffer();
    size_t buffer_offset = m.allocator->mappable ? m.buffer_offset() : m.staging_buffer_offset();

    if (vkdev->info.support_VK_KHR_push_descriptor)
        return host_compute_barrier(buffer, buffer_offset, m.total() * m.elemsize);

    record_type r;
    r.type = 14;
    r.host_compute_barrier.buffer = buffer;
    r.host_compute_barrier.offset = buffer_offset;
    r.host_compute_barrier.size = m.total() * m.elemsize;
    delayed_records.push_back(r);
}

void VkCompute::record_compute_host_barrier(const VkMat& m)
{
    m.data->state = 1;

    if (!m.allocator->mappable && !m.staging_data)
        return;

    VkBuffer buffer = m.allocator->mappable ? m.buffer() : m.staging_buffer();
    size_t buffer_offset = m.allocator->mappable ? m.buffer_offset() : m.staging_buffer_offset();

    if (vkdev->info.support_VK_KHR_push_descriptor)
        return compute_host_barrier(buffer, buffer_offset, m.total() * m.elemsize);

    record_type r;
    r.type = 15;
    r.compute_host_barrier.buffer = buffer;
    r.compute_host_barrier.offset = buffer_offset;
    r.compute_host_barrier.size = m.total() * m.elemsize;
    delayed_records.push_back(r);
}

void VkCompute::record_prepare_transfer_barrier(const VkMat& m)
{
    if (m.data->state == 1)
        return record_host_transfer_barrier(m);

    if (m.data->state == 2)
        return record_transfer_transfer_barrier(m);

    if (m.data->state == 3)
        return record_compute_transfer_barrier(m);

    m.data->state = 2;
}

void VkCompute::record_prepare_compute_barrier(const VkMat& m)
{
    if (m.data->state == 1)
        return record_host_compute_barrier(m);

    if (m.data->state == 2)
        return record_transfer_compute_barrier(m);

    if (m.data->state == 3)
        return record_compute_compute_barrier(m);

    m.data->state = 3;
}

void VkCompute::record_prepare_host_barrier(const VkMat& m)
{
    if (m.data->state == 2)
        return record_transfer_host_barrier(m);

    if (m.data->state == 3)
        return record_compute_host_barrier(m);

    m.data->state = 1;
}

void VkCompute::record_initial_image_compute_barrier(const VkImageMat& im)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
        return initial_image_compute_barrier(im.image());

    record_type r;
    r.type = 11;
    r.initial_image_compute_barrier.image = im.image();
    delayed_records.push_back(r);
}

#if __ANDROID_API__ >= 26
void VkCompute::record_update_import_android_hardware_buffer_bindings(VkPipelineLayout pipeline_layout, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, VkSampler sampler, const VkImageMat& im, const VkMat& m)
{
    VkDescriptorImageInfo descriptorImageInfo;
    descriptorImageInfo.sampler = sampler;
    descriptorImageInfo.imageView = im.imageview();
    descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo descriptorBufferInfo;
    descriptorBufferInfo.buffer = m.buffer();
    descriptorBufferInfo.offset = m.buffer_offset();
    descriptorBufferInfo.range = m.total() * m.elemsize;

    if (vkdev->info.support_VK_KHR_push_descriptor)
    {
        return update_import_android_hardware_buffer_bindings(pipeline_layout, descriptor_update_template, descriptorImageInfo, descriptorBufferInfo);
    }

    // create new descriptor_pool and descriptorset
    VkDescriptorPool descriptor_pool;
    {
        VkDescriptorPoolSize poolSizes[2];
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 1;

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo;
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.pNext = 0;
        descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        descriptorPoolCreateInfo.maxSets = 1;
        descriptorPoolCreateInfo.poolSizeCount = 2;
        descriptorPoolCreateInfo.pPoolSizes = poolSizes;

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

//     fprintf(stderr, "update descriptorset %p\n", descriptorset);

    if (vkdev->info.support_VK_KHR_descriptor_update_template)
    {
        struct ImportAndroidHardwareBufferDescriptorInfo
        {
            VkDescriptorImageInfo imageInfo;
            VkDescriptorBufferInfo bufferInfo;
            VkDescriptorBufferInfo buffer4Info;
        };

        ImportAndroidHardwareBufferDescriptorInfo info;
        info.imageInfo = descriptorImageInfo;
        info.bufferInfo = descriptorBufferInfo;
        info.buffer4Info = descriptorBufferInfo;

        vkdev->vkUpdateDescriptorSetWithTemplateKHR(vkdev->vkdevice(), descriptorset, descriptor_update_template, &info);
    }
    else
    {
        VkWriteDescriptorSet writeDescriptorSets[3];
        writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[0].pNext = 0;
        writeDescriptorSets[0].dstSet = descriptorset;
        writeDescriptorSets[0].dstBinding = 0;
        writeDescriptorSets[0].dstArrayElement = 0;
        writeDescriptorSets[0].descriptorCount = 1;
        writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writeDescriptorSets[0].pImageInfo = &descriptorImageInfo;
        writeDescriptorSets[0].pBufferInfo = 0;
        writeDescriptorSets[0].pTexelBufferView = 0;
        writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[1].pNext = 0;
        writeDescriptorSets[1].dstSet = descriptorset;
        writeDescriptorSets[1].dstBinding = 1;
        writeDescriptorSets[1].dstArrayElement = 0;
        writeDescriptorSets[1].descriptorCount = 1;
        writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSets[1].pImageInfo = 0;
        writeDescriptorSets[1].pBufferInfo = &descriptorBufferInfo;
        writeDescriptorSets[1].pTexelBufferView = 0;
        writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[2].pNext = 0;
        writeDescriptorSets[2].dstSet = descriptorset;
        writeDescriptorSets[2].dstBinding = 2;
        writeDescriptorSets[2].dstArrayElement = 0;
        writeDescriptorSets[2].descriptorCount = 1;
        writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSets[2].pImageInfo = 0;
        writeDescriptorSets[2].pBufferInfo = &descriptorBufferInfo;
        writeDescriptorSets[2].pTexelBufferView = 0;

        vkUpdateDescriptorSets(vkdev->vkdevice(), 3, writeDescriptorSets, 0, 0);
    }

    record_type r;
    r.type = 3;
    r.bind_descriptorset.pipeline_layout = pipeline_layout;
    r.bind_descriptorset.descriptorset = descriptorset;
    delayed_records.push_back(r);
}
#endif // __ANDROID_API__ >= 26

int VkCompute::submit_and_wait()
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
    {
        end_command_buffer();

        return queue_submit_and_wait_fence();
    }

    begin_command_buffer();

#if NCNN_BENCHMARK
    reset_query_pool();
#endif // NCNN_BENCHMARK

    // handle delayed records
    for (size_t i=0; i<delayed_records.size(); i++)
    {
        const record_type& r = delayed_records[i];

        switch (r.type)
        {
        case 0:
            copy_buffer(r.copy.src, r.copy.src_offset, r.copy.dst, r.copy.dst_offset, r.copy.size);
            break;
        case 1:
            copy_buffer_regions(r.copy_regions.src, r.copy_regions.dst, r.regions);
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
            transfer_compute_barrier(r.transfer_compute_barrier.buffer, r.transfer_compute_barrier.offset, r.transfer_compute_barrier.size);
            break;
        case 7:
            compute_transfer_barrier(r.compute_transfer_barrier.buffer, r.compute_transfer_barrier.offset, r.compute_transfer_barrier.size);
            break;
        case 8:
            compute_compute_barrier(r.compute_compute_barrier.buffer, r.compute_compute_barrier.offset, r.compute_compute_barrier.size);
            break;
        case 9:
            transfer_transfer_barrier(r.compute_compute_barrier.buffer, r.compute_compute_barrier.offset, r.compute_compute_barrier.size);
            break;
#if NCNN_BENCHMARK
        case 10:
            write_timestamp(r.write_timestamp.query);
            break;
#endif // NCNN_BENCHMARK
        case 11:
            initial_image_compute_barrier(r.initial_image_compute_barrier.image);
            break;
        case 12:
            host_transfer_barrier(r.host_transfer_barrier.buffer, r.host_transfer_barrier.offset, r.host_transfer_barrier.size);
            break;
        case 13:
            transfer_host_barrier(r.transfer_host_barrier.buffer, r.transfer_host_barrier.offset, r.transfer_host_barrier.size);
            break;
        case 14:
            host_compute_barrier(r.host_compute_barrier.buffer, r.host_compute_barrier.offset, r.host_compute_barrier.size);
            break;
        case 15:
            compute_host_barrier(r.compute_host_barrier.buffer, r.compute_host_barrier.offset, r.compute_host_barrier.size);
            break;
        case 16:
            queue_transfer_acquire_barrier(r.queue_transfer_acquire_barrier.buffer, r.queue_transfer_acquire_barrier.offset, r.queue_transfer_acquire_barrier.size, r.queue_transfer_acquire_barrier.src_queue_family_index);
            break;
        }
    }

    end_command_buffer();

    delayed_records.clear();

    return queue_submit_and_wait_fence();
}

int VkCompute::reset()
{
//     fprintf(stderr, "cmd reset\n");

    VkResult ret = vkResetCommandBuffer(command_buffer, 0);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkResetCommandBuffer failed %d\n", ret);
        return -1;
    }

    ret = vkResetFences(vkdev->vkdevice(), 1, &fence);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkResetFences failed %d\n", ret);
        return -1;
    }

    if (vkdev->info.support_VK_KHR_push_descriptor)
    {
        begin_command_buffer();

#if NCNN_BENCHMARK
        reset_query_pool();
#endif // NCNN_BENCHMARK
    }

    return 0;
}

#if NCNN_BENCHMARK
int VkCompute::create_query_pool(uint32_t _query_count)
{
    query_count = _query_count;

    VkQueryPoolCreateInfo queryPoolCreateInfo;
    queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryPoolCreateInfo.pNext = 0;
    queryPoolCreateInfo.flags = 0;
    queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolCreateInfo.queryCount = query_count;
    queryPoolCreateInfo.pipelineStatistics = 0;

    VkResult ret = vkCreateQueryPool(vkdev->vkdevice(), &queryPoolCreateInfo, 0, &query_pool);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateQueryPool failed %d\n", ret);
        return -1;
    }

    if (vkdev->info.support_VK_KHR_push_descriptor)
    {
        reset_query_pool();
    }

    return 0;
}

int VkCompute::get_query_pool_results(uint32_t first_query, uint32_t query_count, std::vector<uint64_t>& results)
{
    if (results.size() < first_query + query_count)
    {
        fprintf(stderr, "results not large enough\n");
        return -1;
    }

    VkResult ret = vkGetQueryPoolResults(vkdev->vkdevice(), query_pool, first_query, query_count,
                                         query_count * sizeof(uint64_t), results.data() + first_query, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (ret != VK_SUCCESS && ret != VK_NOT_READY)
    {
        fprintf(stderr, "vkGetQueryPoolResults failed %d\n", ret);
        return -1;
    }

    return 0;
}
#endif // NCNN_BENCHMARK

void VkCompute::copy_buffer(VkBuffer src, size_t src_offset, VkBuffer dst, size_t dst_offset, size_t size)
{
//     fprintf(stderr, "cmd copy %p[+%lu] to %p[+%lu] %lu\n", src, src_offset, dst, dst_offset, size);

    VkBufferCopy region;
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = size;

    vkCmdCopyBuffer(command_buffer, src, dst, 1, &region);
}

void VkCompute::copy_buffer_regions(VkBuffer src, VkBuffer dst, const std::vector<VkBufferCopy>& regions)
{
//     fprintf(stderr, "cmd copy regions %p to %p\n", src, dst);

    vkCmdCopyBuffer(command_buffer, src, dst, regions.size(), regions.data());
}

void VkCompute::bind_pipeline(VkPipeline pipeline)
{
//     fprintf(stderr, "cmd bind_pipeline %p\n", pipeline);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
}

void VkCompute::bind_descriptorset(VkPipelineLayout pipeline_layout, VkDescriptorSet descriptorset)
{
//     fprintf(stderr, "cmd bind_descriptorset %p %p\n", pipeline_layout, descriptorset);

    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &descriptorset, 0, 0);
}

void VkCompute::update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const std::vector<VkDescriptorBufferInfo>& descriptorBufferInfos)
{
//     fprintf(stderr, "cmd update_bindings %p %p\n", pipeline_layout, descriptor_update_template);

    vkdev->vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, descriptor_update_template, pipeline_layout, 0, descriptorBufferInfos.data());
}

void VkCompute::push_constants(VkPipelineLayout pipeline_layout, const std::vector<vk_constant_type>& constants)
{
//     fprintf(stderr, "cmd push_constants %p\n", pipeline_layout);

    vkCmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constants.size() * sizeof(vk_constant_type), constants.data());
}

void VkCompute::dispatch(const uint32_t* group_count_xyz)
{
//     fprintf(stderr, "cmd dispatch %d %d %d\n", group_count_xyz[0], group_count_xyz[1], group_count_xyz[2]);

    vkCmdDispatch(command_buffer, group_count_xyz[0], group_count_xyz[1], group_count_xyz[2]);
}

void VkCompute::transfer_compute_barrier(VkBuffer buffer, size_t offset, size_t size)
{
//     fprintf(stderr, "cmd transfer_compute_barrier %p[+%lu] %lu\n", buffer, offset, size);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = offset;
    bufferBarrier.size = size;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

void VkCompute::compute_transfer_barrier(VkBuffer buffer, size_t offset, size_t size)
{
//     fprintf(stderr, "cmd compute_transfer_barrier %p[+%lu] %lu\n", buffer, offset, size);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = offset;
    bufferBarrier.size = size;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

void VkCompute::compute_compute_barrier(VkBuffer buffer, size_t offset, size_t size)
{
//     fprintf(stderr, "cmd compute_compute_barrier %p[+%lu] %lu\n", buffer, offset, size);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = offset;
    bufferBarrier.size = size;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

void VkCompute::transfer_transfer_barrier(VkBuffer buffer, size_t offset, size_t size)
{
//     fprintf(stderr, "cmd transfer_transfer_barrier %p[+%lu] %lu\n", buffer, offset, size);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = offset;
    bufferBarrier.size = size;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

void VkCompute::host_transfer_barrier(VkBuffer buffer, size_t offset, size_t size)
{
//     fprintf(stderr, "cmd host_transfer_barrier %p[+%lu] %lu\n", buffer, offset, size);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = offset;
    bufferBarrier.size = size;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_HOST_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

void VkCompute::transfer_host_barrier(VkBuffer buffer, size_t offset, size_t size)
{
//     fprintf(stderr, "cmd transfer_host_barrier %p[+%lu] %lu\n", buffer, offset, size);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = offset;
    bufferBarrier.size = size;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_HOST_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

void VkCompute::host_compute_barrier(VkBuffer buffer, size_t offset, size_t size)
{
//     fprintf(stderr, "cmd host_compute_barrier %p[+%lu] %lu\n", buffer, offset, size);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = offset;
    bufferBarrier.size = size;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_HOST_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

void VkCompute::compute_host_barrier(VkBuffer buffer, size_t offset, size_t size)
{
//     fprintf(stderr, "cmd compute_host_barrier %p[+%lu] %lu\n", buffer, offset, size);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bufferBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = offset;
    bufferBarrier.size = size;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_HOST_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

void VkCompute::queue_transfer_acquire_barrier(VkBuffer buffer, size_t offset, size_t size, uint32_t src_queue_family_index)
{
//     fprintf(stderr, "cmd queue_transfer_acquire_barrier %p[+%lu] %lu   %lu -> %lu\n", buffer, offset, size, src_queue_family_index, queue_family_index);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = 0;
    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    bufferBarrier.srcQueueFamilyIndex = src_queue_family_index;
    bufferBarrier.dstQueueFamilyIndex = queue_family_index;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = offset;
    bufferBarrier.size = size;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

void VkCompute::initial_image_compute_barrier(VkImage image)
{
//     fprintf(stderr, "cmd initial_image_compute_barrier %p %lu %lu\n", image, oldlayout, newlayout);

    VkImageMemoryBarrier imageBarrier;
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.pNext = 0;
    imageBarrier.srcAccessMask = 0;
    imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageBarrier.image = image;
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 0, 0, 1, &imageBarrier);
}

#if __ANDROID_API__ >= 26
void VkCompute::update_import_android_hardware_buffer_bindings(VkPipelineLayout pipeline_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const VkDescriptorImageInfo& descriptorImageInfo, const VkDescriptorBufferInfo& descriptorBufferInfo)
{
    struct ImportAndroidHardwareBufferDescriptorInfo
    {
        VkDescriptorImageInfo imageInfo;
        VkDescriptorBufferInfo bufferInfo;
        VkDescriptorBufferInfo buffer4Info;
    };

    ImportAndroidHardwareBufferDescriptorInfo info;
    info.imageInfo = descriptorImageInfo;
    info.bufferInfo = descriptorBufferInfo;
    info.buffer4Info = descriptorBufferInfo;

    vkdev->vkCmdPushDescriptorSetWithTemplateKHR(command_buffer, descriptor_update_template, pipeline_layout, 0, &info);
}
#endif // __ANDROID_API__ >= 26

#if NCNN_BENCHMARK
void VkCompute::reset_query_pool()
{
//     fprintf(stderr, "cmd reset_query_pool\n");

    if (query_pool)
        vkCmdResetQueryPool(command_buffer, query_pool, 0, query_count);
}

void VkCompute::write_timestamp(uint32_t query)
{
//     fprintf(stderr, "cmd write_timestamp %u\n", query);

    if (query_pool)
        vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query);
}
#endif // NCNN_BENCHMARK

VkTransfer::VkTransfer(const VulkanDevice* _vkdev) : Command(_vkdev, _vkdev->info.transfer_queue_family_index)
{
    buffer_offset_alignment = vkdev->info.buffer_offset_alignment;
    staging_data = 0;
}

VkTransfer::~VkTransfer()
{
}

void VkTransfer::record_upload(const Mat& src, VkMat& dst, const Option& opt)
{
    if (src.elemsize / src.elempack == 4)
    {
        if (opt.use_fp16_storage || (opt.use_fp16_packed && src.elempack % 4 == 0))
        {
            Mat src_fp16;
            cast_float32_to_float16(src, src_fp16);

            record_upload(src_fp16, dst, opt);

            return;
        }
    }

    Mat src_flattened = src.reshape(src.w * src.h * src.c);

    dst.create_like(src_flattened, weight_vkallocator, staging_vkallocator);

    // set weight blob as readonly
    dst.data->state = 4;

    // we can skip queue transfer and staging buffer allocation
    // only on unified memory architecture and unified compute/transfer queue
    // which is usually the case on integrated gpu / cpu
    if (dst.allocator->mappable && queue_family_index == vkdev->info.compute_queue_family_index)
    {
        dst.upload(src_flattened);

        return;
    }

    record_type r;
    r.size = src_flattened.total() * src_flattened.elemsize;
    r.mat = src_flattened;
    r.vkmat = dst;
    delayed_records.push_back(r);
}

int VkTransfer::submit_and_wait()
{
    if (delayed_records.empty())
        return 0;

    int transfer_count = delayed_records.size();

    // solve staging buffer size
    size_t staging_buffer_size = 0;
    for (int i=0; i<transfer_count; i++)
    {
        const record_type& r = delayed_records[i];
        staging_buffer_size += alignSize(r.size, buffer_offset_alignment);
    }

    // allocate staging buffer
    staging_data = staging_vkallocator->fastMalloc(staging_buffer_size);

    // copy upload data
    size_t mapped_ptr_offset = 0;
    for (int i=0; i<transfer_count; i++)
    {
        const record_type& r = delayed_records[i];

        memcpy((unsigned char*)staging_data->mapped_ptr + mapped_ptr_offset, r.mat.data, r.size);

        mapped_ptr_offset += alignSize(r.size, buffer_offset_alignment);
    }

    staging_vkallocator->flush(staging_data);

    begin_command_buffer();

//     fprintf(stderr, "cmd transfer %p %lu\n", staging_data->buffer, staging_buffer_size);

    // handle delayed records
    size_t staging_buffer_offset = 0;
    for (int i=0; i<transfer_count; i++)
    {
        const record_type& r = delayed_records[i];

        copy_buffer(staging_data->buffer, staging_buffer_offset, r.vkmat.buffer(), r.vkmat.buffer_offset(), r.size);

        staging_buffer_offset += alignSize(r.size, buffer_offset_alignment);
    }

    // owner transfer release
    for (int i=0; i<transfer_count; i++)
    {
        const record_type& r = delayed_records[i];

        queue_transfer_release_barrier(r.vkmat.buffer(), r.vkmat.buffer_offset(), r.size, vkdev->info.compute_queue_family_index);
    }

    end_command_buffer();

    int ret = queue_submit_and_wait_fence();

    // compute queue owner transfer acquire
    {
        VkCompute cmd(vkdev);

        for (int i=0; i<transfer_count; i++)
        {
            const record_type& r = delayed_records[i];

            cmd.record_queue_transfer_acquire(r.vkmat, queue_family_index);
        }

        cmd.submit_and_wait();
    }

    // deallocate staging buffer
    staging_vkallocator->fastFree(staging_data);
    staging_data = 0;

    delayed_records.clear();

    return ret;
}

void VkTransfer::copy_buffer(VkBuffer src, size_t src_offset, VkBuffer dst, size_t dst_offset, size_t size)
{
//     fprintf(stderr, "cmd copy %p to %p\n", src, dst);

    VkBufferCopy region;
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = size;

    vkCmdCopyBuffer(command_buffer, src, dst, 1, &region);
}

void VkTransfer::copy_buffer_regions(VkBuffer src, VkBuffer dst, const std::vector<VkBufferCopy>& regions)
{
//     fprintf(stderr, "cmd copy regions %p to %p\n", src, dst);

    vkCmdCopyBuffer(command_buffer, src, dst, regions.size(), regions.data());
}

void VkTransfer::queue_transfer_release_barrier(VkBuffer buffer, size_t offset, size_t size, uint32_t dst_queue_family_index)
{
//     fprintf(stderr, "cmd queue_transfer_release_barrier %p[+%lu] %lu   %lu -> %lu\n", buffer, offset, size, queue_family_index, dst_queue_family_index);

    VkBufferMemoryBarrier bufferBarrier;
    bufferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bufferBarrier.pNext = 0;
    bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bufferBarrier.dstAccessMask = 0;
    bufferBarrier.srcQueueFamilyIndex = queue_family_index;
    bufferBarrier.dstQueueFamilyIndex = dst_queue_family_index;
    bufferBarrier.buffer = buffer;
    bufferBarrier.offset = offset;
    bufferBarrier.size = size;

    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

    vkCmdPipelineBarrier(command_buffer, srcStageMask, dstStageMask, 0, 0, 0, 1, &bufferBarrier, 0, 0);
}

} // namespace ncnn

#endif // NCNN_VULKAN
