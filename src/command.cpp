// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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
#include <algorithm>
#include "option.h"
#include "pipeline.h"

namespace ncnn {

VkCompute::VkCompute(const VulkanDevice* _vkdev) : vkdev(_vkdev)
{
    compute_command_pool = 0;
    compute_command_buffer = 0;
    compute_command_fence = 0;

#if NCNN_BENCHMARK
    query_count = 0;
    query_pool = 0;
#endif // NCNN_BENCHMARK

    init();
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
        vkResetCommandBuffer(compute_command_buffer, 0);

        vkDestroyQueryPool(vkdev->vkdevice(), query_pool, 0);
    }
#endif // NCNN_BENCHMARK

    vkDestroyFence(vkdev->vkdevice(), compute_command_fence, 0);

    vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &compute_command_buffer);
    vkDestroyCommandPool(vkdev->vkdevice(), compute_command_pool, 0);
}

void VkCompute::record_upload(const Mat& src, VkMat& dst, const Option& opt)
{
//     fprintf(stderr, "record_upload\n");

    // create dst
    dst.create_like(src, opt.blob_vkallocator);

    if (dst.allocator->mappable)
    {
        // memcpy src to device
        memcpy(dst.mapped_ptr(), src.data, src.total() * src.elemsize);
        dst.allocator->flush(dst.data);

        // mark device host-write @ null
        dst.data->access_flags = VK_ACCESS_HOST_WRITE_BIT;
        dst.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;

        return;
    }

    // create staging
    VkMat dst_staging;
    dst_staging.create_like(src, opt.staging_vkallocator);

    // memcpy src to staging
    memcpy(dst_staging.mapped_ptr(), src.data, src.total() * src.elemsize);
    dst_staging.allocator->flush(dst_staging.data);

    // barrier staging host-write @ null to transfer-read @ compute
    {
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = dst_staging.buffer();
        barriers[0].offset = dst_staging.buffer_offset();
        barriers[0].size = dst_staging.buffer_capacity();

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            record r;
            r.type = record::TYPE_buffer_barrers;
            r.command_buffer = compute_command_buffer;
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            delayed_records.push_back(r);
        }
    }

    // record staging to device
    {
        VkBufferCopy* regions = new VkBufferCopy[1];
        regions[0].srcOffset = dst_staging.buffer_offset();
        regions[0].dstOffset = dst.buffer_offset();
        regions[0].size = std::min(dst_staging.buffer_capacity(), dst.buffer_capacity());

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdCopyBuffer(compute_command_buffer, dst_staging.buffer(), dst.buffer(), 1, regions);
            delete[] regions;
        }
        else
        {
            record r;
            r.type = record::TYPE_copy_buffer;
            r.command_buffer = compute_command_buffer;
            r.copy_buffer.src = dst_staging.buffer();
            r.copy_buffer.dst = dst.buffer();
            r.copy_buffer.region_count = 1;
            r.copy_buffer.regions = regions;
            delayed_records.push_back(r);
        }
    }

    // mark device transfer-write @ queue
    dst.data->access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
    dst.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;

    // stash staging
    upload_staging_buffers.push_back(dst_staging);
}

void VkCompute::record_download(const VkMat& src, Mat& dst, const Option& opt)
{
//     fprintf(stderr, "record_download\n");

    // create dst
    dst.create_like(src, opt.blob_allocator);

    if (src.allocator->mappable)
    {
        // barrier device any @ compute to host-read @ compute
        if (src.data->access_flags != VK_ACCESS_HOST_READ_BIT || src.data->stage_flags != VK_PIPELINE_STAGE_HOST_BIT)
        {
            VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
            barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barriers[0].pNext = 0;
            barriers[0].srcAccessMask = src.data->access_flags;
            barriers[0].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].buffer = src.buffer();
            barriers[0].offset = src.buffer_offset();
            barriers[0].size = src.buffer_capacity();

            VkPipelineStageFlags src_stage = src.data->stage_flags;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_HOST_BIT;

            if (vkdev->info.support_VK_KHR_push_descriptor)
            {
                vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
                delete[] barriers;
            }
            else
            {
                record r;
                r.type = record::TYPE_buffer_barrers;
                r.command_buffer = compute_command_buffer;
                r.buffer_barrers.src_stage = src_stage;
                r.buffer_barrers.dst_stage = dst_stage;
                r.buffer_barrers.barrier_count = 1;
                r.buffer_barrers.barriers = barriers;
                delayed_records.push_back(r);
            }

            // mark device host-read @ any
            src.data->access_flags = VK_ACCESS_HOST_READ_BIT;
            src.data->stage_flags = VK_PIPELINE_STAGE_HOST_BIT;
        }

        // stash download post buffer and mat
        download_post_buffers.push_back(src);
        download_post_mats.push_back(dst);

        // post memcpy device to dst
        {
            record r;
            r.type = record::TYPE_post_download;
            r.command_buffer = 0;
            r.post_download.download_post_buffer_mat_offset = download_post_buffers.size() - 1;
            delayed_records.push_back(r);
        }

        return;
    }

    if (src.data->access_flags != VK_ACCESS_TRANSFER_READ_BIT || src.data->stage_flags != VK_PIPELINE_STAGE_TRANSFER_BIT)
    {
        // barrier device any @ compute to transfer-read @ compute
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = src.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = src.buffer();
        barriers[0].offset = src.buffer_offset();
        barriers[0].size = src.buffer_capacity();

        VkPipelineStageFlags src_stage = src.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            record r;
            r.type = record::TYPE_buffer_barrers;
            r.command_buffer = compute_command_buffer;
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            delayed_records.push_back(r);
        }

        // mark device transfer-read @ transfer
        src.data->access_flags = VK_ACCESS_TRANSFER_READ_BIT;
        src.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // create staging
    VkMat src_staging;
    src_staging.create_like(src, opt.staging_vkallocator);

    // record device to staging
    {
        VkBufferCopy* regions = new VkBufferCopy[1];
        regions[0].srcOffset = src.buffer_offset();
        regions[0].dstOffset = src_staging.buffer_offset();
        regions[0].size = std::min(src.buffer_capacity(), src_staging.buffer_capacity());

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdCopyBuffer(compute_command_buffer, src.buffer(), src_staging.buffer(), 1, regions);
            delete[] regions;
        }
        else
        {
            record r;
            r.type = record::TYPE_copy_buffer;
            r.command_buffer = compute_command_buffer;
            r.copy_buffer.src = src.buffer();
            r.copy_buffer.dst = src_staging.buffer();
            r.copy_buffer.region_count = 1;
            r.copy_buffer.regions = regions;
            delayed_records.push_back(r);
        }
    }

    // barrier staging transfer-write @ compute to host-read @ compute
    {
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barriers[0].dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = src_staging.buffer();
        barriers[0].offset = src_staging.buffer_offset();
        barriers[0].size = src_staging.buffer_capacity();

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_HOST_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            record r;
            r.type = record::TYPE_buffer_barrers;
            r.command_buffer = compute_command_buffer;
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            delayed_records.push_back(r);
        }
    }

    // stash download post buffer and mat
    download_post_buffers.push_back(src_staging);
    download_post_mats.push_back(dst);

    // post memcpy device to dst
    {
        record r;
        r.type = record::TYPE_post_download;
        r.command_buffer = 0;
        r.post_download.download_post_buffer_mat_offset = download_post_buffers.size() - 1;
        delayed_records.push_back(r);
    }
}

void VkCompute::record_clone(const VkMat& src, VkMat& dst, const Option& opt)
{
//     fprintf(stderr, "record_clone\n");

    if (src.data->access_flags != VK_ACCESS_TRANSFER_READ_BIT || src.data->stage_flags != VK_PIPELINE_STAGE_TRANSFER_BIT)
    {
        // barrier device any @ compute to transfer-read @ compute
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = src.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = src.buffer();
        barriers[0].offset = src.buffer_offset();
        barriers[0].size = src.buffer_capacity();

        VkPipelineStageFlags src_stage = src.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            record r;
            r.type = record::TYPE_buffer_barrers;
            r.command_buffer = compute_command_buffer;
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            delayed_records.push_back(r);
        }

        // mark device transfer-read @ transfer
        src.data->access_flags = VK_ACCESS_TRANSFER_READ_BIT;
        src.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    // create dst
    dst.create_like(src, opt.blob_vkallocator);

    // record device to staging
    {
        VkBufferCopy* regions = new VkBufferCopy[1];
        regions[0].srcOffset = src.buffer_offset();
        regions[0].dstOffset = dst.buffer_offset();
        regions[0].size = std::min(src.buffer_capacity(), dst.buffer_capacity());

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdCopyBuffer(compute_command_buffer, src.buffer(), dst.buffer(), 1, regions);
            delete[] regions;
        }
        else
        {
            record r;
            r.type = record::TYPE_copy_buffer;
            r.command_buffer = compute_command_buffer;
            r.copy_buffer.src = src.buffer();
            r.copy_buffer.dst = dst.buffer();
            r.copy_buffer.region_count = 1;
            r.copy_buffer.regions = regions;
            delayed_records.push_back(r);
        }
    }

    // mark device transfer-write @ transfer
    dst.data->access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
    dst.data->stage_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher)
{
//     fprintf(stderr, "record_pipeline %p\n", pipeline);

    const size_t binding_count = bindings.size();
    const size_t constant_count = constants.size();

    for (size_t i=0; i<binding_count; i++)
    {
        const VkMat& binding = bindings[i];

        if (binding.data->access_flags & VK_ACCESS_SHADER_WRITE_BIT || binding.data->stage_flags != VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
        {
            // barrier device any @ compute/null to shader-readwrite @ compute
            VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
            barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barriers[0].pNext = 0;
            barriers[0].srcAccessMask = binding.data->access_flags;
            barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[0].buffer = binding.buffer();
            barriers[0].offset = binding.buffer_offset();
            barriers[0].size = binding.buffer_capacity();

            VkPipelineStageFlags src_stage = binding.data->stage_flags;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            if (vkdev->info.support_VK_KHR_push_descriptor)
            {
                vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
                delete[] barriers;
            }
            else
            {
                record r;
                r.type = record::TYPE_buffer_barrers;
                r.command_buffer = compute_command_buffer;
                r.buffer_barrers.src_stage = src_stage;
                r.buffer_barrers.dst_stage = dst_stage;
                r.buffer_barrers.barrier_count = 1;
                r.buffer_barrers.barriers = barriers;
                delayed_records.push_back(r);
            }

            // mark device shader-readwrite @ compute
            binding.data->access_flags = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            binding.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        }
    }

    // record bind pipeline
    {
        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdBindPipeline(compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
        }
        else
        {
            record r;
            r.type = record::TYPE_bind_pipeline;
            r.command_buffer = compute_command_buffer;
            r.bind_pipeline.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            r.bind_pipeline.pipeline = pipeline->pipeline;
            delayed_records.push_back(r);
        }
    }

    // record update bindings
    if (binding_count > 0)
    {
        std::vector<VkDescriptorBufferInfo> descriptorBufferInfos(binding_count);
        for (size_t i=0; i<binding_count; i++)
        {
            descriptorBufferInfos[i].buffer = bindings[i].buffer();
            descriptorBufferInfos[i].offset = bindings[i].buffer_offset();
            descriptorBufferInfos[i].range = bindings[i].total() * bindings[i].elemsize;
        }

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkdev->vkCmdPushDescriptorSetWithTemplateKHR(compute_command_buffer, pipeline->descriptor_update_template, pipeline->pipeline_layout, 0, descriptorBufferInfos.data());
        }
        else
        {
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
                descriptorSetAllocateInfo.pSetLayouts = &pipeline->descriptorset_layout;

                VkResult ret = vkAllocateDescriptorSets(vkdev->vkdevice(), &descriptorSetAllocateInfo, &descriptorset);
                if (ret != VK_SUCCESS)
                {
                    fprintf(stderr, "vkAllocateDescriptorSets failed %d\n", ret);
                    return;
                }
            }
            descriptorsets.push_back(descriptorset);

            if (vkdev->info.support_VK_KHR_descriptor_update_template)
            {
                vkdev->vkUpdateDescriptorSetWithTemplateKHR(vkdev->vkdevice(), descriptorset, pipeline->descriptor_update_template, descriptorBufferInfos.data());
            }
            else
            {
                std::vector<VkWriteDescriptorSet> writeDescriptorSets(binding_count);
                for (size_t i=0; i<binding_count; i++)
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

            record r;
            r.type = record::TYPE_bind_descriptorsets;
            r.command_buffer = compute_command_buffer;
            r.bind_descriptorsets.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            r.bind_descriptorsets.pipeline_layout = pipeline->pipeline_layout;
            r.bind_descriptorsets.descriptorset_count = 1;
            r.bind_descriptorsets.descriptorset_offset = descriptorsets.size() - 1;
            delayed_records.push_back(r);
        }
    }

    // record push constants
    if (constant_count > 0)
    {
        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdPushConstants(compute_command_buffer, pipeline->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, constant_count * sizeof(vk_constant_type), constants.data());
        }
        else
        {
            uint32_t size = constant_count * sizeof(vk_constant_type);
            unsigned char* constant_values = new unsigned char[size];
            memcpy(constant_values, constants.data(), size);

            record r;
            r.type = record::TYPE_push_constants;
            r.command_buffer = compute_command_buffer;
            r.push_constants.pipeline_layout = pipeline->pipeline_layout;
            r.push_constants.stage_flags = VK_SHADER_STAGE_COMPUTE_BIT;
            r.push_constants.size = size;
            r.push_constants.values = constant_values;
            delayed_records.push_back(r);
        }
    }

    // record dispatch
    {
        uint32_t group_count_x = (dispatcher.w + pipeline->local_size_x - 1) / pipeline->local_size_x;
        uint32_t group_count_y = (dispatcher.h + pipeline->local_size_y - 1) / pipeline->local_size_y;
        uint32_t group_count_z = (dispatcher.c + pipeline->local_size_z - 1) / pipeline->local_size_z;

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdDispatch(compute_command_buffer, group_count_x, group_count_y, group_count_z);
        }
        else
        {
            record r;
            r.type = record::TYPE_dispatch;
            r.command_buffer = compute_command_buffer;
            r.dispatch.group_count_x = group_count_x;
            r.dispatch.group_count_y = group_count_y;
            r.dispatch.group_count_z = group_count_z;
            delayed_records.push_back(r);
        }
    }
}

#if NCNN_BENCHMARK
void VkCompute::record_write_timestamp(uint32_t query)
{
    if (vkdev->info.support_VK_KHR_push_descriptor)
    {
        if (query_pool)
            vkCmdWriteTimestamp(compute_command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, query);
    }
    else
    {
        record r;
        r.type = record::TYPE_write_timestamp;
        r.command_buffer = compute_command_buffer;
        r.write_timestamp.query = query;
        delayed_records.push_back(r);
    }
}
#endif // NCNN_BENCHMARK

#if __ANDROID_API__ >= 26
void VkCompute::record_import_android_hardware_buffer(const ImportAndroidHardwareBufferPipeline* pipeline, const VkImageMat& src, const VkMat& dst)
{
    // image layout transform undefined @ null to general @ compute
    {
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = 0;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = src.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            record r;
            r.type = record::TYPE_image_barrers;
            r.command_buffer = compute_command_buffer;
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            delayed_records.push_back(r);
        }
    }

    // record bind pipeline
    {
        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdBindPipeline(compute_command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline);
        }
        else
        {
            record r;
            r.type = record::TYPE_bind_pipeline;
            r.command_buffer = compute_command_buffer;
            r.bind_pipeline.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            r.bind_pipeline.pipeline = pipeline->pipeline;
            delayed_records.push_back(r);
        }
    }

    // record update bindings
    {
        VkDescriptorImageInfo descriptorImageInfo;
        descriptorImageInfo.sampler = pipeline->sampler;
        descriptorImageInfo.imageView = src.imageview();
        descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo descriptorBufferInfo;
        descriptorBufferInfo.buffer = dst.buffer();
        descriptorBufferInfo.offset = dst.buffer_offset();
        descriptorBufferInfo.range = dst.total() * dst.elemsize;

        if (vkdev->info.support_VK_KHR_push_descriptor)
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

            vkdev->vkCmdPushDescriptorSetWithTemplateKHR(compute_command_buffer, pipeline->descriptor_update_template, pipeline->pipeline_layout, 0, &info);
        }
        else
        {
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
                descriptorSetAllocateInfo.pSetLayouts = &pipeline->descriptorset_layout;

                VkResult ret = vkAllocateDescriptorSets(vkdev->vkdevice(), &descriptorSetAllocateInfo, &descriptorset);
                if (ret != VK_SUCCESS)
                {
                    fprintf(stderr, "vkAllocateDescriptorSets failed %d\n", ret);
                    return;
                }
            }
            descriptorsets.push_back(descriptorset);

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

                vkdev->vkUpdateDescriptorSetWithTemplateKHR(vkdev->vkdevice(), descriptorset, pipeline->descriptor_update_template, &info);
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

            record r;
            r.type = record::TYPE_bind_descriptorsets;
            r.command_buffer = compute_command_buffer;
            r.bind_descriptorsets.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
            r.bind_descriptorsets.pipeline_layout = pipeline->pipeline_layout;
            r.bind_descriptorsets.descriptorset_count = 1;
            r.bind_descriptorsets.descriptorset_offset = descriptorsets.size() - 1;
            delayed_records.push_back(r);
        }
    }

    // record dispatch
    {
        uint32_t group_count_x = (dst.w + pipeline->local_size_x - 1) / pipeline->local_size_x;
        uint32_t group_count_y = (dst.h + pipeline->local_size_y - 1) / pipeline->local_size_y;
        uint32_t group_count_z = (dst.c + pipeline->local_size_z - 1) / pipeline->local_size_z;

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdDispatch(compute_command_buffer, group_count_x, group_count_y, group_count_z);
        }
        else
        {
            record r;
            r.type = record::TYPE_dispatch;
            r.command_buffer = compute_command_buffer;
            r.dispatch.group_count_x = group_count_x;
            r.dispatch.group_count_y = group_count_y;
            r.dispatch.group_count_z = group_count_z;
            delayed_records.push_back(r);
        }
    }
}
#endif // __ANDROID_API__ >= 26

int VkCompute::submit_and_wait()
{
//     fprintf(stderr, "submit_and_wait\n");

    if (!vkdev->info.support_VK_KHR_push_descriptor)
    {
        begin_command_buffer();

#if NCNN_BENCHMARK
        if (query_pool)
            vkCmdResetQueryPool(compute_command_buffer, query_pool, 0, query_count);
#endif // NCNN_BENCHMARK

        const size_t record_count = delayed_records.size();

        // handle delayed records
        for (size_t i=0; i<record_count; i++)
        {
            const record& r = delayed_records[i];

            switch (r.type)
            {
            case record::TYPE_copy_buffer:
            {
                vkCmdCopyBuffer(r.command_buffer, r.copy_buffer.src, r.copy_buffer.dst, r.copy_buffer.region_count, r.copy_buffer.regions);
                delete[] r.copy_buffer.regions;
                break;
            }
            case record::TYPE_bind_pipeline:
            {
                vkCmdBindPipeline(r.command_buffer, r.bind_pipeline.bind_point, r.bind_pipeline.pipeline);
                break;
            }
            case record::TYPE_bind_descriptorsets:
            {
                vkCmdBindDescriptorSets(r.command_buffer, r.bind_descriptorsets.bind_point, r.bind_descriptorsets.pipeline_layout, 0, r.bind_descriptorsets.descriptorset_count, &descriptorsets[r.bind_descriptorsets.descriptorset_offset], 0, 0);
                break;
            }
            case record::TYPE_push_constants:
            {
                vkCmdPushConstants(r.command_buffer, r.push_constants.pipeline_layout, r.push_constants.stage_flags, 0, r.push_constants.size, r.push_constants.values);
                break;
            }
            case record::TYPE_dispatch:
            {
                vkCmdDispatch(r.command_buffer, r.dispatch.group_count_x, r.dispatch.group_count_y, r.dispatch.group_count_z);
                break;
            }
            case record::TYPE_memory_barrers:
            {
                vkCmdPipelineBarrier(r.command_buffer, r.memory_barrers.src_stage, r.memory_barrers.dst_stage, 0, r.memory_barrers.barrier_count, r.memory_barrers.barriers, 0, 0, 0, 0);
                delete[] r.memory_barrers.barriers;
                break;
            }
            case record::TYPE_buffer_barrers:
            {
                vkCmdPipelineBarrier(r.command_buffer, r.buffer_barrers.src_stage, r.buffer_barrers.dst_stage, 0, 0, 0, r.buffer_barrers.barrier_count, r.buffer_barrers.barriers, 0, 0);
                delete[] r.buffer_barrers.barriers;
                break;
            }
            case record::TYPE_image_barrers:
            {
                vkCmdPipelineBarrier(r.command_buffer, r.image_barrers.src_stage, r.image_barrers.dst_stage, 0, 0, 0, 0, 0, r.image_barrers.barrier_count, r.image_barrers.barriers);
                delete[] r.image_barrers.barriers;
                break;
            }
#if NCNN_BENCHMARK
            case record::TYPE_write_timestamp:
            {
                if (query_pool)
                    vkCmdWriteTimestamp(r.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, query_pool, r.write_timestamp.query);
                break;
            }
#endif // NCNN_BENCHMARK
            case record::TYPE_post_download:
            default:
                break;

            }
        }
    }

    // end command buffer
    {
        end_command_buffer();
    }

    // acquire queue and reclaim on return
    VkQueue compute_queue = vkdev->acquire_queue(vkdev->info.compute_queue_family_index);
    if (compute_queue == 0)
    {
        fprintf(stderr, "out of compute queue\n");
        return -1;
    }

    // submit compute
    {
        VkSubmitInfo submitInfo;
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = 0;
        submitInfo.waitSemaphoreCount = 0;
        submitInfo.pWaitSemaphores = 0;
        submitInfo.pWaitDstStageMask = 0;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &compute_command_buffer;
        submitInfo.signalSemaphoreCount = 0;
        submitInfo.pSignalSemaphores = 0;

        VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkQueueSubmit failed %d\n", ret);
            vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
            return -1;
        }
    }

    vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);

    // wait
    {
        VkResult ret = vkWaitForFences(vkdev->vkdevice(), 1, &compute_command_fence, VK_TRUE, UINT64_MAX);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkWaitForFences failed %d\n", ret);
            return -1;
        }
    }

    // handle delayed post records
    for (size_t i=0; i<delayed_records.size(); i++)
    {
        const record& r = delayed_records[i];

        switch (r.type)
        {
        case record::TYPE_post_download:
        {
            const VkMat& src = download_post_buffers[r.post_download.download_post_buffer_mat_offset];
            Mat& dst = download_post_mats[r.post_download.download_post_buffer_mat_offset];

            src.allocator->invalidate(src.data);
            memcpy(dst.data, src.mapped_ptr(), dst.total() * dst.elemsize);
            break;
        }
        case record::TYPE_copy_buffer:
        case record::TYPE_bind_pipeline:
        case record::TYPE_bind_descriptorsets:
        case record::TYPE_push_constants:
        case record::TYPE_dispatch:
        case record::TYPE_memory_barrers:
        case record::TYPE_buffer_barrers:
        case record::TYPE_image_barrers:
        default:
            break;

        }
    }

    delayed_records.clear();

    return 0;
}

int VkCompute::reset()
{
    // reset command buffer and fence
    {
        VkResult ret = vkResetCommandBuffer(compute_command_buffer, 0);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkResetCommandBuffer failed %d\n", ret);
            return -1;
        }
    }
    {
        VkResult ret = vkResetFences(vkdev->vkdevice(), 1, &compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkResetFences failed %d\n", ret);
            return -1;
        }
    }

    if (vkdev->info.support_VK_KHR_push_descriptor)
    {
        begin_command_buffer();

#if NCNN_BENCHMARK
        if (query_pool)
            vkCmdResetQueryPool(compute_command_buffer, query_pool, 0, query_count);
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
        if (query_pool)
            vkCmdResetQueryPool(compute_command_buffer, query_pool, 0, query_count);
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

int VkCompute::init()
{
    // compute_command_pool
    {
        VkCommandPoolCreateInfo commandPoolCreateInfo;
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.pNext = 0;
        commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolCreateInfo.queueFamilyIndex = vkdev->info.compute_queue_family_index;

        VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &compute_command_pool);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkCreateCommandPool failed %d\n", ret);
            return -1;
        }
    }

    // compute_command_buffer
    {
        VkCommandBufferAllocateInfo commandBufferAllocateInfo;
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.pNext = 0;
        commandBufferAllocateInfo.commandPool = compute_command_pool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;

        VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &compute_command_buffer);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkAllocateCommandBuffers failed %d\n", ret);
            return -1;
        }
    }

    // compute_command_fence
    {
        VkFenceCreateInfo fenceCreateInfo;
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.pNext = 0;
        fenceCreateInfo.flags = 0;

        VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkCreateFence failed %d\n", ret);
            return -1;
        }
    }

    if (vkdev->info.support_VK_KHR_push_descriptor)
    {
        begin_command_buffer();

#if NCNN_BENCHMARK
        if (query_pool)
            vkCmdResetQueryPool(compute_command_buffer, query_pool, 0, query_count);
#endif // NCNN_BENCHMARK
    }

    return 0;
}

int VkCompute::begin_command_buffer()
{
    VkCommandBufferBeginInfo commandBufferBeginInfo;
    commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    commandBufferBeginInfo.pNext = 0;
    commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    commandBufferBeginInfo.pInheritanceInfo = 0;

    VkResult ret = vkBeginCommandBuffer(compute_command_buffer, &commandBufferBeginInfo);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkBeginCommandBuffer failed %d\n", ret);
        return -1;
    }

    return 0;
}

int VkCompute::end_command_buffer()
{
    VkResult ret = vkEndCommandBuffer(compute_command_buffer);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkEndCommandBuffer failed %d\n", ret);
        return -1;
    }

    return 0;
}

VkTransfer::VkTransfer(const VulkanDevice* _vkdev) : vkdev(_vkdev)
{
    compute_command_pool = 0;
    transfer_command_pool = 0;

    upload_command_buffer = 0;
    compute_command_buffer = 0;

    upload_compute_semaphore = 0;

    upload_command_fence = 0;
    compute_command_fence = 0;

    init();
}

VkTransfer::~VkTransfer()
{
    vkDestroyFence(vkdev->vkdevice(), compute_command_fence, 0);

    vkFreeCommandBuffers(vkdev->vkdevice(), compute_command_pool, 1, &compute_command_buffer);
    vkDestroyCommandPool(vkdev->vkdevice(), compute_command_pool, 0);

    if (!vkdev->info.unified_compute_transfer_queue)
    {
        vkDestroyFence(vkdev->vkdevice(), upload_command_fence, 0);

        vkDestroySemaphore(vkdev->vkdevice(), upload_compute_semaphore, 0);

        vkFreeCommandBuffers(vkdev->vkdevice(), transfer_command_pool, 1, &upload_command_buffer);
        vkDestroyCommandPool(vkdev->vkdevice(), transfer_command_pool, 0);
    }
}

void VkTransfer::record_upload(const Mat& src, VkMat& dst, const Option& opt)
{
//     fprintf(stderr, "record_upload src = %d | %d %d %d @ %d\n", src.dims, src.w, src.h, src.c, src.elempack);

    // NOTE keep the hack here ?
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

    // create dst
    dst.create_like(src_flattened, opt.blob_vkallocator);

    if (dst.allocator->mappable)
    {
        // memcpy src_flattened to device
        memcpy(dst.mapped_ptr(), src_flattened.data, src_flattened.total() * src_flattened.elemsize);
        dst.allocator->flush(dst.data);

        // barrier device host-write @ null to shader-read @ compute
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.buffer_offset();
            barrier.size = dst.buffer_capacity();

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }

        // mark device shader-readwrite @ compute
        dst.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
        dst.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        return;
    }

    // create staging
    VkMat dst_staging;
    dst_staging.create_like(src_flattened, opt.staging_vkallocator);

    // memcpy src_flattened to staging
    memcpy(dst_staging.mapped_ptr(), src_flattened.data, src_flattened.total() * src_flattened.elemsize);

    VkCommandBuffer command_buffer;
    if (vkdev->info.unified_compute_transfer_queue)
    {
        command_buffer = compute_command_buffer;
    }
    else
    {
        command_buffer = upload_command_buffer;
    }

    // barrier staging host-write @ null to transfer-read @ queue
    {
        VkBufferMemoryBarrier barrier;
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.pNext = 0;
        barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = dst_staging.buffer();
        barrier.offset = dst_staging.buffer_offset();
        barrier.size = dst_staging.buffer_capacity();

        VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_HOST_BIT;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
    }

    // record staging to device
    {
        VkBufferCopy region;
        region.srcOffset = dst_staging.buffer_offset();
        region.dstOffset = dst.buffer_offset();
        region.size = std::min(dst_staging.buffer_capacity(), dst.buffer_capacity());

        vkCmdCopyBuffer(command_buffer, dst_staging.buffer(), dst.buffer(), 1, &region);
    }

    if (vkdev->info.unified_compute_transfer_queue)
    {
        // barrier device transfer-write @ compute to shader-read @ compute
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.buffer_offset();
            barrier.size = dst.buffer_capacity();

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }
    else
    {
        // queue ownership transfer any @ transfer to shader-read @ compute

        // release
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = 0;
            barrier.srcQueueFamilyIndex = vkdev->info.transfer_queue_family_index;
            barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.buffer_offset();
            barrier.size = dst.buffer_capacity();

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

            vkCmdPipelineBarrier(upload_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }

        // acquire
        {
            VkBufferMemoryBarrier barrier;
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.pNext = 0;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.srcQueueFamilyIndex = vkdev->info.transfer_queue_family_index;
            barrier.dstQueueFamilyIndex = vkdev->info.compute_queue_family_index;
            barrier.buffer = dst.buffer();
            barrier.offset = dst.buffer_offset();
            barrier.size = dst.buffer_capacity();

            VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, &barrier, 0, 0);
        }
    }

    // mark device shader-readwrite @ compute
    dst.data->access_flags = VK_ACCESS_SHADER_READ_BIT;
    dst.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

    // stash staging
    upload_staging_buffers.push_back(dst_staging);
}

int VkTransfer::submit_and_wait()
{
//     fprintf(stderr, "submit_and_wait\n");

    // end command buffer
    {
        end_command_buffer();
    }

    VkQueue compute_queue = vkdev->acquire_queue(vkdev->info.compute_queue_family_index);
    if (compute_queue == 0)
    {
        fprintf(stderr, "out of compute queue\n");
        return -1;
    }

    if (vkdev->info.unified_compute_transfer_queue)
    {
        // submit compute
        {
            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = 0;
            submitInfo.pWaitDstStageMask = 0;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &compute_command_buffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = 0;

            VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, compute_command_fence);
            if (ret != VK_SUCCESS)
            {
                fprintf(stderr, "vkQueueSubmit failed %d\n", ret);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
                return -1;
            }
        }
    }
    else
    {
        VkQueue transfer_queue = vkdev->acquire_queue(vkdev->info.transfer_queue_family_index);
        if (transfer_queue == 0)
        {
            fprintf(stderr, "out of transfer queue\n");
            vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
            return -1;
        }

        // submit upload compute
        {
            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 0;
            submitInfo.pWaitSemaphores = 0;
            submitInfo.pWaitDstStageMask = 0;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &upload_command_buffer;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &upload_compute_semaphore;

            VkResult ret = vkQueueSubmit(transfer_queue, 1, &submitInfo, upload_command_fence);
            if (ret != VK_SUCCESS)
            {
                fprintf(stderr, "vkQueueSubmit failed %d\n", ret);
                vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index, transfer_queue);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
                return -1;
            }
        }
        {
            VkPipelineStageFlags wait_dst_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;// FIXME

            VkSubmitInfo submitInfo;
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = 0;
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = &upload_compute_semaphore;
            submitInfo.pWaitDstStageMask = &wait_dst_stage;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &compute_command_buffer;
            submitInfo.signalSemaphoreCount = 0;
            submitInfo.pSignalSemaphores = 0;

            VkResult ret = vkQueueSubmit(compute_queue, 1, &submitInfo, compute_command_fence);
            if (ret != VK_SUCCESS)
            {
                fprintf(stderr, "vkQueueSubmit failed %d\n", ret);
                vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index, transfer_queue);
                vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);
                return -1;
            }
        }

        vkdev->reclaim_queue(vkdev->info.transfer_queue_family_index, transfer_queue);
    }

    vkdev->reclaim_queue(vkdev->info.compute_queue_family_index, compute_queue);

    // wait
    if (vkdev->info.unified_compute_transfer_queue)
    {
        VkResult ret = vkWaitForFences(vkdev->vkdevice(), 1, &compute_command_fence, VK_TRUE, UINT64_MAX);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkWaitForFences failed %d\n", ret);
            return -1;
        }
    }
    else
    {
        VkFence fences[2] = { upload_command_fence, compute_command_fence };

        VkResult ret = vkWaitForFences(vkdev->vkdevice(), 2, fences, VK_TRUE, UINT64_MAX);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkWaitForFences failed %d\n", ret);
            return -1;
        }
    }

    return 0;
}

int VkTransfer::init()
{
    // compute_command_pool
    {
        VkCommandPoolCreateInfo commandPoolCreateInfo;
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.pNext = 0;
        commandPoolCreateInfo.flags = 0;
        commandPoolCreateInfo.queueFamilyIndex = vkdev->info.compute_queue_family_index;

        VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &compute_command_pool);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkCreateCommandPool failed %d\n", ret);
            return -1;
        }
    }

    // compute_command_buffer
    {
        VkCommandBufferAllocateInfo commandBufferAllocateInfo;
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.pNext = 0;
        commandBufferAllocateInfo.commandPool = compute_command_pool;
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1;

        VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &compute_command_buffer);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkAllocateCommandBuffers failed %d\n", ret);
            return -1;
        }
    }

    // compute_command_fence
    {
        VkFenceCreateInfo fenceCreateInfo;
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.pNext = 0;
        fenceCreateInfo.flags = 0;

        VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &compute_command_fence);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkCreateFence failed %d\n", ret);
            return -1;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue)
    {
        // transfer_command_pool
        {
            VkCommandPoolCreateInfo commandPoolCreateInfo;
            commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            commandPoolCreateInfo.pNext = 0;
            commandPoolCreateInfo.flags = 0;
            commandPoolCreateInfo.queueFamilyIndex = vkdev->info.transfer_queue_family_index;

            VkResult ret = vkCreateCommandPool(vkdev->vkdevice(), &commandPoolCreateInfo, 0, &transfer_command_pool);
            if (ret != VK_SUCCESS)
            {
                fprintf(stderr, "vkCreateCommandPool failed %d\n", ret);
                return -1;
            }
        }

        // upload_command_buffer
        {
            VkCommandBufferAllocateInfo commandBufferAllocateInfo;
            commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            commandBufferAllocateInfo.pNext = 0;
            commandBufferAllocateInfo.commandPool = transfer_command_pool;
            commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            commandBufferAllocateInfo.commandBufferCount = 1;

            VkResult ret = vkAllocateCommandBuffers(vkdev->vkdevice(), &commandBufferAllocateInfo, &upload_command_buffer);
            if (ret != VK_SUCCESS)
            {
                fprintf(stderr, "vkAllocateCommandBuffers failed %d\n", ret);
                return -1;
            }
        }

        // upload_compute_semaphore
        {
            VkSemaphoreCreateInfo semaphoreCreateInfo;
            semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            semaphoreCreateInfo.pNext = 0;
            semaphoreCreateInfo.flags = 0;

            VkResult ret = vkCreateSemaphore(vkdev->vkdevice(), &semaphoreCreateInfo, 0, &upload_compute_semaphore);
            if (ret != VK_SUCCESS)
            {
                fprintf(stderr, "vkCreateSemaphore failed %d\n", ret);
                return -1;
            }
        }

        // upload_command_fence
        {
            VkFenceCreateInfo fenceCreateInfo;
            fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceCreateInfo.pNext = 0;
            fenceCreateInfo.flags = 0;

            VkResult ret = vkCreateFence(vkdev->vkdevice(), &fenceCreateInfo, 0, &upload_command_fence);
            if (ret != VK_SUCCESS)
            {
                fprintf(stderr, "vkCreateFence failed %d\n", ret);
                return -1;
            }
        }
    }

    begin_command_buffer();

    return 0;
}

int VkTransfer::begin_command_buffer()
{
    {
        VkCommandBufferBeginInfo commandBufferBeginInfo;
        commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        commandBufferBeginInfo.pNext = 0;
        commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        commandBufferBeginInfo.pInheritanceInfo = 0;

        VkResult ret = vkBeginCommandBuffer(compute_command_buffer, &commandBufferBeginInfo);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkBeginCommandBuffer failed %d\n", ret);
            return -1;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue)
    {
        {
            VkCommandBufferBeginInfo commandBufferBeginInfo;
            commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            commandBufferBeginInfo.pNext = 0;
            commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            commandBufferBeginInfo.pInheritanceInfo = 0;

            VkResult ret = vkBeginCommandBuffer(upload_command_buffer, &commandBufferBeginInfo);
            if (ret != VK_SUCCESS)
            {
                fprintf(stderr, "vkBeginCommandBuffer failed %d\n", ret);
                return -1;
            }
        }
    }

    return 0;
}

int VkTransfer::end_command_buffer()
{
    {
        VkResult ret = vkEndCommandBuffer(compute_command_buffer);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkEndCommandBuffer failed %d\n", ret);
            return -1;
        }
    }

    if (!vkdev->info.unified_compute_transfer_queue)
    {
        {
            VkResult ret = vkEndCommandBuffer(upload_command_buffer);
            if (ret != VK_SUCCESS)
            {
                fprintf(stderr, "vkEndCommandBuffer failed %d\n", ret);
                return -1;
            }
        }
    }

    return 0;
}

} // namespace ncnn

#endif // NCNN_VULKAN
