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

#ifndef NCNN_COMMAND_H
#define NCNN_COMMAND_H

#include "platform.h"

#if NCNN_VULKAN

#include <vector>
#include <vulkan/vulkan.h>
#include "mat.h"
#include "pipeline.h"

namespace ncnn {

class Command
{
public:
    Command(VulkanDevice* vkdev, uint32_t queue_index);
    ~Command();

protected:
    int create_command_pool();
    int create_command_buffer();

    // record issue
    int begin_command_buffer();
    int end_command_buffer();
    int queue_submit();
    int wait_fence();

protected:
    VulkanDevice* vkdev;
    uint32_t queue_index;

    VkQueue queue;

    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    VkFence fence;
};

class VkCompute : public Command
{
public:
    VkCompute(VulkanDevice* vkdev);
    ~VkCompute();

    int begin();

    void record_upload(const VkMat& m);

    void record_download(const VkMat& m);

    void record_clone(const VkMat& src, const VkMat& dst);

    void record_copy_region(const VkMat& src, const VkMat& dst, const VkBufferCopy& region);

    void record_copy_regions(const VkMat& src, const VkMat& dst, const std::vector<VkBufferCopy>& regions);

    void record_transfer_compute_barrier(const VkMat& m);

    void record_compute_transfer_barrier(const VkMat& m);

    void record_compute_compute_barrier(const VkMat& m);

    void record_prepare_transfer_barrier(const VkMat& m);

    void record_prepare_compute_barrier(const VkMat& m);

    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& m);

    int end();

    int submit();

    int wait();

protected:
    // record pipeline things
    void record_bind_pipeline(VkPipeline pipeline);
    void record_update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const std::vector<VkMat>& bindings);
    void record_push_constants(VkPipelineLayout pipeline_layout, const std::vector<vk_constant_type>& constants);
    void record_dispatch(const uint32_t* group_count_xyz);

protected:
    // recording issue
    void copy_buffer(VkBuffer src, size_t src_offset, VkBuffer dst, size_t dst_offset, size_t size);
    void copy_buffer_regions(VkBuffer src, VkBuffer dst, const std::vector<VkBufferCopy>& regions);
    void bind_pipeline(VkPipeline pipeline);
    void bind_descriptorset(VkPipelineLayout pipeline_layout, VkDescriptorSet descriptorset);
    void update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const std::vector<VkDescriptorBufferInfo>& descriptorBufferInfos);
    void push_constants(VkPipelineLayout pipeline_layout, const std::vector<vk_constant_type>& constants);
    void dispatch(const uint32_t* group_count_xyz);
    void transfer_compute_barrier(VkBuffer buffer, size_t offset, size_t size);
    void compute_transfer_barrier(VkBuffer buffer, size_t offset, size_t size);
    void compute_compute_barrier(VkBuffer buffer, size_t offset, size_t size);

protected:
    // delayed record
    // the good-old path for device without VK_KHR_push_descriptor
    std::vector<VkDescriptorPool> descriptor_pools;
    std::vector<VkDescriptorSet> descriptorsets;
    struct record_type
    {
        // 0=begin
        // 1=copy
        // 2=copy regions
        // 3=bind pipeline
        // 4=bind descriptorset
        // 5=push constants
        // 6=dispatch
        // 7=transfer-compute barrier
        // 8=compute-transfer barrier
        // 9=compute-compute barrier
        // 10=end
        int type;

        union
        {
        struct { VkBuffer src; size_t src_offset; VkBuffer dst; size_t dst_offset; size_t size; } copy;
        struct { VkBuffer src; VkBuffer dst; } copy_regions;
        struct { VkPipeline pipeline; } bind_pipeline;
        struct { VkPipelineLayout pipeline_layout; VkDescriptorSet descriptorset; } bind_descriptorset;
        struct { VkPipelineLayout pipeline_layout; } push_constants;
        struct { uint32_t group_count_xyz[3]; } dispatch;
        struct { VkBuffer buffer; size_t offset; size_t size; } transfer_compute_barrier;
        struct { VkBuffer buffer; size_t offset; size_t size; } compute_transfer_barrier;
        struct { VkBuffer buffer; size_t offset; size_t size; } compute_compute_barrier;
        };

        std::vector<VkBufferCopy> regions;
        std::vector<vk_constant_type> constants;
    };
    std::vector<record_type> delayed_records;
};

class VkTransfer : public Command
{
public:
    VkTransfer(VulkanDevice* vkdev);
    ~VkTransfer();

    void record_upload(const Mat& src, VkMat& dst);

    void record_download(const VkMat& src, Mat& dst);

    int submit();

    int wait();

public:
    VkAllocator* weight_vkallocator;
    VkAllocator* staging_vkallocator;

protected:
    // recording issue
    void copy_buffer(VkBuffer src, size_t src_offset, VkBuffer dst, size_t dst_offset, size_t size);
    void copy_buffer_regions(VkBuffer src, VkBuffer dst, const std::vector<VkBufferCopy>& regions);

protected:
    VkBufferMemory* staging_data;
    void* mapped_ptr;

    // delayed record
    struct record_type
    {
        // 0=upload
        // 1=download
        int type;

        size_t size;

        union
        {
        struct { void* src; VkBuffer dst; size_t dst_offset; } upload;
        struct { VkBuffer src; size_t src_offset; void* dst; } download;
        };
    };
    std::vector<record_type> delayed_records;
};

} // namespace ncnn

#endif // NCNN_VULKAN

#endif // NCNN_COMMAND_H
