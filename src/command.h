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
    Command(const VulkanDevice* vkdev, uint32_t queue_family_index);
    virtual ~Command();

protected:
    int create_command_pool();
    int create_command_buffer();

    // record issue
    int begin_command_buffer();
    int end_command_buffer();
    int queue_submit_and_wait_fence();

protected:
    const VulkanDevice* vkdev;
    uint32_t queue_family_index;

    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;

    VkFence fence;
};

class VkCompute : public Command
{
public:
    VkCompute(const VulkanDevice* vkdev);
    ~VkCompute();

    void record_upload(const VkMat& m);

    void record_download(const VkMat& m);

    void record_clone(const VkMat& src, const VkMat& dst);

    void record_copy_region(const VkMat& src, const VkMat& dst, const VkBufferCopy& region);

    void record_copy_regions(const VkMat& src, const VkMat& dst, const std::vector<VkBufferCopy>& regions);

    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& m);

#if NCNN_BENCHMARK
    void record_write_timestamp(uint32_t query);
#endif // NCNN_BENCHMARK

    void record_queue_transfer_acquire(const VkMat& m, uint32_t src_queue_family_index);

#if __ANDROID_API__ >= 26
    void record_import_android_hardware_buffer(const ImportAndroidHardwareBufferPipeline* pipeline, const VkImageMat& im, const VkMat& m);
#endif // __ANDROID_API__ >= 26

    int submit_and_wait();

    int reset();

#if NCNN_BENCHMARK
    int create_query_pool(uint32_t query_count);

    int get_query_pool_results(uint32_t first_query, uint32_t query_count, std::vector<uint64_t>& results);
#endif // NCNN_BENCHMARK

protected:
    // record pipeline things
    void record_bind_pipeline(VkPipeline pipeline);
    void record_update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const std::vector<VkMat>& bindings);
    void record_push_constants(VkPipelineLayout pipeline_layout, const std::vector<vk_constant_type>& constants);
    void record_dispatch(const uint32_t* group_count_xyz);

    // record barrier things
    void record_transfer_compute_barrier(const VkMat& m);
    void record_compute_transfer_barrier(const VkMat& m);
    void record_compute_compute_barrier(const VkMat& m);
    void record_transfer_transfer_barrier(const VkMat& m);
    void record_host_transfer_barrier(const VkMat& m);
    void record_transfer_host_barrier(const VkMat& m);
    void record_host_compute_barrier(const VkMat& m);
    void record_compute_host_barrier(const VkMat& m);

    // record prepare things
    void record_prepare_transfer_barrier(const VkMat& m);
    void record_prepare_compute_barrier(const VkMat& m);
    void record_prepare_host_barrier(const VkMat& m);

    void record_initial_image_compute_barrier(const VkImageMat& im);

#if __ANDROID_API__ >= 26
    void record_update_import_android_hardware_buffer_bindings(VkPipelineLayout pipeline_layout, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, VkSampler sampler, const VkImageMat& im, const VkMat& m);
#endif // __ANDROID_API__ >= 26

#if NCNN_BENCHMARK
    void reset_query_pool();
#endif // NCNN_BENCHMARK

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
    void transfer_transfer_barrier(VkBuffer buffer, size_t offset, size_t size);
    void host_transfer_barrier(VkBuffer buffer, size_t offset, size_t size);
    void transfer_host_barrier(VkBuffer buffer, size_t offset, size_t size);
    void host_compute_barrier(VkBuffer buffer, size_t offset, size_t size);
    void compute_host_barrier(VkBuffer buffer, size_t offset, size_t size);
    void queue_transfer_acquire_barrier(VkBuffer buffer, size_t offset, size_t size, uint32_t src_queue_family_index);
    void initial_image_compute_barrier(VkImage image);
#if __ANDROID_API__ >= 26
    void update_import_android_hardware_buffer_bindings(VkPipelineLayout pipeline_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const VkDescriptorImageInfo& descriptorImageInfo, const VkDescriptorBufferInfo& descriptorBufferInfo);
#endif // __ANDROID_API__ >= 26
#if NCNN_BENCHMARK
    void write_timestamp(uint32_t query);
#endif // NCNN_BENCHMARK

protected:
    // delayed record
    // the good-old path for device without VK_KHR_push_descriptor
    std::vector<VkDescriptorPool> descriptor_pools;
    std::vector<VkDescriptorSet> descriptorsets;
    struct record_type
    {
        // 0=copy
        // 1=copy regions
        // 2=bind pipeline
        // 3=bind descriptorset
        // 4=push constants
        // 5=dispatch
        // 6=transfer-compute barrier
        // 7=compute-transfer barrier
        // 8=compute-compute barrier
        // 9=transfer-transfer barrier
        // 10=write timestamp
        // 11=initial image compute barrier
        // 12=host-transfer barrier
        // 13=transfer-host barrier
        // 14=host-compute barrier
        // 15=compute-host barrier
        // 16=queue-transfer-acquire barrier
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
        struct { VkBuffer buffer; size_t offset; size_t size; } transfer_transfer_barrier;
#if NCNN_BENCHMARK
        struct { uint32_t query; } write_timestamp;
#endif // NCNN_BENCHMARK
        struct { VkImage image; } initial_image_compute_barrier;
        struct { VkBuffer buffer; size_t offset; size_t size; } host_transfer_barrier;
        struct { VkBuffer buffer; size_t offset; size_t size; } transfer_host_barrier;
        struct { VkBuffer buffer; size_t offset; size_t size; } host_compute_barrier;
        struct { VkBuffer buffer; size_t offset; size_t size; } compute_host_barrier;
        struct { VkBuffer buffer; size_t offset; size_t size; size_t src_queue_family_index; } queue_transfer_acquire_barrier;
        };

        std::vector<VkBufferCopy> regions;
        std::vector<vk_constant_type> constants;
    };
    std::vector<record_type> delayed_records;

#if NCNN_BENCHMARK
    uint32_t query_count;
    VkQueryPool query_pool;
#endif // NCNN_BENCHMARK
};

class VkTransfer : public Command
{
public:
    VkTransfer(const VulkanDevice* vkdev);
    ~VkTransfer();

    void record_upload(const Mat& src, VkMat& dst, const Option& opt);

    int submit_and_wait();

public:
    VkAllocator* weight_vkallocator;
    VkAllocator* staging_vkallocator;

protected:
    // recording issue
    void copy_buffer(VkBuffer src, size_t src_offset, VkBuffer dst, size_t dst_offset, size_t size);
    void copy_buffer_regions(VkBuffer src, VkBuffer dst, const std::vector<VkBufferCopy>& regions);
    void queue_transfer_release_barrier(VkBuffer buffer, size_t offset, size_t size, uint32_t target_queue_family_index);

protected:
    size_t buffer_offset_alignment;
    VkBufferMemory* staging_data;

    // delayed record
    struct record_type
    {
        size_t size;
        Mat mat;
        VkMat vkmat;
    };
    std::vector<record_type> delayed_records;
};

} // namespace ncnn

#endif // NCNN_VULKAN

#endif // NCNN_COMMAND_H
