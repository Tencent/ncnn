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

#ifndef NCNN_COMMAND_H
#define NCNN_COMMAND_H

#include "platform.h"

#if NCNN_VULKAN

#include <vector>
#include <vulkan/vulkan.h>
#include "mat.h"

namespace ncnn {

class Pipeline;
class VkCompute
{
public:
    VkCompute(const VulkanDevice* vkdev);
    virtual ~VkCompute();

public:
    void record_upload(const Mat& src, VkMat& dst, const Option& opt);

    void record_download(const VkMat& src, Mat& dst, const Option& opt);

    void record_clone(const VkMat& src, VkMat& dst, const Option& opt);

    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher);

#if NCNN_BENCHMARK
    void record_write_timestamp(uint32_t query);
#endif // NCNN_BENCHMARK

#if __ANDROID_API__ >= 26
    void record_import_android_hardware_buffer(const ImportAndroidHardwareBufferPipeline* pipeline, const VkImageMat& src, const VkMat& dst);
#endif // __ANDROID_API__ >= 26

    int submit_and_wait();

    int reset();

#if NCNN_BENCHMARK
    int create_query_pool(uint32_t query_count);

    int get_query_pool_results(uint32_t first_query, uint32_t query_count, std::vector<uint64_t>& results);
#endif // NCNN_BENCHMARK

protected:
    int init();
    int begin_command_buffer();
    int end_command_buffer();

protected:
    const VulkanDevice* vkdev;

    VkCommandPool compute_command_pool;

    VkCommandBuffer compute_command_buffer;

    VkFence compute_command_fence;

    std::vector<VkMat> upload_staging_buffers;
    std::vector<VkMat> download_post_buffers;
    std::vector<Mat> download_post_mats;

    // the good-old path for device without VK_KHR_push_descriptor
    std::vector<VkDescriptorPool> descriptor_pools;
    std::vector<VkDescriptorSet> descriptorsets;

    struct record
    {
        enum
        {
            TYPE_copy_buffer,
            TYPE_bind_pipeline,
            TYPE_bind_descriptorsets,
            TYPE_push_constants,
            TYPE_dispatch,
            TYPE_memory_barrers,
            TYPE_buffer_barrers,
            TYPE_image_barrers,

#if NCNN_BENCHMARK
            TYPE_write_timestamp,
#endif // NCNN_BENCHMARK

            TYPE_post_download,
        };

        int type;
        VkCommandBuffer command_buffer;

        union
        {
        struct { VkBuffer src; VkBuffer dst; uint32_t region_count; const VkBufferCopy* regions; } copy_buffer;

        struct { VkPipelineBindPoint bind_point; VkPipeline pipeline; } bind_pipeline;
        struct { VkPipelineBindPoint bind_point; VkPipelineLayout pipeline_layout; uint32_t descriptorset_count; uint32_t descriptorset_offset; } bind_descriptorsets;
        struct { VkPipelineLayout pipeline_layout; VkShaderStageFlags stage_flags; uint32_t size; const void* values; } push_constants;

        struct { uint32_t group_count_x; uint32_t group_count_y; uint32_t group_count_z; } dispatch;

        struct { VkPipelineStageFlags src_stage; VkPipelineStageFlags dst_stage; uint32_t barrier_count; const VkMemoryBarrier* barriers; } memory_barrers;
        struct { VkPipelineStageFlags src_stage; VkPipelineStageFlags dst_stage; uint32_t barrier_count; const VkBufferMemoryBarrier* barriers; } buffer_barrers;
        struct { VkPipelineStageFlags src_stage; VkPipelineStageFlags dst_stage; uint32_t barrier_count; const VkImageMemoryBarrier* barriers; } image_barrers;

#if NCNN_BENCHMARK
        struct { uint32_t query; } write_timestamp;
#endif // NCNN_BENCHMARK

        struct { uint32_t download_post_buffer_mat_offset; } post_download;
        };
    };

    std::vector<record> delayed_records;

#if NCNN_BENCHMARK
    uint32_t query_count;
    VkQueryPool query_pool;
#endif // NCNN_BENCHMARK
};

class VkTransfer
{
public:
    VkTransfer(const VulkanDevice* vkdev);
    ~VkTransfer();

public:
    void record_upload(const Mat& src, VkMat& dst, const Option& opt);

    int submit_and_wait();

protected:
    int init();
    int begin_command_buffer();
    int end_command_buffer();

protected:
    const VulkanDevice* vkdev;

    VkCommandPool compute_command_pool;
    VkCommandPool transfer_command_pool;

    VkCommandBuffer upload_command_buffer;
    VkCommandBuffer compute_command_buffer;

    VkSemaphore upload_compute_semaphore;

    VkFence upload_command_fence;
    VkFence compute_command_fence;

    std::vector<VkMat> upload_staging_buffers;
};

} // namespace ncnn

#endif // NCNN_VULKAN

#endif // NCNN_COMMAND_H
