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

#include "mat.h"

namespace ncnn {

class Pipeline;
#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
class ImportAndroidHardwareBufferPipeline;
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API
class VkComputePrivate;
class NCNN_EXPORT VkCompute
{
public:
    explicit VkCompute(const VulkanDevice* vkdev);
    virtual ~VkCompute();

public:
    void record_upload(const Mat& src, VkMat& dst, const Option& opt);

    void record_upload(const Mat& src, VkImageMat& dst, const Option& opt);

    void record_download(const VkMat& src, Mat& dst, const Option& opt);

    void record_download(const VkImageMat& src, Mat& dst, const Option& opt);

    void record_buffer_to_image(const VkMat& src, VkImageMat& dst, const Option& opt);

    void record_image_to_buffer(const VkImageMat& src, VkMat& dst, const Option& opt);

    void record_clone(const Mat& src, VkMat& dst, const Option& opt);

    void record_clone(const Mat& src, VkImageMat& dst, const Option& opt);

    void record_clone(const VkMat& src, Mat& dst, const Option& opt);

    void record_clone(const VkImageMat& src, Mat& dst, const Option& opt);

    void record_clone(const VkMat& src, VkMat& dst, const Option& opt);

    void record_clone(const VkImageMat& src, VkImageMat& dst, const Option& opt);

    void record_clone(const VkMat& src, VkImageMat& dst, const Option& opt);

    void record_clone(const VkImageMat& src, VkMat& dst, const Option& opt);

    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher);

    void record_pipeline(const Pipeline* pipeline, const std::vector<VkImageMat>& bindings, const std::vector<vk_constant_type>& constants, const VkImageMat& dispatcher);

    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher);
    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const VkImageMat& dispatcher);
    void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const Mat& dispatcher);

#if NCNN_BENCHMARK
    void record_write_timestamp(uint32_t query);
#endif // NCNN_BENCHMARK

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
    void record_import_android_hardware_buffer(const ImportAndroidHardwareBufferPipeline* pipeline, const VkImageMat& src, const VkMat& dst);

    void record_import_android_hardware_buffer(const ImportAndroidHardwareBufferPipeline* pipeline, const VkImageMat& src, const VkImageMat& dst);
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API

    int submit_and_wait();

    int reset();

#if NCNN_BENCHMARK
    int create_query_pool(uint32_t query_count);

    int get_query_pool_results(uint32_t first_query, uint32_t query_count, std::vector<uint64_t>& results);
#endif // NCNN_BENCHMARK

protected:
    const VulkanDevice* vkdev;

    void barrier_readwrite(const VkMat& binding);
    void barrier_readwrite(const VkImageMat& binding);
    void barrier_readonly(const VkImageMat& binding);

private:
    VkComputePrivate* const d;
};

class VkTransferPrivate;
class NCNN_EXPORT VkTransfer
{
public:
    explicit VkTransfer(const VulkanDevice* vkdev);
    virtual ~VkTransfer();

public:
    void record_upload(const Mat& src, VkMat& dst, const Option& opt, bool flatten = true);

    void record_upload(const Mat& src, VkImageMat& dst, const Option& opt);

    int submit_and_wait();

protected:
    const VulkanDevice* vkdev;

private:
    VkTransferPrivate* const d;
};

} // namespace ncnn

#endif // NCNN_VULKAN

#endif // NCNN_COMMAND_H
