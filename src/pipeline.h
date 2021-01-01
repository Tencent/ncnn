// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef NCNN_PIPELINE_H
#define NCNN_PIPELINE_H

#include "mat.h"
#include "platform.h"
#if NCNN_VULKAN
#include "gpu.h"

#include <vulkan/vulkan.h>
#endif // NCNN_VULKAN

namespace ncnn {

#if NCNN_VULKAN
class Option;
class PipelinePrivate;
class NCNN_EXPORT Pipeline
{
public:
    explicit Pipeline(const VulkanDevice* vkdev);
    virtual ~Pipeline();

public:
    void set_optimal_local_size_xyz(int w = 4, int h = 4, int c = 4);
    void set_optimal_local_size_xyz(const Mat& local_size_xyz);
    void set_local_size_xyz(int w, int h, int c);

    int create(const uint32_t* spv_data, size_t spv_data_size, const std::vector<vk_specialization_type>& specializations);

    int create(int shader_type_index, const Option& opt, const std::vector<vk_specialization_type>& specializations);

public:
    VkShaderModule shader_module() const;
    VkDescriptorSetLayout descriptorset_layout() const;
    VkPipelineLayout pipeline_layout() const;
    VkPipeline pipeline() const;
    VkDescriptorUpdateTemplateKHR descriptor_update_template() const;

    const ShaderInfo& shader_info() const;

    uint32_t local_size_x() const;
    uint32_t local_size_y() const;
    uint32_t local_size_z() const;

protected:
    void set_shader_module(VkShaderModule shader_module);
    void set_descriptorset_layout(VkDescriptorSetLayout descriptorset_layout);
    void set_pipeline_layout(VkPipelineLayout pipeline_layout);
    void set_pipeline(VkPipeline pipeline);
    void set_descriptor_update_template(VkDescriptorUpdateTemplateKHR descriptor_update_template);

    void set_shader_info(const ShaderInfo& shader_info);

public:
    const VulkanDevice* vkdev;

private:
    Pipeline(const Pipeline&);
    Pipeline& operator=(const Pipeline&);

private:
    PipelinePrivate* const d;
};

#if NCNN_PLATFORM_API
#if __ANDROID_API__ >= 26
class VkCompute;
class NCNN_EXPORT ImportAndroidHardwareBufferPipeline : private Pipeline
{
public:
    explicit ImportAndroidHardwareBufferPipeline(const VulkanDevice* vkdev);
    virtual ~ImportAndroidHardwareBufferPipeline();

    int create(VkAndroidHardwareBufferImageAllocator* ahb_im_allocator, int type_to, int rotate_from, const Option& opt);
    int create(VkAndroidHardwareBufferImageAllocator* ahb_im_allocator, int type_to, int rotate_from, int target_width, int target_height, const Option& opt);
    void destroy();

    friend class VkCompute;

protected:
    int create_shader_module(const Option& opt);
    int create_sampler(VkAndroidHardwareBufferImageAllocator* ahb_im_allocator);
    int create_descriptorset_layout();

public:
    int type_to;
    int rotate_from;
    bool need_resize;

    VkSampler sampler;
};
#endif // __ANDROID_API__ >= 26
#endif // NCNN_PLATFORM_API

#endif // NCNN_VULKAN

} // namespace ncnn

#endif // NCNN_PIPELINE_H
