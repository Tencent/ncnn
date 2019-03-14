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

#include "platform.h"
#include "mat.h"
#if NCNN_VULKAN
#include <vulkan/vulkan.h>
#include "gpu.h"
#endif // NCNN_VULKAN

namespace ncnn {

#if NCNN_VULKAN
class Pipeline
{
public:
    Pipeline(const VulkanDevice* vkdev);
    ~Pipeline();

public:
    void set_optimal_local_size_xyz(int w = 32, int h = 32, int c = 32);

    int create(const char* name, const std::vector<vk_specialization_type>& specializations,
               int binding_count, int push_constant_count);
    void destroy();

protected:
    int create_descriptorset_layout(int binding_count);
    int create_pipeline_layout(int push_constant_count);
    int create_pipeline(const char* name, const std::vector<vk_specialization_type>& specializations);
    int create_descriptor_update_template(int binding_count);

public:
    const VulkanDevice* vkdev;

    // shared among each layer type instance
    VkShaderModule shader_module;

    VkDescriptorSetLayout descriptorset_layout;
    VkPipelineLayout pipeline_layout;

    // op forward TODO use pipeline cache ?
    VkPipeline pipeline;

    VkDescriptorUpdateTemplateKHR descriptor_update_template;

    uint32_t local_size_x;
    uint32_t local_size_y;
    uint32_t local_size_z;
};
#endif // NCNN_VULKAN

} // namespace ncnn

#endif // NCNN_PIPELINE_H
