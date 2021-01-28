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

#ifndef NCNN_PIPELINECACHE_H
#define NCNN_PIPELINECACHE_H

#include "platform.h"

#if NCNN_VULKAN
#include <vulkan/vulkan.h>
#endif // NCNN_VULKAN

#include "mat.h"
#include "gpu.h"

namespace ncnn {

#if NCNN_VULKAN

class VulkanDevice;
class PipelineCachePrivate;
class NCNN_EXPORT PipelineCache
{
public:
    explicit PipelineCache(const VulkanDevice* _vkdev);

    virtual ~PipelineCache();

    void clear();

    int get_pipeline(const uint32_t* spv_data, size_t spv_data_size, const std::vector<vk_specialization_type>& specializations,
                     uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                     VkShaderModule* shader_module,
                     VkDescriptorSetLayout* descriptorset_layout,
                     VkPipelineLayout* pipeline_layout,
                     VkPipeline* pipeline,
                     VkDescriptorUpdateTemplateKHR* descriptor_update_template,
                     ShaderInfo& shader_info) const;

    int get_pipeline(int shader_type_index, const Option& opt, const std::vector<vk_specialization_type>& specializations,
                     uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                     VkShaderModule* shader_module,
                     VkDescriptorSetLayout* descriptorset_layout,
                     VkPipelineLayout* pipeline_layout,
                     VkPipeline* pipeline,
                     VkDescriptorUpdateTemplateKHR* descriptor_update_template,
                     ShaderInfo& shader_info) const;

protected:
    int create_shader_module(int shader_type_index, const Option& opt, uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                             VkShaderModule* _shader_module, ShaderInfo& si) const;

    int new_pipeline(VkShaderModule shader_module, const ShaderInfo& shader_info, const std::vector<vk_specialization_type>& specializations,
                     VkDescriptorSetLayout* descriptorset_layout,
                     VkPipelineLayout* pipeline_layout,
                     VkPipeline* pipeline,
                     VkDescriptorUpdateTemplateKHR* descriptor_update_template) const;

protected:
    const VulkanDevice* vkdev;

private:
    PipelineCache(const PipelineCache&);
    PipelineCache& operator=(const PipelineCache&);

private:
    PipelineCachePrivate* const d;
};

#endif // NCNN_VULKAN

} // namespace ncnn

#endif // NCNN_PIPELINECACHE_H
