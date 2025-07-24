// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_PIPELINECACHE_H
#define NCNN_PIPELINECACHE_H

#include "platform.h"

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
                     uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z, uint32_t subgroup_size,
                     VkShaderModule* shader_module,
                     VkDescriptorSetLayout* descriptorset_layout,
                     VkPipelineLayout* pipeline_layout,
                     VkPipeline* pipeline,
                     VkDescriptorUpdateTemplateKHR* descriptor_update_template,
                     ShaderInfo& shader_info) const;

    int get_pipeline(int shader_type_index, const Option& opt, const std::vector<vk_specialization_type>& specializations,
                     uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z, uint32_t subgroup_size,
                     VkShaderModule* shader_module,
                     VkDescriptorSetLayout* descriptorset_layout,
                     VkPipelineLayout* pipeline_layout,
                     VkPipeline* pipeline,
                     VkDescriptorUpdateTemplateKHR* descriptor_update_template,
                     ShaderInfo& shader_info) const;

protected:
    int create_shader_module(int shader_type_index, const Option& opt,
                             uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                             VkShaderModule* _shader_module, ShaderInfo& si) const;

    int new_pipeline(VkShaderModule shader_module, const ShaderInfo& shader_info, const std::vector<vk_specialization_type>& specializations, uint32_t subgroup_size,
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
