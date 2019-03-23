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

#include "pipeline.h"
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "mat.h"
#include <string>

namespace ncnn {

#if NCNN_VULKAN
Pipeline::Pipeline(const VulkanDevice* _vkdev) : vkdev(_vkdev)
{
    descriptorset_layout = 0;
    pipeline_layout = 0;
    pipeline = 0;
    descriptor_update_template = 0;

    local_size_x = 1;
    local_size_y = 1;
    local_size_z = 1;
}

Pipeline::~Pipeline()
{
    destroy();
}

int Pipeline::create(const char* name, const std::vector<vk_specialization_type>& specializations, int binding_count, int push_constant_count)
{
    create_descriptorset_layout(binding_count);

    create_pipeline_layout(push_constant_count);

    create_pipeline(name, specializations);

    if (vkdev->info.support_VK_KHR_descriptor_update_template)
    {
        create_descriptor_update_template(binding_count);
    }

    return 0;
}

void Pipeline::destroy()
{
    if (vkdev->info.support_VK_KHR_descriptor_update_template)
    {
        if (descriptor_update_template)
        {
            vkdev->vkDestroyDescriptorUpdateTemplateKHR(vkdev->vkdevice(), descriptor_update_template, 0);
            descriptor_update_template = 0;
        }
    }

    if (pipeline)
    {
        vkDestroyPipeline(vkdev->vkdevice(), pipeline, 0);
        pipeline = 0;
    }

    if (pipeline_layout)
    {
        vkDestroyPipelineLayout(vkdev->vkdevice(), pipeline_layout, 0);
        pipeline_layout = 0;
    }

    if (descriptorset_layout)
    {
        vkDestroyDescriptorSetLayout(vkdev->vkdevice(), descriptorset_layout, 0);
        descriptorset_layout = 0;
    }
}

void Pipeline::set_optimal_local_size_xyz(int w, int h, int c)
{
    if (c > 0)
    {
        local_size_z = vkdev->info.max_workgroup_size[2];
        while ((uint32_t)c < local_size_z)
        {
            local_size_z /= 2;
        }
    }
    else
    {
        local_size_z = std::min((uint32_t)128, vkdev->info.max_workgroup_size[2]);
    }

    uint32_t max_local_size_xy = vkdev->info.max_workgroup_invocations / local_size_z;

    if (h == w || (h < 0 && w < 0))
    {
        uint32_t local_size_xy = sqrt(max_local_size_xy);
        uint32_t local_size_xy_prefer = 128;
        while (local_size_xy < local_size_xy_prefer)
        {
            local_size_xy_prefer /= 2;
        }
        local_size_x = local_size_xy_prefer;
        local_size_y = local_size_xy_prefer;
    }
    if (h > 0 && w > 0)
    {
        if (h > w)
        {
            float ps = h / (float)w;
            float local_size_xy = sqrt(max_local_size_xy / ps);
            local_size_y = local_size_xy * ps;
            local_size_x = std::max((uint32_t)local_size_xy, (uint32_t)1);
        }
        else
        {
            float ps = w / (float)h;
            float local_size_xy = sqrt(max_local_size_xy / ps);
            local_size_y = std::max((uint32_t)local_size_xy, (uint32_t)1);
            local_size_x = local_size_xy * ps;
        }

        uint32_t local_size_y_prefer = std::min((uint32_t)128, vkdev->info.max_workgroup_size[1]);
        while (local_size_y < local_size_y_prefer)
        {
            local_size_y_prefer /= 2;
        }

        uint32_t local_size_x_prefer = std::min((uint32_t)128, vkdev->info.max_workgroup_size[0]);
        while (local_size_x < local_size_x_prefer)
        {
            local_size_x_prefer /= 2;
        }

        local_size_y = local_size_y_prefer;
        local_size_x = local_size_x_prefer;
    }
    else if (h > 0)
    {
        local_size_y = std::min(max_local_size_xy, vkdev->info.max_workgroup_size[1]);
        while ((uint32_t)h < local_size_y)
        {
            local_size_y /= 2;
        }

        uint32_t max_local_size_x = max_local_size_xy / local_size_y;
        local_size_x = std::min(max_local_size_x, vkdev->info.max_workgroup_size[0]);
    }
    else if (w > 0)
    {
        local_size_x = std::min(max_local_size_xy, vkdev->info.max_workgroup_size[0]);
        while ((uint32_t)w < local_size_x)
        {
            local_size_x /= 2;
        }

        uint32_t max_local_size_y = max_local_size_xy / local_size_x;
        local_size_y = std::min(max_local_size_y, vkdev->info.max_workgroup_size[1]);
    }

//     fprintf(stderr, "local size = %d %d %d\n", local_size_x, local_size_y, local_size_z);
}

int Pipeline::create_descriptorset_layout(int binding_count)
{
    if (binding_count == 0)
    {
        descriptorset_layout = 0;
        return 0;
    }

    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings(binding_count);
    for (int i=0; i<binding_count; i++)
    {
        descriptorSetLayoutBindings[i].binding = i;
        descriptorSetLayoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[i].descriptorCount = 1;
        descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        descriptorSetLayoutBindings[i].pImmutableSamplers = 0;
    }

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.pNext = 0;
    descriptorSetLayoutCreateInfo.flags = 0;
    descriptorSetLayoutCreateInfo.bindingCount = binding_count;
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();

    if (vkdev->info.support_VK_KHR_push_descriptor)
    {
        descriptorSetLayoutCreateInfo.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    }

    VkResult ret = vkCreateDescriptorSetLayout(vkdev->vkdevice(), &descriptorSetLayoutCreateInfo, 0, &descriptorset_layout);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateDescriptorSetLayout failed %d\n", ret);
        return -1;
    }

    return 0;
}

int Pipeline::create_pipeline_layout(int push_constant_count)
{
    VkPushConstantRange pushConstantRange;
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(int) * push_constant_count;

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.pNext = 0;
    pipelineLayoutCreateInfo.flags = 0;

    if (descriptorset_layout)
    {
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorset_layout;
    }
    else
    {
    pipelineLayoutCreateInfo.setLayoutCount = 0;
    pipelineLayoutCreateInfo.pSetLayouts = 0;
    }

    if (push_constant_count > 0)
    {
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
    }
    else
    {
    pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    pipelineLayoutCreateInfo.pPushConstantRanges = 0;
    }

    VkResult ret = vkCreatePipelineLayout(vkdev->vkdevice(), &pipelineLayoutCreateInfo, 0, &pipeline_layout);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreatePipelineLayout failed %d\n", ret);
        return -1;
    }

    return 0;
}

int Pipeline::create_pipeline(const char* _name, const std::vector<vk_specialization_type>& specializations)
{
    std::string name = _name;

    if (vkdev->info.support_fp16_arithmetic)
    {
        name += "_fp16a";
    }
    else if (vkdev->info.support_fp16_storage)
    {
        name += "_fp16s";
    }

    VkShaderModule shader_module = vkdev->get_shader_module(name.c_str());

    const int specialization_count = specializations.size();

    // +3 for local_size_xyz
    std::vector<VkSpecializationMapEntry> specializationMapEntries;
    specializationMapEntries.resize(specialization_count + 3);

    for (int i=0; i<specialization_count; i++)
    {
        specializationMapEntries[i].constantID = i;
        specializationMapEntries[i].offset = i * sizeof(vk_specialization_type);
        specializationMapEntries[i].size = sizeof(vk_specialization_type);
    }

    std::vector<vk_specialization_type> specialization_data = specializations;

    // append local_size_xyz specialization
    {
        VkSpecializationMapEntry* local_size_xyz_entries = specializationMapEntries.data() + specialization_count;

        local_size_xyz_entries[0].constantID = 233;
        local_size_xyz_entries[0].offset = (specialization_count+0) * sizeof(vk_specialization_type);
        local_size_xyz_entries[0].size = sizeof(vk_specialization_type);

        local_size_xyz_entries[1].constantID = 234;
        local_size_xyz_entries[1].offset = (specialization_count+1) * sizeof(vk_specialization_type);
        local_size_xyz_entries[1].size = sizeof(vk_specialization_type);

        local_size_xyz_entries[2].constantID = 235;
        local_size_xyz_entries[2].offset = (specialization_count+2) * sizeof(vk_specialization_type);
        local_size_xyz_entries[2].size = sizeof(vk_specialization_type);

        specialization_data.resize(specialization_count + 3);
        specialization_data[ specialization_count+0 ].u32 = local_size_x;
        specialization_data[ specialization_count+1 ].u32 = local_size_y;
        specialization_data[ specialization_count+2 ].u32 = local_size_z;
    }

    VkSpecializationInfo specializationInfo;
    specializationInfo.mapEntryCount = specializationMapEntries.size();
    specializationInfo.pMapEntries = specializationMapEntries.data();
    specializationInfo.dataSize = specialization_data.size() * sizeof(vk_specialization_type);
    specializationInfo.pData = specialization_data.data();

    VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo;
    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfo.pNext = 0;
    pipelineShaderStageCreateInfo.flags = 0;
    pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineShaderStageCreateInfo.module = shader_module;
    pipelineShaderStageCreateInfo.pName = name.c_str();
    pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;

    VkComputePipelineCreateInfo computePipelineCreateInfo;
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = 0;
    computePipelineCreateInfo.flags = 0;
    computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
    computePipelineCreateInfo.layout = pipeline_layout;
    computePipelineCreateInfo.basePipelineHandle = 0;
    computePipelineCreateInfo.basePipelineIndex = 0;

    VkResult ret = vkCreateComputePipelines(vkdev->vkdevice(), 0, 1, &computePipelineCreateInfo, 0, &pipeline);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateComputePipelines failed %d\n", ret);
        return -1;
    }

    return 0;
}

int Pipeline::create_descriptor_update_template(int binding_count)
{
    if (binding_count == 0)
    {
        descriptor_update_template = 0;
        return 0;
    }

    std::vector<VkDescriptorUpdateTemplateEntryKHR> descriptorUpdateTemplateEntries(binding_count);
    for (int i=0; i<binding_count; i++)// TODO do not update weights
    {
        descriptorUpdateTemplateEntries[i].dstBinding = i;
        descriptorUpdateTemplateEntries[i].dstArrayElement = 0;
        descriptorUpdateTemplateEntries[i].descriptorCount = 1;
        descriptorUpdateTemplateEntries[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorUpdateTemplateEntries[i].offset = i * sizeof(VkDescriptorBufferInfo);
        descriptorUpdateTemplateEntries[i].stride = sizeof(VkDescriptorBufferInfo);
    }

    VkDescriptorUpdateTemplateCreateInfoKHR descriptorUpdateTemplateCreateInfo;
    descriptorUpdateTemplateCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR;
    descriptorUpdateTemplateCreateInfo.pNext = 0;
    descriptorUpdateTemplateCreateInfo.flags = 0;
    descriptorUpdateTemplateCreateInfo.descriptorUpdateEntryCount = binding_count;// TODO do not update weights
    descriptorUpdateTemplateCreateInfo.pDescriptorUpdateEntries = descriptorUpdateTemplateEntries.data();
    if (vkdev->info.support_VK_KHR_push_descriptor)
    {
    descriptorUpdateTemplateCreateInfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
    }
    else
    {
    descriptorUpdateTemplateCreateInfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET_KHR;
    }
    // descriptorSetLayout should be ignored if VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR
    // FIXME HACK WARNING TODO NOTE but crash on radv if set NULL  :(
    descriptorUpdateTemplateCreateInfo.descriptorSetLayout = descriptorset_layout;
    descriptorUpdateTemplateCreateInfo.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    descriptorUpdateTemplateCreateInfo.pipelineLayout = pipeline_layout;
    descriptorUpdateTemplateCreateInfo.set = 0;

    VkResult ret = vkdev->vkCreateDescriptorUpdateTemplateKHR(vkdev->vkdevice(), &descriptorUpdateTemplateCreateInfo, 0, &descriptor_update_template);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateDescriptorUpdateTemplateKHR failed %d\n", ret);
        return -1;
    }

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
