// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "layer.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include "cpu.h"

namespace ncnn {

Option::Option()
{
    lightmode = true;
    num_threads = get_cpu_count();
    blob_allocator = 0;
    workspace_allocator = 0;

#if NCNN_VULKAN
    vkdev = 0;
    blob_vkallocator = 0;
    workspace_vkallocator = 0;
#endif // NCNN_VULKAN
}

static Option g_default_option;

const Option& get_default_option()
{
    return g_default_option;
}

int set_default_option(const Option& opt)
{
    if (opt.num_threads <= 0)
    {
        fprintf(stderr, "invalid option num_threads %d\n", opt.num_threads);
        return -1;
    }

    g_default_option = opt;

    return 0;
}

Layer::Layer()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = false;
}

Layer::~Layer()
{
}

int Layer::load_param(const ParamDict& /*pd*/)
{
    return 0;
}

int Layer::load_model(const ModelBin& /*mb*/)
{
    return 0;
}

int Layer::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    if (!support_inplace)
        return -1;

    top_blobs = bottom_blobs;
    for (int i = 0; i < (int)top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blobs[i].clone(opt.blob_allocator);
        if (top_blobs[i].empty())
            return -100;
    }

    return forward_inplace(top_blobs, opt);
}

int Layer::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (!support_inplace)
        return -1;

    top_blob = bottom_blob.clone(opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    return forward_inplace(top_blob, opt);
}

int Layer::forward_inplace(std::vector<Mat>& /*bottom_top_blobs*/, const Option& /*opt*/) const
{
    return -1;
}

int Layer::forward_inplace(Mat& /*bottom_top_blob*/, const Option& /*opt*/) const
{
    return -1;
}

#if NCNN_VULKAN
int Layer::create_vulkan_pipeline()
{
    create_descriptorset_layout();

    create_pipeline_layout();

    create_pipeline();

    create_descriptor_update_template();

    return 0;
}

int Layer::destroy_vulkan_pipeline()
{
    vkdev->vkDestroyDescriptorUpdateTemplateKHR(vkdev->vkdevice(), descriptor_update_template, 0);

    vkDestroyPipeline(vkdev->vkdevice(), pipeline, 0);

    vkDestroyPipelineLayout(vkdev->vkdevice(), pipeline_layout, 0);

    vkDestroyDescriptorSetLayout(vkdev->vkdevice(), descriptorset_layout, 0);

    return 0;
}

int Layer::forward(const std::vector<VkMat>& /*bottom_blobs*/, std::vector<VkMat>& /*top_blobs*/, Command& /*cmd*/, const Option& /*opt*/) const
{
    return -1;
}

int Layer::forward(const VkMat& /*bottom_blob*/, VkMat& /*top_blob*/, Command& /*cmd*/, const Option& /*opt*/) const
{
    return -1;
}

int Layer::forward_inplace(std::vector<VkMat>& /*bottom_top_blobs*/, Command& /*cmd*/, const Option& /*opt*/) const
{
    return -1;
}

int Layer::forward_inplace(VkMat& /*bottom_top_blob*/, Command& /*cmd*/, const Option& /*opt*/) const
{
    return -1;
}

int Layer::create_descriptorset_layout()
{
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
    descriptorSetLayoutCreateInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    descriptorSetLayoutCreateInfo.bindingCount = binding_count;
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();

    VkResult ret = vkCreateDescriptorSetLayout(vkdev->vkdevice(), &descriptorSetLayoutCreateInfo, 0, &descriptorset_layout);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateDescriptorSetLayout failed %d\n", ret);
        return -1;
    }

    return 0;
}

int Layer::create_pipeline_layout()
{
    VkPushConstantRange pushConstantRange;
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(int) * push_constant_count;

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.pNext = 0;
    pipelineLayoutCreateInfo.flags = 0;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorset_layout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

    VkResult ret = vkCreatePipelineLayout(vkdev->vkdevice(), &pipelineLayoutCreateInfo, 0, &pipeline_layout);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreatePipelineLayout failed %d\n", ret);
        return -1;
    }

    return 0;
}

int Layer::create_pipeline()
{
    const int specialization_count = specializations.size();

    // +3 for local_size_xyz
    std::vector<VkSpecializationMapEntry> specializationMapEntries;
    specializationMapEntries.resize(specialization_count + 3);

    for (int i=0; i<specialization_count; i++)
    {
        specializationMapEntries[i].constantID = i;
        specializationMapEntries[i].offset = i * sizeof(int);
        specializationMapEntries[i].size = sizeof(int);
    }

    std::vector<int> specialization_data = specializations;

    // append local_size_xyz specialization
    VkSpecializationMapEntry* local_size_xyz_entries = specializationMapEntries.data() + specialization_count;

    local_size_xyz_entries[0].constantID = 233;
    local_size_xyz_entries[0].offset = (specialization_count+0) * sizeof(int);
    local_size_xyz_entries[0].size = sizeof(int);

    local_size_xyz_entries[1].constantID = 234;
    local_size_xyz_entries[1].offset = (specialization_count+1) * sizeof(int);
    local_size_xyz_entries[1].size = sizeof(int);

    local_size_xyz_entries[2].constantID = 235;
    local_size_xyz_entries[2].offset = (specialization_count+2) * sizeof(int);
    local_size_xyz_entries[2].size = sizeof(int);

    specialization_data.push_back(local_size_x);
    specialization_data.push_back(local_size_y);
    specialization_data.push_back(local_size_z);

    VkSpecializationInfo specializationInfo;
    specializationInfo.mapEntryCount = specializationMapEntries.size();
    specializationInfo.pMapEntries = specializationMapEntries.data();
    specializationInfo.dataSize = specialization_data.size() * sizeof(int);
    specializationInfo.pData = specialization_data.data();

    VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo;
    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfo.pNext = 0;
    pipelineShaderStageCreateInfo.flags = 0;
    pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineShaderStageCreateInfo.module = shader_module;
    pipelineShaderStageCreateInfo.pName = "main";
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

int Layer::create_descriptor_update_template()
{
    std::vector<VkDescriptorUpdateTemplateEntry> descriptorUpdateTemplateEntries(binding_count);
    for (int i=0; i<binding_count; i++)// TODO do not update weights
    {
        descriptorUpdateTemplateEntries[i].dstBinding = i;
        descriptorUpdateTemplateEntries[i].dstArrayElement = 0;
        descriptorUpdateTemplateEntries[i].descriptorCount = 1;
        descriptorUpdateTemplateEntries[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorUpdateTemplateEntries[i].offset = i * sizeof(VkDescriptorBufferInfo);
        descriptorUpdateTemplateEntries[i].stride = sizeof(VkDescriptorBufferInfo);
    }

    VkDescriptorUpdateTemplateCreateInfo descriptorUpdateTemplateCreateInfo;
    descriptorUpdateTemplateCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
    descriptorUpdateTemplateCreateInfo.pNext = 0;
    descriptorUpdateTemplateCreateInfo.flags = 0;
    descriptorUpdateTemplateCreateInfo.descriptorUpdateEntryCount = binding_count;// TODO do not update weights
    descriptorUpdateTemplateCreateInfo.pDescriptorUpdateEntries = descriptorUpdateTemplateEntries.data();
    descriptorUpdateTemplateCreateInfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
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

#include "layer_declaration.h"

static const layer_registry_entry layer_registry[] =
{
#include "layer_registry.h"
};

static const int layer_registry_entry_count = sizeof(layer_registry) / sizeof(layer_registry_entry);

#if NCNN_STRING
int layer_to_index(const char* type)
{
    for (int i=0; i<layer_registry_entry_count; i++)
    {
        if (strcmp(type, layer_registry[i].name) == 0)
            return i;
    }

    return -1;
}

Layer* create_layer(const char* type)
{
    int index = layer_to_index(type);
    if (index == -1)
        return 0;

    return create_layer(index);
}
#endif // NCNN_STRING

Layer* create_layer(int index)
{
    if (index < 0 || index >= layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    return layer_creator();
}

} // namespace ncnn
