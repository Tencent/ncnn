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

#ifndef NCNN_GPU_H
#define NCNN_GPU_H

#include "platform.h"

#if NCNN_VULKAN

#include <vulkan/vulkan.h>
#include <vector>

namespace ncnn {

// instance
int create_gpu_instance();
void destroy_gpu_instance();

// get info
int get_gpu_count();
int get_default_gpu_index();

class GpuInfo
{
public:
    // vulkan physical device
    VkPhysicalDevice physical_device;

    // 0 = discrete gpu
    // 1 = integrated gpu
    // 2 = virtual gpu
    // 3 = cpu
    int type;

    // hardware capability
    int max_shared_memory_size;
    int max_workgroup_count[3];
    int max_workgroup_invocations;
    int max_workgroup_size[3];

    // runtime
    uint32_t compute_queue_index;
    uint32_t transfer_queue_index;

    uint32_t unified_memory_index;
    uint32_t device_local_memory_index;
    uint32_t host_visible_memory_index;

    // extension capability
    int support_VK_KHR_descriptor_update_template;
    int support_VK_KHR_push_descriptor;
    int support_VK_AMD_gpu_shader_half_float;
};

const GpuInfo& get_gpu_info(int device_index = get_default_gpu_index());

// class VkAllocator;
// class VkMat;
class VulkanDevice
{
public:
    VulkanDevice(int device_index = get_default_gpu_index());
    ~VulkanDevice();

    const GpuInfo& info;

    VkDevice vkdevice() const { return device; }

    VkShaderModule get_shader_module(int type_index) const;

    // VK_KHR_descriptor_update_template
    PFN_vkCreateDescriptorUpdateTemplateKHR vkCreateDescriptorUpdateTemplateKHR;
    PFN_vkDestroyDescriptorUpdateTemplateKHR vkDestroyDescriptorUpdateTemplateKHR;
    PFN_vkUpdateDescriptorSetWithTemplateKHR vkUpdateDescriptorSetWithTemplateKHR;
    PFN_vkCmdPushDescriptorSetWithTemplateKHR vkCmdPushDescriptorSetWithTemplateKHR;

    // support_VK_KHR_push_descriptor
    PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR;

protected:
    // shader management
    int create_shader_module();
    void destroy_shader_module();

    // device extension
    int init_device_extension();

private:
    VkDevice device;

    std::vector<VkShaderModule> shader_modules;
};

} // namespace ncnn

#endif // NCNN_VULKAN

#endif // NCNN_GPU_H
