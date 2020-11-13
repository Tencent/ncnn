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

#include "mat.h"

#include <vulkan/vulkan.h>

namespace ncnn {

// instance
int create_gpu_instance();
void destroy_gpu_instance();

// instance extension capability
extern int support_VK_KHR_external_memory_capabilities;
extern int support_VK_KHR_get_physical_device_properties2;
extern int support_VK_KHR_get_surface_capabilities2;
extern int support_VK_KHR_surface;
extern int support_VK_EXT_debug_utils;
#if __ANDROID_API__ >= 26
extern int support_VK_KHR_android_surface;
#endif // __ANDROID_API__ >= 26

// VK_KHR_external_memory_capabilities
extern PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR vkGetPhysicalDeviceExternalBufferPropertiesKHR;

// VK_KHR_get_physical_device_properties2
extern PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR;
extern PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR;
extern PFN_vkGetPhysicalDeviceFormatProperties2KHR vkGetPhysicalDeviceFormatProperties2KHR;
extern PFN_vkGetPhysicalDeviceImageFormatProperties2KHR vkGetPhysicalDeviceImageFormatProperties2KHR;
extern PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR vkGetPhysicalDeviceQueueFamilyProperties2KHR;
extern PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR;
extern PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR vkGetPhysicalDeviceSparseImageFormatProperties2KHR;

// VK_KHR_get_surface_capabilities2
extern PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR vkGetPhysicalDeviceSurfaceCapabilities2KHR;
extern PFN_vkGetPhysicalDeviceSurfaceFormats2KHR vkGetPhysicalDeviceSurfaceFormats2KHR;

// VK_KHR_surface
extern PFN_vkDestroySurfaceKHR vkDestroySurfaceKHR;
extern PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR;
extern PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR;
extern PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR;
extern PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR;

#if __ANDROID_API__ >= 26
// VK_KHR_android_surface
extern PFN_vkCreateAndroidSurfaceKHR vkCreateAndroidSurfaceKHR;
#endif // __ANDROID_API__ >= 26

// get info
int get_gpu_count();
int get_default_gpu_index();

class GpuInfo
{
public:
    // vulkan physical device
    VkPhysicalDevice physical_device;

    // memory properties
    VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;

    // info
    uint32_t api_version;
    uint32_t driver_version;
    uint32_t vendor_id;
    uint32_t device_id;
    std::string device_name;
    uint8_t pipeline_cache_uuid[VK_UUID_SIZE];

    // 0 = discrete gpu
    // 1 = integrated gpu
    // 2 = virtual gpu
    // 3 = cpu
    int type;

    // hardware limit
    uint32_t max_shared_memory_size;
    uint32_t max_workgroup_count[3];
    uint32_t max_workgroup_invocations;
    uint32_t max_workgroup_size[3];
    size_t memory_map_alignment;
    size_t buffer_offset_alignment;
    size_t non_coherent_atom_size;
    size_t buffer_image_granularity;
    uint32_t max_image_dimension_1d;
    uint32_t max_image_dimension_2d;
    uint32_t max_image_dimension_3d;
    float timestamp_period;

    // runtime
    uint32_t compute_queue_family_index;
    uint32_t graphics_queue_family_index;
    uint32_t transfer_queue_family_index;

    uint32_t compute_queue_count;
    uint32_t graphics_queue_count;
    uint32_t transfer_queue_count;

    // property
    bool unified_compute_transfer_queue;

    // subgroup
    uint32_t subgroup_size;
    bool support_subgroup_basic;
    bool support_subgroup_vote;
    bool support_subgroup_ballot;
    bool support_subgroup_shuffle;

    // bug is not feature
    bool bug_storage_buffer_no_l1;
    bool bug_corrupted_online_pipeline_cache;

    // but sometimes bug is a feature
    bool bug_implicit_fp16_arithmetic;

    // fp16 and int8 feature
    bool support_fp16_packed;
    bool support_fp16_storage;
    bool support_fp16_arithmetic;
    bool support_int8_storage;
    bool support_int8_arithmetic;

    // ycbcr conversion feature
    bool support_ycbcr_conversion;

    // extension capability
    int support_VK_KHR_8bit_storage;
    int support_VK_KHR_16bit_storage;
    int support_VK_KHR_bind_memory2;
    int support_VK_KHR_dedicated_allocation;
    int support_VK_KHR_descriptor_update_template;
    int support_VK_KHR_external_memory;
    int support_VK_KHR_get_memory_requirements2;
    int support_VK_KHR_maintenance1;
    int support_VK_KHR_push_descriptor;
    int support_VK_KHR_sampler_ycbcr_conversion;
    int support_VK_KHR_shader_float16_int8;
    int support_VK_KHR_shader_float_controls;
    int support_VK_KHR_storage_buffer_storage_class;
    int support_VK_KHR_swapchain;
    int support_VK_EXT_memory_budget;
    int support_VK_EXT_queue_family_foreign;
#if __ANDROID_API__ >= 26
    int support_VK_ANDROID_external_memory_android_hardware_buffer;
#endif // __ANDROID_API__ >= 26
};

const GpuInfo& get_gpu_info(int device_index = get_default_gpu_index());

class VkAllocator;
class VkCompute;
class Layer;
class Packing_vulkan;
class Option;
class PipelineCache;
class VulkanDevice
{
public:
    VulkanDevice(int device_index = get_default_gpu_index());
    ~VulkanDevice();

    const GpuInfo& info;

    VkDevice vkdevice() const
    {
        return device;
    }

    VkShaderModule compile_shader_module(const uint32_t* spv_data, size_t spv_data_size) const;

    // with fixed workgroup size
    VkShaderModule compile_shader_module(const uint32_t* spv_data, size_t spv_data_size, uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z) const;

    // helper for creating pipeline
    int create_descriptorset_layout(int binding_count, const int* binding_types, VkDescriptorSetLayout* descriptorset_layout) const;
    int create_pipeline_layout(int push_constant_count, VkDescriptorSetLayout descriptorset_layout, VkPipelineLayout* pipeline_layout) const;
    int create_pipeline(VkShaderModule shader_module, VkPipelineLayout pipeline_layout, const std::vector<vk_specialization_type>& specializations, VkPipeline* pipeline) const;
    int create_descriptor_update_template(int binding_count, const int* binding_types, VkDescriptorSetLayout descriptorset_layout, VkPipelineLayout pipeline_layout, VkDescriptorUpdateTemplateKHR* descriptor_update_template) const;

    uint32_t find_memory_index(uint32_t memory_type_bits, VkFlags required, VkFlags preferred, VkFlags preferred_not) const;
    bool is_mappable(uint32_t memory_type_index) const;
    bool is_coherent(uint32_t memory_type_index) const;

    VkQueue acquire_queue(uint32_t queue_family_index) const;
    void reclaim_queue(uint32_t queue_family_index, VkQueue queue) const;

    // allocator on this device
    VkAllocator* acquire_blob_allocator() const;
    void reclaim_blob_allocator(VkAllocator* allocator) const;

    VkAllocator* acquire_staging_allocator() const;
    void reclaim_staging_allocator(VkAllocator* allocator) const;

    // immutable sampler for texelfetch
    const VkSampler* immutable_texelfetch_sampler() const;

    // dummy buffer image
    VkMat get_dummy_buffer() const;
    VkImageMat get_dummy_image() const;

    // pipeline cache on this device
    const PipelineCache* get_pipeline_cache() const;

    // test image allocation
    bool shape_support_image_storage(const Mat& shape) const;

    // current gpu heap memory budget in MB
    uint32_t get_heap_budget() const;

    // utility operator
    void convert_packing(const VkMat& src, VkMat& dst, int dst_elempack, VkCompute& cmd, const Option& opt) const;
    void convert_packing(const VkImageMat& src, VkImageMat& dst, int dst_elempack, VkCompute& cmd, const Option& opt) const;
    void convert_packing(const VkMat& src, VkImageMat& dst, int dst_elempack, VkCompute& cmd, const Option& opt) const;
    void convert_packing(const VkImageMat& src, VkMat& dst, int dst_elempack, VkCompute& cmd, const Option& opt) const;

    // VK_KHR_bind_memory2
    PFN_vkBindBufferMemory2KHR vkBindBufferMemory2KHR;
    PFN_vkBindImageMemory2KHR vkBindImageMemory2KHR;

    // VK_KHR_descriptor_update_template
    PFN_vkCreateDescriptorUpdateTemplateKHR vkCreateDescriptorUpdateTemplateKHR;
    PFN_vkDestroyDescriptorUpdateTemplateKHR vkDestroyDescriptorUpdateTemplateKHR;
    PFN_vkUpdateDescriptorSetWithTemplateKHR vkUpdateDescriptorSetWithTemplateKHR;

    // VK_KHR_get_memory_requirements2
    PFN_vkGetImageMemoryRequirements2KHR vkGetImageMemoryRequirements2KHR;
    PFN_vkGetBufferMemoryRequirements2KHR vkGetBufferMemoryRequirements2KHR;
    PFN_vkGetImageSparseMemoryRequirements2KHR vkGetImageSparseMemoryRequirements2KHR;

    // VK_KHR_maintenance1
    PFN_vkTrimCommandPoolKHR vkTrimCommandPoolKHR;

    // VK_KHR_push_descriptor
    PFN_vkCmdPushDescriptorSetWithTemplateKHR vkCmdPushDescriptorSetWithTemplateKHR;
    PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR;

    // VK_KHR_sampler_ycbcr_conversion
    PFN_vkCreateSamplerYcbcrConversionKHR vkCreateSamplerYcbcrConversionKHR;
    PFN_vkDestroySamplerYcbcrConversionKHR vkDestroySamplerYcbcrConversionKHR;

    // VK_KHR_swapchain
    PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR;
    PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR;
    PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR;
    PFN_vkAcquireNextImageKHR vkAcquireNextImageKHR;
    PFN_vkQueuePresentKHR vkQueuePresentKHR;

#if __ANDROID_API__ >= 26
    // VK_ANDROID_external_memory_android_hardware_buffer
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAndroidHardwareBufferPropertiesANDROID;
    PFN_vkGetMemoryAndroidHardwareBufferANDROID vkGetMemoryAndroidHardwareBufferANDROID;
#endif // __ANDROID_API__ >= 26

protected:
    // device extension
    int init_device_extension();

    // dummy buffer and image
    int create_dummy_buffer_image();
    void destroy_dummy_buffer_image();

    // utility operator
    const ncnn::Packing_vulkan* get_utility_operator(int storage_type_from, int storage_type_to, int cast_type_from_index, int cast_type_to_index, int packing_type_to_index) const;
    void destroy_utility_operator();

private:
    VkDevice device;

    // hardware queue
    mutable std::vector<VkQueue> compute_queues;
    mutable std::vector<VkQueue> graphics_queues;
    mutable std::vector<VkQueue> transfer_queues;
    mutable Mutex queue_lock;

    // default blob allocator for each queue
    mutable std::vector<VkAllocator*> blob_allocators;
    mutable Mutex blob_allocator_lock;

    // default staging allocator for each queue
    mutable std::vector<VkAllocator*> staging_allocators;
    mutable Mutex staging_allocator_lock;

    // nearest sampler for texelfetch
    VkSampler texelfetch_sampler;

    // dummy buffer and image
    VkAllocator* dummy_allocator;
    VkMat dummy_buffer;
    VkImageMat dummy_image;

    // device-wide pipeline cache
    PipelineCache* pipeline_cache;

    // utility operator
    // from buffer | image
    // to buffer | image
    // from fp32-b/i | fp16p-b/i | fp16s-b/i
    // to fp32-b/i | fp16p-b/i | fp16s-b/i
    // to pack1 | pack4 | pack8
    mutable ncnn::Packing_vulkan* uop_packing[2][2][3][3][3];
    mutable Mutex uop_lock;
};

VulkanDevice* get_gpu_device(int device_index = get_default_gpu_index());

// online spirv compilation
int compile_spirv_module(const char* comp_data, int comp_data_size, const Option& opt, std::vector<uint32_t>& spirv);
int compile_spirv_module(int shader_type_index, const Option& opt, std::vector<uint32_t>& spirv);

// info from spirv
class ShaderInfo
{
public:
    int specialization_count;
    int binding_count;
    int push_constant_count;

    // 0 = null
    // 1 = storage buffer
    // 2 = storage image
    // 3 = combined image sampler
    int binding_types[16]; // 16 is large enough I think ...
};

int resolve_shader_info(const uint32_t* spv_data, size_t spv_data_size, ShaderInfo& shader_info);

} // namespace ncnn

#endif // NCNN_VULKAN

#endif // NCNN_GPU_H
