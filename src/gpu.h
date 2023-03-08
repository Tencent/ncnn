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

#include "vulkan_header_fix.h"

namespace ncnn {

// instance
NCNN_EXPORT int create_gpu_instance();
NCNN_EXPORT void destroy_gpu_instance();

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

// VK_NV_cooperative_matrix
extern PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV vkGetPhysicalDeviceCooperativeMatrixPropertiesNV;

// get info
NCNN_EXPORT int get_gpu_count();
NCNN_EXPORT int get_default_gpu_index();

class GpuInfoPrivate;
class NCNN_EXPORT GpuInfo
{
public:
    explicit GpuInfo();
    virtual ~GpuInfo();

    // vulkan physical device
    VkPhysicalDevice physical_device() const;

    // memory properties
    const VkPhysicalDeviceMemoryProperties& physical_device_memory_properties() const;

    // info
    uint32_t api_version() const;
    uint32_t driver_version() const;
    uint32_t vendor_id() const;
    uint32_t device_id() const;
    const char* device_name() const;
    uint8_t* pipeline_cache_uuid() const;

    // 0 = discrete gpu
    // 1 = integrated gpu
    // 2 = virtual gpu
    // 3 = cpu
    int type() const;

    // hardware limit
    uint32_t max_shared_memory_size() const;
    uint32_t max_workgroup_count_x() const;
    uint32_t max_workgroup_count_y() const;
    uint32_t max_workgroup_count_z() const;
    uint32_t max_workgroup_invocations() const;
    uint32_t max_workgroup_size_x() const;
    uint32_t max_workgroup_size_y() const;
    uint32_t max_workgroup_size_z() const;
    size_t memory_map_alignment() const;
    size_t buffer_offset_alignment() const;
    size_t non_coherent_atom_size() const;
    size_t buffer_image_granularity() const;
    uint32_t max_image_dimension_1d() const;
    uint32_t max_image_dimension_2d() const;
    uint32_t max_image_dimension_3d() const;
    float timestamp_period() const;

    // runtime
    uint32_t compute_queue_family_index() const;
    uint32_t graphics_queue_family_index() const;
    uint32_t transfer_queue_family_index() const;

    uint32_t compute_queue_count() const;
    uint32_t graphics_queue_count() const;
    uint32_t transfer_queue_count() const;

    // property
    bool unified_compute_transfer_queue() const;

    // subgroup
    uint32_t subgroup_size() const;
    bool support_subgroup_basic() const;
    bool support_subgroup_vote() const;
    bool support_subgroup_ballot() const;
    bool support_subgroup_shuffle() const;

    // bug is not feature
    bool bug_storage_buffer_no_l1() const;
    bool bug_corrupted_online_pipeline_cache() const;
    bool bug_buffer_image_load_zero() const;

    // but sometimes bug is a feature
    bool bug_implicit_fp16_arithmetic() const;

    // fp16 and int8 feature
    bool support_fp16_packed() const;
    bool support_fp16_storage() const;
    bool support_fp16_arithmetic() const;
    bool support_int8_packed() const;
    bool support_int8_storage() const;
    bool support_int8_arithmetic() const;

    // ycbcr conversion feature
    bool support_ycbcr_conversion() const;

    // cooperative matrix feature
    bool support_cooperative_matrix() const;
    bool support_cooperative_matrix_16_8_8() const;

    // extension capability
    int support_VK_KHR_8bit_storage() const;
    int support_VK_KHR_16bit_storage() const;
    int support_VK_KHR_bind_memory2() const;
    int support_VK_KHR_create_renderpass2() const;
    int support_VK_KHR_dedicated_allocation() const;
    int support_VK_KHR_descriptor_update_template() const;
    int support_VK_KHR_external_memory() const;
    int support_VK_KHR_get_memory_requirements2() const;
    int support_VK_KHR_maintenance1() const;
    int support_VK_KHR_maintenance2() const;
    int support_VK_KHR_maintenance3() const;
    int support_VK_KHR_multiview() const;
    int support_VK_KHR_portability_subset() const;
    int support_VK_KHR_push_descriptor() const;
    int support_VK_KHR_sampler_ycbcr_conversion() const;
    int support_VK_KHR_shader_float16_int8() const;
    int support_VK_KHR_shader_float_controls() const;
    int support_VK_KHR_storage_buffer_storage_class() const;
    int support_VK_KHR_swapchain() const;
    int support_VK_EXT_descriptor_indexing() const;
    int support_VK_EXT_memory_budget() const;
    int support_VK_EXT_queue_family_foreign() const;
#if __ANDROID_API__ >= 26
    int support_VK_ANDROID_external_memory_android_hardware_buffer() const;
#endif // __ANDROID_API__ >= 26
    int support_VK_NV_cooperative_matrix() const;

private:
    GpuInfo(const GpuInfo&);
    GpuInfo& operator=(const GpuInfo&);

private:
    friend int create_gpu_instance();
    GpuInfoPrivate* const d;
};

NCNN_EXPORT const GpuInfo& get_gpu_info(int device_index = get_default_gpu_index());

class VkAllocator;
class VkCompute;
class Option;
class PipelineCache;
class VulkanDevicePrivate;
class NCNN_EXPORT VulkanDevice
{
public:
    VulkanDevice(int device_index = get_default_gpu_index());
    ~VulkanDevice();

    const GpuInfo& info;

    VkDevice vkdevice() const;

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
    VkImageMat get_dummy_image_readonly() const;

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

    // VK_KHR_create_renderpass2
    PFN_vkCmdBeginRenderPass2KHR vkCmdBeginRenderPass2KHR;
    PFN_vkCmdEndRenderPass2KHR vkCmdEndRenderPass2KHR;
    PFN_vkCmdNextSubpass2KHR vkCmdNextSubpass2KHR;
    PFN_vkCreateRenderPass2KHR vkCreateRenderPass2KHR;

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

    // VK_KHR_maintenance3
    PFN_vkGetDescriptorSetLayoutSupportKHR vkGetDescriptorSetLayoutSupportKHR;

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

private:
    VulkanDevice(const VulkanDevice&);
    VulkanDevice& operator=(const VulkanDevice&);

private:
    VulkanDevicePrivate* const d;
};

NCNN_EXPORT VulkanDevice* get_gpu_device(int device_index = get_default_gpu_index());

// online spirv compilation
NCNN_EXPORT int compile_spirv_module(const char* comp_string, const Option& opt, std::vector<uint32_t>& spirv);
NCNN_EXPORT int compile_spirv_module(const char* comp_data, int comp_data_size, const Option& opt, std::vector<uint32_t>& spirv);
NCNN_EXPORT int compile_spirv_module(int shader_type_index, const Option& opt, std::vector<uint32_t>& spirv);

// info from spirv
class NCNN_EXPORT ShaderInfo
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

    int reserved_0;
    int reserved_1;
    int reserved_2;
    int reserved_3;
};

NCNN_EXPORT int resolve_shader_info(const uint32_t* spv_data, size_t spv_data_size, ShaderInfo& shader_info);

} // namespace ncnn

#endif // NCNN_VULKAN

#endif // NCNN_GPU_H
