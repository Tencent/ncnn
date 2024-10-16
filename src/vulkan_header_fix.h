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

#ifndef NCNN_VULKAN_HEADER_FIX_H
#define NCNN_VULKAN_HEADER_FIX_H

// #include "platform.h"

// This header contains new structure and function declearation to fix build with old vulkan sdk

#ifndef VK_KHR_maintenance1
#define VK_KHR_maintenance1 1
typedef VkFlags VkCommandPoolTrimFlags;
typedef VkCommandPoolTrimFlags VkCommandPoolTrimFlagsKHR;
typedef void(VKAPI_PTR* PFN_vkTrimCommandPool)(VkDevice device, VkCommandPool commandPool, VkCommandPoolTrimFlags flags);
typedef PFN_vkTrimCommandPool PFN_vkTrimCommandPoolKHR;
#endif // VK_KHR_maintenance1

#ifndef VK_KHR_get_physical_device_properties2
#define VK_KHR_get_physical_device_properties2                    1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2              (VkStructureType)1000059000
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2            (VkStructureType)1000059001
#define VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2                     (VkStructureType)1000059002
#define VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2               (VkStructureType)1000059003
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2     (VkStructureType)1000059004
#define VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2               (VkStructureType)1000059005
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2     (VkStructureType)1000059006
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR          VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2
#define VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2_KHR                 VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2
#define VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2_KHR           VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2_KHR VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2
#define VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2_KHR           VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2
typedef struct VkPhysicalDeviceFeatures2
{
    VkStructureType sType;
    void* pNext;
    VkPhysicalDeviceFeatures features;
} VkPhysicalDeviceFeatures2;
typedef struct VkPhysicalDeviceProperties2
{
    VkStructureType sType;
    void* pNext;
    VkPhysicalDeviceProperties properties;
} VkPhysicalDeviceProperties2;
typedef struct VkFormatProperties2
{
    VkStructureType sType;
    void* pNext;
    VkFormatProperties formatProperties;
} VkFormatProperties2;
typedef struct VkImageFormatProperties2
{
    VkStructureType sType;
    void* pNext;
    VkImageFormatProperties imageFormatProperties;
} VkImageFormatProperties2;
typedef struct VkPhysicalDeviceImageFormatInfo2
{
    VkStructureType sType;
    const void* pNext;
    VkFormat format;
    VkImageType type;
    VkImageTiling tiling;
    VkImageUsageFlags usage;
    VkImageCreateFlags flags;
} VkPhysicalDeviceImageFormatInfo2;
typedef struct VkQueueFamilyProperties2
{
    VkStructureType sType;
    void* pNext;
    VkQueueFamilyProperties queueFamilyProperties;
} VkQueueFamilyProperties2;
typedef struct VkPhysicalDeviceMemoryProperties2
{
    VkStructureType sType;
    void* pNext;
    VkPhysicalDeviceMemoryProperties memoryProperties;
} VkPhysicalDeviceMemoryProperties2;
typedef VkPhysicalDeviceFeatures2 VkPhysicalDeviceFeatures2KHR;
typedef VkPhysicalDeviceProperties2 VkPhysicalDeviceProperties2KHR;
typedef VkFormatProperties2 VkFormatProperties2KHR;
typedef VkImageFormatProperties2 VkImageFormatProperties2KHR;
typedef VkPhysicalDeviceImageFormatInfo2 VkPhysicalDeviceImageFormatInfo2KHR;
typedef VkQueueFamilyProperties2 VkQueueFamilyProperties2KHR;
typedef VkPhysicalDeviceMemoryProperties2 VkPhysicalDeviceMemoryProperties2KHR;
typedef void(VKAPI_PTR* PFN_vkGetPhysicalDeviceFeatures2)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceFeatures2* pFeatures);
typedef void(VKAPI_PTR* PFN_vkGetPhysicalDeviceProperties2)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceProperties2* pProperties);
typedef void(VKAPI_PTR* PFN_vkGetPhysicalDeviceFormatProperties2)(VkPhysicalDevice physicalDevice, VkFormat format, VkFormatProperties2* pFormatProperties);
typedef VkResult(VKAPI_PTR* PFN_vkGetPhysicalDeviceImageFormatProperties2)(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceImageFormatInfo2* pImageFormatInfo, VkImageFormatProperties2* pImageFormatProperties);
typedef void(VKAPI_PTR* PFN_vkGetPhysicalDeviceQueueFamilyProperties2)(VkPhysicalDevice physicalDevice, uint32_t* pQueueFamilyPropertyCount, VkQueueFamilyProperties2* pQueueFamilyProperties);
typedef void(VKAPI_PTR* PFN_vkGetPhysicalDeviceMemoryProperties2)(VkPhysicalDevice physicalDevice, VkPhysicalDeviceMemoryProperties2* pMemoryProperties);
typedef PFN_vkGetPhysicalDeviceFeatures2 PFN_vkGetPhysicalDeviceFeatures2KHR;
typedef PFN_vkGetPhysicalDeviceProperties2 PFN_vkGetPhysicalDeviceProperties2KHR;
typedef PFN_vkGetPhysicalDeviceFormatProperties2 PFN_vkGetPhysicalDeviceFormatProperties2KHR;
typedef PFN_vkGetPhysicalDeviceImageFormatProperties2 PFN_vkGetPhysicalDeviceImageFormatProperties2KHR;
typedef PFN_vkGetPhysicalDeviceQueueFamilyProperties2 PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR;
typedef PFN_vkGetPhysicalDeviceMemoryProperties2 PFN_vkGetPhysicalDeviceMemoryProperties2KHR;
#endif // VK_KHR_get_physical_device_properties2

#ifndef VK_KHR_external_memory_capabilities
#define VK_KHR_external_memory_capabilities                              1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO     (VkStructureType)1000071000
#define VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES               (VkStructureType)1000071001
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO           (VkStructureType)1000071002
#define VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES                     (VkStructureType)1000071003
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES                  (VkStructureType)1000071004
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO_KHR VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO
#define VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES_KHR           VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO_KHR       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO
#define VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES_KHR                 VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR              VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES
#define VK_LUID_SIZE                                                     8
#define VK_LUID_SIZE_KHR                                                 VK_LUID_SIZE
typedef enum VkExternalMemoryHandleTypeFlagBits
{
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT = 0x00000001,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT = 0x00000002,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT = 0x00000004,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT = 0x00000008,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT = 0x00000010,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT = 0x00000020,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT = 0x00000040,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT = 0x00000200,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID = 0x00000400,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT = 0x00000080,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT = 0x00000100,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA = 0x00000800,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_RDMA_ADDRESS_BIT_NV = 0x00001000,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkExternalMemoryHandleTypeFlagBits;
typedef VkFlags VkExternalMemoryHandleTypeFlags;
typedef enum VkExternalMemoryFeatureFlagBits
{
    VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT = 0x00000001,
    VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT = 0x00000002,
    VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT = 0x00000004,
    VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT_KHR = VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT,
    VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT_KHR = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT,
    VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT_KHR = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT,
    VK_EXTERNAL_MEMORY_FEATURE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkExternalMemoryFeatureFlagBits;
typedef VkFlags VkExternalMemoryFeatureFlags;
typedef struct VkExternalMemoryProperties
{
    VkExternalMemoryFeatureFlags externalMemoryFeatures;
    VkExternalMemoryHandleTypeFlags exportFromImportedHandleTypes;
    VkExternalMemoryHandleTypeFlags compatibleHandleTypes;
} VkExternalMemoryProperties;
typedef struct VkPhysicalDeviceExternalImageFormatInfo
{
    VkStructureType sType;
    const void* pNext;
    VkExternalMemoryHandleTypeFlagBits handleType;
} VkPhysicalDeviceExternalImageFormatInfo;
typedef struct VkExternalImageFormatProperties
{
    VkStructureType sType;
    void* pNext;
    VkExternalMemoryProperties externalMemoryProperties;
} VkExternalImageFormatProperties;
typedef struct VkPhysicalDeviceExternalBufferInfo
{
    VkStructureType sType;
    const void* pNext;
    VkBufferCreateFlags flags;
    VkBufferUsageFlags usage;
    VkExternalMemoryHandleTypeFlagBits handleType;
} VkPhysicalDeviceExternalBufferInfo;
typedef struct VkExternalBufferProperties
{
    VkStructureType sType;
    void* pNext;
    VkExternalMemoryProperties externalMemoryProperties;
} VkExternalBufferProperties;
typedef struct VkPhysicalDeviceIDProperties
{
    VkStructureType sType;
    void* pNext;
    uint8_t deviceUUID[VK_UUID_SIZE];
    uint8_t driverUUID[VK_UUID_SIZE];
    uint8_t deviceLUID[VK_LUID_SIZE];
    uint32_t deviceNodeMask;
    VkBool32 deviceLUIDValid;
} VkPhysicalDeviceIDProperties;
typedef VkExternalMemoryHandleTypeFlags VkExternalMemoryHandleTypeFlagsKHR;
typedef VkExternalMemoryHandleTypeFlagBits VkExternalMemoryHandleTypeFlagBitsKHR;
typedef VkExternalMemoryFeatureFlags VkExternalMemoryFeatureFlagsKHR;
typedef VkExternalMemoryFeatureFlagBits VkExternalMemoryFeatureFlagBitsKHR;
typedef VkExternalMemoryProperties VkExternalMemoryPropertiesKHR;
typedef VkPhysicalDeviceExternalImageFormatInfo VkPhysicalDeviceExternalImageFormatInfoKHR;
typedef VkExternalImageFormatProperties VkExternalImageFormatPropertiesKHR;
typedef VkPhysicalDeviceExternalBufferInfo VkPhysicalDeviceExternalBufferInfoKHR;
typedef VkExternalBufferProperties VkExternalBufferPropertiesKHR;
typedef VkPhysicalDeviceIDProperties VkPhysicalDeviceIDPropertiesKHR;
typedef void(VKAPI_PTR* PFN_vkGetPhysicalDeviceExternalBufferProperties)(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalBufferInfo* pExternalBufferInfo, VkExternalBufferProperties* pExternalBufferProperties);
typedef PFN_vkGetPhysicalDeviceExternalBufferProperties PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR;
#endif // VK_KHR_external_memory_capabilities

#ifndef VK_KHR_external_memory
#define VK_KHR_external_memory                                   1
#define VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO     (VkStructureType)1000072000
#define VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO      (VkStructureType)1000072001
#define VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO            (VkStructureType)1000072002
#define VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO_KHR VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO
#define VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO_KHR  VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO
#define VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR        VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO
#define VK_QUEUE_FAMILY_EXTERNAL_KHR                             (~0U - 1)
typedef struct VkExternalMemoryImageCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkExternalMemoryHandleTypeFlags handleTypes;
} VkExternalMemoryImageCreateInfo;
typedef struct VkExternalMemoryBufferCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkExternalMemoryHandleTypeFlags handleTypes;
} VkExternalMemoryBufferCreateInfo;
typedef struct VkExportMemoryAllocateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkExternalMemoryHandleTypeFlags handleTypes;
} VkExportMemoryAllocateInfo;
typedef VkExternalMemoryImageCreateInfo VkExternalMemoryImageCreateInfoKHR;
typedef VkExternalMemoryBufferCreateInfo VkExternalMemoryBufferCreateInfoKHR;
typedef VkExportMemoryAllocateInfo VkExportMemoryAllocateInfoKHR;
#endif // VK_KHR_external_memory

#ifndef VK_KHR_descriptor_update_template
#define VK_KHR_descriptor_update_template                            1
#define VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO     (VkStructureType)1000085000
#define VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkDescriptorUpdateTemplate)
typedef VkDescriptorUpdateTemplate VkDescriptorUpdateTemplateKHR;
typedef enum VkDescriptorUpdateTemplateType
{
    VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET = 0,
    VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR = 1,
    VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET_KHR = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET,
    VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_MAX_ENUM = 0x7FFFFFFF
} VkDescriptorUpdateTemplateType;
typedef VkFlags VkDescriptorUpdateTemplateCreateFlags;
typedef struct VkDescriptorUpdateTemplateEntry
{
    uint32_t dstBinding;
    uint32_t dstArrayElement;
    uint32_t descriptorCount;
    VkDescriptorType descriptorType;
    size_t offset;
    size_t stride;
} VkDescriptorUpdateTemplateEntry;
typedef struct VkDescriptorUpdateTemplateCreateInfo
{
    VkStructureType sType;
    void* pNext;
    VkDescriptorUpdateTemplateCreateFlags flags;
    uint32_t descriptorUpdateEntryCount;
    const VkDescriptorUpdateTemplateEntry* pDescriptorUpdateEntries;
    VkDescriptorUpdateTemplateType templateType;
    VkDescriptorSetLayout descriptorSetLayout;
    VkPipelineBindPoint pipelineBindPoint;
    VkPipelineLayout pipelineLayout;
    uint32_t set;
} VkDescriptorUpdateTemplateCreateInfo;
typedef VkDescriptorUpdateTemplateType VkDescriptorUpdateTemplateTypeKHR;
typedef VkDescriptorUpdateTemplateCreateFlags VkDescriptorUpdateTemplateCreateFlagsKHR;
typedef VkDescriptorUpdateTemplateEntry VkDescriptorUpdateTemplateEntryKHR;
typedef VkDescriptorUpdateTemplateCreateInfo VkDescriptorUpdateTemplateCreateInfoKHR;
typedef VkResult(VKAPI_PTR* PFN_vkCreateDescriptorUpdateTemplate)(VkDevice device, const VkDescriptorUpdateTemplateCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorUpdateTemplate* pDescriptorUpdateTemplate);
typedef void(VKAPI_PTR* PFN_vkDestroyDescriptorUpdateTemplate)(VkDevice device, VkDescriptorUpdateTemplate descriptorUpdateTemplate, const VkAllocationCallbacks* pAllocator);
typedef void(VKAPI_PTR* PFN_vkUpdateDescriptorSetWithTemplate)(VkDevice device, VkDescriptorSet descriptorSet, VkDescriptorUpdateTemplate descriptorUpdateTemplate, const void* pData);
typedef PFN_vkCreateDescriptorUpdateTemplate PFN_vkCreateDescriptorUpdateTemplateKHR;
typedef PFN_vkDestroyDescriptorUpdateTemplate PFN_vkDestroyDescriptorUpdateTemplateKHR;
typedef PFN_vkUpdateDescriptorSetWithTemplate PFN_vkUpdateDescriptorSetWithTemplateKHR;
#endif // VK_KHR_descriptor_update_template

#ifndef VK_KHR_push_descriptor
#define VK_KHR_push_descriptor                                           1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR (VkStructureType)1000080000
typedef struct VkPhysicalDevicePushDescriptorPropertiesKHR
{
    VkStructureType sType;
    void* pNext;
    uint32_t maxPushDescriptors;
} VkPhysicalDevicePushDescriptorPropertiesKHR;
typedef void(VKAPI_PTR* PFN_vkCmdPushDescriptorSetKHR)(VkCommandBuffer commandBuffer, VkPipelineBindPoint pipelineBindPoint, VkPipelineLayout layout, uint32_t set, uint32_t descriptorWriteCount, const VkWriteDescriptorSet* pDescriptorWrites);
typedef void(VKAPI_PTR* PFN_vkCmdPushDescriptorSetWithTemplateKHR)(VkCommandBuffer commandBuffer, VkDescriptorUpdateTemplate descriptorUpdateTemplate, VkPipelineLayout layout, uint32_t set, const void* pData);
#endif // VK_KHR_push_descriptor

#ifndef VK_KHR_get_surface_capabilities2
#define VK_KHR_get_surface_capabilities2                     1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR (VkStructureType)1000119000
#define VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR         (VkStructureType)1000119001
#define VK_STRUCTURE_TYPE_SURFACE_FORMAT_2_KHR               (VkStructureType)1000119002
typedef struct VkPhysicalDeviceSurfaceInfo2KHR
{
    VkStructureType sType;
    const void* pNext;
    VkSurfaceKHR surface;
} VkPhysicalDeviceSurfaceInfo2KHR;
typedef struct VkSurfaceCapabilities2KHR
{
    VkStructureType sType;
    void* pNext;
    VkSurfaceCapabilitiesKHR surfaceCapabilities;
} VkSurfaceCapabilities2KHR;
typedef struct VkSurfaceFormat2KHR
{
    VkStructureType sType;
    void* pNext;
    VkSurfaceFormatKHR surfaceFormat;
} VkSurfaceFormat2KHR;
typedef VkResult(VKAPI_PTR* PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR)(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSurfaceInfo2KHR* pSurfaceInfo, VkSurfaceCapabilities2KHR* pSurfaceCapabilities);
typedef VkResult(VKAPI_PTR* PFN_vkGetPhysicalDeviceSurfaceFormats2KHR)(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceSurfaceInfo2KHR* pSurfaceInfo, uint32_t* pSurfaceFormatCount, VkSurfaceFormat2KHR* pSurfaceFormats);
#endif // VK_KHR_get_surface_capabilities2

#ifndef VK_KHR_external_memory_capabilities
#define VK_KHR_external_memory_capabilities                              1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO     (VkStructureType)1000071000
#define VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES               (VkStructureType)1000071001
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO           (VkStructureType)1000071002
#define VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES                     (VkStructureType)1000071003
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES                  (VkStructureType)1000071004
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO_KHR VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO
#define VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES_KHR           VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO_KHR       VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO
#define VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES_KHR                 VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES_KHR              VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES
#define VK_LUID_SIZE                                                     8U
#define VK_LUID_SIZE_KHR                                                 VK_LUID_SIZE
typedef enum VkExternalMemoryHandleTypeFlagBits
{
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT = 0x00000001,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT = 0x00000002,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT = 0x00000004,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT = 0x00000008,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT = 0x00000010,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT = 0x00000020,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT = 0x00000040,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT = 0x00000200,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID = 0x00000400,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT = 0x00000080,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT = 0x00000100,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_ZIRCON_VMO_BIT_FUCHSIA = 0x00000800,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_RDMA_ADDRESS_BIT_NV = 0x00001000,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT_KHR = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkExternalMemoryHandleTypeFlagBits;
typedef VkFlags VkExternalMemoryHandleTypeFlags;
typedef enum VkExternalMemoryFeatureFlagBits
{
    VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT = 0x00000001,
    VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT = 0x00000002,
    VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT = 0x00000004,
    VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT_KHR = VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT,
    VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT_KHR = VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT,
    VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT_KHR = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT,
    VK_EXTERNAL_MEMORY_FEATURE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkExternalMemoryFeatureFlagBits;
typedef VkFlags VkExternalMemoryFeatureFlags;
typedef struct VkExternalMemoryProperties
{
    VkExternalMemoryFeatureFlags externalMemoryFeatures;
    VkExternalMemoryHandleTypeFlags exportFromImportedHandleTypes;
    VkExternalMemoryHandleTypeFlags compatibleHandleTypes;
} VkExternalMemoryProperties;
typedef struct VkPhysicalDeviceExternalImageFormatInfo
{
    VkStructureType sType;
    const void* pNext;
    VkExternalMemoryHandleTypeFlagBits handleType;
} VkPhysicalDeviceExternalImageFormatInfo;
typedef struct VkExternalImageFormatProperties
{
    VkStructureType sType;
    void* pNext;
    VkExternalMemoryProperties externalMemoryProperties;
} VkExternalImageFormatProperties;
typedef struct VkPhysicalDeviceExternalBufferInfo
{
    VkStructureType sType;
    const void* pNext;
    VkBufferCreateFlags flags;
    VkBufferUsageFlags usage;
    VkExternalMemoryHandleTypeFlagBits handleType;
} VkPhysicalDeviceExternalBufferInfo;
typedef struct VkExternalBufferProperties
{
    VkStructureType sType;
    void* pNext;
    VkExternalMemoryProperties externalMemoryProperties;
} VkExternalBufferProperties;
typedef struct VkPhysicalDeviceIDProperties
{
    VkStructureType sType;
    void* pNext;
    uint8_t deviceUUID[VK_UUID_SIZE];
    uint8_t driverUUID[VK_UUID_SIZE];
    uint8_t deviceLUID[VK_LUID_SIZE];
    uint32_t deviceNodeMask;
    VkBool32 deviceLUIDValid;
} VkPhysicalDeviceIDProperties;
typedef VkExternalMemoryHandleTypeFlags VkExternalMemoryHandleTypeFlagsKHR;
typedef VkExternalMemoryHandleTypeFlagBits VkExternalMemoryHandleTypeFlagBitsKHR;
typedef VkExternalMemoryFeatureFlags VkExternalMemoryFeatureFlagsKHR;
typedef VkExternalMemoryFeatureFlagBits VkExternalMemoryFeatureFlagBitsKHR;
typedef VkExternalMemoryProperties VkExternalMemoryPropertiesKHR;
typedef VkPhysicalDeviceExternalImageFormatInfo VkPhysicalDeviceExternalImageFormatInfoKHR;
typedef VkExternalImageFormatProperties VkExternalImageFormatPropertiesKHR;
typedef VkPhysicalDeviceExternalBufferInfo VkPhysicalDeviceExternalBufferInfoKHR;
typedef VkExternalBufferProperties VkExternalBufferPropertiesKHR;
typedef VkPhysicalDeviceIDProperties VkPhysicalDeviceIDPropertiesKHR;
typedef void(VKAPI_PTR* PFN_vkGetPhysicalDeviceExternalBufferProperties)(VkPhysicalDevice physicalDevice, const VkPhysicalDeviceExternalBufferInfo* pExternalBufferInfo, VkExternalBufferProperties* pExternalBufferProperties);
typedef PFN_vkGetPhysicalDeviceExternalBufferProperties PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR;
#endif // VK_KHR_external_memory_capabilities

#ifndef VK_KHR_get_memory_requirements2
#define VK_KHR_get_memory_requirements2                         1
#define VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2     (VkStructureType)1000146000
#define VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2      (VkStructureType)1000146001
#define VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2                 (VkStructureType)1000146003
#define VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2
#define VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2_KHR  VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2
#define VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR             VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2
typedef struct VkBufferMemoryRequirementsInfo2
{
    VkStructureType sType;
    const void* pNext;
    VkBuffer buffer;
} VkBufferMemoryRequirementsInfo2;
typedef struct VkImageMemoryRequirementsInfo2
{
    VkStructureType sType;
    const void* pNext;
    VkImage image;
} VkImageMemoryRequirementsInfo2;
typedef struct VkMemoryRequirements2
{
    VkStructureType sType;
    void* pNext;
    VkMemoryRequirements memoryRequirements;
} VkMemoryRequirements2;
typedef VkBufferMemoryRequirementsInfo2 VkBufferMemoryRequirementsInfo2KHR;
typedef VkImageMemoryRequirementsInfo2 VkImageMemoryRequirementsInfo2KHR;
typedef VkMemoryRequirements2 VkMemoryRequirements2KHR;
typedef void(VKAPI_PTR* PFN_vkGetImageMemoryRequirements2)(VkDevice device, const VkImageMemoryRequirementsInfo2* pInfo, VkMemoryRequirements2* pMemoryRequirements);
typedef void(VKAPI_PTR* PFN_vkGetBufferMemoryRequirements2)(VkDevice device, const VkBufferMemoryRequirementsInfo2* pInfo, VkMemoryRequirements2* pMemoryRequirements);
typedef PFN_vkGetImageMemoryRequirements2 PFN_vkGetImageMemoryRequirements2KHR;
typedef PFN_vkGetBufferMemoryRequirements2 PFN_vkGetBufferMemoryRequirements2KHR;
#endif // VK_KHR_get_memory_requirements2

#ifndef VK_KHR_dedicated_allocation
#define VK_KHR_dedicated_allocation                          1
#define VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS      (VkStructureType)1000127000
#define VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO     (VkStructureType)1000127001
#define VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR  VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS
#define VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO
typedef struct VkMemoryDedicatedRequirements
{
    VkStructureType sType;
    void* pNext;
    VkBool32 prefersDedicatedAllocation;
    VkBool32 requiresDedicatedAllocation;
} VkMemoryDedicatedRequirements;
typedef struct VkMemoryDedicatedAllocateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkImage image;
    VkBuffer buffer;
} VkMemoryDedicatedAllocateInfo;
typedef VkMemoryDedicatedRequirements VkMemoryDedicatedRequirementsKHR;
typedef VkMemoryDedicatedAllocateInfo VkMemoryDedicatedAllocateInfoKHR;
#endif // VK_KHR_dedicated_allocation

#ifndef VK_KHR_16bit_storage
#define VK_KHR_16bit_storage                                         1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES     (VkStructureType)1000083000
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES
typedef struct VkPhysicalDevice16BitStorageFeatures
{
    VkStructureType sType;
    void* pNext;
    VkBool32 storageBuffer16BitAccess;
    VkBool32 uniformAndStorageBuffer16BitAccess;
    VkBool32 storagePushConstant16;
    VkBool32 storageInputOutput16;
} VkPhysicalDevice16BitStorageFeatures;
typedef VkPhysicalDevice16BitStorageFeatures VkPhysicalDevice16BitStorageFeaturesKHR;
#endif // VK_KHR_16bit_storage

#ifndef VK_KHR_bind_memory2
#define VK_KHR_bind_memory2                           1
#define VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO     (VkStructureType)1000157000
#define VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO      (VkStructureType)1000157001
#define VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO_KHR VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO
#define VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO_KHR  VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO
typedef struct VkBindBufferMemoryInfo
{
    VkStructureType sType;
    const void* pNext;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize memoryOffset;
} VkBindBufferMemoryInfo;
typedef struct VkBindImageMemoryInfo
{
    VkStructureType sType;
    const void* pNext;
    VkImage image;
    VkDeviceMemory memory;
    VkDeviceSize memoryOffset;
} VkBindImageMemoryInfo;
typedef VkBindBufferMemoryInfo VkBindBufferMemoryInfoKHR;
typedef VkBindImageMemoryInfo VkBindImageMemoryInfoKHR;
typedef VkResult(VKAPI_PTR* PFN_vkBindBufferMemory2)(VkDevice device, uint32_t bindInfoCount, const VkBindBufferMemoryInfo* pBindInfos);
typedef VkResult(VKAPI_PTR* PFN_vkBindImageMemory2)(VkDevice device, uint32_t bindInfoCount, const VkBindImageMemoryInfo* pBindInfos);
typedef PFN_vkBindBufferMemory2 PFN_vkBindBufferMemory2KHR;
typedef PFN_vkBindImageMemory2 PFN_vkBindImageMemory2KHR;
#endif // VK_KHR_bind_memory2

#ifndef VK_KHR_sampler_ycbcr_conversion
#define VK_KHR_sampler_ycbcr_conversion                                         1
#define VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO                  (VkStructureType)1000156000
#define VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO                         (VkStructureType)1000156001
#define VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO                          (VkStructureType)1000156002
#define VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO                  (VkStructureType)1000156003
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES     (VkStructureType)1000156004
#define VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES      (VkStructureType)1000156005
#define VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO_KHR              VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO
#define VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO_KHR                     VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO
#define VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO_KHR                      VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO
#define VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO_KHR              VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES_KHR VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES
#define VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES_KHR  VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES
VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkSamplerYcbcrConversion)
typedef VkSamplerYcbcrConversion VkSamplerYcbcrConversionKHR;
typedef enum VkSamplerYcbcrModelConversion
{
    VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY = 0,
    VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY = 1,
    VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709 = 2,
    VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601 = 3,
    VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020 = 4,
    VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY_KHR = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
    VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY_KHR = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_IDENTITY,
    VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709_KHR = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709,
    VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601_KHR = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601,
    VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020_KHR = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_2020,
    VK_SAMPLER_YCBCR_MODEL_CONVERSION_MAX_ENUM = 0x7FFFFFFF
} VkSamplerYcbcrModelConversion;
typedef enum VkSamplerYcbcrRange
{
    VK_SAMPLER_YCBCR_RANGE_ITU_FULL = 0,
    VK_SAMPLER_YCBCR_RANGE_ITU_NARROW = 1,
    VK_SAMPLER_YCBCR_RANGE_ITU_FULL_KHR = VK_SAMPLER_YCBCR_RANGE_ITU_FULL,
    VK_SAMPLER_YCBCR_RANGE_ITU_NARROW_KHR = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW,
    VK_SAMPLER_YCBCR_RANGE_MAX_ENUM = 0x7FFFFFFF
} VkSamplerYcbcrRange;
typedef enum VkChromaLocation
{
    VK_CHROMA_LOCATION_COSITED_EVEN = 0,
    VK_CHROMA_LOCATION_MIDPOINT = 1,
    VK_CHROMA_LOCATION_COSITED_EVEN_KHR = VK_CHROMA_LOCATION_COSITED_EVEN,
    VK_CHROMA_LOCATION_MIDPOINT_KHR = VK_CHROMA_LOCATION_MIDPOINT,
    VK_CHROMA_LOCATION_MAX_ENUM = 0x7FFFFFFF
} VkChromaLocation;
typedef struct VkSamplerYcbcrConversionCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    VkFormat format;
    VkSamplerYcbcrModelConversion ycbcrModel;
    VkSamplerYcbcrRange ycbcrRange;
    VkComponentMapping components;
    VkChromaLocation xChromaOffset;
    VkChromaLocation yChromaOffset;
    VkFilter chromaFilter;
    VkBool32 forceExplicitReconstruction;
} VkSamplerYcbcrConversionCreateInfo;
typedef struct VkSamplerYcbcrConversionInfo
{
    VkStructureType sType;
    const void* pNext;
    VkSamplerYcbcrConversion conversion;
} VkSamplerYcbcrConversionInfo;
typedef struct VkBindImagePlaneMemoryInfo
{
    VkStructureType sType;
    const void* pNext;
    VkImageAspectFlagBits planeAspect;
} VkBindImagePlaneMemoryInfo;
typedef struct VkImagePlaneMemoryRequirementsInfo
{
    VkStructureType sType;
    const void* pNext;
    VkImageAspectFlagBits planeAspect;
} VkImagePlaneMemoryRequirementsInfo;
typedef struct VkPhysicalDeviceSamplerYcbcrConversionFeatures
{
    VkStructureType sType;
    void* pNext;
    VkBool32 samplerYcbcrConversion;
} VkPhysicalDeviceSamplerYcbcrConversionFeatures;
typedef struct VkSamplerYcbcrConversionImageFormatProperties
{
    VkStructureType sType;
    void* pNext;
    uint32_t combinedImageSamplerDescriptorCount;
} VkSamplerYcbcrConversionImageFormatProperties;
typedef VkSamplerYcbcrModelConversion VkSamplerYcbcrModelConversionKHR;
typedef VkSamplerYcbcrRange VkSamplerYcbcrRangeKHR;
typedef VkChromaLocation VkChromaLocationKHR;
typedef VkSamplerYcbcrConversionCreateInfo VkSamplerYcbcrConversionCreateInfoKHR;
typedef VkSamplerYcbcrConversionInfo VkSamplerYcbcrConversionInfoKHR;
typedef VkBindImagePlaneMemoryInfo VkBindImagePlaneMemoryInfoKHR;
typedef VkImagePlaneMemoryRequirementsInfo VkImagePlaneMemoryRequirementsInfoKHR;
typedef VkPhysicalDeviceSamplerYcbcrConversionFeatures VkPhysicalDeviceSamplerYcbcrConversionFeaturesKHR;
typedef VkSamplerYcbcrConversionImageFormatProperties VkSamplerYcbcrConversionImageFormatPropertiesKHR;
typedef VkResult(VKAPI_PTR* PFN_vkCreateSamplerYcbcrConversion)(VkDevice device, const VkSamplerYcbcrConversionCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSamplerYcbcrConversion* pYcbcrConversion);
typedef void(VKAPI_PTR* PFN_vkDestroySamplerYcbcrConversion)(VkDevice device, VkSamplerYcbcrConversion ycbcrConversion, const VkAllocationCallbacks* pAllocator);
typedef PFN_vkCreateSamplerYcbcrConversion PFN_vkCreateSamplerYcbcrConversionKHR;
typedef PFN_vkDestroySamplerYcbcrConversion PFN_vkDestroySamplerYcbcrConversionKHR;
#endif // VK_KHR_sampler_ycbcr_conversion

#if VK_HEADER_VERSION < 70
// the unpublished VK_KHR_subgroup
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES (VkStructureType)1000094000
typedef enum VkSubgroupFeatureFlagBits
{
    VK_SUBGROUP_FEATURE_BASIC_BIT = 0x00000001,
    VK_SUBGROUP_FEATURE_VOTE_BIT = 0x00000002,
    VK_SUBGROUP_FEATURE_ARITHMETIC_BIT = 0x00000004,
    VK_SUBGROUP_FEATURE_BALLOT_BIT = 0x00000008,
    VK_SUBGROUP_FEATURE_SHUFFLE_BIT = 0x00000010,
    VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT = 0x00000020,
    VK_SUBGROUP_FEATURE_CLUSTERED_BIT = 0x00000040,
    VK_SUBGROUP_FEATURE_QUAD_BIT = 0x00000080,
    VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV = 0x00000100,
    VK_SUBGROUP_FEATURE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkSubgroupFeatureFlagBits;
typedef VkFlags VkSubgroupFeatureFlags;
typedef struct VkPhysicalDeviceSubgroupProperties
{
    VkStructureType sType;
    void* pNext;
    uint32_t subgroupSize;
    VkShaderStageFlags supportedStages;
    VkSubgroupFeatureFlags supportedOperations;
    VkBool32 quadOperationsInAllStages;
} VkPhysicalDeviceSubgroupProperties;

// the unpublished VK_KHR_protected_memory
#define VK_STRUCTURE_TYPE_PROTECTED_SUBMIT_INFO                       (VkStructureType)1000145000
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_FEATURES   (VkStructureType)1000145001
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROTECTED_MEMORY_PROPERTIES (VkStructureType)1000145002
#define VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2                         (VkStructureType)1000145003
#define VK_MEMORY_PROPERTY_PROTECTED_BIT                              (VkMemoryPropertyFlagBits)0x00000020
#define VK_IMAGE_CREATE_PROTECTED_BIT                                 (VkImageCreateFlagBits)0x00000800
#define VK_BUFFER_CREATE_PROTECTED_BIT                                (VkBufferCreateFlagBits)0x00000008
#define VK_QUEUE_PROTECTED_BIT                                        (VkQueueFlagBits)0x00000010
#define VK_COMMAND_POOL_CREATE_PROTECTED_BIT                          (VkCommandPoolCreateFlagBits)0x00000004
typedef enum VkDeviceQueueCreateFlagBits
{
    VK_DEVICE_QUEUE_CREATE_PROTECTED_BIT = 0x00000001,
    VK_DEVICE_QUEUE_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkDeviceQueueCreateFlagBits;
typedef VkFlags VkDeviceQueueCreateFlags;
typedef struct VkPhysicalDeviceProtectedMemoryFeatures
{
    VkStructureType sType;
    void* pNext;
    VkBool32 protectedMemory;
} VkPhysicalDeviceProtectedMemoryFeatures;
typedef struct VkPhysicalDeviceProtectedMemoryProperties
{
    VkStructureType sType;
    void* pNext;
    VkBool32 protectedNoFault;
} VkPhysicalDeviceProtectedMemoryProperties;
typedef struct VkProtectedSubmitInfo
{
    VkStructureType sType;
    const void* pNext;
    VkBool32 protectedSubmit;
} VkProtectedSubmitInfo;
#endif // VK_HEADER_VERSION < 70

#if VK_HEADER_VERSION < 204
typedef uint64_t VkFlags64;
#endif // VK_HEADER_VERSION < 204

#ifndef VK_KHR_format_feature_flags2
#define VK_KHR_format_feature_flags2              1
#define VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3     (VkStructureType)1000360000
#define VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3_KHR VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3
typedef VkFlags64 VkFormatFeatureFlags2;
typedef VkFlags64 VkFormatFeatureFlagBits2;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT = 0x00000001ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT_KHR = 0x00000001ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT = 0x00000002ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT_KHR = 0x00000002ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT = 0x00000004ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT_KHR = 0x00000004ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT = 0x00000008ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT_KHR = 0x00000008ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT = 0x00000010ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT_KHR = 0x00000010ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT = 0x00000020ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT_KHR = 0x00000020ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT = 0x00000040ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT_KHR = 0x00000040ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT = 0x00000080ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT_KHR = 0x00000080ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT = 0x00000100ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT_KHR = 0x00000100ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT = 0x00000200ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT_KHR = 0x00000200ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_BLIT_SRC_BIT = 0x00000400ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_BLIT_SRC_BIT_KHR = 0x00000400ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_BLIT_DST_BIT = 0x00000800ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_BLIT_DST_BIT_KHR = 0x00000800ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT = 0x00001000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT_KHR = 0x00001000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_CUBIC_BIT = 0x00002000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_CUBIC_BIT_EXT = 0x00002000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT = 0x00004000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT_KHR = 0x00004000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT = 0x00008000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT_KHR = 0x00008000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT = 0x00010000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT_KHR = 0x00010000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_MIDPOINT_CHROMA_SAMPLES_BIT = 0x00020000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_MIDPOINT_CHROMA_SAMPLES_BIT_KHR = 0x00020000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT = 0x00040000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT_KHR = 0x00040000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT = 0x00080000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT_KHR = 0x00080000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT = 0x00100000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_BIT_KHR = 0x00100000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT = 0x00200000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_CHROMA_RECONSTRUCTION_EXPLICIT_FORCEABLE_BIT_KHR = 0x00200000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_DISJOINT_BIT = 0x00400000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_DISJOINT_BIT_KHR = 0x00400000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_COSITED_CHROMA_SAMPLES_BIT = 0x00800000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_COSITED_CHROMA_SAMPLES_BIT_KHR = 0x00800000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT = 0x80000000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT_KHR = 0x80000000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT = 0x100000000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT_KHR = 0x100000000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT = 0x200000000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT_KHR = 0x200000000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_ACCELERATION_STRUCTURE_VERTEX_BUFFER_BIT_KHR = 0x20000000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_FRAGMENT_DENSITY_MAP_BIT_EXT = 0x01000000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR = 0x40000000ULL;
static const VkFormatFeatureFlagBits2 VK_FORMAT_FEATURE_2_LINEAR_COLOR_ATTACHMENT_BIT_NV = 0x4000000000ULL;
typedef struct VkFormatProperties3
{
    VkStructureType sType;
    void* pNext;
    VkFormatFeatureFlags2 linearTilingFeatures;
    VkFormatFeatureFlags2 optimalTilingFeatures;
    VkFormatFeatureFlags2 bufferFeatures;
} VkFormatProperties3;
typedef VkFormatFeatureFlags2 VkFormatFeatureFlags2KHR;
typedef VkFormatFeatureFlagBits2 VkFormatFeatureFlagBits2KHR;
typedef VkFormatProperties3 VkFormatProperties3KHR;
#endif // VK_KHR_format_feature_flags2

#ifndef VK_ANDROID_external_memory_android_hardware_buffer
#define VK_ANDROID_external_memory_android_hardware_buffer                    1
#define VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_USAGE_ANDROID               (VkStructureType)1000129000
#define VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID          (VkStructureType)1000129001
#define VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID   (VkStructureType)1000129002
#define VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID         (VkStructureType)1000129003
#define VK_STRUCTURE_TYPE_MEMORY_GET_ANDROID_HARDWARE_BUFFER_INFO_ANDROID     (VkStructureType)1000129004
#define VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID                             (VkStructureType)1000129005
#define VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_2_ANDROID (VkStructureType)1000129006
struct AHardwareBuffer;
typedef struct VkAndroidHardwareBufferUsageANDROID
{
    VkStructureType sType;
    void* pNext;
    uint64_t androidHardwareBufferUsage;
} VkAndroidHardwareBufferUsageANDROID;
typedef struct VkAndroidHardwareBufferPropertiesANDROID
{
    VkStructureType sType;
    void* pNext;
    VkDeviceSize allocationSize;
    uint32_t memoryTypeBits;
} VkAndroidHardwareBufferPropertiesANDROID;
typedef struct VkAndroidHardwareBufferFormatPropertiesANDROID
{
    VkStructureType sType;
    void* pNext;
    VkFormat format;
    uint64_t externalFormat;
    VkFormatFeatureFlags formatFeatures;
    VkComponentMapping samplerYcbcrConversionComponents;
    VkSamplerYcbcrModelConversion suggestedYcbcrModel;
    VkSamplerYcbcrRange suggestedYcbcrRange;
    VkChromaLocation suggestedXChromaOffset;
    VkChromaLocation suggestedYChromaOffset;
} VkAndroidHardwareBufferFormatPropertiesANDROID;
typedef struct VkImportAndroidHardwareBufferInfoANDROID
{
    VkStructureType sType;
    const void* pNext;
    struct AHardwareBuffer* buffer;
} VkImportAndroidHardwareBufferInfoANDROID;
typedef struct VkMemoryGetAndroidHardwareBufferInfoANDROID
{
    VkStructureType sType;
    const void* pNext;
    VkDeviceMemory memory;
} VkMemoryGetAndroidHardwareBufferInfoANDROID;
typedef struct VkExternalFormatANDROID
{
    VkStructureType sType;
    void* pNext;
    uint64_t externalFormat;
} VkExternalFormatANDROID;
typedef struct VkAndroidHardwareBufferFormatProperties2ANDROID
{
    VkStructureType sType;
    void* pNext;
    VkFormat format;
    uint64_t externalFormat;
    VkFormatFeatureFlags2 formatFeatures;
    VkComponentMapping samplerYcbcrConversionComponents;
    VkSamplerYcbcrModelConversion suggestedYcbcrModel;
    VkSamplerYcbcrRange suggestedYcbcrRange;
    VkChromaLocation suggestedXChromaOffset;
    VkChromaLocation suggestedYChromaOffset;
} VkAndroidHardwareBufferFormatProperties2ANDROID;
typedef VkResult(VKAPI_PTR* PFN_vkGetAndroidHardwareBufferPropertiesANDROID)(VkDevice device, const struct AHardwareBuffer* buffer, VkAndroidHardwareBufferPropertiesANDROID* pProperties);
typedef VkResult(VKAPI_PTR* PFN_vkGetMemoryAndroidHardwareBufferANDROID)(VkDevice device, const VkMemoryGetAndroidHardwareBufferInfoANDROID* pInfo, struct AHardwareBuffer** pBuffer);
#endif // VK_ANDROID_external_memory_android_hardware_buffer

#ifndef VK_KHR_maintenance3
#define VK_KHR_maintenance3                                        1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES (VkStructureType)1000168000
#define VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT            (VkStructureType)1000168001
typedef struct VkPhysicalDeviceMaintenance3Properties
{
    VkStructureType sType;
    void* pNext;
    uint32_t maxPerSetDescriptors;
    VkDeviceSize maxMemoryAllocationSize;
} VkPhysicalDeviceMaintenance3Properties;
typedef struct VkDescriptorSetLayoutSupport
{
    VkStructureType sType;
    void* pNext;
    VkBool32 supported;
} VkDescriptorSetLayoutSupport;
typedef VkPhysicalDeviceMaintenance3Properties VkPhysicalDeviceMaintenance3PropertiesKHR;
typedef VkDescriptorSetLayoutSupport VkDescriptorSetLayoutSupportKHR;
typedef void(VKAPI_PTR* PFN_vkGetDescriptorSetLayoutSupport)(VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo, VkDescriptorSetLayoutSupport* pSupport);
typedef PFN_vkGetDescriptorSetLayoutSupport PFN_vkGetDescriptorSetLayoutSupportKHR;
#endif // VK_KHR_maintenance3

#ifndef VK_KHR_8bit_storage
#define VK_KHR_8bit_storage                                         1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES     (VkStructureType)1000177000
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES
typedef struct VkPhysicalDevice8BitStorageFeatures
{
    VkStructureType sType;
    void* pNext;
    VkBool32 storageBuffer8BitAccess;
    VkBool32 uniformAndStorageBuffer8BitAccess;
    VkBool32 storagePushConstant8;
} VkPhysicalDevice8BitStorageFeatures;
typedef VkPhysicalDevice8BitStorageFeatures VkPhysicalDevice8BitStorageFeaturesKHR;
#endif // VK_KHR_8bit_storage

#ifndef VK_KHR_shader_float16_int8
#define VK_KHR_shader_float16_int8                                  1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES     (VkStructureType)1000082000
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES
typedef struct VkPhysicalDeviceFloat16Int8FeaturesKHR
{
    VkStructureType sType;
    void* pNext;
    VkBool32 shaderFloat16;
    VkBool32 shaderInt8;
} VkPhysicalDeviceFloat16Int8FeaturesKHR;
#endif // VK_KHR_shader_float16_int8

#ifndef VK_EXT_memory_budget
#define VK_EXT_memory_budget                                           1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT (VkStructureType)1000237000
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT (VkStructureType)1000238000
#define VK_STRUCTURE_TYPE_MEMORY_PRIORITY_ALLOCATE_INFO_EXT            (VkStructureType)1000238001
typedef uint64_t VkDeviceAddress;
typedef struct VkPhysicalDeviceMemoryBudgetPropertiesEXT
{
    VkStructureType sType;
    void* pNext;
    VkDeviceSize heapBudget[VK_MAX_MEMORY_HEAPS];
    VkDeviceSize heapUsage[VK_MAX_MEMORY_HEAPS];
} VkPhysicalDeviceMemoryBudgetPropertiesEXT;
typedef struct VkPhysicalDeviceMemoryPriorityFeaturesEXT
{
    VkStructureType sType;
    void* pNext;
    VkBool32 memoryPriority;
} VkPhysicalDeviceMemoryPriorityFeaturesEXT;
typedef struct VkMemoryPriorityAllocateInfoEXT
{
    VkStructureType sType;
    const void* pNext;
    float priority;
} VkMemoryPriorityAllocateInfoEXT;
#endif // VK_EXT_memory_budget

#ifndef VK_EXT_buffer_device_address
#define VK_EXT_buffer_device_address                                  1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_ADDRESS_FEATURES_EXT (VkStructureType)1000244000
#define VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_EXT              (VkStructureType)1000244001
#define VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_CREATE_INFO_EXT       (VkStructureType)1000244002
#define VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_EXT                 (VkBufferUsageFlagBits)0x00020000
typedef struct VkPhysicalDeviceBufferAddressFeaturesEXT
{
    VkStructureType sType;
    void* pNext;
    VkBool32 bufferDeviceAddress;
    VkBool32 bufferDeviceAddressCaptureReplay;
    VkBool32 bufferDeviceAddressMultiDevice;
} VkPhysicalDeviceBufferAddressFeaturesEXT;
typedef struct VkBufferDeviceAddressInfoEXT
{
    VkStructureType sType;
    const void* pNext;
    VkBuffer buffer;
} VkBufferDeviceAddressInfoEXT;
typedef struct VkBufferDeviceAddressCreateInfoEXT
{
    VkStructureType sType;
    const void* pNext;
    VkDeviceSize deviceAddress;
} VkBufferDeviceAddressCreateInfoEXT;
typedef VkDeviceAddress(VKAPI_PTR* PFN_vkGetBufferDeviceAddressEXT)(VkDevice device, const VkBufferDeviceAddressInfoEXT* pInfo);
#endif // VK_EXT_buffer_device_address

#ifndef VK_EXT_validation_features
#define VK_EXT_validation_features                             1
#define VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT              (VkStructureType)1000247000
#define VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_EXT (VkBufferCreateFlagBits)0x00020000
typedef enum VkValidationFeatureEnableEXT
{
    VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT = 0,
    VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT = 1,
    VK_VALIDATION_FEATURE_ENABLE_BEGIN_RANGE_EXT = VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
    VK_VALIDATION_FEATURE_ENABLE_END_RANGE_EXT = VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT,
    VK_VALIDATION_FEATURE_ENABLE_RANGE_SIZE_EXT = (VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT - VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT + 1),
    VK_VALIDATION_FEATURE_ENABLE_MAX_ENUM_EXT = 0x7FFFFFFF
} VkValidationFeatureEnableEXT;
typedef enum VkValidationFeatureDisableEXT
{
    VK_VALIDATION_FEATURE_DISABLE_ALL_EXT = 0,
    VK_VALIDATION_FEATURE_DISABLE_SHADERS_EXT = 1,
    VK_VALIDATION_FEATURE_DISABLE_THREAD_SAFETY_EXT = 2,
    VK_VALIDATION_FEATURE_DISABLE_API_PARAMETERS_EXT = 3,
    VK_VALIDATION_FEATURE_DISABLE_OBJECT_LIFETIMES_EXT = 4,
    VK_VALIDATION_FEATURE_DISABLE_CORE_CHECKS_EXT = 5,
    VK_VALIDATION_FEATURE_DISABLE_UNIQUE_HANDLES_EXT = 6,
    VK_VALIDATION_FEATURE_DISABLE_BEGIN_RANGE_EXT = VK_VALIDATION_FEATURE_DISABLE_ALL_EXT,
    VK_VALIDATION_FEATURE_DISABLE_END_RANGE_EXT = VK_VALIDATION_FEATURE_DISABLE_UNIQUE_HANDLES_EXT,
    VK_VALIDATION_FEATURE_DISABLE_RANGE_SIZE_EXT = (VK_VALIDATION_FEATURE_DISABLE_UNIQUE_HANDLES_EXT - VK_VALIDATION_FEATURE_DISABLE_ALL_EXT + 1),
    VK_VALIDATION_FEATURE_DISABLE_MAX_ENUM_EXT = 0x7FFFFFFF
} VkValidationFeatureDisableEXT;
typedef struct VkValidationFeaturesEXT
{
    VkStructureType sType;
    const void* pNext;
    uint32_t enabledValidationFeatureCount;
    const VkValidationFeatureEnableEXT* pEnabledValidationFeatures;
    uint32_t disabledValidationFeatureCount;
    const VkValidationFeatureDisableEXT* pDisabledValidationFeatures;
} VkValidationFeaturesEXT;
#endif // VK_EXT_validation_features

#ifndef VK_NV_cooperative_matrix
#define VK_NV_cooperative_matrix                                           1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV   (VkStructureType)1000249000
#define VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_NV                 (VkStructureType)1000249001
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_NV (VkStructureType)1000249002
typedef enum VkComponentTypeNV
{
    VK_COMPONENT_TYPE_FLOAT16_NV = 0,
    VK_COMPONENT_TYPE_FLOAT32_NV = 1,
    VK_COMPONENT_TYPE_FLOAT64_NV = 2,
    VK_COMPONENT_TYPE_SINT8_NV = 3,
    VK_COMPONENT_TYPE_SINT16_NV = 4,
    VK_COMPONENT_TYPE_SINT32_NV = 5,
    VK_COMPONENT_TYPE_SINT64_NV = 6,
    VK_COMPONENT_TYPE_UINT8_NV = 7,
    VK_COMPONENT_TYPE_UINT16_NV = 8,
    VK_COMPONENT_TYPE_UINT32_NV = 9,
    VK_COMPONENT_TYPE_UINT64_NV = 10,
    VK_COMPONENT_TYPE_BEGIN_RANGE_NV = VK_COMPONENT_TYPE_FLOAT16_NV,
    VK_COMPONENT_TYPE_END_RANGE_NV = VK_COMPONENT_TYPE_UINT64_NV,
    VK_COMPONENT_TYPE_RANGE_SIZE_NV = (VK_COMPONENT_TYPE_UINT64_NV - VK_COMPONENT_TYPE_FLOAT16_NV + 1),
    VK_COMPONENT_TYPE_MAX_ENUM_NV = 0x7FFFFFFF
} VkComponentTypeNV;
typedef enum VkScopeNV
{
    VK_SCOPE_DEVICE_NV = 1,
    VK_SCOPE_WORKGROUP_NV = 2,
    VK_SCOPE_SUBGROUP_NV = 3,
    VK_SCOPE_QUEUE_FAMILY_NV = 5,
    VK_SCOPE_BEGIN_RANGE_NV = VK_SCOPE_DEVICE_NV,
    VK_SCOPE_END_RANGE_NV = VK_SCOPE_QUEUE_FAMILY_NV,
    VK_SCOPE_RANGE_SIZE_NV = (VK_SCOPE_QUEUE_FAMILY_NV - VK_SCOPE_DEVICE_NV + 1),
    VK_SCOPE_MAX_ENUM_NV = 0x7FFFFFFF
} VkScopeNV;
typedef struct VkCooperativeMatrixPropertiesNV
{
    VkStructureType sType;
    void* pNext;
    uint32_t MSize;
    uint32_t NSize;
    uint32_t KSize;
    VkComponentTypeNV AType;
    VkComponentTypeNV BType;
    VkComponentTypeNV CType;
    VkComponentTypeNV DType;
    VkScopeNV scope;
} VkCooperativeMatrixPropertiesNV;
typedef struct VkPhysicalDeviceCooperativeMatrixFeaturesNV
{
    VkStructureType sType;
    void* pNext;
    VkBool32 cooperativeMatrix;
    VkBool32 cooperativeMatrixRobustBufferAccess;
} VkPhysicalDeviceCooperativeMatrixFeaturesNV;
typedef struct VkPhysicalDeviceCooperativeMatrixPropertiesNV
{
    VkStructureType sType;
    void* pNext;
    VkShaderStageFlags cooperativeMatrixSupportedStages;
} VkPhysicalDeviceCooperativeMatrixPropertiesNV;
typedef VkResult(VKAPI_PTR* PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV)(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkCooperativeMatrixPropertiesNV* pProperties);
#endif // VK_NV_cooperative_matrix

#ifndef VK_AMD_device_coherent_memory
#define VK_AMD_device_coherent_memory                                  1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD (VkStructureType)1000229000
#define VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD                     (VkMemoryPropertyFlagBits)0x00000040
#define VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD                     (VkMemoryPropertyFlagBits)0x00000040
typedef struct VkPhysicalDeviceCoherentMemoryFeaturesAMD
{
    VkStructureType sType;
    void* pNext;
    VkBool32 deviceCoherentMemory;
} VkPhysicalDeviceCoherentMemoryFeaturesAMD;
#endif // VK_AMD_device_coherent_memory

#ifndef VK_KHR_buffer_device_address
#define VK_KHR_buffer_device_address                                         1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES     (VkStructureType)1000257000
#define VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO                         (VkStructureType)1000244001
#define VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO          (VkStructureType)1000257002
#define VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO        (VkStructureType)1000257003
#define VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO          (VkStructureType)1000257004
#define VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT                   (VkBufferCreateFlagBits)0x00020000
#define VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT                            (VkBufferUsageFlagBits)0x00020000
#define VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT                                (VkMemoryAllocateFlagBits)0x00000002
#define VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT                 (VkMemoryAllocateFlagBits)0x00000004
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES
#define VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR                     VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO
#define VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO_KHR      VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO
#define VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO_KHR    VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO
#define VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO_KHR      VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO
#define VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR               VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT
#define VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
#define VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR                            VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
#define VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_KHR             VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT
typedef struct VkPhysicalDeviceBufferDeviceAddressFeatures
{
    VkStructureType sType;
    void* pNext;
    VkBool32 bufferDeviceAddress;
    VkBool32 bufferDeviceAddressCaptureReplay;
    VkBool32 bufferDeviceAddressMultiDevice;
} VkPhysicalDeviceBufferDeviceAddressFeatures;
typedef struct VkBufferDeviceAddressInfo
{
    VkStructureType sType;
    const void* pNext;
    VkBuffer buffer;
} VkBufferDeviceAddressInfo;
typedef struct VkBufferOpaqueCaptureAddressCreateInfo
{
    VkStructureType sType;
    const void* pNext;
    uint64_t opaqueCaptureAddress;
} VkBufferOpaqueCaptureAddressCreateInfo;
typedef struct VkMemoryOpaqueCaptureAddressAllocateInfo
{
    VkStructureType sType;
    const void* pNext;
    uint64_t opaqueCaptureAddress;
} VkMemoryOpaqueCaptureAddressAllocateInfo;
typedef struct VkDeviceMemoryOpaqueCaptureAddressInfo
{
    VkStructureType sType;
    const void* pNext;
    VkDeviceMemory memory;
} VkDeviceMemoryOpaqueCaptureAddressInfo;
typedef VkPhysicalDeviceBufferDeviceAddressFeatures VkPhysicalDeviceBufferDeviceAddressFeaturesKHR;
typedef VkBufferDeviceAddressInfo VkBufferDeviceAddressInfoKHR;
typedef VkBufferOpaqueCaptureAddressCreateInfo VkBufferOpaqueCaptureAddressCreateInfoKHR;
typedef VkMemoryOpaqueCaptureAddressAllocateInfo VkMemoryOpaqueCaptureAddressAllocateInfoKHR;
typedef VkDeviceMemoryOpaqueCaptureAddressInfo VkDeviceMemoryOpaqueCaptureAddressInfoKHR;
typedef VkDeviceAddress(VKAPI_PTR* PFN_vkGetBufferDeviceAddress)(VkDevice device, const VkBufferDeviceAddressInfo* pInfo);
typedef uint64_t(VKAPI_PTR* PFN_vkGetBufferOpaqueCaptureAddress)(VkDevice device, const VkBufferDeviceAddressInfo* pInfo);
typedef uint64_t(VKAPI_PTR* PFN_vkGetDeviceMemoryOpaqueCaptureAddress)(VkDevice device, const VkDeviceMemoryOpaqueCaptureAddressInfo* pInfo);
typedef PFN_vkGetBufferDeviceAddress PFN_vkGetBufferDeviceAddressKHR;
typedef PFN_vkGetBufferOpaqueCaptureAddress PFN_vkGetBufferOpaqueCaptureAddressKHR;
typedef PFN_vkGetDeviceMemoryOpaqueCaptureAddress PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR;
#endif // VK_KHR_buffer_device_address

#ifndef VK_KHR_portability_enumeration
#define VK_KHR_portability_enumeration 1
typedef enum VkInstanceCreateFlagBits
{
    VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR = 0x00000001,
    VK_INSTANCE_CREATE_FLAG_BITS_MAX_ENUM = 0x7FFFFFFF
} VkInstanceCreateFlagBits;
#endif // VK_KHR_portability_enumeration

#ifndef VK_KHR_cooperative_matrix
#define VK_KHR_cooperative_matrix                                           1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR   (VkStructureType)1000506000
#define VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR                 (VkStructureType)1000506001
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR (VkStructureType)1000506002
typedef enum VkComponentTypeKHR
{
    VK_COMPONENT_TYPE_FLOAT16_KHR = 0,
    VK_COMPONENT_TYPE_FLOAT32_KHR = 1,
    VK_COMPONENT_TYPE_FLOAT64_KHR = 2,
    VK_COMPONENT_TYPE_SINT8_KHR = 3,
    VK_COMPONENT_TYPE_SINT16_KHR = 4,
    VK_COMPONENT_TYPE_SINT32_KHR = 5,
    VK_COMPONENT_TYPE_SINT64_KHR = 6,
    VK_COMPONENT_TYPE_UINT8_KHR = 7,
    VK_COMPONENT_TYPE_UINT16_KHR = 8,
    VK_COMPONENT_TYPE_UINT32_KHR = 9,
    VK_COMPONENT_TYPE_UINT64_KHR = 10,
    VK_COMPONENT_TYPE_MAX_ENUM_KHR = 0x7FFFFFFF
} VkComponentTypeKHR;
typedef enum VkScopeKHR
{
    VK_SCOPE_DEVICE_KHR = 1,
    VK_SCOPE_WORKGROUP_KHR = 2,
    VK_SCOPE_SUBGROUP_KHR = 3,
    VK_SCOPE_QUEUE_FAMILY_KHR = 5,
    VK_SCOPE_MAX_ENUM_KHR = 0x7FFFFFFF
} VkScopeKHR;
typedef struct VkCooperativeMatrixPropertiesKHR
{
    VkStructureType sType;
    void* pNext;
    uint32_t MSize;
    uint32_t NSize;
    uint32_t KSize;
    VkComponentTypeKHR AType;
    VkComponentTypeKHR BType;
    VkComponentTypeKHR CType;
    VkComponentTypeKHR ResultType;
    VkBool32 saturatingAccumulation;
    VkScopeKHR scope;
} VkCooperativeMatrixPropertiesKHR;
typedef struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR
{
    VkStructureType sType;
    void* pNext;
    VkBool32 cooperativeMatrix;
    VkBool32 cooperativeMatrixRobustBufferAccess;
} VkPhysicalDeviceCooperativeMatrixFeaturesKHR;
typedef struct VkPhysicalDeviceCooperativeMatrixPropertiesKHR
{
    VkStructureType sType;
    void* pNext;
    VkShaderStageFlags cooperativeMatrixSupportedStages;
} VkPhysicalDeviceCooperativeMatrixPropertiesKHR;
typedef VkResult(VKAPI_PTR* PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)(VkPhysicalDevice physicalDevice, uint32_t* pPropertyCount, VkCooperativeMatrixPropertiesKHR* pProperties);
#endif // VK_KHR_cooperative_matrix

#ifndef VK_KHR_driver_properties
#define VK_KHR_driver_properties                                1
#define VK_MAX_DRIVER_NAME_SIZE                                 256U
#define VK_MAX_DRIVER_INFO_SIZE                                 256U
#define VK_MAX_DRIVER_NAME_SIZE_KHR                             VK_MAX_DRIVER_NAME_SIZE
#define VK_MAX_DRIVER_INFO_SIZE_KHR                             VK_MAX_DRIVER_INFO_SIZE
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES     (VkStructureType)1000196000
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES
typedef enum VkDriverId
{
    VK_DRIVER_ID_AMD_PROPRIETARY = 1,
    VK_DRIVER_ID_AMD_OPEN_SOURCE = 2,
    VK_DRIVER_ID_MESA_RADV = 3,
    VK_DRIVER_ID_NVIDIA_PROPRIETARY = 4,
    VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS = 5,
    VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA = 6,
    VK_DRIVER_ID_IMAGINATION_PROPRIETARY = 7,
    VK_DRIVER_ID_QUALCOMM_PROPRIETARY = 8,
    VK_DRIVER_ID_ARM_PROPRIETARY = 9,
    VK_DRIVER_ID_GOOGLE_SWIFTSHADER = 10,
    VK_DRIVER_ID_GGP_PROPRIETARY = 11,
    VK_DRIVER_ID_BROADCOM_PROPRIETARY = 12,
    VK_DRIVER_ID_MESA_LLVMPIPE = 13,
    VK_DRIVER_ID_MOLTENVK = 14,
    VK_DRIVER_ID_COREAVI_PROPRIETARY = 15,
    VK_DRIVER_ID_JUICE_PROPRIETARY = 16,
    VK_DRIVER_ID_VERISILICON_PROPRIETARY = 17,
    VK_DRIVER_ID_MESA_TURNIP = 18,
    VK_DRIVER_ID_MESA_V3DV = 19,
    VK_DRIVER_ID_MESA_PANVK = 20,
    VK_DRIVER_ID_SAMSUNG_PROPRIETARY = 21,
    VK_DRIVER_ID_MESA_VENUS = 22,
    VK_DRIVER_ID_MESA_DOZEN = 23,
    VK_DRIVER_ID_MESA_NVK = 24,
    VK_DRIVER_ID_IMAGINATION_OPEN_SOURCE_MESA = 25,
    VK_DRIVER_ID_MESA_AGXV = 26,
    VK_DRIVER_ID_MAX_ENUM = 0x7FFFFFFF
} VkDriverId;
typedef struct VkConformanceVersion
{
    uint8_t major;
    uint8_t minor;
    uint8_t subminor;
    uint8_t patch;
} VkConformanceVersion;
typedef struct VkPhysicalDeviceDriverProperties
{
    VkStructureType sType;
    void* pNext;
    VkDriverId driverID;
    char driverName[VK_MAX_DRIVER_NAME_SIZE];
    char driverInfo[VK_MAX_DRIVER_INFO_SIZE];
    VkConformanceVersion conformanceVersion;
} VkPhysicalDeviceDriverProperties;
typedef VkDriverId VkDriverIdKHR;
typedef VkConformanceVersion VkConformanceVersionKHR;
typedef VkPhysicalDeviceDriverProperties VkPhysicalDeviceDriverPropertiesKHR;
#endif // VK_KHR_driver_properties

#endif // NCNN_VULKAN_HEADER_FIX_H
