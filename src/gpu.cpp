// Copyright 2018 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu.h"

#if NCNN_VULKAN

#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "glslang/SPIRV/GlslangToSpv.h"
#if NCNN_SYSTEM_GLSLANG
#include "glslang/Public/ShaderLang.h"
#else
#include "glslang/glslang/Public/ShaderLang.h"
#endif

#include "layer/vulkan/shader/vulkan_activation.comp.hex.h"

#include "command.h"
#include "layer.h"
#include "layer_type.h"
#include "mat.h"
#include "pipelinecache.h"

// There is known issue that vkDestroyDebugUtilsMessengerEXT crash on exit when vulkan validation layer enabled
// upstream fix https://github.com/KhronosGroup/Vulkan-Loader/pull/539
#define ENABLE_VALIDATION_LAYER 0

namespace ncnn {

// global
static Mutex g_instance_lock;

class __ncnn_vulkan_instance_holder
{
public:
    __ncnn_vulkan_instance_holder()
    {
        instance = 0;
        instance_api_version = 0;
        created = 0;
        glslang_initialized = false;

#if NCNN_VULKAN_LOADER
        libvulkan = 0;
#if defined __ANDROID__
        hvkdi = 0;
#endif
#endif // NCNN_VULKAN_LOADER

#if ENABLE_VALIDATION_LAYER
        callback = 0;
#endif
    }

    ~__ncnn_vulkan_instance_holder()
    {
        destroy_gpu_instance();
    }

    operator VkInstance()
    {
        return instance;
    }

    VkInstance instance;
    uint32_t instance_api_version;
    int created;
    bool glslang_initialized;

#if ENABLE_VALIDATION_LAYER
    VkDebugUtilsMessengerEXT callback;
#endif
};
static __ncnn_vulkan_instance_holder g_instance;

static int g_gpu_count = 0;
static int g_default_gpu_index = -1;

// NOTE 32 is large enough i think ...
#define NCNN_MAX_GPU_COUNT 32
static GpuInfo* g_gpu_infos[NCNN_MAX_GPU_COUNT] = {0};

// default vulkan device
static Mutex g_default_vkdev_lock;
static VulkanDevice* g_default_vkdev[NCNN_MAX_GPU_COUNT] = {0};

struct layer_shader_registry_entry
{
    const char* comp_data;
    int comp_data_size;
};

#include "layer_shader_spv_data.h"

static const layer_shader_registry_entry layer_shader_registry[] = {
#include "layer_shader_registry.h"
};

static const int layer_shader_registry_entry_count = sizeof(layer_shader_registry) / sizeof(layer_shader_registry_entry);

// vulkan core
PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = 0;
PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = 0;
PFN_vkAllocateMemory vkAllocateMemory = 0;
PFN_vkBeginCommandBuffer vkBeginCommandBuffer = 0;
PFN_vkBindBufferMemory vkBindBufferMemory = 0;
PFN_vkBindImageMemory vkBindImageMemory = 0;
PFN_vkCmdBeginQuery vkCmdBeginQuery = 0;
PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = 0;
PFN_vkCmdBindIndexBuffer vkCmdBindIndexBuffer = 0;
PFN_vkCmdBindPipeline vkCmdBindPipeline = 0;
PFN_vkCmdCopyBuffer vkCmdCopyBuffer = 0;
PFN_vkCmdCopyBufferToImage vkCmdCopyBufferToImage = 0;
PFN_vkCmdCopyImage vkCmdCopyImage = 0;
PFN_vkCmdCopyImageToBuffer vkCmdCopyImageToBuffer = 0;
PFN_vkCmdCopyQueryPoolResults vkCmdCopyQueryPoolResults = 0;
PFN_vkCmdDispatch vkCmdDispatch = 0;
PFN_vkCmdDispatchIndirect vkCmdDispatchIndirect = 0;
PFN_vkCmdEndQuery vkCmdEndQuery = 0;
PFN_vkCmdExecuteCommands vkCmdExecuteCommands = 0;
PFN_vkCmdFillBuffer vkCmdFillBuffer = 0;
PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = 0;
PFN_vkCmdPushConstants vkCmdPushConstants = 0;
PFN_vkCmdResetQueryPool vkCmdResetQueryPool = 0;
PFN_vkCmdResolveImage vkCmdResolveImage = 0;
PFN_vkCmdUpdateBuffer vkCmdUpdateBuffer = 0;
PFN_vkCmdWriteTimestamp vkCmdWriteTimestamp = 0;
PFN_vkCreateBuffer vkCreateBuffer = 0;
PFN_vkCreateBufferView vkCreateBufferView = 0;
PFN_vkCreateCommandPool vkCreateCommandPool = 0;
PFN_vkCreateComputePipelines vkCreateComputePipelines = 0;
PFN_vkCreateDescriptorPool vkCreateDescriptorPool = 0;
PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = 0;
PFN_vkCreateDevice vkCreateDevice = 0;
PFN_vkCreateFence vkCreateFence = 0;
PFN_vkCreateImage vkCreateImage = 0;
PFN_vkCreateImageView vkCreateImageView = 0;
PFN_vkCreatePipelineCache vkCreatePipelineCache = 0;
PFN_vkCreatePipelineLayout vkCreatePipelineLayout = 0;
PFN_vkCreateQueryPool vkCreateQueryPool = 0;
PFN_vkCreateSampler vkCreateSampler = 0;
PFN_vkCreateSemaphore vkCreateSemaphore = 0;
PFN_vkCreateShaderModule vkCreateShaderModule = 0;
PFN_vkDestroyBuffer vkDestroyBuffer = 0;
PFN_vkDestroyBufferView vkDestroyBufferView = 0;
PFN_vkDestroyCommandPool vkDestroyCommandPool = 0;
PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = 0;
PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = 0;
PFN_vkDestroyDevice vkDestroyDevice = 0;
PFN_vkDestroyFence vkDestroyFence = 0;
PFN_vkDestroyImage vkDestroyImage = 0;
PFN_vkDestroyImageView vkDestroyImageView = 0;
PFN_vkDestroyInstance vkDestroyInstance = 0;
PFN_vkDestroyPipeline vkDestroyPipeline = 0;
PFN_vkDestroyPipelineCache vkDestroyPipelineCache = 0;
PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = 0;
PFN_vkDestroyQueryPool vkDestroyQueryPool = 0;
PFN_vkDestroySampler vkDestroySampler = 0;
PFN_vkDestroySemaphore vkDestroySemaphore = 0;
PFN_vkDestroyShaderModule vkDestroyShaderModule = 0;
PFN_vkDeviceWaitIdle vkDeviceWaitIdle = 0;
PFN_vkEndCommandBuffer vkEndCommandBuffer = 0;
PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties = 0;
PFN_vkEnumerateDeviceLayerProperties vkEnumerateDeviceLayerProperties = 0;
PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = 0;
PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges = 0;
PFN_vkFreeCommandBuffers vkFreeCommandBuffers = 0;
PFN_vkFreeDescriptorSets vkFreeDescriptorSets = 0;
PFN_vkFreeMemory vkFreeMemory = 0;
PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = 0;
PFN_vkGetDeviceMemoryCommitment vkGetDeviceMemoryCommitment = 0;
PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr = 0;
PFN_vkGetDeviceQueue vkGetDeviceQueue = 0;
PFN_vkGetFenceStatus vkGetFenceStatus = 0;
PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements = 0;
PFN_vkGetImageSubresourceLayout vkGetImageSubresourceLayout = 0;
PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures = 0;
PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties = 0;
PFN_vkGetPhysicalDeviceImageFormatProperties vkGetPhysicalDeviceImageFormatProperties = 0;
PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties = 0;
PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = 0;
PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties = 0;
PFN_vkGetPipelineCacheData vkGetPipelineCacheData = 0;
PFN_vkGetQueryPoolResults vkGetQueryPoolResults = 0;
PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges = 0;
PFN_vkMapMemory vkMapMemory = 0;
PFN_vkMergePipelineCaches vkMergePipelineCaches = 0;
PFN_vkQueueSubmit vkQueueSubmit = 0;
PFN_vkQueueWaitIdle vkQueueWaitIdle = 0;
PFN_vkResetCommandBuffer vkResetCommandBuffer = 0;
PFN_vkResetCommandPool vkResetCommandPool = 0;
PFN_vkResetDescriptorPool vkResetDescriptorPool = 0;
PFN_vkResetFences vkResetFences = 0;
PFN_vkUnmapMemory vkUnmapMemory = 0;
PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = 0;
PFN_vkWaitForFences vkWaitForFences = 0;

int support_VK_KHR_external_memory_capabilities = 0;
int support_VK_KHR_get_physical_device_properties2 = 0;
int support_VK_KHR_get_surface_capabilities2 = 0;
int support_VK_KHR_portability_enumeration = 0;
int support_VK_KHR_surface = 0;
int support_VK_EXT_debug_utils = 0;
int support_VK_EXT_validation_features = 0;
int support_VK_EXT_validation_flags = 0;
#if __ANDROID_API__ >= 26
int support_VK_KHR_android_surface = 0;
#endif // __ANDROID_API__ >= 26

// VK_KHR_cooperative_matrix
PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = 0;

// VK_KHR_external_memory_capabilities
PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR vkGetPhysicalDeviceExternalBufferPropertiesKHR = 0;

// VK_KHR_get_physical_device_properties2
PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR = 0;
PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR = 0;
PFN_vkGetPhysicalDeviceFormatProperties2KHR vkGetPhysicalDeviceFormatProperties2KHR = 0;
PFN_vkGetPhysicalDeviceImageFormatProperties2KHR vkGetPhysicalDeviceImageFormatProperties2KHR = 0;
PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR vkGetPhysicalDeviceQueueFamilyProperties2KHR = 0;
PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR = 0;

// VK_KHR_get_surface_capabilities2
PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR vkGetPhysicalDeviceSurfaceCapabilities2KHR = 0;
PFN_vkGetPhysicalDeviceSurfaceFormats2KHR vkGetPhysicalDeviceSurfaceFormats2KHR = 0;

// VK_KHR_surface
PFN_vkDestroySurfaceKHR vkDestroySurfaceKHR = 0;
PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR = 0;
PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR = 0;
PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR = 0;
PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR = 0;

#if __ANDROID_API__ >= 26
// VK_KHR_android_surface
PFN_vkCreateAndroidSurfaceKHR vkCreateAndroidSurfaceKHR = 0;
#endif // __ANDROID_API__ >= 26

// VK_NV_cooperative_matrix
PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV vkGetPhysicalDeviceCooperativeMatrixPropertiesNV = 0;

// VK_NV_cooperative_matrix2
PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV = 0;

// VK_NV_cooperative_vector
PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV vkGetPhysicalDeviceCooperativeVectorPropertiesNV = 0;

class GpuInfoPrivate
{
public:
    void query_features();
    void query_properties();
    void query_queue_properties();
    int query_extensions();
    void query_extension_features();
    void query_extension_properties();

public:
    int device_index;

    // physical device
    VkPhysicalDevice physicalDevice;

    // features
    VkPhysicalDeviceFeatures physicalDevicefeatures;

    // properties
    VkPhysicalDeviceProperties physicalDeviceProperties;

    // memory properties
    VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;

    // extension properties
    std::vector<VkExtensionProperties> deviceExtensionProperties;

    // 0 = discrete gpu
    // 1 = integrated gpu
    // 2 = virtual gpu
    // 3 = cpu
    int type;

    // runtime
    uint32_t compute_queue_family_index;
    uint32_t transfer_queue_family_index;

    uint32_t compute_queue_count;
    uint32_t transfer_queue_count;

    // property
    bool unified_compute_transfer_queue;

    // bug is not feature
    bool bug_storage_buffer_no_l1;
    bool bug_corrupted_online_pipeline_cache;
    bool bug_buffer_image_load_zero;

    // but sometimes bug is a feature
    bool bug_implicit_fp16_arithmetic;

    // cooperative matrix
    bool support_cooperative_matrix_8_8_16;
    bool support_cooperative_matrix_16_8_8;
    bool support_cooperative_matrix_16_8_16;
    bool support_cooperative_matrix_16_16_16;

    // extension capability
    int support_VK_KHR_8bit_storage;
    int support_VK_KHR_16bit_storage;
    int support_VK_KHR_bind_memory2;
    int support_VK_KHR_buffer_device_address;
    int support_VK_KHR_create_renderpass2;
    int support_VK_KHR_cooperative_matrix;
    int support_VK_KHR_dedicated_allocation;
    int support_VK_KHR_descriptor_update_template;
    int support_VK_KHR_driver_properties;
    int support_VK_KHR_external_memory;
    int support_VK_KHR_get_memory_requirements2;
    int support_VK_KHR_maintenance1;
    int support_VK_KHR_maintenance2;
    int support_VK_KHR_maintenance3;
    int support_VK_KHR_multiview;
    int support_VK_KHR_portability_subset;
    int support_VK_KHR_push_descriptor;
    int support_VK_KHR_robustness2;
    int support_VK_KHR_sampler_ycbcr_conversion;
    int support_VK_KHR_shader_bfloat16;
    int support_VK_KHR_shader_float16_int8;
    int support_VK_KHR_shader_float_controls;
    int support_VK_KHR_shader_float_controls2;
    int support_VK_KHR_shader_integer_dot_product;
    int support_VK_KHR_shader_non_semantic_info;
    int support_VK_KHR_shader_subgroup_extended_types;
    int support_VK_KHR_shader_subgroup_rotate;
    int support_VK_KHR_storage_buffer_storage_class;
    int support_VK_KHR_swapchain;
    int support_VK_KHR_vulkan_memory_model;
    int support_VK_KHR_zero_initialize_workgroup_memory;
    int support_VK_EXT_buffer_device_address;
    int support_VK_EXT_descriptor_indexing;
    int support_VK_EXT_memory_budget;
    int support_VK_EXT_memory_priority;
    int support_VK_EXT_queue_family_foreign;
    int support_VK_EXT_robustness2;
    int support_VK_EXT_shader_atomic_float;
    int support_VK_EXT_shader_atomic_float2;
    int support_VK_EXT_shader_float8;
    int support_VK_EXT_subgroup_size_control;
    int support_VK_AMD_device_coherent_memory;
#if __ANDROID_API__ >= 26
    int support_VK_ANDROID_external_memory_android_hardware_buffer;
#endif // __ANDROID_API__ >= 26
    int support_VK_NV_cooperative_matrix;
    int support_VK_NV_cooperative_matrix2;
    int support_VK_NV_cooperative_vector;

    // extension features
    void* queryExtensionFeatures;
    VkPhysicalDevice8BitStorageFeaturesKHR query8BitStorageFeatures;
    VkPhysicalDevice16BitStorageFeaturesKHR query16BitStorageFeatures;
    VkPhysicalDeviceFloat16Int8FeaturesKHR queryFloat16Int8Features;
    VkPhysicalDeviceSamplerYcbcrConversionFeaturesKHR querySamplerYcbcrConversionFeatures;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR queryCooperativeMatrixFeatures;
    VkPhysicalDeviceCooperativeMatrixFeaturesNV queryCooperativeMatrixFeaturesNV;
    VkPhysicalDeviceCooperativeMatrix2FeaturesNV queryCooperativeMatrix2FeaturesNV;
    VkPhysicalDeviceCooperativeVectorFeaturesNV queryCooperativeVectorFeaturesNV;
    VkPhysicalDeviceRobustness2FeaturesKHR queryRobustness2Features;
    VkPhysicalDeviceShaderBfloat16FeaturesKHR queryShaderBfloat16Features;
    VkPhysicalDeviceShaderFloat8FeaturesEXT queryShaderFloat8Features;
    VkPhysicalDeviceShaderFloatControls2FeaturesKHR queryShaderFloatControls2Features;
    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR queryShaderIntegerDotProductFeatures;
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT querySubgroupSizeControlFeatures;
    VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR queryShaderSubgroupRotateFeatures;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT queryShaderAtomicFloatFeatures;
    VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT queryShaderAtomicFloat2Features;
    VkPhysicalDeviceVulkanMemoryModelFeaturesKHR queryVulkanMemoryModelFeatures;

    // extension properties
    void* queryExtensionProperties;
    VkPhysicalDeviceFloatControlsPropertiesKHR queryFloatControlsProperties;
    VkPhysicalDeviceRobustness2PropertiesKHR queryRobustness2Properties;
    VkPhysicalDeviceShaderIntegerDotProductProperties queryShaderIntegerDotProductProperties;
    VkPhysicalDeviceSubgroupProperties querySubgroupProperties;
    VkPhysicalDeviceDriverPropertiesKHR queryDriverProperties;
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT querySubgroupSizeControlProperties;
    VkPhysicalDeviceCooperativeMatrix2PropertiesNV queryCooperativeMatrix2PropertiesNV;
    VkPhysicalDeviceCooperativeVectorPropertiesNV queryCooperativeVectorPropertiesNV;

    // extension sub properties
    std::vector<VkCooperativeMatrixPropertiesKHR> queryCooperativeMatrixSubProperties;
    std::vector<VkCooperativeMatrixPropertiesNV> queryCooperativeMatrixSubPropertiesNV;
    std::vector<VkCooperativeMatrixFlexibleDimensionsPropertiesNV> queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV;
    std::vector<VkCooperativeVectorPropertiesNV> queryCooperativeVectorSubPropertiesNV;
};

void GpuInfoPrivate::query_features()
{
    vkGetPhysicalDeviceFeatures(physicalDevice, &physicalDevicefeatures);
}

void GpuInfoPrivate::query_properties()
{
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

    // NCNN_LOGE("[%u] apiVersion = %u.%u.%u", i, VK_VERSION_MAJOR(physicalDeviceProperties.apiVersion),
    //     VK_VERSION_MINOR(physicalDeviceProperties.apiVersion), VK_VERSION_PATCH(physicalDeviceProperties.apiVersion));
    // NCNN_LOGE("[%u] driverVersion = %u.%u.%u", i, VK_VERSION_MAJOR(physicalDeviceProperties.driverVersion),
    //     VK_VERSION_MINOR(physicalDeviceProperties.driverVersion), VK_VERSION_PATCH(physicalDeviceProperties.driverVersion));
    // NCNN_LOGE("[%u] vendorID = %x", i, physicalDeviceProperties.vendorID);
    // NCNN_LOGE("[%u] deviceID = %x", i, physicalDeviceProperties.deviceID);
    // NCNN_LOGE("[%u] deviceType = %x", i, physicalDeviceProperties.deviceType);
    // NCNN_LOGE("[%u] deviceName = %s", i, physicalDeviceProperties.deviceName);
    // NCNN_LOGE("[%u] pipelineCacheUUID = %u", i, physicalDeviceProperties.pipelineCacheUUID);

    // device type
    {
        type = -1;
        if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            type = 0;
        if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            type = 1;
        if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
            type = 2;
        if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
            type = 3;
    }

    // mali
    // t760 = 0x13b5 0x7500001 / 0x7501000
    // t860 = 0x13b5 0x8602000
    // t880 = 0x13b5 0x8800020
    // g31  = 0x13b5 0x70930000
    // g51  = 0x13b5 0x70901010
    // g52  = 0x13b5 0x74021000 / 0x72120000
    // g71  = 0x13b5 0x60a00002
    // g72  = 0x13b5 0x62210001
    // g76  = 0x13b5 0x72110000
    // g77  = 0x13b5 0x90800011

    // adreno
    // 506 = 0x5143 0x5000600
    // 510 = 0x5143 0x5010000
    // 512 = 0x5143 0x5010200
    // 530 = 0x5143 0x5030004
    // 540 = 0x5143 0x5040001
    // 616 = 0x5143 0x6010600
    // 630 = 0x5143 0x6030001
    // 640 = 0x5143 0x6040001
    // 650 = 0x5143 0x6050002

    bug_storage_buffer_no_l1 = false;
    bug_corrupted_online_pipeline_cache = false;
    bug_implicit_fp16_arithmetic = false;
    bug_buffer_image_load_zero = false;

    if (physicalDeviceProperties.vendorID == 0x5143 && physicalDeviceProperties.apiVersion < VK_MAKE_VERSION(1, 0, 66))
    {
        // qcom adreno with old buggy driver cannot share created pipeline properly
        bug_corrupted_online_pipeline_cache = true;
    }

    if (physicalDeviceProperties.vendorID == 0x5143 && !(physicalDeviceProperties.deviceID == 0x6040001 || physicalDeviceProperties.deviceID == 0x6050002))
    {
        // NOTE but qcom855/qcom855plus/qcom865 are known exceptions
        // qcom adreno storage buffer without L1 cache
        bug_storage_buffer_no_l1 = true;
    }

    if (physicalDeviceProperties.vendorID == 0x5143 && physicalDeviceProperties.apiVersion < VK_MAKE_VERSION(1, 1, 87))
    {
        // HACK buffer2image before image-read dependency does not work properly
        // even promised with full image memory barrier on old adreno driver
        // TODO figure out a proper workaround without hurt speed too much
        // TODO only for old drivers
        bug_buffer_image_load_zero = true;
    }

    if (physicalDeviceProperties.vendorID == 0x13b5
            && (physicalDeviceProperties.deviceID == 0x7500001
                || physicalDeviceProperties.deviceID == 0x7501000
                || physicalDeviceProperties.deviceID == 0x8602000
                || physicalDeviceProperties.deviceID == 0x8800020
                || physicalDeviceProperties.deviceID == 0x70930000
                || physicalDeviceProperties.deviceID == 0x70901010
                || physicalDeviceProperties.deviceID == 0x72120000
                || physicalDeviceProperties.deviceID == 0x74021000
                || physicalDeviceProperties.deviceID == 0x60a00002
                || physicalDeviceProperties.deviceID == 0x62210001))
    {
        // NOTE rk3288/rk3399/t880/g31/g51/g52/g71/g72
        // however, g76/g77 has explicit fp16 arithmetic
        // arm mali driver accept spirv with fp16 arithmetic
        bug_implicit_fp16_arithmetic = true;
    }

    if (physicalDeviceProperties.vendorID == 0x5143
            && (physicalDeviceProperties.deviceID == 0x6030001
                || physicalDeviceProperties.deviceID == 0x6040001
                || physicalDeviceProperties.deviceID == 0x6050002))
    {
        // TODO enable devices other than qcom845/qcom855/qcom855plus/qcom865
        // qcom adreno driver accept spirv with fp16 arithmetic
        bug_implicit_fp16_arithmetic = true;
    }
}

static uint32_t find_device_compute_queue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties)
{
    // first try, compute only queue
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if ((queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
                && !(queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
        {
            return i;
        }
    }

    // second try, any queue with compute and graphics
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if ((queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
                && (queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
        {
            return i;
        }
    }

    // third try, any queue with compute
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if (queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
        {
            return i;
        }
    }

    //     NCNN_LOGE("no compute queue");
    return -1;
}

static uint32_t find_device_transfer_queue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties)
{
    // first try, transfer only queue
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if ((queueFamilyProperty.queueFlags & VK_QUEUE_TRANSFER_BIT)
                && !(queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
                && !(queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
        {
            return i;
        }
    }

    // second try, any queue with transfer
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if (queueFamilyProperty.queueFlags & VK_QUEUE_TRANSFER_BIT)
        {
            return i;
        }
    }

    // third try, use compute queue
    uint32_t compute_queue_index = find_device_compute_queue(queueFamilyProperties);
    if (compute_queue_index != (uint32_t)-1)
    {
        return compute_queue_index;
    }

    //     NCNN_LOGE("no transfer queue");
    return -1;
}

void GpuInfoPrivate::query_queue_properties()
{
    // find compute queue
    uint32_t queueFamilyPropertiesCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, 0);

    std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertiesCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, queueFamilyProperties.data());

    compute_queue_family_index = find_device_compute_queue(queueFamilyProperties);
    transfer_queue_family_index = find_device_transfer_queue(queueFamilyProperties);

    compute_queue_count = queueFamilyProperties[compute_queue_family_index].queueCount;
    transfer_queue_count = queueFamilyProperties[transfer_queue_family_index].queueCount;

    unified_compute_transfer_queue = compute_queue_family_index == transfer_queue_family_index;
}

int GpuInfoPrivate::query_extensions()
{
    // get device extension
    uint32_t deviceExtensionPropertyCount = 0;
    VkResult ret = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionPropertyCount, NULL);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEnumerateDeviceExtensionProperties failed %d", ret);
        return -1;
    }

    deviceExtensionProperties.resize(deviceExtensionPropertyCount);
    ret = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionPropertyCount, deviceExtensionProperties.data());
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEnumerateDeviceExtensionProperties failed %d", ret);
        return -1;
    }

    // extension capability
    support_VK_KHR_8bit_storage = 0;
    support_VK_KHR_16bit_storage = 0;
    support_VK_KHR_bind_memory2 = 0;
    support_VK_KHR_buffer_device_address = 0;
    support_VK_KHR_create_renderpass2 = 0;
    support_VK_KHR_cooperative_matrix = 0;
    support_VK_KHR_dedicated_allocation = 0;
    support_VK_KHR_descriptor_update_template = 0;
    support_VK_KHR_driver_properties = 0;
    support_VK_KHR_external_memory = 0;
    support_VK_KHR_get_memory_requirements2 = 0;
    support_VK_KHR_maintenance1 = 0;
    support_VK_KHR_maintenance2 = 0;
    support_VK_KHR_maintenance3 = 0;
    support_VK_KHR_multiview = 0;
    support_VK_KHR_portability_subset = 0;
    support_VK_KHR_push_descriptor = 0;
    support_VK_KHR_robustness2 = 0;
    support_VK_KHR_sampler_ycbcr_conversion = 0;
    support_VK_KHR_shader_bfloat16 = 0;
    support_VK_KHR_shader_float16_int8 = 0;
    support_VK_KHR_shader_float_controls = 0;
    support_VK_KHR_shader_float_controls2 = 0;
    support_VK_KHR_shader_integer_dot_product = 0;
    support_VK_KHR_shader_non_semantic_info = 0;
    support_VK_KHR_shader_subgroup_extended_types = 0;
    support_VK_KHR_shader_subgroup_rotate = 0;
    support_VK_KHR_storage_buffer_storage_class = 0;
    support_VK_KHR_swapchain = 0;
    support_VK_KHR_vulkan_memory_model = 0;
    support_VK_KHR_zero_initialize_workgroup_memory = 0;
    support_VK_EXT_buffer_device_address = 0;
    support_VK_EXT_descriptor_indexing = 0;
    support_VK_EXT_memory_budget = 0;
    support_VK_EXT_memory_priority = 0;
    support_VK_EXT_queue_family_foreign = 0;
    support_VK_EXT_robustness2 = 0;
    support_VK_EXT_shader_atomic_float = 0;
    support_VK_EXT_shader_atomic_float2 = 0;
    support_VK_EXT_shader_float8 = 0;
    support_VK_EXT_subgroup_size_control = 0;
    support_VK_AMD_device_coherent_memory = 0;
#if __ANDROID_API__ >= 26
    support_VK_ANDROID_external_memory_android_hardware_buffer = 0;
#endif // __ANDROID_API__ >= 26
    support_VK_NV_cooperative_matrix = 0;
    support_VK_NV_cooperative_matrix2 = 0;
    support_VK_NV_cooperative_vector = 0;
    for (uint32_t j = 0; j < deviceExtensionPropertyCount; j++)
    {
        const VkExtensionProperties& exp = deviceExtensionProperties[j];
        // NCNN_LOGE("device extension %s = %u", exp.extensionName, exp.specVersion);

        if (strcmp(exp.extensionName, "VK_KHR_8bit_storage") == 0)
            support_VK_KHR_8bit_storage = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_16bit_storage") == 0)
            support_VK_KHR_16bit_storage = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_bind_memory2") == 0)
            support_VK_KHR_bind_memory2 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_buffer_device_address") == 0)
            support_VK_KHR_buffer_device_address = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_create_renderpass2") == 0)
            support_VK_KHR_create_renderpass2 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_cooperative_matrix") == 0)
            support_VK_KHR_cooperative_matrix = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_dedicated_allocation") == 0)
            support_VK_KHR_dedicated_allocation = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_descriptor_update_template") == 0)
            support_VK_KHR_descriptor_update_template = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_driver_properties") == 0)
            support_VK_KHR_driver_properties = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_external_memory") == 0)
            support_VK_KHR_external_memory = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_get_memory_requirements2") == 0)
            support_VK_KHR_get_memory_requirements2 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_maintenance1") == 0)
            support_VK_KHR_maintenance1 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_maintenance2") == 0)
            support_VK_KHR_maintenance2 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_maintenance3") == 0)
            support_VK_KHR_maintenance3 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_multiview") == 0)
            support_VK_KHR_multiview = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_portability_subset") == 0)
            support_VK_KHR_portability_subset = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_push_descriptor") == 0)
            support_VK_KHR_push_descriptor = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_robustness2") == 0)
            support_VK_KHR_robustness2 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_sampler_ycbcr_conversion") == 0)
            support_VK_KHR_sampler_ycbcr_conversion = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_shader_bfloat16") == 0)
            support_VK_KHR_shader_bfloat16 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_shader_float16_int8") == 0)
            support_VK_KHR_shader_float16_int8 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_shader_float_controls") == 0)
            support_VK_KHR_shader_float_controls = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_shader_float_controls2") == 0)
            support_VK_KHR_shader_float_controls2 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_shader_integer_dot_product") == 0)
            support_VK_KHR_shader_integer_dot_product = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_shader_non_semantic_info") == 0)
            support_VK_KHR_shader_non_semantic_info = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_shader_subgroup_extended_types") == 0)
            support_VK_KHR_shader_subgroup_extended_types = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_shader_subgroup_rotate") == 0)
            support_VK_KHR_shader_subgroup_rotate = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_storage_buffer_storage_class") == 0)
            support_VK_KHR_storage_buffer_storage_class = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_swapchain") == 0)
            support_VK_KHR_swapchain = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_vulkan_memory_model") == 0)
            support_VK_KHR_vulkan_memory_model = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_zero_initialize_workgroup_memory") == 0)
            support_VK_KHR_zero_initialize_workgroup_memory = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_buffer_device_address") == 0)
            support_VK_EXT_buffer_device_address = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_descriptor_indexing") == 0)
            support_VK_EXT_descriptor_indexing = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_memory_budget") == 0)
            support_VK_EXT_memory_budget = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_memory_priority") == 0)
            support_VK_EXT_memory_priority = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_queue_family_foreign") == 0)
            support_VK_EXT_queue_family_foreign = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_robustness2") == 0)
            support_VK_EXT_robustness2 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_shader_atomic_float") == 0)
            support_VK_EXT_shader_atomic_float = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_shader_atomic_float2") == 0)
            support_VK_EXT_shader_atomic_float2 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_shader_float8") == 0)
            support_VK_EXT_shader_float8 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_subgroup_size_control") == 0)
            support_VK_EXT_subgroup_size_control = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_AMD_device_coherent_memory") == 0)
            support_VK_AMD_device_coherent_memory = exp.specVersion;
#if __ANDROID_API__ >= 26
        else if (strcmp(exp.extensionName, "VK_ANDROID_external_memory_android_hardware_buffer") == 0)
            support_VK_ANDROID_external_memory_android_hardware_buffer = exp.specVersion;
#endif // __ANDROID_API__ >= 26
        else if (strcmp(exp.extensionName, "VK_NV_cooperative_matrix") == 0)
            support_VK_NV_cooperative_matrix = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_NV_cooperative_matrix2") == 0)
            support_VK_NV_cooperative_matrix2 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_NV_cooperative_vector") == 0)
            support_VK_NV_cooperative_vector = exp.specVersion;
    }

    if (support_VK_KHR_buffer_device_address)
    {
        // we prefer khr extension
        support_VK_EXT_buffer_device_address = 0;
    }

    if (support_VK_KHR_cooperative_matrix)
    {
        // we prefer khr extension
        support_VK_NV_cooperative_matrix = 0;
    }

    if (support_VK_KHR_robustness2)
    {
        // we prefer khr extension
        support_VK_EXT_robustness2 = 0;
    }

    return 0;
}

void GpuInfoPrivate::query_extension_features()
{
    queryExtensionFeatures = 0;

    // query int8 storage
    memset(&query8BitStorageFeatures, 0, sizeof(query8BitStorageFeatures));
    query8BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR;
    query8BitStorageFeatures.pNext = 0;
    if (support_VK_KHR_8bit_storage)
    {
        query8BitStorageFeatures.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &query8BitStorageFeatures;
    }

    // query fp16/int16 storage
    memset(&query16BitStorageFeatures, 0, sizeof(query16BitStorageFeatures));
    query16BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;
    query16BitStorageFeatures.pNext = 0;
    if (support_VK_KHR_16bit_storage)
    {
        query16BitStorageFeatures.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &query16BitStorageFeatures;
    }

    // query fp16/int8 arithmetic
    memset(&queryFloat16Int8Features, 0, sizeof(queryFloat16Int8Features));
    queryFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
    queryFloat16Int8Features.pNext = 0;
    if (support_VK_KHR_shader_float16_int8)
    {
        queryFloat16Int8Features.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryFloat16Int8Features;
    }

    // query ycbcr_conversion
    memset(&querySamplerYcbcrConversionFeatures, 0, sizeof(querySamplerYcbcrConversionFeatures));
    querySamplerYcbcrConversionFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES_KHR;
    querySamplerYcbcrConversionFeatures.pNext = 0;
    if (support_VK_KHR_sampler_ycbcr_conversion)
    {
        querySamplerYcbcrConversionFeatures.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &querySamplerYcbcrConversionFeatures;
    }

    // query cooperative_matrix
    memset(&queryCooperativeMatrixFeatures, 0, sizeof(queryCooperativeMatrixFeatures));
    queryCooperativeMatrixFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    queryCooperativeMatrixFeatures.pNext = 0;
    if (support_VK_KHR_cooperative_matrix)
    {
        queryCooperativeMatrixFeatures.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryCooperativeMatrixFeatures;
    }

    // query nv cooperative matrix
    memset(&queryCooperativeMatrixFeaturesNV, 0, sizeof(queryCooperativeMatrixFeaturesNV));
    queryCooperativeMatrixFeaturesNV.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_NV;
    queryCooperativeMatrixFeaturesNV.pNext = 0;
    if (support_VK_NV_cooperative_matrix)
    {
        queryCooperativeMatrixFeaturesNV.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryCooperativeMatrixFeaturesNV;
    }

    // query nv cooperative matrix2
    memset(&queryCooperativeMatrix2FeaturesNV, 0, sizeof(queryCooperativeMatrix2FeaturesNV));
    queryCooperativeMatrix2FeaturesNV.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_FEATURES_NV;
    queryCooperativeMatrix2FeaturesNV.pNext = 0;
    if (support_VK_NV_cooperative_matrix2)
    {
        queryCooperativeMatrix2FeaturesNV.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryCooperativeMatrix2FeaturesNV;
    }

    // query nv cooperative vector
    memset(&queryCooperativeVectorFeaturesNV, 0, sizeof(queryCooperativeVectorFeaturesNV));
    queryCooperativeVectorFeaturesNV.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV;
    queryCooperativeVectorFeaturesNV.pNext = 0;
    if (support_VK_NV_cooperative_vector)
    {
        queryCooperativeVectorFeaturesNV.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryCooperativeVectorFeaturesNV;
    }

    // query robustness2
    memset(&queryRobustness2Features, 0, sizeof(queryRobustness2Features));
    queryRobustness2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_KHR;
    queryRobustness2Features.pNext = 0;
    if (support_VK_KHR_robustness2 || support_VK_EXT_robustness2)
    {
        queryRobustness2Features.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryRobustness2Features;
    }

    // query bfloat16
    memset(&queryShaderBfloat16Features, 0, sizeof(queryShaderBfloat16Features));
    queryShaderBfloat16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    queryShaderBfloat16Features.pNext = 0;
    if (support_VK_KHR_shader_bfloat16)
    {
        queryShaderBfloat16Features.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryShaderBfloat16Features;
    }

    // query float8
    memset(&queryShaderFloat8Features, 0, sizeof(queryShaderFloat8Features));
    queryShaderFloat8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    queryShaderFloat8Features.pNext = 0;
    if (support_VK_EXT_shader_float8)
    {
        queryShaderFloat8Features.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryShaderFloat8Features;
    }

    // query float controls 2
    memset(&queryShaderFloatControls2Features, 0, sizeof(queryShaderFloatControls2Features));
    queryShaderFloatControls2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES_KHR;
    queryShaderFloatControls2Features.pNext = 0;
    if (support_VK_KHR_shader_float_controls2)
    {
        queryShaderFloatControls2Features.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryShaderFloatControls2Features;
    }

    // query integer dot product
    memset(&queryShaderIntegerDotProductFeatures, 0, sizeof(queryShaderIntegerDotProductFeatures));
    queryShaderIntegerDotProductFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    queryShaderIntegerDotProductFeatures.pNext = 0;
    if (support_VK_KHR_shader_integer_dot_product)
    {
        queryShaderIntegerDotProductFeatures.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryShaderIntegerDotProductFeatures;
    }

    // query subgroup size control
    memset(&querySubgroupSizeControlFeatures, 0, sizeof(querySubgroupSizeControlFeatures));
    querySubgroupSizeControlFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    querySubgroupSizeControlFeatures.pNext = 0;
    if (support_VK_EXT_subgroup_size_control >= 2)
    {
        querySubgroupSizeControlFeatures.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &querySubgroupSizeControlFeatures;
    }

    // query subgroup rotate
    memset(&queryShaderSubgroupRotateFeatures, 0, sizeof(queryShaderSubgroupRotateFeatures));
    queryShaderSubgroupRotateFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES_KHR;
    queryShaderSubgroupRotateFeatures.pNext = 0;
    if (support_VK_KHR_shader_subgroup_rotate)
    {
        queryShaderSubgroupRotateFeatures.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryShaderSubgroupRotateFeatures;
    }

    // query atomic float
    memset(&queryShaderAtomicFloatFeatures, 0, sizeof(queryShaderAtomicFloatFeatures));
    queryShaderAtomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    queryShaderAtomicFloatFeatures.pNext = 0;
    if (support_VK_EXT_shader_atomic_float)
    {
        queryShaderAtomicFloatFeatures.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryShaderAtomicFloatFeatures;
    }

    // query atomic float2
    memset(&queryShaderAtomicFloat2Features, 0, sizeof(queryShaderAtomicFloat2Features));
    queryShaderAtomicFloat2Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT;
    queryShaderAtomicFloat2Features.pNext = 0;
    if (support_VK_EXT_shader_atomic_float2)
    {
        queryShaderAtomicFloat2Features.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryShaderAtomicFloat2Features;
    }

    // query vulkan memory model
    memset(&queryVulkanMemoryModelFeatures, 0, sizeof(queryVulkanMemoryModelFeatures));
    queryVulkanMemoryModelFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES_KHR;
    queryVulkanMemoryModelFeatures.pNext = 0;
    if (support_VK_KHR_vulkan_memory_model)
    {
        queryVulkanMemoryModelFeatures.pNext = queryExtensionFeatures;
        queryExtensionFeatures = &queryVulkanMemoryModelFeatures;
    }

    if (support_VK_KHR_get_physical_device_properties2)
    {
        VkPhysicalDeviceFeatures2KHR queryFeatures;
        queryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR;
        queryFeatures.pNext = queryExtensionFeatures;

        vkGetPhysicalDeviceFeatures2KHR(physicalDevice, &queryFeatures);
    }

    // apply known blacklist
    if (physicalDeviceProperties.vendorID == 0x13b5 && physicalDeviceProperties.apiVersion < VK_MAKE_VERSION(1, 0, 82))
    {
        // the 16bit_storage implementation of arm mali driver is buggy :[
        query16BitStorageFeatures.storageBuffer16BitAccess = VK_FALSE;
    }

    if (physicalDeviceProperties.vendorID == 0x10002 && physicalDeviceProperties.deviceID == 0x70006214 && physicalDeviceProperties.apiVersion == VK_MAKE_VERSION(1, 1, 82))
    {
        // the 16bit_storage implementation of vivante gc1700 driver is buggy :[
        query16BitStorageFeatures.storageBuffer16BitAccess = VK_FALSE;
    }

    if (bug_implicit_fp16_arithmetic)
    {
        // force capability on as long as the driver accept spirv with fp16 arithmetic :D
        queryFloat16Int8Features.shaderFloat16 = VK_TRUE;
    }

    if (physicalDeviceProperties.vendorID == 0x5143 && !query16BitStorageFeatures.storageBuffer16BitAccess)
    {
        // fp16 arithmetic yields wrong result on old adreno drivers :(
        queryFloat16Int8Features.shaderFloat16 = VK_FALSE;
    }
}

void GpuInfoPrivate::query_extension_properties()
{
    queryExtensionProperties = 0;

    // query float controls
    memset(&queryFloatControlsProperties, 0, sizeof(queryFloatControlsProperties));
    queryFloatControlsProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
    queryFloatControlsProperties.pNext = 0;
    if (support_VK_KHR_shader_float_controls)
    {
        queryFloatControlsProperties.pNext = queryExtensionProperties;
        queryExtensionProperties = &queryFloatControlsProperties;
    }

    // query integer dot product
    memset(&queryShaderIntegerDotProductProperties, 0, sizeof(queryShaderIntegerDotProductProperties));
    queryShaderIntegerDotProductProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR;
    queryShaderIntegerDotProductProperties.pNext = 0;
    if (support_VK_KHR_shader_integer_dot_product)
    {
        queryShaderIntegerDotProductProperties.pNext = queryExtensionProperties;
        queryExtensionProperties = &queryShaderIntegerDotProductProperties;
    }

    // query subgroup
    memset(&querySubgroupProperties, 0, sizeof(querySubgroupProperties));
    querySubgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    querySubgroupProperties.pNext = 0;
    if (VK_VERSION_MAJOR(g_instance.instance_api_version) >= 1 && VK_VERSION_MINOR(g_instance.instance_api_version) >= 1)
    {
        querySubgroupProperties.pNext = queryExtensionProperties;
        queryExtensionProperties = &querySubgroupProperties;
    }
    else
    {
        querySubgroupProperties.subgroupSize = 64;
        if (physicalDeviceProperties.vendorID == 0x5143) // qcom adreno prefer very large workgroup :P
            querySubgroupProperties.subgroupSize = 128;
        if (physicalDeviceProperties.vendorID == 0x13b5) // arm mali
            querySubgroupProperties.subgroupSize = 16;
        if (physicalDeviceProperties.vendorID == 0x1010) // imgtec powervr
            querySubgroupProperties.subgroupSize = 32;
        if (physicalDeviceProperties.vendorID == 0x1002) // amd
            querySubgroupProperties.subgroupSize = 64;
        if (physicalDeviceProperties.vendorID == 0x10de) // nvidia
            querySubgroupProperties.subgroupSize = 32;
        if (physicalDeviceProperties.vendorID == 0x8086) // intel
            querySubgroupProperties.subgroupSize = 32;
    }

    // query driver properties
    memset(&queryDriverProperties, 0, sizeof(queryDriverProperties));
    queryDriverProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES_KHR;
    queryDriverProperties.pNext = 0;
    if (support_VK_KHR_driver_properties)
    {
        queryDriverProperties.pNext = queryExtensionProperties;
        queryExtensionProperties = &queryDriverProperties;
    }

    // query robustness2
    memset(&queryRobustness2Properties, 0, sizeof(queryRobustness2Properties));
    queryRobustness2Properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_KHR;
    queryRobustness2Properties.pNext = 0;
    if (support_VK_KHR_robustness2 || support_VK_EXT_robustness2)
    {
        queryRobustness2Properties.pNext = queryExtensionProperties;
        queryExtensionProperties = &queryRobustness2Properties;
    }

    // query subgroup size control
    memset(&querySubgroupSizeControlProperties, 0, sizeof(querySubgroupSizeControlProperties));
    querySubgroupSizeControlProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
    querySubgroupSizeControlProperties.pNext = 0;
    if (support_VK_EXT_subgroup_size_control)
    {
        querySubgroupSizeControlProperties.pNext = queryExtensionProperties;
        queryExtensionProperties = &querySubgroupSizeControlProperties;
    }

    // query nv cooperative matrix2
    memset(&queryCooperativeMatrix2PropertiesNV, 0, sizeof(queryCooperativeMatrix2PropertiesNV));
    queryCooperativeMatrix2PropertiesNV.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_2_PROPERTIES_NV;
    queryCooperativeMatrix2PropertiesNV.pNext = 0;
    if (support_VK_NV_cooperative_matrix2)
    {
        queryCooperativeMatrix2PropertiesNV.pNext = queryExtensionProperties;
        queryExtensionProperties = &queryCooperativeMatrix2PropertiesNV;
    }

    // query nv cooperative vector
    memset(&queryCooperativeVectorPropertiesNV, 0, sizeof(queryCooperativeVectorPropertiesNV));
    queryCooperativeVectorPropertiesNV.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_PROPERTIES_NV;
    queryCooperativeVectorPropertiesNV.pNext = 0;
    if (support_VK_NV_cooperative_vector)
    {
        queryCooperativeVectorPropertiesNV.pNext = queryExtensionProperties;
        queryExtensionProperties = &queryCooperativeVectorPropertiesNV;
    }

    if (support_VK_KHR_get_physical_device_properties2)
    {
        VkPhysicalDeviceProperties2KHR queryProperties;
        queryProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
        queryProperties.pNext = queryExtensionProperties;

        vkGetPhysicalDeviceProperties2KHR(physicalDevice, &queryProperties);

        // append subgroup rotate
        if (support_VK_KHR_shader_subgroup_rotate)
        {
            if (queryShaderSubgroupRotateFeatures.shaderSubgroupRotate)
                querySubgroupProperties.supportedOperations |= VK_SUBGROUP_FEATURE_ROTATE_BIT_KHR;
            if (queryShaderSubgroupRotateFeatures.shaderSubgroupRotateClustered)
                querySubgroupProperties.supportedOperations |= VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT_KHR;
        }
    }

    if (!support_VK_EXT_subgroup_size_control)
    {
        querySubgroupSizeControlProperties.minSubgroupSize = querySubgroupProperties.subgroupSize;
        querySubgroupSizeControlProperties.maxSubgroupSize = querySubgroupProperties.subgroupSize;
        querySubgroupSizeControlProperties.maxComputeWorkgroupSubgroups = std::max(physicalDeviceProperties.limits.maxComputeWorkGroupInvocations / querySubgroupProperties.subgroupSize, 1u);
    }

    // query supported cooperative matrix types and operations
    queryCooperativeMatrixSubProperties.clear();
    queryCooperativeMatrixSubPropertiesNV.clear();
    support_cooperative_matrix_8_8_16 = false;
    support_cooperative_matrix_16_8_8 = false;
    support_cooperative_matrix_16_8_16 = false;
    support_cooperative_matrix_16_16_16 = false;
    if (support_VK_KHR_cooperative_matrix && queryCooperativeMatrixFeatures.cooperativeMatrix)
    {
        uint32_t propertyCount = 0;
        VkResult ret = vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &propertyCount, 0);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR failed %d", ret);
        }

        queryCooperativeMatrixSubProperties.resize(propertyCount);
        for (uint32_t j = 0; j < propertyCount; j++)
        {
            memset(&queryCooperativeMatrixSubProperties[j], 0, sizeof(queryCooperativeMatrixSubProperties[j]));
            queryCooperativeMatrixSubProperties[j].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
            queryCooperativeMatrixSubProperties[j].pNext = 0;
        }
        ret = vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physicalDevice, &propertyCount, queryCooperativeMatrixSubProperties.data());
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR failed %d", ret);
        }

        for (uint32_t j = 0; j < propertyCount; j++)
        {
            const VkCooperativeMatrixPropertiesKHR& cmp = queryCooperativeMatrixSubProperties[j];
            // NCNN_LOGE("cpm %2d %2d %2d  %d %d %d %d  %d", cmp.MSize, cmp.NSize, cmp.KSize, cmp.AType, cmp.BType, cmp.CType, cmp.ResultType, cmp.scope);

            if (cmp.MSize == 8 && cmp.NSize == 8 && cmp.KSize == 16
                    && cmp.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_KHR
                    && cmp.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && cmp.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR
                    && cmp.scope == VK_SCOPE_SUBGROUP_KHR)
            {
                support_cooperative_matrix_8_8_16 = true;
            }
            if (cmp.MSize == 16 && cmp.NSize == 8 && cmp.KSize == 8
                    && cmp.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_KHR
                    && cmp.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && cmp.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR
                    && cmp.scope == VK_SCOPE_SUBGROUP_KHR)
            {
                support_cooperative_matrix_16_8_8 = true;
            }
            if (cmp.MSize == 16 && cmp.NSize == 8 && cmp.KSize == 16
                    && cmp.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_KHR
                    && cmp.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && cmp.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR
                    && cmp.scope == VK_SCOPE_SUBGROUP_KHR)
            {
                support_cooperative_matrix_16_8_16 = true;
            }
            if (cmp.MSize == 16 && cmp.NSize == 16 && cmp.KSize == 16
                    && cmp.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_KHR
                    && cmp.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && cmp.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR
                    && cmp.scope == VK_SCOPE_SUBGROUP_KHR)
            {
                support_cooperative_matrix_16_16_16 = true;
            }
        }
    }
    else if (support_VK_NV_cooperative_matrix && queryCooperativeMatrixFeaturesNV.cooperativeMatrix)
    {
        uint32_t propertyCount = 0;
        VkResult ret = vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(physicalDevice, &propertyCount, 0);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkGetPhysicalDeviceCooperativeMatrixPropertiesNV failed %d", ret);
        }

        queryCooperativeMatrixSubPropertiesNV.resize(propertyCount);
        for (uint32_t j = 0; j < propertyCount; j++)
        {
            memset(&queryCooperativeMatrixSubPropertiesNV[j], 0, sizeof(queryCooperativeMatrixSubPropertiesNV[j]));
            queryCooperativeMatrixSubPropertiesNV[j].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_NV;
            queryCooperativeMatrixSubPropertiesNV[j].pNext = 0;
        }
        ret = vkGetPhysicalDeviceCooperativeMatrixPropertiesNV(physicalDevice, &propertyCount, queryCooperativeMatrixSubPropertiesNV.data());
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkGetPhysicalDeviceCooperativeMatrixPropertiesNV failed %d", ret);
        }

        for (uint32_t j = 0; j < propertyCount; j++)
        {
            const VkCooperativeMatrixPropertiesNV& cmp = queryCooperativeMatrixSubPropertiesNV[j];
            // NCNN_LOGE("cpm %2d %2d %2d  %d %d %d %d  %d", cmp.MSize, cmp.NSize, cmp.KSize, cmp.AType, cmp.BType, cmp.CType, cmp.DType, cmp.scope);

            if (cmp.MSize == 8 && cmp.NSize == 8 && cmp.KSize == 16
                    && cmp.AType == VK_COMPONENT_TYPE_FLOAT16_NV && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_NV
                    && cmp.CType == VK_COMPONENT_TYPE_FLOAT32_NV && cmp.DType == VK_COMPONENT_TYPE_FLOAT32_NV
                    && cmp.scope == VK_SCOPE_SUBGROUP_NV)
            {
                support_cooperative_matrix_8_8_16 = true;
            }
            if (cmp.MSize == 16 && cmp.NSize == 8 && cmp.KSize == 8
                    && cmp.AType == VK_COMPONENT_TYPE_FLOAT16_NV && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_NV
                    && cmp.CType == VK_COMPONENT_TYPE_FLOAT32_NV && cmp.DType == VK_COMPONENT_TYPE_FLOAT32_NV
                    && cmp.scope == VK_SCOPE_SUBGROUP_NV)
            {
                support_cooperative_matrix_16_8_8 = true;
            }
            if (cmp.MSize == 16 && cmp.NSize == 8 && cmp.KSize == 16
                    && cmp.AType == VK_COMPONENT_TYPE_FLOAT16_NV && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_NV
                    && cmp.CType == VK_COMPONENT_TYPE_FLOAT32_NV && cmp.DType == VK_COMPONENT_TYPE_FLOAT32_NV
                    && cmp.scope == VK_SCOPE_SUBGROUP_NV)
            {
                support_cooperative_matrix_16_8_16 = true;
            }
            if (cmp.MSize == 16 && cmp.NSize == 16 && cmp.KSize == 16
                    && cmp.AType == VK_COMPONENT_TYPE_FLOAT16_NV && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_NV
                    && cmp.CType == VK_COMPONENT_TYPE_FLOAT32_NV && cmp.DType == VK_COMPONENT_TYPE_FLOAT32_NV
                    && cmp.scope == VK_SCOPE_SUBGROUP_NV)
            {
                support_cooperative_matrix_16_16_16 = true;
            }
        }
    }

    // query supported cooperative matrix2 types and operations
    queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV.clear();
    if (support_VK_NV_cooperative_matrix2 && queryCooperativeMatrix2FeaturesNV.cooperativeMatrixFlexibleDimensions)
    {
        uint32_t propertyCount = 0;
        VkResult ret = vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(physicalDevice, &propertyCount, 0);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV failed %d", ret);
        }

        queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV.resize(propertyCount);
        for (uint32_t j = 0; j < propertyCount; j++)
        {
            memset(&queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV[j], 0, sizeof(queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV[j]));
            queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV[j].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_FLEXIBLE_DIMENSIONS_PROPERTIES_NV;
            queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV[j].pNext = 0;
        }
        ret = vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV(physicalDevice, &propertyCount, queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV.data());
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV failed %d", ret);
        }

        for (uint32_t j = 0; j < propertyCount; j++)
        {
            const VkCooperativeMatrixFlexibleDimensionsPropertiesNV& cmfdp = queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV[j];
            // NCNN_LOGE("cmfdp %2d %2d %2d  %d %d %d %d  %d %d %d", cmfdp.MGranularity, cmfdp.NGranularity, cmfdp.KGranularity, cmfdp.AType, cmfdp.BType, cmfdp.CType, cmfdp.ResultType, cmfdp.saturatingAccumulation, cmfdp.scope, cmfdp.workgroupInvocations);
        }
    }

    // query supported cooperative vector types and operations
    queryCooperativeVectorSubPropertiesNV.clear();
    if (support_VK_NV_cooperative_vector && queryCooperativeVectorFeaturesNV.cooperativeVector)
    {
        uint32_t propertyCount = 0;
        VkResult ret = vkGetPhysicalDeviceCooperativeVectorPropertiesNV(physicalDevice, &propertyCount, 0);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkGetPhysicalDeviceCooperativeVectorPropertiesNV failed %d", ret);
        }

        queryCooperativeVectorSubPropertiesNV.resize(propertyCount);
        for (uint32_t j = 0; j < propertyCount; j++)
        {
            memset(&queryCooperativeVectorSubPropertiesNV[j], 0, sizeof(queryCooperativeVectorSubPropertiesNV[j]));
            queryCooperativeVectorSubPropertiesNV[j].sType = VK_STRUCTURE_TYPE_COOPERATIVE_VECTOR_PROPERTIES_NV;
            queryCooperativeVectorSubPropertiesNV[j].pNext = 0;
        }
        ret = vkGetPhysicalDeviceCooperativeVectorPropertiesNV(physicalDevice, &propertyCount, queryCooperativeVectorSubPropertiesNV.data());
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkGetPhysicalDeviceCooperativeVectorPropertiesNV failed %d", ret);
        }

        for (uint32_t j = 0; j < propertyCount; j++)
        {
            const VkCooperativeVectorPropertiesNV& cvp = queryCooperativeVectorSubPropertiesNV[j];
            // NCNN_LOGE("cvp %d %d %d %d %d  %d", cvp.inputType, cvp.inputInterpretation, cvp.matrixInterpretation, cvp.biasInterpretation, cvp.resultType, cvp.transpose);
        }
    }

    if (queryDriverProperties.driverID == VK_DRIVER_ID_MESA_TURNIP)
    {
        // turnip crash when compiling large shader with full subgroup
        querySubgroupSizeControlFeatures.computeFullSubgroups = VK_FALSE;
    }
}

GpuInfo::GpuInfo()
    : d(new GpuInfoPrivate)
{
}

GpuInfo::~GpuInfo()
{
    delete d;
}

GpuInfo::GpuInfo(const GpuInfo&)
    : d(0)
{
}

GpuInfo& GpuInfo::operator=(const GpuInfo&)
{
    return *this;
}

int GpuInfo::device_index() const
{
    return d->device_index;
}

VkPhysicalDevice GpuInfo::physicalDevice() const
{
    return d->physicalDevice;
}

VkPhysicalDevice GpuInfo::physical_device() const
{
    return d->physicalDevice;
}

const VkPhysicalDeviceFeatures& GpuInfo::physicalDevicefeatures() const
{
    return d->physicalDevicefeatures;
}

const VkPhysicalDeviceProperties& GpuInfo::physicalDeviceProperties() const
{
    return d->physicalDeviceProperties;
}

const VkPhysicalDeviceMemoryProperties& GpuInfo::physicalDeviceMemoryProperties() const
{
    return d->physicalDeviceMemoryProperties;
}

const VkPhysicalDeviceMemoryProperties& GpuInfo::physical_device_memory_properties() const
{
    return d->physicalDeviceMemoryProperties;
}

const std::vector<VkExtensionProperties>& GpuInfo::deviceExtensionProperties() const
{
    return d->deviceExtensionProperties;
}

uint32_t GpuInfo::api_version() const
{
    return d->physicalDeviceProperties.apiVersion;
}

uint32_t GpuInfo::driver_version() const
{
    return d->physicalDeviceProperties.driverVersion;
}

uint32_t GpuInfo::vendor_id() const
{
    return d->physicalDeviceProperties.vendorID;
}

uint32_t GpuInfo::device_id() const
{
    return d->physicalDeviceProperties.deviceID;
}

const char* GpuInfo::device_name() const
{
    return d->physicalDeviceProperties.deviceName;
}

uint8_t* GpuInfo::pipeline_cache_uuid() const
{
    return d->physicalDeviceProperties.pipelineCacheUUID;
}

uint32_t GpuInfo::driver_id() const
{
    return d->queryDriverProperties.driverID;
}

const char* GpuInfo::driver_name() const
{
    return d->queryDriverProperties.driverName;
}

int GpuInfo::type() const
{
    return d->type;
}

uint32_t GpuInfo::max_shared_memory_size() const
{
    return d->physicalDeviceProperties.limits.maxComputeSharedMemorySize;
}

uint32_t GpuInfo::max_workgroup_count_x() const
{
    return d->physicalDeviceProperties.limits.maxComputeWorkGroupCount[0];
}

uint32_t GpuInfo::max_workgroup_count_y() const
{
    return d->physicalDeviceProperties.limits.maxComputeWorkGroupCount[1];
}

uint32_t GpuInfo::max_workgroup_count_z() const
{
    return d->physicalDeviceProperties.limits.maxComputeWorkGroupCount[2];
}

uint32_t GpuInfo::max_workgroup_invocations() const
{
    return d->physicalDeviceProperties.limits.maxComputeWorkGroupInvocations;
}

uint32_t GpuInfo::max_workgroup_size_x() const
{
    return d->physicalDeviceProperties.limits.maxComputeWorkGroupSize[0];
}

uint32_t GpuInfo::max_workgroup_size_y() const
{
    return d->physicalDeviceProperties.limits.maxComputeWorkGroupSize[1];
}

uint32_t GpuInfo::max_workgroup_size_z() const
{
    return d->physicalDeviceProperties.limits.maxComputeWorkGroupSize[2];
}

size_t GpuInfo::memory_map_alignment() const
{
    return d->physicalDeviceProperties.limits.minMemoryMapAlignment;
}

size_t GpuInfo::buffer_offset_alignment() const
{
    return d->physicalDeviceProperties.limits.minStorageBufferOffsetAlignment;
}

size_t GpuInfo::non_coherent_atom_size() const
{
    return d->physicalDeviceProperties.limits.nonCoherentAtomSize;
}

size_t GpuInfo::buffer_image_granularity() const
{
    return d->physicalDeviceProperties.limits.bufferImageGranularity;
}

uint32_t GpuInfo::max_image_dimension_1d() const
{
    return d->physicalDeviceProperties.limits.maxImageDimension1D;
}

uint32_t GpuInfo::max_image_dimension_2d() const
{
    return d->physicalDeviceProperties.limits.maxImageDimension2D;
}

uint32_t GpuInfo::max_image_dimension_3d() const
{
    return d->physicalDeviceProperties.limits.maxImageDimension3D;
}

float GpuInfo::timestamp_period() const
{
    return d->physicalDeviceProperties.limits.timestampPeriod;
}

uint32_t GpuInfo::compute_queue_family_index() const
{
    return d->compute_queue_family_index;
}

uint32_t GpuInfo::transfer_queue_family_index() const
{
    return d->transfer_queue_family_index;
}

uint32_t GpuInfo::compute_queue_count() const
{
    return d->compute_queue_count;
}

uint32_t GpuInfo::transfer_queue_count() const
{
    return d->transfer_queue_count;
}

bool GpuInfo::unified_compute_transfer_queue() const
{
    return d->unified_compute_transfer_queue;
}

uint32_t GpuInfo::subgroup_size() const
{
    return d->querySubgroupProperties.subgroupSize;
}

uint32_t GpuInfo::min_subgroup_size() const
{
    return d->querySubgroupSizeControlProperties.minSubgroupSize;
}

uint32_t GpuInfo::max_subgroup_size() const
{
    return d->querySubgroupSizeControlProperties.maxSubgroupSize;
}

uint32_t GpuInfo::max_compute_workgroup_subgroups() const
{
    return d->querySubgroupSizeControlProperties.maxComputeWorkgroupSubgroups;
}

bool GpuInfo::support_subgroup_size_control() const
{
    return d->querySubgroupSizeControlFeatures.subgroupSizeControl;
}

bool GpuInfo::support_compute_full_subgroups() const
{
    return d->querySubgroupSizeControlFeatures.computeFullSubgroups;
}

uint32_t GpuInfo::support_subgroup_ops() const
{
    return d->querySubgroupProperties.supportedOperations;
}

bool GpuInfo::bug_storage_buffer_no_l1() const
{
    return d->bug_storage_buffer_no_l1;
}

bool GpuInfo::bug_corrupted_online_pipeline_cache() const
{
    return d->bug_corrupted_online_pipeline_cache;
}

bool GpuInfo::bug_buffer_image_load_zero() const
{
    return d->bug_buffer_image_load_zero;
}

bool GpuInfo::bug_implicit_fp16_arithmetic() const
{
    return d->bug_implicit_fp16_arithmetic;
}

bool GpuInfo::support_fp16_packed() const
{
    return true;
}

bool GpuInfo::support_fp16_storage() const
{
    return d->query16BitStorageFeatures.storageBuffer16BitAccess;
}

bool GpuInfo::support_fp16_uniform() const
{
    return d->query16BitStorageFeatures.uniformAndStorageBuffer16BitAccess;
}

bool GpuInfo::support_fp16_arithmetic() const
{
    return d->queryFloat16Int8Features.shaderFloat16;
}

bool GpuInfo::support_int8_packed() const
{
    return true;
}

bool GpuInfo::support_int8_storage() const
{
    return d->query8BitStorageFeatures.storageBuffer8BitAccess;
}

bool GpuInfo::support_int8_uniform() const
{
    return d->query8BitStorageFeatures.uniformAndStorageBuffer8BitAccess;
}

bool GpuInfo::support_int8_arithmetic() const
{
    return d->queryFloat16Int8Features.shaderInt8;
}

bool GpuInfo::support_fp16_image() const
{
    return d->physicalDevicefeatures.shaderStorageImageExtendedFormats;
}

bool GpuInfo::support_int8_image() const
{
    return d->physicalDevicefeatures.shaderStorageImageExtendedFormats;
}

bool GpuInfo::support_fp_fast_math() const
{
    return d->queryShaderFloatControls2Features.shaderFloatControls2;
}

bool GpuInfo::support_ycbcr_conversion() const
{
    return d->querySamplerYcbcrConversionFeatures.samplerYcbcrConversion;
}

bool GpuInfo::support_cooperative_matrix() const
{
    return d->queryCooperativeMatrixFeatures.cooperativeMatrix || d->queryCooperativeMatrixFeaturesNV.cooperativeMatrix;
}

bool GpuInfo::support_cooperative_matrix_8_8_16() const
{
    return d->support_cooperative_matrix_8_8_16;
}

bool GpuInfo::support_cooperative_matrix_16_8_8() const
{
    return d->support_cooperative_matrix_16_8_8;
}

bool GpuInfo::support_cooperative_matrix_16_8_16() const
{
    return d->support_cooperative_matrix_16_8_16;
}

bool GpuInfo::support_cooperative_matrix_16_16_16() const
{
    return d->support_cooperative_matrix_16_16_16;
}

int GpuInfo::support_VK_KHR_8bit_storage() const
{
    return d->support_VK_KHR_8bit_storage;
}

int GpuInfo::support_VK_KHR_16bit_storage() const
{
    return d->support_VK_KHR_16bit_storage;
}

int GpuInfo::support_VK_KHR_bind_memory2() const
{
    return d->support_VK_KHR_bind_memory2;
}

int GpuInfo::support_VK_KHR_buffer_device_address() const
{
    return d->support_VK_KHR_buffer_device_address;
}

int GpuInfo::support_VK_KHR_create_renderpass2() const
{
    return d->support_VK_KHR_create_renderpass2;
}

int GpuInfo::support_VK_KHR_cooperative_matrix() const
{
    return d->support_VK_KHR_cooperative_matrix;
}

int GpuInfo::support_VK_KHR_dedicated_allocation() const
{
    return d->support_VK_KHR_dedicated_allocation;
}

int GpuInfo::support_VK_KHR_descriptor_update_template() const
{
    return d->support_VK_KHR_descriptor_update_template;
}

int GpuInfo::support_VK_KHR_driver_properties() const
{
    return d->support_VK_KHR_driver_properties;
}

int GpuInfo::support_VK_KHR_external_memory() const
{
    return d->support_VK_KHR_external_memory;
}

int GpuInfo::support_VK_KHR_get_memory_requirements2() const
{
    return d->support_VK_KHR_get_memory_requirements2;
}

int GpuInfo::support_VK_KHR_maintenance1() const
{
    return d->support_VK_KHR_maintenance1;
}

int GpuInfo::support_VK_KHR_maintenance2() const
{
    return d->support_VK_KHR_maintenance2;
}

int GpuInfo::support_VK_KHR_maintenance3() const
{
    return d->support_VK_KHR_maintenance3;
}

int GpuInfo::support_VK_KHR_multiview() const
{
    return d->support_VK_KHR_multiview;
}

int GpuInfo::support_VK_KHR_portability_subset() const
{
    return d->support_VK_KHR_portability_subset;
}

int GpuInfo::support_VK_KHR_push_descriptor() const
{
    return d->support_VK_KHR_push_descriptor;
}

int GpuInfo::support_VK_KHR_robustness2() const
{
    return d->support_VK_KHR_robustness2;
}

int GpuInfo::support_VK_KHR_sampler_ycbcr_conversion() const
{
    return d->support_VK_KHR_sampler_ycbcr_conversion;
}

int GpuInfo::support_VK_KHR_shader_bfloat16() const
{
    return d->support_VK_KHR_shader_bfloat16;
}

int GpuInfo::support_VK_KHR_shader_float16_int8() const
{
    return d->support_VK_KHR_shader_float16_int8;
}

int GpuInfo::support_VK_KHR_shader_float_controls() const
{
    return d->support_VK_KHR_shader_float_controls;
}

int GpuInfo::support_VK_KHR_shader_float_controls2() const
{
    return d->support_VK_KHR_shader_float_controls2;
}

int GpuInfo::support_VK_KHR_shader_integer_dot_product() const
{
    return d->support_VK_KHR_shader_integer_dot_product;
}

int GpuInfo::support_VK_KHR_shader_non_semantic_info() const
{
    return d->support_VK_KHR_shader_non_semantic_info;
}

int GpuInfo::support_VK_KHR_shader_subgroup_extended_types() const
{
    return d->support_VK_KHR_shader_subgroup_extended_types;
}

int GpuInfo::support_VK_KHR_shader_subgroup_rotate() const
{
    return d->support_VK_KHR_shader_subgroup_rotate;
}

int GpuInfo::support_VK_KHR_storage_buffer_storage_class() const
{
    return d->support_VK_KHR_storage_buffer_storage_class;
}

int GpuInfo::support_VK_KHR_swapchain() const
{
    return d->support_VK_KHR_swapchain;
}

int GpuInfo::support_VK_KHR_vulkan_memory_model() const
{
    return d->support_VK_KHR_vulkan_memory_model;
}

int GpuInfo::support_VK_KHR_zero_initialize_workgroup_memory() const
{
    return d->support_VK_KHR_zero_initialize_workgroup_memory;
}

int GpuInfo::support_VK_EXT_buffer_device_address() const
{
    return d->support_VK_EXT_buffer_device_address;
}

int GpuInfo::support_VK_EXT_descriptor_indexing() const
{
    return d->support_VK_EXT_descriptor_indexing;
}

int GpuInfo::support_VK_EXT_memory_budget() const
{
    return d->support_VK_EXT_memory_budget;
}

int GpuInfo::support_VK_EXT_memory_priority() const
{
    return d->support_VK_EXT_memory_priority;
}

int GpuInfo::support_VK_EXT_queue_family_foreign() const
{
    return d->support_VK_EXT_queue_family_foreign;
}

int GpuInfo::support_VK_EXT_robustness2() const
{
    return d->support_VK_EXT_robustness2;
}

int GpuInfo::support_VK_EXT_shader_atomic_float() const
{
    return d->support_VK_EXT_shader_atomic_float;
}

int GpuInfo::support_VK_EXT_shader_atomic_float2() const
{
    return d->support_VK_EXT_shader_atomic_float2;
}

int GpuInfo::support_VK_EXT_shader_float8() const
{
    return d->support_VK_EXT_shader_float8;
}

int GpuInfo::support_VK_EXT_subgroup_size_control() const
{
    return d->support_VK_EXT_subgroup_size_control;
}

int GpuInfo::support_VK_AMD_device_coherent_memory() const
{
    return d->support_VK_AMD_device_coherent_memory;
}

#if __ANDROID_API__ >= 26
int GpuInfo::support_VK_ANDROID_external_memory_android_hardware_buffer() const
{
    return d->support_VK_ANDROID_external_memory_android_hardware_buffer;
}
#endif // __ANDROID_API__ >= 26

int GpuInfo::support_VK_NV_cooperative_matrix() const
{
    return d->support_VK_NV_cooperative_matrix;
}

int GpuInfo::support_VK_NV_cooperative_matrix2() const
{
    return d->support_VK_NV_cooperative_matrix2;
}

int GpuInfo::support_VK_NV_cooperative_vector() const
{
    return d->support_VK_NV_cooperative_vector;
}

const void* GpuInfo::queryExtensionFeatures() const
{
    return d->queryExtensionFeatures;
}

const VkPhysicalDevice8BitStorageFeaturesKHR& GpuInfo::query8BitStorageFeatures() const
{
    return d->query8BitStorageFeatures;
}

const VkPhysicalDevice16BitStorageFeaturesKHR& GpuInfo::query16BitStorageFeatures() const
{
    return d->query16BitStorageFeatures;
}

const VkPhysicalDeviceFloat16Int8FeaturesKHR& GpuInfo::queryFloat16Int8Features() const
{
    return d->queryFloat16Int8Features;
}

const VkPhysicalDeviceSamplerYcbcrConversionFeaturesKHR& GpuInfo::querySamplerYcbcrConversionFeatures() const
{
    return d->querySamplerYcbcrConversionFeatures;
}

const VkPhysicalDeviceCooperativeMatrixFeaturesKHR& GpuInfo::queryCooperativeMatrixFeatures() const
{
    return d->queryCooperativeMatrixFeatures;
}

const VkPhysicalDeviceCooperativeMatrixFeaturesNV& GpuInfo::queryCooperativeMatrixFeaturesNV() const
{
    return d->queryCooperativeMatrixFeaturesNV;
}

const VkPhysicalDeviceCooperativeMatrix2FeaturesNV& GpuInfo::queryCooperativeMatrix2FeaturesNV() const
{
    return d->queryCooperativeMatrix2FeaturesNV;
}

const VkPhysicalDeviceCooperativeVectorFeaturesNV& GpuInfo::queryCooperativeVectorFeaturesNV() const
{
    return d->queryCooperativeVectorFeaturesNV;
}

const VkPhysicalDeviceRobustness2FeaturesKHR& GpuInfo::queryRobustness2Features() const
{
    return d->queryRobustness2Features;
}

const VkPhysicalDeviceSubgroupSizeControlFeaturesEXT& GpuInfo::querySubgroupSizeControlFeatures() const
{
    return d->querySubgroupSizeControlFeatures;
}

const VkPhysicalDeviceShaderBfloat16FeaturesKHR& GpuInfo::queryShaderBfloat16Features() const
{
    return d->queryShaderBfloat16Features;
}

const VkPhysicalDeviceShaderFloat8FeaturesEXT& GpuInfo::queryShaderFloat8Features() const
{
    return d->queryShaderFloat8Features;
}

const VkPhysicalDeviceShaderFloatControls2FeaturesKHR& GpuInfo::queryShaderFloatControls2Features() const
{
    return d->queryShaderFloatControls2Features;
}

const VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR& GpuInfo::queryShaderIntegerDotProductFeatures() const
{
    return d->queryShaderIntegerDotProductFeatures;
}

const VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR& GpuInfo::queryShaderSubgroupRotateFeatures() const
{
    return d->queryShaderSubgroupRotateFeatures;
}

const VkPhysicalDeviceShaderAtomicFloatFeaturesEXT& GpuInfo::queryShaderAtomicFloatFeatures() const
{
    return d->queryShaderAtomicFloatFeatures;
}

const VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT& GpuInfo::queryShaderAtomicFloat2Features() const
{
    return d->queryShaderAtomicFloat2Features;
}

const VkPhysicalDeviceVulkanMemoryModelFeaturesKHR& GpuInfo::queryVulkanMemoryModelFeatures() const
{
    return d->queryVulkanMemoryModelFeatures;
}

const void* GpuInfo::queryExtensionProperties() const
{
    return d->queryExtensionProperties;
}

const VkPhysicalDeviceCooperativeMatrix2PropertiesNV& GpuInfo::queryCooperativeMatrix2PropertiesNV() const
{
    return d->queryCooperativeMatrix2PropertiesNV;
}

const VkPhysicalDeviceCooperativeVectorPropertiesNV& GpuInfo::queryCooperativeVectorPropertiesNV() const
{
    return d->queryCooperativeVectorPropertiesNV;
}

const VkPhysicalDeviceDriverPropertiesKHR& GpuInfo::queryDriverProperties() const
{
    return d->queryDriverProperties;
}

const VkPhysicalDeviceFloatControlsPropertiesKHR& GpuInfo::queryFloatControlsProperties() const
{
    return d->queryFloatControlsProperties;
}

const VkPhysicalDeviceRobustness2PropertiesKHR& GpuInfo::queryRobustness2Properties() const
{
    return d->queryRobustness2Properties;
}

const VkPhysicalDeviceShaderIntegerDotProductProperties& GpuInfo::queryShaderIntegerDotProductProperties() const
{
    return d->queryShaderIntegerDotProductProperties;
}

const VkPhysicalDeviceSubgroupProperties& GpuInfo::querySubgroupProperties() const
{
    return d->querySubgroupProperties;
}

const VkPhysicalDeviceSubgroupSizeControlPropertiesEXT& GpuInfo::querySubgroupSizeControlProperties() const
{
    return d->querySubgroupSizeControlProperties;
}

const std::vector<VkCooperativeMatrixPropertiesKHR>& GpuInfo::queryCooperativeMatrixSubProperties() const
{
    return d->queryCooperativeMatrixSubProperties;
}

const std::vector<VkCooperativeMatrixPropertiesNV>& GpuInfo::queryCooperativeMatrixSubPropertiesNV() const
{
    return d->queryCooperativeMatrixSubPropertiesNV;
}

const std::vector<VkCooperativeMatrixFlexibleDimensionsPropertiesNV>& GpuInfo::queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV() const
{
    return d->queryCooperativeMatrixFlexibleDimensionsSubPropertiesNV;
}

const std::vector<VkCooperativeVectorPropertiesNV>& GpuInfo::queryCooperativeVectorSubPropertiesNV() const
{
    return d->queryCooperativeVectorSubPropertiesNV;
}

void GpuInfo::get_optimal_cooperative_matrix_mnk(int M, int N, int K, VkComponentTypeKHR type, VkComponentTypeKHR acctype, VkScopeKHR scope, int& coopmat_M, int& coopmat_N, int& coopmat_K) const
{
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;

    // collect mnk candidates
    std::vector<VkCooperativeMatrixPropertiesKHR> mnk_properties;

    if (d->support_VK_KHR_cooperative_matrix && d->queryCooperativeMatrixFeatures.cooperativeMatrix)
    {
        for (size_t i = 0; i < d->queryCooperativeMatrixSubProperties.size(); i++)
        {
            const VkCooperativeMatrixPropertiesKHR& cmp = d->queryCooperativeMatrixSubProperties[i];

            if (cmp.AType == type && cmp.BType == type
                    && cmp.CType == acctype && cmp.ResultType == acctype
                    && cmp.scope == scope)
            {
                mnk_properties.push_back(cmp);
            }
        }
    }
    else if (d->support_VK_NV_cooperative_matrix && d->queryCooperativeMatrixFeaturesNV.cooperativeMatrix)
    {
        for (size_t i = 0; i < d->queryCooperativeMatrixSubPropertiesNV.size(); i++)
        {
            const VkCooperativeMatrixPropertiesNV& cmp = d->queryCooperativeMatrixSubPropertiesNV[i];

            if (cmp.AType == (VkComponentTypeNV)type && cmp.BType == (VkComponentTypeNV)type
                    && cmp.CType == (VkComponentTypeNV)acctype && cmp.DType == (VkComponentTypeNV)acctype
                    && cmp.scope == (VkScopeNV)scope)
            {
                VkCooperativeMatrixPropertiesKHR cmp_khr;
                cmp_khr.MSize = cmp.MSize;
                cmp_khr.NSize = cmp.NSize;
                cmp_khr.KSize = cmp.KSize;

                mnk_properties.push_back(cmp_khr);
            }
        }
    }

    if (mnk_properties.empty() && (acctype == VK_COMPONENT_TYPE_FLOAT16_KHR || acctype == VK_COMPONENT_TYPE_BFLOAT16_KHR))
    {
        // try acctype fp32
        return get_optimal_cooperative_matrix_mnk(M, N, K, type, VK_COMPONENT_TYPE_FLOAT32_KHR, scope, coopmat_M, coopmat_N, coopmat_K);
    }

    if (mnk_properties.empty())
        return;

    // find the optimal, prefer the first mnk tuple with same cost
    double min_cost = DBL_MAX;
    for (size_t i = 0; i < mnk_properties.size(); i++)
    {
        const VkCooperativeMatrixPropertiesKHR& cmp = mnk_properties[i];

        const int M_pad = (M + cmp.MSize - 1) / cmp.MSize * cmp.MSize;
        const int N_pad = (N + cmp.NSize - 1) / cmp.NSize * cmp.NSize;
        const int K_pad = (K + cmp.KSize - 1) / cmp.KSize * cmp.KSize;

        double cost = M_pad * N_pad * K_pad - M * N * K;
        if (cost < min_cost)
        {
            min_cost = cost;
            coopmat_M = cmp.MSize;
            coopmat_N = cmp.NSize;
            coopmat_K = cmp.KSize;
        }
    }
}

static int init_instance_core()
{
    vkAllocateCommandBuffers = (PFN_vkAllocateCommandBuffers)vkGetInstanceProcAddr(g_instance, "vkAllocateCommandBuffers");
    vkAllocateDescriptorSets = (PFN_vkAllocateDescriptorSets)vkGetInstanceProcAddr(g_instance, "vkAllocateDescriptorSets");
    vkAllocateMemory = (PFN_vkAllocateMemory)vkGetInstanceProcAddr(g_instance, "vkAllocateMemory");
    vkBeginCommandBuffer = (PFN_vkBeginCommandBuffer)vkGetInstanceProcAddr(g_instance, "vkBeginCommandBuffer");
    vkBindBufferMemory = (PFN_vkBindBufferMemory)vkGetInstanceProcAddr(g_instance, "vkBindBufferMemory");
    vkBindImageMemory = (PFN_vkBindImageMemory)vkGetInstanceProcAddr(g_instance, "vkBindImageMemory");
    vkCmdBeginQuery = (PFN_vkCmdBeginQuery)vkGetInstanceProcAddr(g_instance, "vkCmdBeginQuery");
    vkCmdBindDescriptorSets = (PFN_vkCmdBindDescriptorSets)vkGetInstanceProcAddr(g_instance, "vkCmdBindDescriptorSets");
    vkCmdBindIndexBuffer = (PFN_vkCmdBindIndexBuffer)vkGetInstanceProcAddr(g_instance, "vkCmdBindIndexBuffer");
    vkCmdBindPipeline = (PFN_vkCmdBindPipeline)vkGetInstanceProcAddr(g_instance, "vkCmdBindPipeline");
    vkCmdCopyBuffer = (PFN_vkCmdCopyBuffer)vkGetInstanceProcAddr(g_instance, "vkCmdCopyBuffer");
    vkCmdCopyBufferToImage = (PFN_vkCmdCopyBufferToImage)vkGetInstanceProcAddr(g_instance, "vkCmdCopyBufferToImage");
    vkCmdCopyImage = (PFN_vkCmdCopyImage)vkGetInstanceProcAddr(g_instance, "vkCmdCopyImage");
    vkCmdCopyImageToBuffer = (PFN_vkCmdCopyImageToBuffer)vkGetInstanceProcAddr(g_instance, "vkCmdCopyImageToBuffer");
    vkCmdCopyQueryPoolResults = (PFN_vkCmdCopyQueryPoolResults)vkGetInstanceProcAddr(g_instance, "vkCmdCopyQueryPoolResults");
    vkCmdDispatch = (PFN_vkCmdDispatch)vkGetInstanceProcAddr(g_instance, "vkCmdDispatch");
    vkCmdDispatchIndirect = (PFN_vkCmdDispatchIndirect)vkGetInstanceProcAddr(g_instance, "vkCmdDispatchIndirect");
    vkCmdEndQuery = (PFN_vkCmdEndQuery)vkGetInstanceProcAddr(g_instance, "vkCmdEndQuery");
    vkCmdExecuteCommands = (PFN_vkCmdExecuteCommands)vkGetInstanceProcAddr(g_instance, "vkCmdExecuteCommands");
    vkCmdFillBuffer = (PFN_vkCmdFillBuffer)vkGetInstanceProcAddr(g_instance, "vkCmdFillBuffer");
    vkCmdPipelineBarrier = (PFN_vkCmdPipelineBarrier)vkGetInstanceProcAddr(g_instance, "vkCmdPipelineBarrier");
    vkCmdPushConstants = (PFN_vkCmdPushConstants)vkGetInstanceProcAddr(g_instance, "vkCmdPushConstants");
    vkCmdResetQueryPool = (PFN_vkCmdResetQueryPool)vkGetInstanceProcAddr(g_instance, "vkCmdResetQueryPool");
    vkCmdResolveImage = (PFN_vkCmdResolveImage)vkGetInstanceProcAddr(g_instance, "vkCmdResolveImage");
    vkCmdUpdateBuffer = (PFN_vkCmdUpdateBuffer)vkGetInstanceProcAddr(g_instance, "vkCmdUpdateBuffer");
    vkCmdWriteTimestamp = (PFN_vkCmdWriteTimestamp)vkGetInstanceProcAddr(g_instance, "vkCmdWriteTimestamp");
    vkCreateBuffer = (PFN_vkCreateBuffer)vkGetInstanceProcAddr(g_instance, "vkCreateBuffer");
    vkCreateBufferView = (PFN_vkCreateBufferView)vkGetInstanceProcAddr(g_instance, "vkCreateBufferView");
    vkCreateCommandPool = (PFN_vkCreateCommandPool)vkGetInstanceProcAddr(g_instance, "vkCreateCommandPool");
    vkCreateComputePipelines = (PFN_vkCreateComputePipelines)vkGetInstanceProcAddr(g_instance, "vkCreateComputePipelines");
    vkCreateDescriptorPool = (PFN_vkCreateDescriptorPool)vkGetInstanceProcAddr(g_instance, "vkCreateDescriptorPool");
    vkCreateDescriptorSetLayout = (PFN_vkCreateDescriptorSetLayout)vkGetInstanceProcAddr(g_instance, "vkCreateDescriptorSetLayout");
    vkCreateDevice = (PFN_vkCreateDevice)vkGetInstanceProcAddr(g_instance, "vkCreateDevice");
    vkCreateFence = (PFN_vkCreateFence)vkGetInstanceProcAddr(g_instance, "vkCreateFence");
    vkCreateImage = (PFN_vkCreateImage)vkGetInstanceProcAddr(g_instance, "vkCreateImage");
    vkCreateImageView = (PFN_vkCreateImageView)vkGetInstanceProcAddr(g_instance, "vkCreateImageView");
    vkCreatePipelineCache = (PFN_vkCreatePipelineCache)vkGetInstanceProcAddr(g_instance, "vkCreatePipelineCache");
    vkCreatePipelineLayout = (PFN_vkCreatePipelineLayout)vkGetInstanceProcAddr(g_instance, "vkCreatePipelineLayout");
    vkCreateQueryPool = (PFN_vkCreateQueryPool)vkGetInstanceProcAddr(g_instance, "vkCreateQueryPool");
    vkCreateSampler = (PFN_vkCreateSampler)vkGetInstanceProcAddr(g_instance, "vkCreateSampler");
    vkCreateSemaphore = (PFN_vkCreateSemaphore)vkGetInstanceProcAddr(g_instance, "vkCreateSemaphore");
    vkCreateShaderModule = (PFN_vkCreateShaderModule)vkGetInstanceProcAddr(g_instance, "vkCreateShaderModule");
    vkDestroyBuffer = (PFN_vkDestroyBuffer)vkGetInstanceProcAddr(g_instance, "vkDestroyBuffer");
    vkDestroyBufferView = (PFN_vkDestroyBufferView)vkGetInstanceProcAddr(g_instance, "vkDestroyBufferView");
    vkDestroyCommandPool = (PFN_vkDestroyCommandPool)vkGetInstanceProcAddr(g_instance, "vkDestroyCommandPool");
    vkDestroyDescriptorPool = (PFN_vkDestroyDescriptorPool)vkGetInstanceProcAddr(g_instance, "vkDestroyDescriptorPool");
    vkDestroyDescriptorSetLayout = (PFN_vkDestroyDescriptorSetLayout)vkGetInstanceProcAddr(g_instance, "vkDestroyDescriptorSetLayout");
    vkDestroyDevice = (PFN_vkDestroyDevice)vkGetInstanceProcAddr(g_instance, "vkDestroyDevice");
    vkDestroyFence = (PFN_vkDestroyFence)vkGetInstanceProcAddr(g_instance, "vkDestroyFence");
    vkDestroyImage = (PFN_vkDestroyImage)vkGetInstanceProcAddr(g_instance, "vkDestroyImage");
    vkDestroyImageView = (PFN_vkDestroyImageView)vkGetInstanceProcAddr(g_instance, "vkDestroyImageView");
    vkDestroyInstance = (PFN_vkDestroyInstance)vkGetInstanceProcAddr(g_instance, "vkDestroyInstance");
    vkDestroyPipeline = (PFN_vkDestroyPipeline)vkGetInstanceProcAddr(g_instance, "vkDestroyPipeline");
    vkDestroyPipelineCache = (PFN_vkDestroyPipelineCache)vkGetInstanceProcAddr(g_instance, "vkDestroyPipelineCache");
    vkDestroyPipelineLayout = (PFN_vkDestroyPipelineLayout)vkGetInstanceProcAddr(g_instance, "vkDestroyPipelineLayout");
    vkDestroyQueryPool = (PFN_vkDestroyQueryPool)vkGetInstanceProcAddr(g_instance, "vkDestroyQueryPool");
    vkDestroySampler = (PFN_vkDestroySampler)vkGetInstanceProcAddr(g_instance, "vkDestroySampler");
    vkDestroySemaphore = (PFN_vkDestroySemaphore)vkGetInstanceProcAddr(g_instance, "vkDestroySemaphore");
    vkDestroyShaderModule = (PFN_vkDestroyShaderModule)vkGetInstanceProcAddr(g_instance, "vkDestroyShaderModule");
    vkDeviceWaitIdle = (PFN_vkDeviceWaitIdle)vkGetInstanceProcAddr(g_instance, "vkDeviceWaitIdle");
    vkEndCommandBuffer = (PFN_vkEndCommandBuffer)vkGetInstanceProcAddr(g_instance, "vkEndCommandBuffer");
    vkEnumerateDeviceExtensionProperties = (PFN_vkEnumerateDeviceExtensionProperties)vkGetInstanceProcAddr(g_instance, "vkEnumerateDeviceExtensionProperties");
    vkEnumerateDeviceLayerProperties = (PFN_vkEnumerateDeviceLayerProperties)vkGetInstanceProcAddr(g_instance, "vkEnumerateDeviceLayerProperties");
    vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)vkGetInstanceProcAddr(g_instance, "vkEnumeratePhysicalDevices");
    vkFlushMappedMemoryRanges = (PFN_vkFlushMappedMemoryRanges)vkGetInstanceProcAddr(g_instance, "vkFlushMappedMemoryRanges");
    vkFreeCommandBuffers = (PFN_vkFreeCommandBuffers)vkGetInstanceProcAddr(g_instance, "vkFreeCommandBuffers");
    vkFreeDescriptorSets = (PFN_vkFreeDescriptorSets)vkGetInstanceProcAddr(g_instance, "vkFreeDescriptorSets");
    vkFreeMemory = (PFN_vkFreeMemory)vkGetInstanceProcAddr(g_instance, "vkFreeMemory");
    vkGetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)vkGetInstanceProcAddr(g_instance, "vkGetBufferMemoryRequirements");
    vkGetDeviceMemoryCommitment = (PFN_vkGetDeviceMemoryCommitment)vkGetInstanceProcAddr(g_instance, "vkGetDeviceMemoryCommitment");
    vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(g_instance, "vkGetDeviceProcAddr");
    vkGetDeviceQueue = (PFN_vkGetDeviceQueue)vkGetInstanceProcAddr(g_instance, "vkGetDeviceQueue");
    vkGetFenceStatus = (PFN_vkGetFenceStatus)vkGetInstanceProcAddr(g_instance, "vkGetFenceStatus");
    vkGetImageMemoryRequirements = (PFN_vkGetImageMemoryRequirements)vkGetInstanceProcAddr(g_instance, "vkGetImageMemoryRequirements");
    vkGetImageSubresourceLayout = (PFN_vkGetImageSubresourceLayout)vkGetInstanceProcAddr(g_instance, "vkGetImageSubresourceLayout");
    vkGetPhysicalDeviceFeatures = (PFN_vkGetPhysicalDeviceFeatures)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceFeatures");
    vkGetPhysicalDeviceFormatProperties = (PFN_vkGetPhysicalDeviceFormatProperties)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceFormatProperties");
    vkGetPhysicalDeviceImageFormatProperties = (PFN_vkGetPhysicalDeviceImageFormatProperties)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceImageFormatProperties");
    vkGetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceMemoryProperties");
    vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceProperties");
    vkGetPhysicalDeviceQueueFamilyProperties = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    vkGetPipelineCacheData = (PFN_vkGetPipelineCacheData)vkGetInstanceProcAddr(g_instance, "vkGetPipelineCacheData");
    vkGetQueryPoolResults = (PFN_vkGetQueryPoolResults)vkGetInstanceProcAddr(g_instance, "vkGetQueryPoolResults");
    vkInvalidateMappedMemoryRanges = (PFN_vkInvalidateMappedMemoryRanges)vkGetInstanceProcAddr(g_instance, "vkInvalidateMappedMemoryRanges");
    vkMapMemory = (PFN_vkMapMemory)vkGetInstanceProcAddr(g_instance, "vkMapMemory");
    vkMergePipelineCaches = (PFN_vkMergePipelineCaches)vkGetInstanceProcAddr(g_instance, "vkMergePipelineCaches");
    vkQueueSubmit = (PFN_vkQueueSubmit)vkGetInstanceProcAddr(g_instance, "vkQueueSubmit");
    vkQueueWaitIdle = (PFN_vkQueueWaitIdle)vkGetInstanceProcAddr(g_instance, "vkQueueWaitIdle");
    vkResetCommandBuffer = (PFN_vkResetCommandBuffer)vkGetInstanceProcAddr(g_instance, "vkResetCommandBuffer");
    vkResetCommandPool = (PFN_vkResetCommandPool)vkGetInstanceProcAddr(g_instance, "vkResetCommandPool");
    vkResetDescriptorPool = (PFN_vkResetDescriptorPool)vkGetInstanceProcAddr(g_instance, "vkResetDescriptorPool");
    vkResetFences = (PFN_vkResetFences)vkGetInstanceProcAddr(g_instance, "vkResetFences");
    vkUnmapMemory = (PFN_vkUnmapMemory)vkGetInstanceProcAddr(g_instance, "vkUnmapMemory");
    vkUpdateDescriptorSets = (PFN_vkUpdateDescriptorSets)vkGetInstanceProcAddr(g_instance, "vkUpdateDescriptorSets");
    vkWaitForFences = (PFN_vkWaitForFences)vkGetInstanceProcAddr(g_instance, "vkWaitForFences");

    return 0;
}

static int init_instance_extension()
{
    if (support_VK_KHR_external_memory_capabilities)
    {
        vkGetPhysicalDeviceExternalBufferPropertiesKHR = (PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceExternalBufferPropertiesKHR");
    }

    if (support_VK_KHR_get_physical_device_properties2)
    {
        vkGetPhysicalDeviceFeatures2KHR = (PFN_vkGetPhysicalDeviceFeatures2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceFeatures2KHR");
        vkGetPhysicalDeviceProperties2KHR = (PFN_vkGetPhysicalDeviceProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceProperties2KHR");
        vkGetPhysicalDeviceFormatProperties2KHR = (PFN_vkGetPhysicalDeviceFormatProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceFormatProperties2KHR");
        vkGetPhysicalDeviceImageFormatProperties2KHR = (PFN_vkGetPhysicalDeviceImageFormatProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceImageFormatProperties2KHR");
        vkGetPhysicalDeviceQueueFamilyProperties2KHR = (PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceQueueFamilyProperties2KHR");
        vkGetPhysicalDeviceMemoryProperties2KHR = (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceMemoryProperties2KHR");
    }

    if (support_VK_KHR_get_surface_capabilities2)
    {
        vkGetPhysicalDeviceSurfaceCapabilities2KHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceCapabilities2KHR");
        vkGetPhysicalDeviceSurfaceFormats2KHR = (PFN_vkGetPhysicalDeviceSurfaceFormats2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceFormats2KHR");
    }

    if (support_VK_KHR_surface)
    {
        vkDestroySurfaceKHR = (PFN_vkDestroySurfaceKHR)vkGetInstanceProcAddr(g_instance, "vkDestroySurfaceKHR");
        vkGetPhysicalDeviceSurfaceSupportKHR = (PFN_vkGetPhysicalDeviceSurfaceSupportKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceSupportKHR");
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
        vkGetPhysicalDeviceSurfaceFormatsKHR = (PFN_vkGetPhysicalDeviceSurfaceFormatsKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceFormatsKHR");
        vkGetPhysicalDeviceSurfacePresentModesKHR = (PFN_vkGetPhysicalDeviceSurfacePresentModesKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfacePresentModesKHR");
    }

#if __ANDROID_API__ >= 26
    if (support_VK_KHR_android_surface)
    {
        vkCreateAndroidSurfaceKHR = (PFN_vkCreateAndroidSurfaceKHR)vkGetInstanceProcAddr(g_instance, "vkCreateAndroidSurfaceKHR");
    }
#endif // __ANDROID_API__ >= 26

    // VK_KHR_cooperative_matrix
    {
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    }

    // VK_NV_cooperative_matrix
    {
        vkGetPhysicalDeviceCooperativeMatrixPropertiesNV = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesNV)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesNV");
    }

    // VK_NV_cooperative_matrix2
    {
        vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV = (PFN_vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV");
    }

    // VK_NV_cooperative_vector
    {
        vkGetPhysicalDeviceCooperativeVectorPropertiesNV = (PFN_vkGetPhysicalDeviceCooperativeVectorPropertiesNV)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceCooperativeVectorPropertiesNV");
    }

    return 0;
}

#if ENABLE_VALIDATION_LAYER
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
    VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* /*pUserData*/)
{
    NCNN_LOGE("validation layer: %s", pCallbackData->pMessage);

    return VK_FALSE;
}

static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback)
{
    PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func)
        return func(instance, pCreateInfo, pAllocator, pCallback);

    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator)
{
    PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func)
        func(instance, callback, pAllocator);
}
#endif // ENABLE_VALIDATION_LAYER

static int find_default_vulkan_device_index()
{
    // first try, discrete gpu
    for (int i = 0; i < g_gpu_count; i++)
    {
        if (g_gpu_infos[i]->type() == 0)
            return i;
    }

    // second try, integrated gpu
    for (int i = 0; i < g_gpu_count; i++)
    {
        if (g_gpu_infos[i]->type() == 1)
            return i;
    }

    // third try, any probed device
    if (g_gpu_count > 0)
        return 0;

    NCNN_LOGE("no vulkan device");
    return -1;
}

int create_gpu_instance(const char* driver_path)
{
    MutexLockGuard lock(g_instance_lock);

    if (g_instance.created != 0)
        return g_instance.instance ? 0 : -1;

    g_instance.created = 1;

    // NCNN_LOGE("create_gpu_instance");

#if NCNN_SIMPLEVK
    // load vulkan driver
    {
        int ret = load_vulkan_driver(driver_path);
        if (ret != 0)
        {
            NCNN_LOGE("load vulkan driver failed");
            return -1;
        }
    }
#else
    if (driver_path)
    {
        NCNN_LOGE("custom vulkan driver is not supported when NCNN_SIMPLEVK is off");
        NCNN_LOGE("will always use the system vulkan driver");
    }
#endif // NCNN_SIMPLEVK

    VkResult ret;

    std::vector<const char*> enabledLayers;

#if ENABLE_VALIDATION_LAYER
    uint32_t instanceLayerPropertyCount;
    ret = vkEnumerateInstanceLayerProperties(&instanceLayerPropertyCount, NULL);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEnumerateInstanceLayerProperties failed %d", ret);
        return -1;
    }

    std::vector<VkLayerProperties> instanceLayerProperties(instanceLayerPropertyCount);
    ret = vkEnumerateInstanceLayerProperties(&instanceLayerPropertyCount, instanceLayerProperties.data());
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEnumerateInstanceLayerProperties failed %d", ret);
        return -1;
    }

    for (uint32_t i = 0; i < instanceLayerPropertyCount; i++)
    {
        const VkLayerProperties& lp = instanceLayerProperties[i];
        //         NCNN_LOGE("instance layer %s = %u", lp.layerName, lp.implementationVersion);

        if (strcmp(lp.layerName, "VK_LAYER_LUNARG_standard_validation") == 0)
        {
            enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");
        }
        if (strcmp(lp.layerName, "VK_LAYER_LUNARG_parameter_validation") == 0)
        {
            enabledLayers.push_back("VK_LAYER_LUNARG_parameter_validation");
        }
        if (strcmp(lp.layerName, "VK_LAYER_KHRONOS_validation") == 0)
        {
            enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
        }
    }
#endif // ENABLE_VALIDATION_LAYER

    std::vector<const char*> enabledExtensions;

    uint32_t instanceExtensionPropertyCount;
    ret = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionPropertyCount, NULL);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEnumerateInstanceExtensionProperties failed %d", ret);
        return -1;
    }

    std::vector<VkExtensionProperties> instanceExtensionProperties(instanceExtensionPropertyCount);
    ret = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionPropertyCount, instanceExtensionProperties.data());
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEnumerateInstanceExtensionProperties failed %d", ret);
        return -1;
    }

    support_VK_KHR_get_physical_device_properties2 = 0;
    support_VK_KHR_get_surface_capabilities2 = 0;
    support_VK_KHR_portability_enumeration = 0;
    support_VK_KHR_surface = 0;
    support_VK_EXT_debug_utils = 0;
    support_VK_EXT_validation_features = 0;
    support_VK_EXT_validation_flags = 0;
#if __ANDROID_API__ >= 26
    support_VK_KHR_android_surface = 0;
#endif // __ANDROID_API__ >= 26
    for (uint32_t j = 0; j < instanceExtensionPropertyCount; j++)
    {
        const VkExtensionProperties& exp = instanceExtensionProperties[j];
        //         NCNN_LOGE("instance extension %s = %u", exp.extensionName, exp.specVersion);

        if (strcmp(exp.extensionName, "VK_KHR_external_memory_capabilities") == 0)
            support_VK_KHR_external_memory_capabilities = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_get_physical_device_properties2") == 0)
            support_VK_KHR_get_physical_device_properties2 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_get_surface_capabilities2") == 0)
            support_VK_KHR_get_surface_capabilities2 = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_portability_enumeration") == 0)
            support_VK_KHR_portability_enumeration = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_KHR_surface") == 0)
            support_VK_KHR_surface = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_debug_utils") == 0)
            support_VK_EXT_debug_utils = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_validation_features") == 0)
            support_VK_EXT_validation_features = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_validation_flags") == 0)
            support_VK_EXT_validation_flags = exp.specVersion;
#if __ANDROID_API__ >= 26
        else if (strcmp(exp.extensionName, "VK_KHR_android_surface") == 0)
            support_VK_KHR_android_surface = exp.specVersion;
#endif // __ANDROID_API__ >= 26
    }

    if (support_VK_EXT_validation_features)
    {
        // we prefer the modern one
        support_VK_EXT_validation_flags = 0;
    }

    if (support_VK_KHR_external_memory_capabilities)
        enabledExtensions.push_back("VK_KHR_external_memory_capabilities");
    if (support_VK_KHR_get_physical_device_properties2)
        enabledExtensions.push_back("VK_KHR_get_physical_device_properties2");
    if (support_VK_KHR_get_surface_capabilities2)
        enabledExtensions.push_back("VK_KHR_get_surface_capabilities2");
    if (support_VK_KHR_portability_enumeration)
        enabledExtensions.push_back("VK_KHR_portability_enumeration");
    if (support_VK_KHR_surface)
        enabledExtensions.push_back("VK_KHR_surface");
#if ENABLE_VALIDATION_LAYER
    if (support_VK_EXT_debug_utils)
        enabledExtensions.push_back("VK_EXT_debug_utils");
    if (support_VK_EXT_validation_features)
        enabledExtensions.push_back("VK_EXT_validation_features");
    if (support_VK_EXT_validation_flags)
        enabledExtensions.push_back("VK_EXT_validation_flags");
#endif // ENABLE_VALIDATION_LAYER
#if __ANDROID_API__ >= 26
    if (support_VK_KHR_android_surface)
        enabledExtensions.push_back("VK_KHR_android_surface");
#endif // __ANDROID_API__ >= 26

    uint32_t instance_api_version = VK_MAKE_VERSION(1, 0, 0);
    typedef VkResult(VKAPI_PTR * PFN_vkEnumerateInstanceVersion)(uint32_t * pApiVersion);
    PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(0, "vkEnumerateInstanceVersion");
    if (vkEnumerateInstanceVersion)
    {
        ret = vkEnumerateInstanceVersion(&instance_api_version);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkEnumerateInstanceVersion failed %d", ret);
            return -1;
        }
    }

    // NCNN_LOGE("instance apiVersion = %u.%u.%u", VK_VERSION_MAJOR(instance_api_version), VK_VERSION_MINOR(instance_api_version), VK_VERSION_PATCH(instance_api_version));

    VkApplicationInfo applicationInfo;
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pNext = 0;
    applicationInfo.pApplicationName = "ncnn";
    applicationInfo.applicationVersion = 0;
    applicationInfo.pEngineName = "ncnn";
    applicationInfo.engineVersion = NCNN_VERSION;
    applicationInfo.apiVersion = instance_api_version;

    void* enabledExtensionFeatures = 0;

#if ENABLE_VALIDATION_LAYER
    std::vector<VkValidationFeatureEnableEXT> enabledValidationFeature;
    enabledValidationFeature.push_back(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT);
    enabledValidationFeature.push_back(VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT);
    enabledValidationFeature.push_back(VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT);
    enabledValidationFeature.push_back(VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT);
    enabledValidationFeature.push_back(VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT);

    VkValidationFeaturesEXT validationFeatures;
    validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
    validationFeatures.pNext = 0;
    validationFeatures.enabledValidationFeatureCount = enabledValidationFeature.size();
    validationFeatures.pEnabledValidationFeatures = enabledValidationFeature.data();
    validationFeatures.disabledValidationFeatureCount = 0;
    validationFeatures.pDisabledValidationFeatures = 0;
    if (support_VK_EXT_validation_features)
    {
        validationFeatures.pNext = enabledExtensionFeatures;
        enabledExtensionFeatures = &validationFeatures;
    }

    VkValidationFlagsEXT validationFlags;
    validationFlags.sType = VK_STRUCTURE_TYPE_VALIDATION_FLAGS_EXT;
    validationFlags.pNext = 0;
    validationFlags.disabledValidationCheckCount = 0;
    validationFlags.pDisabledValidationChecks = 0;
    if (support_VK_EXT_validation_flags)
    {
        validationFlags.pNext = enabledExtensionFeatures;
        enabledExtensionFeatures = &validationFlags;
    }
#endif // ENABLE_VALIDATION_LAYER

    VkInstanceCreateInfo instanceCreateInfo;
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pNext = enabledExtensionFeatures;
    instanceCreateInfo.flags = 0;
    if (support_VK_KHR_portability_enumeration)
        instanceCreateInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    instanceCreateInfo.pApplicationInfo = &applicationInfo;
    instanceCreateInfo.enabledLayerCount = enabledLayers.size();
    instanceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
    instanceCreateInfo.enabledExtensionCount = enabledExtensions.size();
    instanceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();

    VkInstance instance = 0;
    ret = vkCreateInstance(&instanceCreateInfo, 0, &instance);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateInstance failed %d", ret);
        return -1;
    }

    g_instance.instance = instance;
    g_instance.instance_api_version = instance_api_version;

    init_instance_core();

#if ENABLE_VALIDATION_LAYER
    if (support_VK_EXT_debug_utils)
    {
        VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = 0;
        ret = CreateDebugUtilsMessengerEXT(g_instance, &createInfo, NULL, &g_instance.callback);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("CreateDebugUtilsMessengerEXT failed %d", ret);
            return -1;
        }
    }
#endif // ENABLE_VALIDATION_LAYER

    init_instance_extension();

    uint32_t physicalDeviceCount = 0;
    ret = vkEnumeratePhysicalDevices(g_instance, &physicalDeviceCount, 0);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEnumeratePhysicalDevices failed %d", ret);
        return -1;
    }

    if (physicalDeviceCount > NCNN_MAX_GPU_COUNT)
        physicalDeviceCount = NCNN_MAX_GPU_COUNT;

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);

    ret = vkEnumeratePhysicalDevices(g_instance, &physicalDeviceCount, physicalDevices.data());
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkEnumeratePhysicalDevices failed %d", ret);
        return -1;
    }

    // find proper device and queue
    int gpu_info_index = 0;
    for (uint32_t i = 0; i < physicalDeviceCount; i++)
    {
        const VkPhysicalDevice& physicalDevice = physicalDevices[i];
        delete g_gpu_infos[gpu_info_index];
        g_gpu_infos[gpu_info_index] = new GpuInfo;

        GpuInfo& gpu_info = *g_gpu_infos[gpu_info_index];

        gpu_info.d->device_index = gpu_info_index;

        gpu_info.d->physicalDevice = physicalDevice;

        gpu_info.d->query_features();
        gpu_info.d->query_properties();

        // device type

        // info
        // NCNN_LOGE("[%u] max_shared_memory_size = %u", i, gpu_info.max_shared_memory_size);
        // NCNN_LOGE("[%u] max_workgroup_count = %u %u %u", i, gpu_info.max_workgroup_count[0], gpu_info.max_workgroup_count[1], gpu_info.max_workgroup_count[2]);
        // NCNN_LOGE("[%u] max_workgroup_invocations = %u", i, gpu_info.max_workgroup_invocations);
        // NCNN_LOGE("[%u] max_workgroup_size = %u %u %u", i, gpu_info.max_workgroup_size[0], gpu_info.max_workgroup_size[1], gpu_info.max_workgroup_size[2]);
        // NCNN_LOGE("[%u] memory_map_alignment = %lu", i, gpu_info.memory_map_alignment);
        // NCNN_LOGE("[%u] buffer_offset_alignment = %lu", i, gpu_info.buffer_offset_alignment);

        gpu_info.d->query_queue_properties();

        // cache memory properties
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &gpu_info.d->physicalDeviceMemoryProperties);

        int rqde = gpu_info.d->query_extensions();
        if (rqde != 0)
        {
            return -1;
        }

        gpu_info.d->query_extension_features();
        gpu_info.d->query_extension_properties();

        NCNN_LOGE("[%u %s]  queueC=%u[%u]  queueT=%u[%u]", i, gpu_info.device_name(),
                  gpu_info.compute_queue_family_index(), gpu_info.compute_queue_count(),
                  gpu_info.transfer_queue_family_index(), gpu_info.transfer_queue_count());

        NCNN_LOGE("[%u %s]  fp16-p/s/u/a=%d/%d/%d/%d  int8-p/s/u/a=%d/%d/%d/%d", i, gpu_info.device_name(),
                  gpu_info.support_fp16_packed(), gpu_info.support_fp16_storage(), gpu_info.support_fp16_uniform(), gpu_info.support_fp16_arithmetic(),
                  gpu_info.support_int8_packed(), gpu_info.support_int8_storage(), gpu_info.support_int8_uniform(), gpu_info.support_int8_arithmetic());

        NCNN_LOGE("[%u %s]  subgroup=%u(%u~%u)  ops=%d/%d/%d/%d/%d/%d/%d/%d/%d/%d", i, gpu_info.device_name(),
                  gpu_info.subgroup_size(), gpu_info.min_subgroup_size(), gpu_info.max_subgroup_size(),
                  (gpu_info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0,
                  (gpu_info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_VOTE_BIT) != 0,
                  (gpu_info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0,
                  (gpu_info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0,
                  (gpu_info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0,
                  (gpu_info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0,
                  (gpu_info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) != 0,
                  (gpu_info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_QUAD_BIT) != 0,
                  (gpu_info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_ROTATE_BIT_KHR) != 0,
                  (gpu_info.support_subgroup_ops() & VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT_KHR) != 0);

        // collect matrix mnk
        std::vector<VkCooperativeMatrixPropertiesKHR> fp16_matrix_properties;
        std::vector<VkCooperativeMatrixPropertiesKHR> int8_matrix_properties;
        std::vector<VkCooperativeMatrixPropertiesKHR> bf16_matrix_properties;
        std::vector<VkCooperativeMatrixPropertiesKHR> fp8_matrix_properties;
        if (gpu_info.support_VK_KHR_cooperative_matrix())
        {
            const std::vector<VkCooperativeMatrixPropertiesKHR>& properties = gpu_info.queryCooperativeMatrixSubProperties();
            for (uint32_t j = 0; j < properties.size(); j++)
            {
                const VkCooperativeMatrixPropertiesKHR& cmp = properties[j];

                if (cmp.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_KHR)
                {
                    bool mnk_hit = false;
                    for (size_t k = 0; k < fp16_matrix_properties.size(); k++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp0 = fp16_matrix_properties[k];
                        if (cmp.MSize == cmp0.MSize && cmp.NSize == cmp0.NSize && cmp.KSize == cmp0.KSize)
                        {
                            mnk_hit = true;
                            break;
                        }
                    }
                    if (!mnk_hit)
                        fp16_matrix_properties.push_back(cmp);
                }
                if ((cmp.AType == VK_COMPONENT_TYPE_SINT8_KHR || cmp.AType == VK_COMPONENT_TYPE_SINT8_PACKED_NV)
                        && (cmp.BType == VK_COMPONENT_TYPE_SINT8_KHR || cmp.BType == VK_COMPONENT_TYPE_SINT8_PACKED_NV))
                {
                    bool mnk_hit = false;
                    for (size_t k = 0; k < int8_matrix_properties.size(); k++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp0 = int8_matrix_properties[k];
                        if (cmp.MSize == cmp0.MSize && cmp.NSize == cmp0.NSize && cmp.KSize == cmp0.KSize)
                        {
                            mnk_hit = true;
                            break;
                        }
                    }
                    if (!mnk_hit)
                        int8_matrix_properties.push_back(cmp);
                }
                if (cmp.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR && cmp.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR)
                {
                    bool mnk_hit = false;
                    for (size_t k = 0; k < bf16_matrix_properties.size(); k++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp0 = bf16_matrix_properties[k];
                        if (cmp.MSize == cmp0.MSize && cmp.NSize == cmp0.NSize && cmp.KSize == cmp0.KSize)
                        {
                            mnk_hit = true;
                            break;
                        }
                    }
                    if (!mnk_hit)
                        bf16_matrix_properties.push_back(cmp);
                }
                if ((cmp.AType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT || cmp.AType == VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT
                        || cmp.AType == VK_COMPONENT_TYPE_FLOAT_E4M3_NV || cmp.AType == VK_COMPONENT_TYPE_FLOAT_E5M2_NV)
                        && (cmp.BType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT || cmp.BType == VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT
                            || cmp.BType == VK_COMPONENT_TYPE_FLOAT_E4M3_NV || cmp.BType == VK_COMPONENT_TYPE_FLOAT_E5M2_NV))
                {
                    bool mnk_hit = false;
                    for (size_t k = 0; k < fp8_matrix_properties.size(); k++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp0 = fp8_matrix_properties[k];
                        if (cmp.MSize == cmp0.MSize && cmp.NSize == cmp0.NSize && cmp.KSize == cmp0.KSize)
                        {
                            mnk_hit = true;
                            break;
                        }
                    }
                    if (!mnk_hit)
                        fp8_matrix_properties.push_back(cmp);
                }
            }
        }
        else if (gpu_info.support_VK_NV_cooperative_matrix())
        {
            const std::vector<VkCooperativeMatrixPropertiesNV>& properties = gpu_info.queryCooperativeMatrixSubPropertiesNV();
            for (uint32_t j = 0; j < properties.size(); j++)
            {
                const VkCooperativeMatrixPropertiesNV& cmp = properties[j];

                if (cmp.AType == VK_COMPONENT_TYPE_FLOAT16_NV && cmp.BType == VK_COMPONENT_TYPE_FLOAT16_NV)
                {
                    bool mnk_hit = false;
                    for (size_t k = 0; k < fp16_matrix_properties.size(); k++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp0 = fp16_matrix_properties[k];
                        if (cmp.MSize == cmp0.MSize && cmp.NSize == cmp0.NSize && cmp.KSize == cmp0.KSize)
                        {
                            mnk_hit = true;
                            break;
                        }
                    }
                    if (!mnk_hit)
                    {
                        VkCooperativeMatrixPropertiesKHR cmp_khr;
                        cmp_khr.MSize = cmp.MSize;
                        cmp_khr.NSize = cmp.NSize;
                        cmp_khr.KSize = cmp.KSize;
                        fp16_matrix_properties.push_back(cmp_khr);
                    }
                }
                if (cmp.AType == VK_COMPONENT_TYPE_SINT8_NV && cmp.BType == VK_COMPONENT_TYPE_SINT8_NV)
                {
                    bool mnk_hit = false;
                    for (size_t k = 0; k < int8_matrix_properties.size(); k++)
                    {
                        const VkCooperativeMatrixPropertiesKHR& cmp0 = int8_matrix_properties[k];
                        if (cmp.MSize == cmp0.MSize && cmp.NSize == cmp0.NSize && cmp.KSize == cmp0.KSize)
                        {
                            mnk_hit = true;
                            break;
                        }
                    }
                    if (!mnk_hit)
                    {
                        VkCooperativeMatrixPropertiesKHR cmp_khr;
                        cmp_khr.MSize = cmp.MSize;
                        cmp_khr.NSize = cmp.NSize;
                        cmp_khr.KSize = cmp.KSize;
                        int8_matrix_properties.push_back(cmp_khr);
                    }
                }
            }
        }

        std::string fp16_matrix_info_str;
        std::string int8_matrix_info_str;
        std::string bf16_matrix_info_str;
        std::string fp8_matrix_info_str;
        {
            for (uint32_t j = 0; j < fp16_matrix_properties.size(); j++)
            {
                const VkCooperativeMatrixPropertiesKHR& cmp = fp16_matrix_properties[j];
                char tmp[64];
                sprintf(tmp, j > 0 ? "/%ux%ux%u" : "%ux%ux%u", cmp.MSize, cmp.NSize, cmp.KSize);
                fp16_matrix_info_str += tmp;
            }
            for (uint32_t j = 0; j < int8_matrix_properties.size(); j++)
            {
                const VkCooperativeMatrixPropertiesKHR& cmp = int8_matrix_properties[j];
                char tmp[64];
                sprintf(tmp, j > 0 ? "/%ux%ux%u" : "%ux%ux%u", cmp.MSize, cmp.NSize, cmp.KSize);
                int8_matrix_info_str += tmp;
            }
            for (uint32_t j = 0; j < bf16_matrix_properties.size(); j++)
            {
                const VkCooperativeMatrixPropertiesKHR& cmp = bf16_matrix_properties[j];
                char tmp[64];
                sprintf(tmp, j > 0 ? "/%ux%ux%u" : "%ux%ux%u", cmp.MSize, cmp.NSize, cmp.KSize);
                bf16_matrix_info_str += tmp;
            }
            for (uint32_t j = 0; j < fp8_matrix_properties.size(); j++)
            {
                const VkCooperativeMatrixPropertiesKHR& cmp = fp8_matrix_properties[j];
                char tmp[64];
                sprintf(tmp, j > 0 ? "/%ux%ux%u" : "%ux%ux%u", cmp.MSize, cmp.NSize, cmp.KSize);
                fp8_matrix_info_str += tmp;
            }

            if (fp16_matrix_info_str.empty())
                fp16_matrix_info_str = "0";
            if (int8_matrix_info_str.empty())
                int8_matrix_info_str = "0";
            if (bf16_matrix_info_str.empty())
                bf16_matrix_info_str = "0";
            if (fp8_matrix_info_str.empty())
                fp8_matrix_info_str = "0";
        }

        NCNN_LOGE("[%u %s]  fp16-cm=%s  int8-cm=%s  bf16-cm=%s  fp8-cm=%s", i, gpu_info.device_name(),
                  fp16_matrix_info_str.c_str(), int8_matrix_info_str.c_str(), bf16_matrix_info_str.c_str(), fp8_matrix_info_str.c_str());

        gpu_info_index++;
    }

    g_gpu_count = gpu_info_index;

    // the default gpu device
    g_default_gpu_index = find_default_vulkan_device_index();

    g_instance.glslang_initialized = glslang::InitializeProcess();

    // the global __ncnn_vulkan_instance_holder destructor will call destroy_gpu_instance() on exit
    // but it seems to be too late for nvidia driver :(
    // driver's internal data structure has been destroyed when called, causing segfault
    // atexit() seems to be helpful for calling it earlier    --- nihui
    static int destroy_gpu_instance_atexit_registered = 0;
    if (!destroy_gpu_instance_atexit_registered)
    {
        atexit(destroy_gpu_instance);
        destroy_gpu_instance_atexit_registered = 1;
    }

    return 0;
}

VkInstance get_gpu_instance()
{
    return (VkInstance)g_instance;
}

void destroy_gpu_instance()
{
    MutexLockGuard lock(g_instance_lock);

    if (g_instance.created == 0)
        return;

    for (int i = 0; i < NCNN_MAX_GPU_COUNT; i++)
    {
        VulkanDevice* vulkan_device = g_default_vkdev[i];
        if (vulkan_device)
        {
            VkDevice vkdev = g_default_vkdev[i]->vkdevice();
            if (vkdev)
            {
                vkDeviceWaitIdle(vkdev);
            }
        }
    }

    // NCNN_LOGE("destroy_gpu_instance");

    if (g_instance.glslang_initialized)
    {
        glslang::FinalizeProcess();
        g_instance.glslang_initialized = false;
    }

    for (int i = 0; i < NCNN_MAX_GPU_COUNT; i++)
    {
        delete g_default_vkdev[i];
        g_default_vkdev[i] = 0;

        delete g_gpu_infos[i];
        g_gpu_infos[i] = 0;
    }

#if ENABLE_VALIDATION_LAYER
    if (support_VK_EXT_debug_utils && g_instance.callback)
    {
        DestroyDebugUtilsMessengerEXT(g_instance, g_instance.callback, NULL);
        g_instance.callback = 0;
    }
#endif // ENABLE_VALIDATION_LAYER

    if (vkDestroyInstance)
    {
        vkDestroyInstance(g_instance, 0);
        vkDestroyInstance = 0;
    }

    g_instance.instance = 0;

#if NCNN_SIMPLEVK
    unload_vulkan_driver();
#endif

    g_instance.created = 0;
}

static void try_create_gpu_instance()
{
    {
        MutexLockGuard lock(g_instance_lock);

        if (g_instance.created != 0)
            return;
    }

    create_gpu_instance();
}

int get_gpu_count()
{
    try_create_gpu_instance();

    return g_gpu_count;
}

int get_default_gpu_index()
{
    try_create_gpu_instance();

    return g_default_gpu_index;
}

const GpuInfo& get_gpu_info(int device_index)
{
    try_create_gpu_instance();

    return *g_gpu_infos[device_index];
}

class VkDummyAllocator : public VkBlobAllocator
{
public:
    // NOTE 16k is large enough I think ...
    VkDummyAllocator(const VulkanDevice* _vkdev)
        : VkBlobAllocator(_vkdev, 16 * 1024)
    {
    }
};

class VkDummyCompute : public VkCompute
{
public:
    VkDummyCompute(const VulkanDevice* _vkdev)
        : VkCompute(_vkdev)
    {
    }

    void record_dummy(const VkMat& buffer)
    {
        barrier_readwrite(buffer);
    }

    void record_dummy(const VkImageMat& image)
    {
        barrier_readwrite(image);
    }

    void record_dummy_readonly(const VkImageMat& image)
    {
        barrier_readonly(image);
    }
};

class VulkanDevicePrivate
{
public:
    VulkanDevicePrivate(VulkanDevice* _vkdev);
    VulkanDevice* const vkdev;

    // dummy buffer and image
    int create_dummy_buffer_image();
    void destroy_dummy_buffer_image();

    // utility operator
    const ncnn::Layer* get_utility_operator(int cast_type_from_index, int cast_type_to_index, int packing_type_to_index) const;
    void destroy_utility_operator();

    VkDevice device;

    // hardware queue
    mutable std::vector<VkQueue> compute_queues;
    mutable std::vector<VkQueue> transfer_queues;
    mutable int free_compute_queue_count;
    mutable int free_transfer_queue_count;
    mutable Mutex compute_queue_lock;
    mutable Mutex transfer_queue_lock;
    mutable ConditionVariable compute_queue_condition;
    mutable ConditionVariable transfer_queue_condition;

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
    VkImageMat dummy_image_readonly;

    // device-wide pipeline cache
    PipelineCache* pipeline_cache;

    // utility operator
    // from fp32 | fp16
    // to fp32 | fp16
    // to pack1 | pack4
    mutable ncnn::Layer* uop_packing[2][2][2];
    // from int8
    // to int8
    // to pack1 | pack4
    mutable ncnn::Layer* uop_packing_int8[2];
    mutable Mutex uop_lock;

    // device is valid and sucessfully initialized
    bool valid;
};

VulkanDevicePrivate::VulkanDevicePrivate(VulkanDevice* _vkdev)
    : vkdev(_vkdev)
{
    device = 0;
    texelfetch_sampler = 0;
    dummy_allocator = 0;
    pipeline_cache = 0;
    valid = false;
    memset(uop_packing, 0, sizeof(uop_packing));
    memset(uop_packing_int8, 0, sizeof(uop_packing_int8));
}

int VulkanDevicePrivate::create_dummy_buffer_image()
{
    dummy_allocator = new VkDummyAllocator(vkdev);

    dummy_buffer.create(1, 4u, dummy_allocator);
    dummy_image.create(1, 4u, dummy_allocator);
#if __APPLE__
    if (vkdev->info.type() == 0)
        dummy_image_readonly.create(1, 4u, dummy_allocator);
#else
    dummy_image_readonly.create(1, 4u, dummy_allocator);
#endif

    VkDummyCompute cmd(vkdev);

    cmd.record_dummy(dummy_buffer);
    cmd.record_dummy(dummy_image);
#if __APPLE__
    if (vkdev->info.type() == 0)
        cmd.record_dummy_readonly(dummy_image_readonly);
#else
    cmd.record_dummy_readonly(dummy_image_readonly);
#endif

    return cmd.submit_and_wait();
}

void VulkanDevicePrivate::destroy_dummy_buffer_image()
{
    dummy_buffer.release();
    dummy_image.release();
#if __APPLE__
    if (vkdev->info.type() == 0)
        dummy_image_readonly.release();
#else
    dummy_image_readonly.release();
#endif

    if (dummy_allocator)
    {
        delete dummy_allocator;
        dummy_allocator = 0;
    }
}

const ncnn::Layer* VulkanDevicePrivate::get_utility_operator(int cast_type_from_index, int cast_type_to_index, int packing_type_to_index) const
{
    bool use_fp16 = (cast_type_from_index == 1 || cast_type_to_index == 1);
    bool use_int8 = (cast_type_from_index == 3 || cast_type_to_index == 3);

    MutexLockGuard lock(uop_lock);

    const ncnn::Layer* cached_uop = 0;
    if (use_int8)
    {
        cached_uop = uop_packing_int8[packing_type_to_index];
    }
    else
    {
        cached_uop = uop_packing[cast_type_from_index][cast_type_to_index][packing_type_to_index];
    }
    if (cached_uop)
        return cached_uop;

    // create uop
    Option opt;
    opt.use_fp16_packed = use_fp16; // fp16p is always supported
    opt.use_fp16_storage = use_fp16 && vkdev->info.support_fp16_storage();
    opt.use_int8_packed = use_int8; // int8p is always supported
    opt.use_int8_storage = use_int8 && vkdev->info.support_int8_storage();

    // fp16/int8 arithmetic are not necessary for packing
    // and may conflict with storage options
    opt.use_fp16_arithmetic = false;
    opt.use_int8_arithmetic = false;

    // do not enable spirv-1.3 from cooperative matrix
    opt.use_cooperative_matrix = false;

    opt.use_vulkan_compute = true;

    // cache uop pipeline as device member explicitly
    opt.pipeline_cache = 0;

    opt.vulkan_device_index = vkdev->info.device_index();

    ncnn::Layer* uop = ncnn::create_layer_vulkan(LayerType::Packing);
    uop->vkdev = vkdev;

    ncnn::ParamDict pd;
    pd.set(0, packing_type_to_index == 0 ? 1 : 4); // out_elempack
    pd.set(2, cast_type_from_index + 1);           // 0=auto 1=fp32 2=fp16 3=int8
    pd.set(3, cast_type_to_index + 1);

    uop->load_param(pd);

    uop->create_pipeline(opt);

    if (use_int8)
    {
        uop_packing_int8[packing_type_to_index] = uop;
    }
    else
    {
        uop_packing[cast_type_from_index][cast_type_to_index][packing_type_to_index] = uop;
    }

    return uop;
}

void VulkanDevicePrivate::destroy_utility_operator()
{
    Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_arithmetic = false;
    opt.use_cooperative_matrix = false;
    opt.pipeline_cache = 0;
    opt.vulkan_device_index = vkdev->info.device_index();

    // from fp32 | fp16
    for (int j0 = 0; j0 < 2; j0++)
    {
        // to fp32 | fp16
        for (int j1 = 0; j1 < 2; j1++)
        {
            bool use_fp16 = (j0 == 1 || j1 == 1);

            opt.use_fp16_packed = use_fp16;
            opt.use_fp16_storage = use_fp16 && vkdev->info.support_fp16_storage();
            opt.use_int8_packed = false;
            opt.use_int8_storage = false;

            // to pack1 | pack4
            for (int k = 0; k < 2; k++)
            {
                ncnn::Layer* uop = uop_packing[j0][j1][k];
                if (!uop)
                    continue;

                uop->destroy_pipeline(opt);

                delete uop;

                uop_packing[j0][j1][k] = 0;
            }
        }
    }

    // int8
    {
        bool use_int8 = true;

        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_int8_packed = use_int8;
        opt.use_int8_storage = use_int8 && vkdev->info.support_int8_storage();

        // to pack1 | pack4
        for (int k = 0; k < 2; k++)
        {
            ncnn::Layer* uop = uop_packing_int8[k];
            if (!uop)
                continue;

            uop->destroy_pipeline(opt);

            delete uop;

            uop_packing_int8[k] = 0;
        }
    }
}

VulkanDevice::VulkanDevice(int device_index)
    : info(get_gpu_info(device_index)), d(new VulkanDevicePrivate(this))
{
    try_create_gpu_instance();

    std::vector<const char*> enabledExtensions;
    if (info.support_VK_KHR_8bit_storage())
        enabledExtensions.push_back("VK_KHR_8bit_storage");
    if (info.support_VK_KHR_16bit_storage())
        enabledExtensions.push_back("VK_KHR_16bit_storage");
    if (info.support_VK_KHR_bind_memory2())
        enabledExtensions.push_back("VK_KHR_bind_memory2");
    if (info.support_VK_KHR_buffer_device_address())
        enabledExtensions.push_back("VK_KHR_buffer_device_address");
    if (info.support_VK_KHR_create_renderpass2())
        enabledExtensions.push_back("VK_KHR_create_renderpass2");
    if (info.support_VK_KHR_cooperative_matrix())
        enabledExtensions.push_back("VK_KHR_cooperative_matrix");
    if (info.support_VK_KHR_dedicated_allocation())
        enabledExtensions.push_back("VK_KHR_dedicated_allocation");
    if (info.support_VK_KHR_descriptor_update_template())
        enabledExtensions.push_back("VK_KHR_descriptor_update_template");
    if (info.support_VK_KHR_driver_properties())
        enabledExtensions.push_back("VK_KHR_driver_properties");
    if (info.support_VK_KHR_external_memory())
        enabledExtensions.push_back("VK_KHR_external_memory");
    if (info.support_VK_KHR_get_memory_requirements2())
        enabledExtensions.push_back("VK_KHR_get_memory_requirements2");
    if (info.support_VK_KHR_maintenance1())
        enabledExtensions.push_back("VK_KHR_maintenance1");
    if (info.support_VK_KHR_maintenance2())
        enabledExtensions.push_back("VK_KHR_maintenance2");
    if (info.support_VK_KHR_maintenance3())
        enabledExtensions.push_back("VK_KHR_maintenance3");
    if (info.support_VK_KHR_multiview())
        enabledExtensions.push_back("VK_KHR_multiview");
    if (info.support_VK_KHR_portability_subset())
        enabledExtensions.push_back("VK_KHR_portability_subset");
    if (info.support_VK_KHR_push_descriptor())
        enabledExtensions.push_back("VK_KHR_push_descriptor");
    if (info.support_VK_KHR_robustness2())
        enabledExtensions.push_back("VK_KHR_robustness2");
    if (info.support_VK_KHR_sampler_ycbcr_conversion())
        enabledExtensions.push_back("VK_KHR_sampler_ycbcr_conversion");
    if (info.support_VK_KHR_shader_bfloat16())
        enabledExtensions.push_back("VK_KHR_shader_bfloat16");
    if (info.support_VK_KHR_shader_float16_int8())
        enabledExtensions.push_back("VK_KHR_shader_float16_int8");
    if (info.support_VK_KHR_shader_float_controls())
        enabledExtensions.push_back("VK_KHR_shader_float_controls");
    if (info.support_VK_KHR_shader_float_controls2())
        enabledExtensions.push_back("VK_KHR_shader_float_controls2");
    if (info.support_VK_KHR_shader_integer_dot_product())
        enabledExtensions.push_back("VK_KHR_shader_integer_dot_product");
    if (info.support_VK_KHR_shader_non_semantic_info())
        enabledExtensions.push_back("VK_KHR_shader_non_semantic_info");
    if (info.support_VK_KHR_shader_subgroup_extended_types())
        enabledExtensions.push_back("VK_KHR_shader_subgroup_extended_types");
    if (info.support_VK_KHR_shader_subgroup_rotate())
        enabledExtensions.push_back("VK_KHR_shader_subgroup_rotate");
    if (info.support_VK_KHR_storage_buffer_storage_class())
        enabledExtensions.push_back("VK_KHR_storage_buffer_storage_class");
    if (info.support_VK_KHR_swapchain())
        enabledExtensions.push_back("VK_KHR_swapchain");
    if (info.support_VK_KHR_vulkan_memory_model())
        enabledExtensions.push_back("VK_KHR_vulkan_memory_model");
    if (info.support_VK_KHR_zero_initialize_workgroup_memory())
        enabledExtensions.push_back("VK_KHR_zero_initialize_workgroup_memory");
    if (info.support_VK_EXT_buffer_device_address())
        enabledExtensions.push_back("VK_EXT_buffer_device_address");
    if (info.support_VK_EXT_descriptor_indexing())
        enabledExtensions.push_back("VK_EXT_descriptor_indexing");
    if (info.support_VK_EXT_memory_budget())
        enabledExtensions.push_back("VK_EXT_memory_budget");
    if (info.support_VK_EXT_memory_priority())
        enabledExtensions.push_back("VK_EXT_memory_priority");
    if (info.support_VK_EXT_queue_family_foreign())
        enabledExtensions.push_back("VK_EXT_queue_family_foreign");
    if (info.support_VK_EXT_robustness2())
        enabledExtensions.push_back("VK_EXT_robustness2");
    if (info.support_VK_EXT_shader_atomic_float())
        enabledExtensions.push_back("VK_EXT_shader_atomic_float");
    if (info.support_VK_EXT_shader_atomic_float2())
        enabledExtensions.push_back("VK_EXT_shader_atomic_float2");
    if (info.support_VK_EXT_shader_float8())
        enabledExtensions.push_back("VK_EXT_shader_float8");
    if (info.support_VK_EXT_subgroup_size_control())
        enabledExtensions.push_back("VK_EXT_subgroup_size_control");
    if (info.support_VK_AMD_device_coherent_memory())
        enabledExtensions.push_back("VK_AMD_device_coherent_memory");
#if __ANDROID_API__ >= 26
    if (info.support_VK_ANDROID_external_memory_android_hardware_buffer())
        enabledExtensions.push_back("VK_ANDROID_external_memory_android_hardware_buffer");
#endif // __ANDROID_API__ >= 26
    if (info.support_VK_NV_cooperative_matrix())
        enabledExtensions.push_back("VK_NV_cooperative_matrix");
    if (info.support_VK_NV_cooperative_matrix2())
        enabledExtensions.push_back("VK_NV_cooperative_matrix2");
    if (info.support_VK_NV_cooperative_vector())
        enabledExtensions.push_back("VK_NV_cooperative_vector");

    const void* enabledExtensionFeatures = info.queryExtensionFeatures();

    std::vector<float> compute_queue_priorities(info.compute_queue_count(), 1.f);   // 0.f ~ 1.f
    std::vector<float> transfer_queue_priorities(info.transfer_queue_count(), 1.f); // 0.f ~ 1.f

    VkDeviceQueueCreateInfo deviceQueueCreateInfos[3];

    VkDeviceQueueCreateInfo deviceComputeQueueCreateInfo;
    deviceComputeQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    deviceComputeQueueCreateInfo.pNext = 0;
    deviceComputeQueueCreateInfo.flags = 0;
    deviceComputeQueueCreateInfo.queueFamilyIndex = info.compute_queue_family_index();
    deviceComputeQueueCreateInfo.queueCount = info.compute_queue_count();
    deviceComputeQueueCreateInfo.pQueuePriorities = compute_queue_priorities.data();

    VkDeviceQueueCreateInfo deviceTransferQueueCreateInfo;
    deviceTransferQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    deviceTransferQueueCreateInfo.pNext = 0;
    deviceTransferQueueCreateInfo.flags = 0;
    deviceTransferQueueCreateInfo.queueFamilyIndex = info.transfer_queue_family_index();
    deviceTransferQueueCreateInfo.queueCount = info.transfer_queue_count();
    deviceTransferQueueCreateInfo.pQueuePriorities = transfer_queue_priorities.data();

    VkDeviceCreateInfo deviceCreateInfo;
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pNext = enabledExtensionFeatures;
    deviceCreateInfo.flags = 0;
    if (info.compute_queue_family_index() == info.transfer_queue_family_index())
    {
        deviceQueueCreateInfos[0] = deviceComputeQueueCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 1;
    }
    else // if (info.compute_queue_family_index() != info.transfer_queue_family_index())
    {
        deviceQueueCreateInfos[0] = deviceComputeQueueCreateInfo;
        deviceQueueCreateInfos[1] = deviceTransferQueueCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 2;
    }

    deviceCreateInfo.pQueueCreateInfos = deviceQueueCreateInfos;
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.ppEnabledLayerNames = 0;
    deviceCreateInfo.enabledExtensionCount = enabledExtensions.size();
    deviceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();
    deviceCreateInfo.pEnabledFeatures = 0; // VkPhysicalDeviceFeatures pointer

    VkResult ret = vkCreateDevice(info.physicalDevice(), &deviceCreateInfo, 0, &d->device);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateDevice failed %d", ret);
        return;
    }

    init_device_extension();

    d->free_compute_queue_count = 0;
    d->free_transfer_queue_count = 0;

    d->free_compute_queue_count = info.compute_queue_count();
    d->compute_queues.resize(info.compute_queue_count());
    d->blob_allocators.resize(info.compute_queue_count());
    d->staging_allocators.resize(info.compute_queue_count());
    for (uint32_t i = 0; i < info.compute_queue_count(); i++)
    {
        vkGetDeviceQueue(d->device, info.compute_queue_family_index(), i, &d->compute_queues[i]);
        d->blob_allocators[i] = new VkBlobAllocator(this);
        d->staging_allocators[i] = new VkStagingAllocator(this);
    }
    if (info.compute_queue_family_index() != info.transfer_queue_family_index())
    {
        d->free_transfer_queue_count = info.transfer_queue_count();
        d->transfer_queues.resize(info.transfer_queue_count());
        for (uint32_t i = 0; i < info.transfer_queue_count(); i++)
        {
            vkGetDeviceQueue(d->device, info.transfer_queue_family_index(), i, &d->transfer_queues[i]);
        }
    }

    // prepare immutable texelfetch sampler
    {
        VkSamplerCreateInfo samplerCreateInfo;
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.pNext = 0;
        samplerCreateInfo.flags = 0;
        samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
        samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerCreateInfo.mipLodBias = 0.0f;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.maxAnisotropy = 1;
        samplerCreateInfo.compareEnable = VK_FALSE;
        samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = 0.0f;
        samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        samplerCreateInfo.unnormalizedCoordinates = VK_TRUE;

        ret = vkCreateSampler(d->device, &samplerCreateInfo, 0, &d->texelfetch_sampler);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateSampler failed %d", ret);
        }
    }

    int cret = d->create_dummy_buffer_image();
    if (cret != 0)
    {
        NCNN_LOGE("VulkanDevice create_dummy_buffer_image failed %d", cret);
        return;
    }

    d->pipeline_cache = new PipelineCache(this);

    d->valid = true;
}

VulkanDevice::~VulkanDevice()
{
    d->destroy_utility_operator();

    d->destroy_dummy_buffer_image();

    if (d->texelfetch_sampler)
    {
        vkDestroySampler(d->device, d->texelfetch_sampler, 0);
    }

    for (size_t i = 0; i < d->blob_allocators.size(); i++)
    {
        delete d->blob_allocators[i];
    }
    d->blob_allocators.clear();
    for (size_t i = 0; i < d->staging_allocators.size(); i++)
    {
        delete d->staging_allocators[i];
    }
    d->staging_allocators.clear();

    if (d->pipeline_cache)
    {
        delete d->pipeline_cache;
    }

    if (d->device)
    {
        vkDestroyDevice(d->device, 0);
    }

    delete d;
}

VulkanDevice::VulkanDevice(const VulkanDevice&)
    : info(get_gpu_info(0)), d(0)
{
}

VulkanDevice& VulkanDevice::operator=(const VulkanDevice&)
{
    return *this;
}

VkDevice VulkanDevice::vkdevice() const
{
    return d->device;
}

bool VulkanDevice::is_valid() const
{
    return d->valid;
}

VkShaderModule VulkanDevice::compile_shader_module(const uint32_t* spv_data, size_t spv_data_size) const
{
    VkShaderModuleCreateInfo shaderModuleCreateInfo;
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pNext = 0;
    shaderModuleCreateInfo.flags = 0;
    shaderModuleCreateInfo.codeSize = spv_data_size;
    shaderModuleCreateInfo.pCode = spv_data;

    VkShaderModule shader_module;
    VkResult ret = vkCreateShaderModule(d->device, &shaderModuleCreateInfo, 0, &shader_module);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateShaderModule failed %d", ret);
        return 0;
    }

    return shader_module;
}

static void inject_local_size_xyz(const uint32_t* code, size_t size, uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z, uint32_t* dstcode, size_t* dstsize)
{
    uint32_t local_size_x_id = -1;
    uint32_t local_size_y_id = -1;
    uint32_t local_size_z_id = -1;
    uint32_t gl_WorkGroupSize_id = -1;

    const uint32_t* p = code;
    uint32_t* dp = dstcode;

    // skip magic version generator bound schema
    memcpy(dp, p, 5 * sizeof(uint32_t));
    p += 5;
    dp += 5;

    // foreach op
    while ((const unsigned char*)p < (const unsigned char*)code + size)
    {
        uint32_t opcode = p[0];

        uint16_t wordcount = opcode >> 16;
        uint16_t op = opcode & 0xffff;

        if (op == 16) // OpExecutionMode
        {
            uint32_t mode = p[2];
            if (mode == 17) // LocalSize
            {
                memcpy(dp, p, wordcount * sizeof(uint32_t));

                // set local_size_xyz
                dp[3] = local_size_x;
                dp[4] = local_size_y;
                dp[5] = local_size_z;

                p += wordcount;
                dp += wordcount;
                continue;
            }
        }
        else if (op == 50) // OpSpecConstant
        {
            uint32_t id = p[2];
            if (id == local_size_x_id || id == local_size_y_id || id == local_size_z_id)
            {
                p += wordcount;
                continue;
            }
        }
        else if (op == 51) // OpSpecConstantComposite
        {
            uint32_t id = p[2];
            if (id == gl_WorkGroupSize_id)
            {
                if (wordcount == 6 && (p[3] == local_size_x_id || p[4] == local_size_y_id || p[5] == local_size_z_id))
                {
                    p += wordcount;
                    continue;
                }
            }
        }
        else if (op == 71) // OpDecorate
        {
            uint32_t id = p[1];
            uint32_t decoration = p[2];
            if (decoration == 1) // SpecId
            {
                uint32_t specid = p[3];
                if (specid == 233) local_size_x_id = id;
                if (specid == 234) local_size_y_id = id;
                if (specid == 235) local_size_z_id = id;
                if (specid == 233 || specid == 234 || specid == 235)
                {
                    p += wordcount;
                    continue;
                }
            }
            else if (decoration == 11) // BuiltIn
            {
                uint32_t builtin = p[3];
                if (builtin == 25) // WorkgroupSize
                {
                    gl_WorkGroupSize_id = id;
                    p += wordcount;
                    continue;
                }
            }
        }

        memcpy(dp, p, wordcount * sizeof(uint32_t));
        p += wordcount;
        dp += wordcount;
    }

    *dstsize = (unsigned char*)dp - (unsigned char*)dstcode;
}

VkShaderModule VulkanDevice::compile_shader_module(const uint32_t* spv_data, size_t spv_data_size, uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z) const
{
    uint32_t* spv_data_modified = (uint32_t*)malloc(spv_data_size);
    size_t spv_data_size_modified = spv_data_size;
    inject_local_size_xyz(spv_data, spv_data_size, local_size_x, local_size_y, local_size_z, spv_data_modified, &spv_data_size_modified);

    VkShaderModule shader_module = compile_shader_module(spv_data_modified, spv_data_size_modified);

    free(spv_data_modified);

    return shader_module;
}

int VulkanDevice::create_descriptorset_layout(int binding_count, const int* binding_types, VkDescriptorSetLayout* descriptorset_layout) const
{
    if (binding_count == 0)
    {
        *descriptorset_layout = 0;
        return 0;
    }

    std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings(binding_count);
    for (int i = 0; i < binding_count; i++)
    {
        int binding_type = binding_types[i];

        descriptorSetLayoutBindings[i].binding = i;
        descriptorSetLayoutBindings[i].descriptorCount = 1;
        descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        if (binding_type == 1)
        {
            descriptorSetLayoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorSetLayoutBindings[i].pImmutableSamplers = 0;
        }
        else if (binding_type == 2)
        {
            descriptorSetLayoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBindings[i].pImmutableSamplers = 0;
        }
        else // if (binding_type == 3)
        {
            descriptorSetLayoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorSetLayoutBindings[i].pImmutableSamplers = immutable_texelfetch_sampler(); // we always use texelfetch
        }
    }

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.pNext = 0;
    descriptorSetLayoutCreateInfo.flags = 0;
    descriptorSetLayoutCreateInfo.bindingCount = binding_count;
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();

    if (info.support_VK_KHR_push_descriptor())
    {
        descriptorSetLayoutCreateInfo.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    }

    VkResult ret = vkCreateDescriptorSetLayout(d->device, &descriptorSetLayoutCreateInfo, 0, descriptorset_layout);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateDescriptorSetLayout failed %d", ret);
        return -1;
    }

    return 0;
}

int VulkanDevice::create_pipeline_layout(int push_constant_count, VkDescriptorSetLayout descriptorset_layout, VkPipelineLayout* pipeline_layout) const
{
    VkPushConstantRange pushConstantRange;
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(vk_constant_type) * push_constant_count;

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

    VkResult ret = vkCreatePipelineLayout(d->device, &pipelineLayoutCreateInfo, 0, pipeline_layout);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreatePipelineLayout failed %d", ret);
        return -1;
    }

    return 0;
}

int VulkanDevice::create_pipeline(VkShaderModule shader_module, VkPipelineLayout pipeline_layout, const std::vector<vk_specialization_type>& specializations, uint32_t subgroup_size, VkPipeline* pipeline) const
{
    const int specialization_count = specializations.size();

    std::vector<VkSpecializationMapEntry> specializationMapEntries(specialization_count);
    for (int i = 0; i < specialization_count; i++)
    {
        specializationMapEntries[i].constantID = i;
        specializationMapEntries[i].offset = i * sizeof(vk_specialization_type);
        specializationMapEntries[i].size = sizeof(vk_specialization_type);
    }

    VkSpecializationInfo specializationInfo;
    specializationInfo.mapEntryCount = specializationMapEntries.size();
    specializationInfo.pMapEntries = specializationMapEntries.data();
    specializationInfo.dataSize = specializations.size() * sizeof(vk_specialization_type);
    specializationInfo.pData = specializations.data();

    VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo;
    pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfo.pNext = 0;
    pipelineShaderStageCreateInfo.flags = 0;
    pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineShaderStageCreateInfo.module = shader_module;
    pipelineShaderStageCreateInfo.pName = "main";
    pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;

    // but full subgroup bits enforce local_size_x be multiple of subgroup size
    // if (info.support_compute_full_subgroups())
    // {
    //     pipelineShaderStageCreateInfo.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
    // }

    void* enabledExtensionFeatures = 0;

    // subgroup size control
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT pipelineShaderStageRequiredSubgroupSizeCreateInfo;
    pipelineShaderStageRequiredSubgroupSizeCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
    pipelineShaderStageRequiredSubgroupSizeCreateInfo.pNext = 0;
    pipelineShaderStageRequiredSubgroupSizeCreateInfo.requiredSubgroupSize = subgroup_size;
    if (info.support_subgroup_size_control())
    {
        // pipelineShaderStageCreateInfo.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT;
        pipelineShaderStageRequiredSubgroupSizeCreateInfo.pNext = enabledExtensionFeatures;
        enabledExtensionFeatures = &pipelineShaderStageRequiredSubgroupSizeCreateInfo;
    }

    pipelineShaderStageCreateInfo.pNext = enabledExtensionFeatures;

    VkComputePipelineCreateInfo computePipelineCreateInfo;
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = 0;
    computePipelineCreateInfo.flags = 0;
    computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
    computePipelineCreateInfo.layout = pipeline_layout;
    computePipelineCreateInfo.basePipelineHandle = 0;
    computePipelineCreateInfo.basePipelineIndex = 0;

    VkResult ret = vkCreateComputePipelines(d->device, 0, 1, &computePipelineCreateInfo, 0, pipeline);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateComputePipelines failed %d", ret);
        return -1;
    }

    return 0;
}

int VulkanDevice::create_descriptor_update_template(int binding_count, const int* binding_types, VkDescriptorSetLayout descriptorset_layout, VkPipelineLayout pipeline_layout, VkDescriptorUpdateTemplateKHR* descriptor_update_template) const
{
    if (binding_count == 0)
    {
        *descriptor_update_template = 0;
        return 0;
    }

    std::vector<VkDescriptorUpdateTemplateEntryKHR> descriptorUpdateTemplateEntries(binding_count);
    size_t offset = 0;
    for (int i = 0; i < binding_count; i++) // TODO do not update weights
    {
        int binding_type = binding_types[i];

        descriptorUpdateTemplateEntries[i].dstBinding = i;
        descriptorUpdateTemplateEntries[i].dstArrayElement = 0;
        descriptorUpdateTemplateEntries[i].descriptorCount = 1;
        descriptorUpdateTemplateEntries[i].offset = offset;

        if (binding_type == 1)
        {
            descriptorUpdateTemplateEntries[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorUpdateTemplateEntries[i].stride = sizeof(VkDescriptorBufferInfo);
        }
        else if (binding_type == 2)
        {
            descriptorUpdateTemplateEntries[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorUpdateTemplateEntries[i].stride = sizeof(VkDescriptorImageInfo);
        }
        else // if (binding_type == 3)
        {
            descriptorUpdateTemplateEntries[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorUpdateTemplateEntries[i].stride = sizeof(VkDescriptorImageInfo);
        }

        offset += descriptorUpdateTemplateEntries[i].stride;
    }

    VkDescriptorUpdateTemplateCreateInfoKHR descriptorUpdateTemplateCreateInfo;
    descriptorUpdateTemplateCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR;
    descriptorUpdateTemplateCreateInfo.pNext = 0;
    descriptorUpdateTemplateCreateInfo.flags = 0;
    descriptorUpdateTemplateCreateInfo.descriptorUpdateEntryCount = binding_count; // TODO do not update weights
    descriptorUpdateTemplateCreateInfo.pDescriptorUpdateEntries = descriptorUpdateTemplateEntries.data();
    if (info.support_VK_KHR_push_descriptor())
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

    VkResult ret = vkCreateDescriptorUpdateTemplateKHR(d->device, &descriptorUpdateTemplateCreateInfo, 0, descriptor_update_template);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateDescriptorUpdateTemplateKHR failed %d", ret);
        return -1;
    }

    return 0;
}

uint32_t VulkanDevice::find_memory_index(uint32_t memory_type_bits, VkFlags required, VkFlags preferred, VkFlags preferred_not) const
{
    const VkPhysicalDeviceMemoryProperties& memory_properties = info.physicalDeviceMemoryProperties();

    // first try, find required and with preferred and without preferred_not
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
    {
        bool is_required = (1 << i) & memory_type_bits;
        if (is_required)
        {
            const VkMemoryType& memoryType = memory_properties.memoryTypes[i];
            if ((memoryType.propertyFlags & required) == required
                    && (preferred && (memoryType.propertyFlags & preferred))
                    && (preferred_not && !(memoryType.propertyFlags & preferred_not)))
            {
                return i;
            }
        }
    }

    // second try, find required and with preferred
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
    {
        bool is_required = (1 << i) & memory_type_bits;
        if (is_required)
        {
            const VkMemoryType& memoryType = memory_properties.memoryTypes[i];
            if ((memoryType.propertyFlags & required) == required
                    && (preferred && (memoryType.propertyFlags & preferred)))
            {
                return i;
            }
        }
    }

    // third try, find required and without preferred_not
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
    {
        bool is_required = (1 << i) & memory_type_bits;
        if (is_required)
        {
            const VkMemoryType& memoryType = memory_properties.memoryTypes[i];
            if ((memoryType.propertyFlags & required) == required
                    && (preferred_not && !(memoryType.propertyFlags & preferred_not)))
            {
                return i;
            }
        }
    }

    // fourth try, find any required
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
    {
        bool is_required = (1 << i) & memory_type_bits;
        if (is_required)
        {
            const VkMemoryType& memoryType = memory_properties.memoryTypes[i];
            if ((memoryType.propertyFlags & required) == required)
            {
                return i;
            }
        }
    }

    NCNN_LOGE("no such memory type %u %u %u %u", memory_type_bits, required, preferred, preferred_not);
    return -1;
}

bool VulkanDevice::is_mappable(uint32_t memory_type_index) const
{
    const VkMemoryType& memoryType = info.physicalDeviceMemoryProperties().memoryTypes[memory_type_index];

    return memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
}

bool VulkanDevice::is_coherent(uint32_t memory_type_index) const
{
    const VkMemoryType& memoryType = info.physicalDeviceMemoryProperties().memoryTypes[memory_type_index];

    return memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
}

VkQueue VulkanDevice::acquire_queue(uint32_t queue_family_index) const
{
    if (queue_family_index != info.compute_queue_family_index() && queue_family_index != info.transfer_queue_family_index())
    {
        NCNN_LOGE("invalid queue_family_index %u", queue_family_index);
        return 0;
    }

    Mutex& queue_lock = queue_family_index == info.compute_queue_family_index() ? d->compute_queue_lock : d->transfer_queue_lock;

    queue_lock.lock();

    ConditionVariable& queue_condition = queue_family_index == info.compute_queue_family_index() ? d->compute_queue_condition : d->transfer_queue_condition;

    int& free_queue_count = queue_family_index == info.compute_queue_family_index() ? d->free_compute_queue_count : d->free_transfer_queue_count;

    while (free_queue_count == 0)
    {
        // no free queues, wait for recleams from other threads
        queue_condition.wait(queue_lock);
    }

    std::vector<VkQueue>& queues = queue_family_index == info.compute_queue_family_index() ? d->compute_queues : d->transfer_queues;

    VkQueue queue = 0;
    for (size_t i = 0; i < queues.size(); i++)
    {
        if (queues[i])
        {
            queue = queues[i];
            queues[i] = 0;
            break;
        }
    }

    if (!queue)
    {
        NCNN_LOGE("FATAL ERROR! out of hardware queue %u", queue_family_index);
    }

    free_queue_count -= 1;

    queue_lock.unlock();

    queue_condition.signal();

    return queue;
}

void VulkanDevice::reclaim_queue(uint32_t queue_family_index, VkQueue queue) const
{
    if (queue_family_index != info.compute_queue_family_index() && queue_family_index != info.transfer_queue_family_index())
    {
        NCNN_LOGE("invalid queue_family_index %u", queue_family_index);
        return;
    }

    Mutex& queue_lock = queue_family_index == info.compute_queue_family_index() ? d->compute_queue_lock : d->transfer_queue_lock;

    queue_lock.lock();

    ConditionVariable& queue_condition = queue_family_index == info.compute_queue_family_index() ? d->compute_queue_condition : d->transfer_queue_condition;

    int& free_queue_count = queue_family_index == info.compute_queue_family_index() ? d->free_compute_queue_count : d->free_transfer_queue_count;

    std::vector<VkQueue>& queues = queue_family_index == info.compute_queue_family_index() ? d->compute_queues : d->transfer_queues;

    size_t i = 0;
    for (; i < queues.size(); i++)
    {
        if (!queues[i])
        {
            queues[i] = queue;
            break;
        }
    }

    if (i == queues.size())
    {
        NCNN_LOGE("FATAL ERROR! reclaim_queue get wild queue %u %p", queue_family_index, queue);
    }

    free_queue_count += 1;

    queue_lock.unlock();

    queue_condition.signal();
}

VkAllocator* VulkanDevice::acquire_blob_allocator() const
{
    MutexLockGuard lock(d->blob_allocator_lock);

    for (int i = 0; i < (int)d->blob_allocators.size(); i++)
    {
        VkAllocator* allocator = d->blob_allocators[i];
        if (allocator)
        {
            d->blob_allocators[i] = 0;
            return allocator;
        }
    }

    // pre-allocated allcator exhausted, create new
    VkAllocator* allocator = new VkBlobAllocator(this);
    d->blob_allocators.push_back(allocator);
    d->blob_allocators[d->blob_allocators.size() - 1] = 0;
    return allocator;
}

void VulkanDevice::reclaim_blob_allocator(VkAllocator* allocator) const
{
    MutexLockGuard lock(d->blob_allocator_lock);

    for (int i = 0; i < (int)d->blob_allocators.size(); i++)
    {
        if (!d->blob_allocators[i])
        {
            d->blob_allocators[i] = allocator;
            return;
        }
    }

    NCNN_LOGE("FATAL ERROR! reclaim_blob_allocator get wild allocator %p", allocator);
}

VkAllocator* VulkanDevice::acquire_staging_allocator() const
{
    MutexLockGuard lock(d->staging_allocator_lock);

    for (int i = 0; i < (int)d->staging_allocators.size(); i++)
    {
        VkAllocator* allocator = d->staging_allocators[i];
        if (allocator)
        {
            d->staging_allocators[i] = 0;
            return allocator;
        }
    }

    // pre-allocated allcator exhausted, create new
    VkAllocator* allocator = new VkStagingAllocator(this);
    d->staging_allocators.push_back(allocator);
    d->staging_allocators[d->staging_allocators.size() - 1] = 0;
    return allocator;
}

void VulkanDevice::reclaim_staging_allocator(VkAllocator* allocator) const
{
    MutexLockGuard lock(d->staging_allocator_lock);

    for (int i = 0; i < (int)d->staging_allocators.size(); i++)
    {
        if (!d->staging_allocators[i])
        {
            d->staging_allocators[i] = allocator;
            return;
        }
    }

    NCNN_LOGE("FATAL ERROR! reclaim_staging_allocator get wild allocator %p", allocator);
}

const VkSampler* VulkanDevice::immutable_texelfetch_sampler() const
{
    return &d->texelfetch_sampler;
}

VkMat VulkanDevice::get_dummy_buffer() const
{
    return d->dummy_buffer;
}

VkImageMat VulkanDevice::get_dummy_image() const
{
    return d->dummy_image;
}

VkImageMat VulkanDevice::get_dummy_image_readonly() const
{
#if __APPLE__
    if (info.type() != 0)
        return d->dummy_image;
#endif
    return d->dummy_image_readonly;
}

const PipelineCache* VulkanDevice::get_pipeline_cache() const
{
    return d->pipeline_cache;
}

bool VulkanDevice::shape_support_image_storage(const Mat& shape) const
{
    int dims = shape.dims;
    int width = shape.w;
    int height = shape.h;
    int depth = shape.c;
    int elempack = shape.elempack;

    // large elempack spills on image w
    if (elempack == 8) width *= 2;
    if (elempack == 16) width *= 4;
    if (elempack == 32) width *= 8;
    if (elempack == 64) width *= 16;

    if (dims == 1)
    {
        if (width > (int)info.max_image_dimension_1d())
        {
            return false;
        }
    }
    else if (dims == 2)
    {
        if (width > (int)info.max_image_dimension_2d() || height > (int)info.max_image_dimension_2d())
        {
            return false;
        }
    }
    else // if (dims == 3)
    {
        if (width > (int)info.max_image_dimension_3d() || height > (int)info.max_image_dimension_3d() || depth > (int)info.max_image_dimension_3d())
        {
            return false;
        }
    }

    return true;
}

uint32_t VulkanDevice::get_heap_budget() const
{
    const VkPhysicalDeviceMemoryProperties& memory_properties = info.physicalDeviceMemoryProperties();

    uint32_t buffer_memory_type_index = d->dummy_allocator->buffer_memory_type_index;
    uint32_t buffer_heap_index = memory_properties.memoryTypes[buffer_memory_type_index].heapIndex;

    if (!info.support_VK_EXT_memory_budget())
    {
        //         NCNN_LOGE("heap budget from assumption\n");
        uint32_t device_local_heap_size = memory_properties.memoryHeaps[buffer_heap_index].size / 1024 / 1024;

        // we usually cannot use all heap
        // 70% for 4G+
        // 50% for 4G-
        return device_local_heap_size >= 4000 ? device_local_heap_size * 0.7 : device_local_heap_size * 0.5;
    }

    VkPhysicalDeviceMemoryBudgetPropertiesEXT memoryBudgetProperties;
    memoryBudgetProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    memoryBudgetProperties.pNext = 0;

    VkPhysicalDeviceMemoryProperties2KHR memoryProperties;
    memoryProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR;
    memoryProperties.pNext = &memoryBudgetProperties;

    vkGetPhysicalDeviceMemoryProperties2KHR(info.physicalDevice(), &memoryProperties);

    return memoryBudgetProperties.heapBudget[buffer_heap_index] / 1024 / 1024;
}

void VulkanDevice::convert_packing(const VkMat& src, VkMat& dst, int dst_elempack, VkCompute& cmd, const Option& opt) const
{
    convert_packing(src, dst, dst_elempack, 0, cmd, opt);
}

void VulkanDevice::convert_packing(const VkMat& src, VkMat& dst, int dst_elempack, int cast_type_to, VkCompute& cmd, const Option& opt) const
{
    int packing_type_to_index = dst_elempack == 1 ? 0 : dst_elempack == 4 ? 1 : 2;

    int cast_type_from_index;
    if (src.elembits() == 32)
    {
        cast_type_from_index = 0;
    }
    else if (src.elembits() == 16)
    {
        cast_type_from_index = 1;
    }
    else // if (src.elembits() == 8)
    {
        cast_type_from_index = 3;
    }

    int cast_type_to_index = cast_type_to ? cast_type_to - 1 : cast_type_from_index;

    // NCNN_LOGE("convert_packing b2b %d %d %d", cast_type_from_index, cast_type_to_index, packing_type_to_index);

    if ((cast_type_from_index == 0 || cast_type_from_index == 1) && (cast_type_to_index == 2 || cast_type_to_index == 3))
    {
        NCNN_LOGE("convert_packing from fp32/fp16 to int32/int8 is not supported");
        return;
    }
    if ((cast_type_from_index == 2 || cast_type_from_index == 3) && (cast_type_to_index == 0 || cast_type_to_index == 1))
    {
        NCNN_LOGE("convert_packing from int32/int8 to fp32/fp16 is not supported");
        return;
    }

    Option opt2 = opt;
    opt2.use_fp16_packed = (cast_type_from_index == 1 || cast_type_to_index == 1);
    opt2.use_fp16_storage = (cast_type_from_index == 1 || cast_type_to_index == 1) && info.support_fp16_storage();
    opt2.use_int8_packed = (cast_type_from_index == 3 || cast_type_to_index == 3);
    opt2.use_int8_storage = (cast_type_from_index == 3 || cast_type_to_index == 3) && info.support_int8_storage();

    const ncnn::Layer* uop = d->get_utility_operator(cast_type_from_index, cast_type_to_index, packing_type_to_index);
    uop->forward(src, dst, cmd, opt2);
}

int VulkanDevice::init_device_extension()
{
    if (info.support_VK_KHR_bind_memory2())
    {
        vkBindBufferMemory2KHR = (PFN_vkBindBufferMemory2KHR)vkGetDeviceProcAddr(d->device, "vkBindBufferMemory2KHR");
        vkBindImageMemory2KHR = (PFN_vkBindImageMemory2KHR)vkGetDeviceProcAddr(d->device, "vkBindImageMemory2KHR");
    }

    if (info.support_VK_KHR_buffer_device_address())
    {
        vkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(d->device, "vkGetBufferDeviceAddressKHR");
        vkGetBufferOpaqueCaptureAddressKHR = (PFN_vkGetBufferOpaqueCaptureAddressKHR)vkGetDeviceProcAddr(d->device, "vkGetBufferOpaqueCaptureAddressKHR");
        vkGetDeviceMemoryOpaqueCaptureAddressKHR = (PFN_vkGetDeviceMemoryOpaqueCaptureAddressKHR)vkGetDeviceProcAddr(d->device, "vkGetDeviceMemoryOpaqueCaptureAddressKHR");
    }

    if (info.support_VK_KHR_descriptor_update_template())
    {
        vkCreateDescriptorUpdateTemplateKHR = (PFN_vkCreateDescriptorUpdateTemplateKHR)vkGetDeviceProcAddr(d->device, "vkCreateDescriptorUpdateTemplateKHR");
        vkDestroyDescriptorUpdateTemplateKHR = (PFN_vkDestroyDescriptorUpdateTemplateKHR)vkGetDeviceProcAddr(d->device, "vkDestroyDescriptorUpdateTemplateKHR");
        vkUpdateDescriptorSetWithTemplateKHR = (PFN_vkUpdateDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(d->device, "vkUpdateDescriptorSetWithTemplateKHR");
    }

    if (info.support_VK_KHR_get_memory_requirements2())
    {
        vkGetImageMemoryRequirements2KHR = (PFN_vkGetImageMemoryRequirements2KHR)vkGetDeviceProcAddr(d->device, "vkGetImageMemoryRequirements2KHR");
        vkGetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2KHR)vkGetDeviceProcAddr(d->device, "vkGetBufferMemoryRequirements2KHR");
    }

    if (info.support_VK_KHR_maintenance1())
    {
        vkTrimCommandPoolKHR = (PFN_vkTrimCommandPoolKHR)vkGetDeviceProcAddr(d->device, "vkTrimCommandPoolKHR");
    }

    if (info.support_VK_KHR_maintenance3())
    {
        vkGetDescriptorSetLayoutSupportKHR = (PFN_vkGetDescriptorSetLayoutSupportKHR)vkGetDeviceProcAddr(d->device, "vkGetDescriptorSetLayoutSupportKHR");
    }

    if (info.support_VK_KHR_push_descriptor())
    {
        if (info.support_VK_KHR_descriptor_update_template())
        {
            vkCmdPushDescriptorSetWithTemplateKHR = (PFN_vkCmdPushDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(d->device, "vkCmdPushDescriptorSetWithTemplateKHR");
        }

        vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(d->device, "vkCmdPushDescriptorSetKHR");
    }

    if (info.support_VK_KHR_sampler_ycbcr_conversion())
    {
        vkCreateSamplerYcbcrConversionKHR = (PFN_vkCreateSamplerYcbcrConversionKHR)vkGetDeviceProcAddr(d->device, "vkCreateSamplerYcbcrConversionKHR");
        vkDestroySamplerYcbcrConversionKHR = (PFN_vkDestroySamplerYcbcrConversionKHR)vkGetDeviceProcAddr(d->device, "vkDestroySamplerYcbcrConversionKHR");
    }

    if (info.support_VK_KHR_swapchain())
    {
        vkCreateSwapchainKHR = (PFN_vkCreateSwapchainKHR)vkGetDeviceProcAddr(d->device, "vkCreateSwapchainKHR");
        vkDestroySwapchainKHR = (PFN_vkDestroySwapchainKHR)vkGetDeviceProcAddr(d->device, "vkDestroySwapchainKHR");
        vkGetSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR)vkGetDeviceProcAddr(d->device, "vkGetSwapchainImagesKHR");
        vkAcquireNextImageKHR = (PFN_vkAcquireNextImageKHR)vkGetDeviceProcAddr(d->device, "vkAcquireNextImageKHR");
        vkQueuePresentKHR = (PFN_vkQueuePresentKHR)vkGetDeviceProcAddr(d->device, "vkQueuePresentKHR");
    }

    if (info.support_VK_EXT_buffer_device_address())
    {
        vkGetBufferDeviceAddressEXT = (PFN_vkGetBufferDeviceAddressEXT)vkGetDeviceProcAddr(d->device, "vkGetBufferDeviceAddressEXT");
    }

#if __ANDROID_API__ >= 26
    if (info.support_VK_ANDROID_external_memory_android_hardware_buffer())
    {
        vkGetAndroidHardwareBufferPropertiesANDROID = (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)vkGetDeviceProcAddr(d->device, "vkGetAndroidHardwareBufferPropertiesANDROID");
        vkGetMemoryAndroidHardwareBufferANDROID = (PFN_vkGetMemoryAndroidHardwareBufferANDROID)vkGetDeviceProcAddr(d->device, "vkGetMemoryAndroidHardwareBufferANDROID");
    }
#endif // __ANDROID_API__ >= 26

    if (info.support_VK_NV_cooperative_vector())
    {
        vkCmdConvertCooperativeVectorMatrixNV = (PFN_vkCmdConvertCooperativeVectorMatrixNV)vkGetDeviceProcAddr(d->device, "vkCmdConvertCooperativeVectorMatrixNV");
        vkConvertCooperativeVectorMatrixNV = (PFN_vkConvertCooperativeVectorMatrixNV)vkGetDeviceProcAddr(d->device, "vkConvertCooperativeVectorMatrixNV");
    }

    return 0;
}

VulkanDevice* get_gpu_device(int device_index)
{
    try_create_gpu_instance();

    if (device_index < 0 || device_index >= g_gpu_count)
        return 0;

    MutexLockGuard lock(g_default_vkdev_lock);

    if (!g_default_vkdev[device_index])
        g_default_vkdev[device_index] = new VulkanDevice(device_index);

    return g_default_vkdev[device_index];
}

static TBuiltInResource get_default_TBuiltInResource()
{
    TBuiltInResource resource;

    resource.maxLights = 32;
    resource.maxClipPlanes = 6;
    resource.maxTextureUnits = 32;
    resource.maxTextureCoords = 32;
    resource.maxVertexAttribs = 64;
    resource.maxVertexUniformComponents = 4096;
    resource.maxVaryingFloats = 64;
    resource.maxVertexTextureImageUnits = 32;
    resource.maxCombinedTextureImageUnits = 80;
    resource.maxTextureImageUnits = 32;
    resource.maxFragmentUniformComponents = 4096;
    resource.maxDrawBuffers = 32;
    resource.maxVertexUniformVectors = 128;
    resource.maxVaryingVectors = 8;
    resource.maxFragmentUniformVectors = 16;
    resource.maxVertexOutputVectors = 16;
    resource.maxFragmentInputVectors = 15;
    resource.minProgramTexelOffset = -8;
    resource.maxProgramTexelOffset = 7;
    resource.maxClipDistances = 8;
    resource.maxComputeWorkGroupCountX = 65535;
    resource.maxComputeWorkGroupCountY = 65535;
    resource.maxComputeWorkGroupCountZ = 65535;
    resource.maxComputeWorkGroupSizeX = 1024;
    resource.maxComputeWorkGroupSizeY = 1024;
    resource.maxComputeWorkGroupSizeZ = 64;
    resource.maxComputeUniformComponents = 1024;
    resource.maxComputeTextureImageUnits = 16;
    resource.maxComputeImageUniforms = 8;
    resource.maxComputeAtomicCounters = 8;
    resource.maxComputeAtomicCounterBuffers = 1;
    resource.maxVaryingComponents = 60;
    resource.maxVertexOutputComponents = 64;
    resource.maxGeometryInputComponents = 64;
    resource.maxGeometryOutputComponents = 128;
    resource.maxFragmentInputComponents = 128;
    resource.maxImageUnits = 8;
    resource.maxCombinedImageUnitsAndFragmentOutputs = 8;
    resource.maxCombinedShaderOutputResources = 8;
    resource.maxImageSamples = 0;
    resource.maxVertexImageUniforms = 0;
    resource.maxTessControlImageUniforms = 0;
    resource.maxTessEvaluationImageUniforms = 0;
    resource.maxGeometryImageUniforms = 0;
    resource.maxFragmentImageUniforms = 8;
    resource.maxCombinedImageUniforms = 8;
    resource.maxGeometryTextureImageUnits = 16;
    resource.maxGeometryOutputVertices = 256;
    resource.maxGeometryTotalOutputComponents = 1024;
    resource.maxGeometryUniformComponents = 1024;
    resource.maxGeometryVaryingComponents = 64;
    resource.maxTessControlInputComponents = 128;
    resource.maxTessControlOutputComponents = 128;
    resource.maxTessControlTextureImageUnits = 16;
    resource.maxTessControlUniformComponents = 1024;
    resource.maxTessControlTotalOutputComponents = 4096;
    resource.maxTessEvaluationInputComponents = 128;
    resource.maxTessEvaluationOutputComponents = 128;
    resource.maxTessEvaluationTextureImageUnits = 16;
    resource.maxTessEvaluationUniformComponents = 1024;
    resource.maxTessPatchComponents = 120;
    resource.maxPatchVertices = 32;
    resource.maxTessGenLevel = 64;
    resource.maxViewports = 16;
    resource.maxVertexAtomicCounters = 0;
    resource.maxTessControlAtomicCounters = 0;
    resource.maxTessEvaluationAtomicCounters = 0;
    resource.maxGeometryAtomicCounters = 0;
    resource.maxFragmentAtomicCounters = 8;
    resource.maxCombinedAtomicCounters = 8;
    resource.maxAtomicCounterBindings = 1;
    resource.maxVertexAtomicCounterBuffers = 0;
    resource.maxTessControlAtomicCounterBuffers = 0;
    resource.maxTessEvaluationAtomicCounterBuffers = 0;
    resource.maxGeometryAtomicCounterBuffers = 0;
    resource.maxFragmentAtomicCounterBuffers = 1;
    resource.maxCombinedAtomicCounterBuffers = 1;
    resource.maxAtomicCounterBufferSize = 16384;
    resource.maxTransformFeedbackBuffers = 4;
    resource.maxTransformFeedbackInterleavedComponents = 64;
    resource.maxCullDistances = 8;
    resource.maxCombinedClipAndCullDistances = 8;
    resource.maxSamples = 4;
    resource.maxMeshOutputVerticesNV = 256;
    resource.maxMeshOutputPrimitivesNV = 512;
    resource.maxMeshWorkGroupSizeX_NV = 32;
    resource.maxMeshWorkGroupSizeY_NV = 1;
    resource.maxMeshWorkGroupSizeZ_NV = 1;
    resource.maxTaskWorkGroupSizeX_NV = 32;
    resource.maxTaskWorkGroupSizeY_NV = 1;
    resource.maxTaskWorkGroupSizeZ_NV = 1;
    resource.maxMeshViewCountNV = 4;

    // TODO compile-time glslang version check
    // resource.maxDualSourceDrawBuffersEXT = 1;

    resource.limits.nonInductiveForLoops = 1;
    resource.limits.whileLoops = 1;
    resource.limits.doWhileLoops = 1;
    resource.limits.generalUniformIndexing = 1;
    resource.limits.generalAttributeMatrixVectorIndexing = 1;
    resource.limits.generalVaryingIndexing = 1;
    resource.limits.generalSamplerIndexing = 1;
    resource.limits.generalVariableIndexing = 1;
    resource.limits.generalConstantMatrixVectorIndexing = 1;

    return resource;
}

class VulkanShaderIncluder : public glslang::TShader::Includer
{
public:
    virtual glslang::TShader::Includer::IncludeResult* includeLocal(const char* headerName, const char* /*includerName*/, size_t /*inclusionDepth*/)
    {
        if (strcmp(headerName, "vulkan_activation.comp") == 0)
        {
            const char* const headerData = vulkan_activation_comp_data;
            const size_t headerLength = sizeof(vulkan_activation_comp_data);
            glslang::TShader::Includer::IncludeResult* r = new glslang::TShader::Includer::IncludeResult(headerName, headerData, headerLength, 0);
            return r;
        }

        return 0;
    }

    virtual void releaseInclude(glslang::TShader::Includer::IncludeResult* r)
    {
        delete r;
    }
};

class DefinitionCollector
{
public:
    template<typename T>
    void append(const char* key, T def)
    {
        definitions.push_back(std::make_pair(key, def));
    }

public:
    struct typed_value
    {
        typed_value(const char* _s)
            : type(0), s(_s)
        {
        }
        typed_value(uint8_t _u8)
            : type(1), u8(_u8)
        {
        }
        typed_value(uint32_t _u32)
            : type(2), u32(_u32)
        {
        }
        typed_value(int32_t _i32)
            : type(3), i32(_i32)
        {
        }
        typed_value(uint64_t _u64)
            : type(4), u64(_u64)
        {
        }
        typed_value(float _f32)
            : type(5), f32(_f32)
        {
        }

        int type;
        union
        {
            const char* s;
            uint8_t u8;
            uint32_t u32;
            int32_t i32;
            uint64_t u64;
            float f32;
        };
    };

    std::vector<std::pair<const char*, typed_value> > definitions;
};

int compile_spirv_module(const char* comp_string, const Option& opt, std::vector<uint32_t>& spirv)
{
    // -1 for omitting the tail '\0'
    int length = strlen(comp_string) - 1;
    return compile_spirv_module(comp_string, length, opt, spirv);
}

int compile_spirv_module(const char* comp_data, int comp_data_size, const Option& opt, std::vector<uint32_t>& spirv)
{
    DefinitionCollector custom_defines;
    DefinitionCollector device_defines;

    if (opt.use_fp16_storage)
    {
        custom_defines.append("sfp", "float16_t");
        custom_defines.append("sfpvec2", "f16vec2");
        custom_defines.append("sfpvec4", "f16vec4");

        if (opt.use_fp16_arithmetic)
        {
            custom_defines.append("sfpmat4", "f16mat4");
        }
    }
    else if (opt.use_fp16_packed)
    {
        custom_defines.append("sfp", "uint");
        custom_defines.append("sfpvec2", "uint");
        custom_defines.append("sfpvec4", "uvec2");
    }
    else
    {
        custom_defines.append("sfp", "float");
        custom_defines.append("sfpvec2", "vec2");
        custom_defines.append("sfpvec4", "vec4");
        custom_defines.append("sfpmat4", "mat4");
    }

    if (opt.use_fp16_arithmetic)
    {
        custom_defines.append("afp", "float16_t");
        custom_defines.append("afpvec2", "f16vec2");
        custom_defines.append("afpvec4", "f16vec4");
        custom_defines.append("afpmat4", "f16mat4");
    }
    else
    {
        custom_defines.append("afp", "float");
        custom_defines.append("afpvec2", "vec2");
        custom_defines.append("afpvec4", "vec4");
        custom_defines.append("afpmat4", "mat4");
    }

    if (opt.use_fp16_storage && opt.use_fp16_uniform && opt.use_fp16_arithmetic)
    {
        custom_defines.append("lfp", "float16_t");
        custom_defines.append("lfpvec4", "f16vec4");
    }
    else if (opt.use_fp16_storage && opt.use_fp16_arithmetic)
    {
        custom_defines.append("lfp", "float");
        custom_defines.append("lfpvec4", "uint64_t");
    }
    else if (opt.use_fp16_storage || opt.use_fp16_packed)
    {
        custom_defines.append("lfp", "float");
        custom_defines.append("lfpvec4", "uvec2");
    }
    else
    {
        custom_defines.append("lfp", "float");
        custom_defines.append("lfpvec4", "vec4");
    }

    if (opt.use_fp16_storage && opt.use_fp16_uniform && opt.use_fp16_arithmetic)
    {
        custom_defines.append("sfp2lfp(v)", "v");
        custom_defines.append("sfp2lfpvec4(v)", "v");

        custom_defines.append("lfp2afp(v)", "v");
        custom_defines.append("lfp2afpvec4(v)", "v");
    }
    else if (opt.use_fp16_storage && opt.use_fp16_arithmetic)
    {
        custom_defines.append("sfp2lfp(v)", "float(v)");
        custom_defines.append("sfp2lfpvec4(v)", "pack64(halfBitsToUInt16(v))");

        custom_defines.append("lfp2afp(v)", "float16_t(v)");
        custom_defines.append("lfp2afpvec4(v)", "int16BitsToHalf(unpack16(v))");
    }
    else if (opt.use_fp16_packed && opt.use_fp16_arithmetic)
    {
        custom_defines.append("sfp2lfp(v)", "v");
        custom_defines.append("sfp2lfpvec4(v)", "v");

        custom_defines.append("lfp2afp(v)", "float16_t(v)");
        custom_defines.append("lfp2afpvec4(v)", "f16vec4(unpackFloat2x16(v.x),unpackFloat2x16(v.y))");
    }
    else if (opt.use_fp16_storage)
    {
        custom_defines.append("sfp2lfp(v)", "float(v)");
        custom_defines.append("sfp2lfpvec4(v)", "uvec2(packHalf2x16(vec4(v).rg),packHalf2x16(vec4(v).ba))");

        custom_defines.append("lfp2afp(v)", "v");
        custom_defines.append("lfp2afpvec4(v)", "vec4(unpackHalf2x16(v.x),unpackHalf2x16(v.y))");
    }
    else if (opt.use_fp16_packed)
    {
        custom_defines.append("sfp2lfp(v)", "v");
        custom_defines.append("sfp2lfpvec4(v)", "v");

        custom_defines.append("lfp2afp(v)", "v");
        custom_defines.append("lfp2afpvec4(v)", "vec4(unpackHalf2x16(v.x),unpackHalf2x16(v.y))");
    }
    else
    {
        custom_defines.append("sfp2lfp(v)", "v");
        custom_defines.append("sfp2lfpvec4(v)", "v");

        custom_defines.append("lfp2afp(v)", "v");
        custom_defines.append("lfp2afpvec4(v)", "v");
    }

    if (opt.use_fp16_storage && opt.use_fp16_arithmetic)
    {
        custom_defines.append("buffer_ld1(buf,i)", "buf[i]");
        custom_defines.append("buffer_st1(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i]=f16vec4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a]);}");
        custom_defines.append("buffer_ld2(buf,i)", "buf[i]");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "buf[i]");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{buf[i4.r]=sbuf[si].r;buf[i4.g]=sbuf[si].g;buf[i4.b]=sbuf[si].b;buf[i4.a]=sbuf[si].a;}");
        custom_defines.append("sfp2afpmat4(v)", "v");
        custom_defines.append("afp2sfpmat4(v)", "v");
    }
    else if (opt.use_fp16_packed && opt.use_fp16_arithmetic)
    {
        // custom_defines.append("buffer_ld1(buf,i)", "float16_t(buf[i])");
        custom_defines.append("buffer_ld1(buf,i)", "float16_t(unpackHalf2x16(buf[(i)/2])[(i)%2])");
        // custom_defines.append("buffer_st1(buf,i,v)", "{buf[i]=float(v);}");
        custom_defines.append("buffer_st1(buf,i,v)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;float _vs=float(v);uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackHalf2x16(_old_v);_v[_im2]=_vs;_new_v=packHalf2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");
        // custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;uint _si=uint(si);uint _sid2=_si/2;uint _sim2=_si%2;float v=unpackHalf2x16(sbuf[_sid2])[_sim2];uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackHalf2x16(_old_v);_v[_im2]=v;_new_v=packHalf2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");

        // custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i]=uvec2(packFloat2x16(f16vec2(sbuf[si4.r],sbuf[si4.g])),packFloat2x16(f16vec2(sbuf[si4.b],sbuf[si4.a])));}");

        custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{uvec4 _si4d2=uvec4(si4)/2;uvec4 _si4m2=uvec4(si4)%2; buf[i]=uvec2(packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.r])[_si4m2.r],unpackHalf2x16(sbuf[_si4d2.g])[_si4m2.g])),packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.b])[_si4m2.b],unpackHalf2x16(sbuf[_si4d2.a])[_si4m2.a])));}");

        custom_defines.append("buffer_ld2(buf,i)", "unpackFloat2x16(buf[i])");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=packFloat2x16(v)}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "f16vec4(unpackFloat2x16(buf[i].x),unpackFloat2x16(buf[i].y))");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=uvec2(packFloat2x16(v.rg),packFloat2x16(v.ba));}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");

        // custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{uvec2 _v=sbuf[si]; f16vec2 _v0=unpackFloat2x16(_v.x);f16vec2 _v1=unpackFloat2x16(_v.y); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g;}");

        custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{uvec2 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y);buffer_st1(buf,i4.r,_v0.r);buffer_st1(buf,i4.g,_v0.g);buffer_st1(buf,i4.b,_v1.r);buffer_st1(buf,i4.a,_v1.g);}");
    }
    else if (opt.use_fp16_storage)
    {
        custom_defines.append("buffer_ld1(buf,i)", "float(buf[i])");
        custom_defines.append("buffer_st1(buf,i,v)", "{buf[i]=float16_t(v);}");
        custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i].r=sbuf[si4.r];buf[i].g=sbuf[si4.g];buf[i].b=sbuf[si4.b];buf[i].a=sbuf[si4.a];}");
        custom_defines.append("buffer_ld2(buf,i)", "vec2(buf[i])");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=f16vec2(v);}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "vec4(buf[i])");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=f16vec4(v);}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{buf[i4.r]=sbuf[si].r;buf[i4.g]=sbuf[si].g;buf[i4.b]=sbuf[si].b;buf[i4.a]=sbuf[si].a;}");
    }
    else if (opt.use_fp16_packed)
    {
        // custom_defines.append("buffer_ld1(buf,i)", "buf[i]");
        custom_defines.append("buffer_ld1(buf,i)", "unpackHalf2x16(buf[(i)/2])[(i)%2]");
        // custom_defines.append("buffer_st1(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_st1(buf,i,v)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;float _vs=float(v);uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackHalf2x16(_old_v);_v[_im2]=_vs;_new_v=packHalf2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");
        // custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;uint _si=uint(si);uint _sid2=_si/2;uint _sim2=_si%2;float v=unpackHalf2x16(sbuf[_sid2])[_sim2];uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackHalf2x16(_old_v);_v[_im2]=v;_new_v=packHalf2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");

        // custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i]=uvec2(packHalf2x16(vec2(sbuf[si4.r],sbuf[si4.g])),packHalf2x16(vec2(sbuf[si4.b],sbuf[si4.a])));}");

        custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{uvec4 _si4d2=uvec4(si4)/2;uvec4 _si4m2=uvec4(si4)%2; buf[i]=uvec2(packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.r])[_si4m2.r],unpackHalf2x16(sbuf[_si4d2.g])[_si4m2.g])),packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.b])[_si4m2.b],unpackHalf2x16(sbuf[_si4d2.a])[_si4m2.a])));}");

        custom_defines.append("buffer_ld2(buf,i)", "unpackHalf2x16(buf[i])");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=packHalf2x16(v)}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "vec4(unpackHalf2x16(buf[i].x),unpackHalf2x16(buf[i].y))");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=uvec2(packHalf2x16(v.rg),packHalf2x16(v.ba));}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");

        // custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{uvec2 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g;}");

        custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{uvec2 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y);buffer_st1(buf,i4.r,_v0.r);buffer_st1(buf,i4.g,_v0.g);buffer_st1(buf,i4.b,_v1.r);buffer_st1(buf,i4.a,_v1.g);}");
    }
    else
    {
        custom_defines.append("buffer_ld1(buf,i)", "buf[i]");
        custom_defines.append("buffer_st1(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i]=vec4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a]);}");
        custom_defines.append("buffer_ld2(buf,i)", "buf[i]");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "buf[i]");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=v;}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{vec4 _v=sbuf[si]; buf[i4.r]=_v.r;buf[i4.g]=_v.g;buf[i4.b]=_v.b;buf[i4.a]=_v.a;}");
        custom_defines.append("sfp2afpmat4(v)", "v");
        custom_defines.append("afp2sfpmat4(v)", "v");
    }

    if (opt.use_int8_storage)
    {
        custom_defines.append("sint8", "int8_t");
    }
    else if (opt.use_int8_packed)
    {
        custom_defines.append("sint8", "int");
    }
    else
    {
        custom_defines.append("sint8", "int");
    }

    custom_defines.append("sint8vec4", "int");

    custom_defines.append("aint8", "int");
    custom_defines.append("aint8vec4", "ivec4");

    custom_defines.append("unpackInt4x8(v)", "ivec4((v<<24)>>24,(v<<16)>>24,(v<<8)>>24,v>>24)");
    custom_defines.append("packInt4x8(v)", "int((uint(v.r)&0xFFu)|((uint(v.g)&0xFFu)<<8)|((uint(v.b)&0xFFu)<<16)|((uint(v.a)&0xFFu)<<24))");

    if (opt.use_int8_storage)
    {
        custom_defines.append("i8buffer_ld1(buf,i)", "int(buf[i])");
        custom_defines.append("i8buffer_st1(buf,i,v)", "{buf[i]=int8_t(v);}");
        custom_defines.append("i8buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
    }
    else
    {
        custom_defines.append("i8buffer_ld1(buf,i)", "int(((buf[(i)/4])<<(24-((i)%4)*8))>>24)");
        custom_defines.append("i8buffer_st1(buf,i,v)", "{uint _i=uint(i);uint _id4=_i/4;uint _im4=_i%4;int _vs=int(v);int _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id4],0,0);ivec4 _v=unpackInt4x8(_old_v);_v[_im4]=_vs;_new_v=packInt4x8(_v);} while(atomicCompSwap(buf[_id4],_old_v,_new_v)!=_old_v);}");
        custom_defines.append("i8buffer_cp1(buf,i,sbuf,si)", "{int _v=i8buffer_ld1(sbuf,si);i8buffer_st1(buf,i,_v);}");
    }

    custom_defines.append("i8buffer_ld4(buf,i)", "unpackInt4x8(buf[i])");
    custom_defines.append("i8buffer_st4(buf,i,v)", "{buf[i]=packInt4x8(v);}");
    custom_defines.append("i8buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");

    custom_defines.append("psc(x)", "(x==0?p.x:x)");

    if (opt.use_fp16_storage)
    {
        custom_defines.append("NCNN_fp16_storage", 1);
    }
    else if (opt.use_fp16_packed)
    {
        custom_defines.append("NCNN_fp16_packed", 1);
    }

    if (opt.use_fp16_uniform)
    {
        custom_defines.append("NCNN_fp16_uniform", 1);
    }

    if (opt.use_fp16_arithmetic)
    {
        custom_defines.append("NCNN_fp16_arithmetic", 1);
    }

    if (opt.use_int8_storage)
    {
        custom_defines.append("NCNN_int8_storage", 1);
    }
    else if (opt.use_int8_packed)
    {
        custom_defines.append("NCNN_int8_packed", 1);
    }

    if (opt.use_int8_uniform)
    {
        custom_defines.append("NCNN_int8_uniform", 1);
    }

    if (opt.use_int8_arithmetic)
    {
        custom_defines.append("NCNN_int8_arithmetic", 1);
    }

    if (opt.use_shader_local_memory)
    {
        custom_defines.append("NCNN_shader_local_memory", 1);
    }

#if __APPLE__
    custom_defines.append("NCNN_moltenvk", 1);
#endif

    custom_defines.append("ncnn_glsl_version", 1);

    bool support_shader_int64 = false;

    // fill device macros
    {
        int device_index = opt.vulkan_device_index;
        if (device_index < 0 || device_index >= get_gpu_count())
            device_index = get_default_gpu_index();

        const GpuInfo& info = get_gpu_info(device_index);

        support_shader_int64 = info.physicalDevicefeatures().shaderInt64;

        // pull in device extensions
        {
            const std::vector<VkExtensionProperties>& properties = info.deviceExtensionProperties();

            for (size_t i = 0; i < properties.size(); i++)
            {
                const VkExtensionProperties& exp = properties[i];
                device_defines.append(exp.extensionName, exp.specVersion);
            }
        }

#define DD_APPEND_FEATURE(X) device_defines.append(#X, features.X ? 1 : 0);

        // pull in device features macros
        {
            const VkPhysicalDeviceFeatures& features = info.physicalDevicefeatures();
            DD_APPEND_FEATURE(robustBufferAccess)
            DD_APPEND_FEATURE(fullDrawIndexUint32)
            DD_APPEND_FEATURE(imageCubeArray)
            DD_APPEND_FEATURE(independentBlend)
            DD_APPEND_FEATURE(geometryShader)
            DD_APPEND_FEATURE(tessellationShader)
            DD_APPEND_FEATURE(sampleRateShading)
            DD_APPEND_FEATURE(dualSrcBlend)
            DD_APPEND_FEATURE(logicOp)
            DD_APPEND_FEATURE(multiDrawIndirect)
            DD_APPEND_FEATURE(drawIndirectFirstInstance)
            DD_APPEND_FEATURE(depthClamp)
            DD_APPEND_FEATURE(depthBiasClamp)
            DD_APPEND_FEATURE(fillModeNonSolid)
            DD_APPEND_FEATURE(depthBounds)
            DD_APPEND_FEATURE(wideLines)
            DD_APPEND_FEATURE(largePoints)
            DD_APPEND_FEATURE(alphaToOne)
            DD_APPEND_FEATURE(multiViewport)
            DD_APPEND_FEATURE(samplerAnisotropy)
            DD_APPEND_FEATURE(textureCompressionETC2)
            DD_APPEND_FEATURE(textureCompressionASTC_LDR)
            DD_APPEND_FEATURE(textureCompressionBC)
            DD_APPEND_FEATURE(occlusionQueryPrecise)
            DD_APPEND_FEATURE(pipelineStatisticsQuery)
            DD_APPEND_FEATURE(vertexPipelineStoresAndAtomics)
            DD_APPEND_FEATURE(fragmentStoresAndAtomics)
            DD_APPEND_FEATURE(shaderTessellationAndGeometryPointSize)
            DD_APPEND_FEATURE(shaderImageGatherExtended)
            DD_APPEND_FEATURE(shaderStorageImageExtendedFormats)
            DD_APPEND_FEATURE(shaderStorageImageMultisample)
            DD_APPEND_FEATURE(shaderStorageImageReadWithoutFormat)
            DD_APPEND_FEATURE(shaderStorageImageWriteWithoutFormat)
            DD_APPEND_FEATURE(shaderUniformBufferArrayDynamicIndexing)
            DD_APPEND_FEATURE(shaderSampledImageArrayDynamicIndexing)
            DD_APPEND_FEATURE(shaderStorageBufferArrayDynamicIndexing)
            DD_APPEND_FEATURE(shaderStorageImageArrayDynamicIndexing)
            DD_APPEND_FEATURE(shaderClipDistance)
            DD_APPEND_FEATURE(shaderCullDistance)
            DD_APPEND_FEATURE(shaderFloat64)
            DD_APPEND_FEATURE(shaderInt64)
            DD_APPEND_FEATURE(shaderInt16)
            DD_APPEND_FEATURE(shaderResourceResidency)
            DD_APPEND_FEATURE(shaderResourceMinLod)
            DD_APPEND_FEATURE(sparseBinding)
            DD_APPEND_FEATURE(sparseResidencyBuffer)
            DD_APPEND_FEATURE(sparseResidencyImage2D)
            DD_APPEND_FEATURE(sparseResidencyImage3D)
            DD_APPEND_FEATURE(sparseResidency2Samples)
            DD_APPEND_FEATURE(sparseResidency4Samples)
            DD_APPEND_FEATURE(sparseResidency8Samples)
            DD_APPEND_FEATURE(sparseResidency16Samples)
            DD_APPEND_FEATURE(sparseResidencyAliased)
            DD_APPEND_FEATURE(variableMultisampleRate)
            DD_APPEND_FEATURE(inheritedQueries)
        }
        if (info.support_VK_KHR_8bit_storage())
        {
            const VkPhysicalDevice8BitStorageFeaturesKHR& features = info.query8BitStorageFeatures();
            DD_APPEND_FEATURE(storageBuffer8BitAccess)
            DD_APPEND_FEATURE(uniformAndStorageBuffer8BitAccess)
            DD_APPEND_FEATURE(storagePushConstant8)
        }
        if (info.support_VK_KHR_16bit_storage())
        {
            const VkPhysicalDevice16BitStorageFeaturesKHR& features = info.query16BitStorageFeatures();
            DD_APPEND_FEATURE(storageBuffer16BitAccess)
            DD_APPEND_FEATURE(uniformAndStorageBuffer16BitAccess)
            DD_APPEND_FEATURE(storagePushConstant16)
            DD_APPEND_FEATURE(storageInputOutput16)
        }
        if (info.support_VK_KHR_robustness2() || info.support_VK_EXT_robustness2())
        {
            const VkPhysicalDeviceRobustness2FeaturesKHR& features = info.queryRobustness2Features();
            DD_APPEND_FEATURE(robustBufferAccess2)
            DD_APPEND_FEATURE(robustImageAccess2)
            DD_APPEND_FEATURE(nullDescriptor)
        }
        if (info.support_VK_KHR_shader_float16_int8())
        {
            const VkPhysicalDeviceFloat16Int8FeaturesKHR& features = info.queryFloat16Int8Features();
            DD_APPEND_FEATURE(shaderFloat16)
            DD_APPEND_FEATURE(shaderInt8)
        }
        if (info.support_VK_KHR_sampler_ycbcr_conversion())
        {
            const VkPhysicalDeviceSamplerYcbcrConversionFeaturesKHR& features = info.querySamplerYcbcrConversionFeatures();
            DD_APPEND_FEATURE(samplerYcbcrConversion)
        }
        if (info.support_VK_KHR_cooperative_matrix())
        {
            const VkPhysicalDeviceCooperativeMatrixFeaturesKHR& features = info.queryCooperativeMatrixFeatures();
            DD_APPEND_FEATURE(cooperativeMatrix)
            DD_APPEND_FEATURE(cooperativeMatrixRobustBufferAccess)
        }
        else if (info.support_VK_NV_cooperative_matrix())
        {
            const VkPhysicalDeviceCooperativeMatrixFeaturesNV& features = info.queryCooperativeMatrixFeaturesNV();
            DD_APPEND_FEATURE(cooperativeMatrix)
            DD_APPEND_FEATURE(cooperativeMatrixRobustBufferAccess)
        }
        if (info.support_VK_NV_cooperative_matrix2())
        {
            const VkPhysicalDeviceCooperativeMatrix2FeaturesNV& features = info.queryCooperativeMatrix2FeaturesNV();
            DD_APPEND_FEATURE(cooperativeMatrixWorkgroupScope)
            DD_APPEND_FEATURE(cooperativeMatrixFlexibleDimensions)
            DD_APPEND_FEATURE(cooperativeMatrixReductions)
            DD_APPEND_FEATURE(cooperativeMatrixConversions)
            DD_APPEND_FEATURE(cooperativeMatrixPerElementOperations)
            DD_APPEND_FEATURE(cooperativeMatrixTensorAddressing)
            DD_APPEND_FEATURE(cooperativeMatrixBlockLoads)
        }
        if (info.support_VK_NV_cooperative_vector())
        {
            const VkPhysicalDeviceCooperativeVectorFeaturesNV& features = info.queryCooperativeVectorFeaturesNV();
            DD_APPEND_FEATURE(cooperativeVector)
            DD_APPEND_FEATURE(cooperativeVectorTraining)
        }
        if (info.support_VK_EXT_subgroup_size_control())
        {
            const VkPhysicalDeviceSubgroupSizeControlFeaturesEXT& features = info.querySubgroupSizeControlFeatures();
            DD_APPEND_FEATURE(subgroupSizeControl)
            DD_APPEND_FEATURE(computeFullSubgroups)
        }
        if (info.support_VK_KHR_shader_bfloat16())
        {
            const VkPhysicalDeviceShaderBfloat16FeaturesKHR& features = info.queryShaderBfloat16Features();
            DD_APPEND_FEATURE(shaderBFloat16Type)
            DD_APPEND_FEATURE(shaderBFloat16DotProduct)
            DD_APPEND_FEATURE(shaderBFloat16CooperativeMatrix)
        }
        if (info.support_VK_EXT_shader_float8())
        {
            const VkPhysicalDeviceShaderFloat8FeaturesEXT& features = info.queryShaderFloat8Features();
            DD_APPEND_FEATURE(shaderFloat8)
            DD_APPEND_FEATURE(shaderFloat8CooperativeMatrix)
        }
        if (info.support_VK_KHR_shader_float_controls2())
        {
            const VkPhysicalDeviceShaderFloatControls2FeaturesKHR& features = info.queryShaderFloatControls2Features();
            DD_APPEND_FEATURE(shaderFloatControls2)
        }
        if (info.support_VK_KHR_shader_integer_dot_product())
        {
            const VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR& features = info.queryShaderIntegerDotProductFeatures();
            DD_APPEND_FEATURE(shaderIntegerDotProduct)
        }
        if (info.support_VK_KHR_shader_subgroup_rotate())
        {
            const VkPhysicalDeviceShaderSubgroupRotateFeaturesKHR& features = info.queryShaderSubgroupRotateFeatures();
            DD_APPEND_FEATURE(shaderSubgroupRotate)
            DD_APPEND_FEATURE(shaderSubgroupRotateClustered)
        }
        if (info.support_VK_EXT_shader_atomic_float())
        {
            const VkPhysicalDeviceShaderAtomicFloatFeaturesEXT& features = info.queryShaderAtomicFloatFeatures();
            DD_APPEND_FEATURE(shaderBufferFloat32Atomics)
            DD_APPEND_FEATURE(shaderBufferFloat32AtomicAdd)
            DD_APPEND_FEATURE(shaderBufferFloat64Atomics)
            DD_APPEND_FEATURE(shaderBufferFloat64AtomicAdd)
            DD_APPEND_FEATURE(shaderSharedFloat32Atomics)
            DD_APPEND_FEATURE(shaderSharedFloat32AtomicAdd)
            DD_APPEND_FEATURE(shaderSharedFloat64Atomics)
            DD_APPEND_FEATURE(shaderSharedFloat64AtomicAdd)
            DD_APPEND_FEATURE(shaderImageFloat32Atomics)
            DD_APPEND_FEATURE(shaderImageFloat32AtomicAdd)
            DD_APPEND_FEATURE(sparseImageFloat32Atomics)
            DD_APPEND_FEATURE(sparseImageFloat32AtomicAdd)
        }
        if (info.support_VK_EXT_shader_atomic_float2())
        {
            const VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT& features = info.queryShaderAtomicFloat2Features();
            DD_APPEND_FEATURE(shaderBufferFloat16Atomics)
            DD_APPEND_FEATURE(shaderBufferFloat16AtomicAdd)
            DD_APPEND_FEATURE(shaderBufferFloat16AtomicMinMax)
            DD_APPEND_FEATURE(shaderBufferFloat32AtomicMinMax)
            DD_APPEND_FEATURE(shaderBufferFloat64AtomicMinMax)
            DD_APPEND_FEATURE(shaderSharedFloat16Atomics)
            DD_APPEND_FEATURE(shaderSharedFloat16AtomicAdd)
            DD_APPEND_FEATURE(shaderSharedFloat16AtomicMinMax)
            DD_APPEND_FEATURE(shaderSharedFloat32AtomicMinMax)
            DD_APPEND_FEATURE(shaderSharedFloat64AtomicMinMax)
            DD_APPEND_FEATURE(shaderImageFloat32AtomicMinMax)
            DD_APPEND_FEATURE(sparseImageFloat32AtomicMinMax)
        }
        if (info.support_VK_KHR_vulkan_memory_model())
        {
            const VkPhysicalDeviceVulkanMemoryModelFeaturesKHR& features = info.queryVulkanMemoryModelFeatures();
            DD_APPEND_FEATURE(vulkanMemoryModel)
            DD_APPEND_FEATURE(vulkanMemoryModelDeviceScope)
            DD_APPEND_FEATURE(vulkanMemoryModelAvailabilityVisibilityChains)
        }

#undef DD_APPEND_FEATURE

#define DD_APPEND_PROPERTY(X) device_defines.append(#X, properties.X);

        // pull in device properties macros
        {
            const VkPhysicalDeviceProperties& properties = info.physicalDeviceProperties();
            DD_APPEND_PROPERTY(apiVersion)
            DD_APPEND_PROPERTY(driverVersion)
            DD_APPEND_PROPERTY(vendorID)
            DD_APPEND_PROPERTY(deviceID)
            DD_APPEND_PROPERTY(deviceType)
            // DD_APPEND_PROPERTY(deviceName)

            // DD_APPEND_PROPERTY(pipelineCacheUUID)

#define DD_APPEND_PROPERTY_LIMIT(X) device_defines.append(#X, properties.limits.X);
#define DD_APPEND_PROPERTY_LIMIT_2(X)                       \
    device_defines.append(#X "_0", properties.limits.X[0]); \
    device_defines.append(#X "_1", properties.limits.X[1]);
#define DD_APPEND_PROPERTY_LIMIT_3(X)                       \
    device_defines.append(#X "_0", properties.limits.X[0]); \
    device_defines.append(#X "_1", properties.limits.X[1]); \
    device_defines.append(#X "_2", properties.limits.X[2]);

            DD_APPEND_PROPERTY_LIMIT(maxImageDimension1D)
            DD_APPEND_PROPERTY_LIMIT(maxImageDimension2D)
            DD_APPEND_PROPERTY_LIMIT(maxImageDimension3D)
            DD_APPEND_PROPERTY_LIMIT(maxImageDimensionCube)
            DD_APPEND_PROPERTY_LIMIT(maxImageArrayLayers)
            DD_APPEND_PROPERTY_LIMIT(maxTexelBufferElements)
            DD_APPEND_PROPERTY_LIMIT(maxUniformBufferRange)
            DD_APPEND_PROPERTY_LIMIT(maxStorageBufferRange)
            DD_APPEND_PROPERTY_LIMIT(maxPushConstantsSize)
            DD_APPEND_PROPERTY_LIMIT(maxMemoryAllocationCount)
            DD_APPEND_PROPERTY_LIMIT(maxSamplerAllocationCount)
            DD_APPEND_PROPERTY_LIMIT(bufferImageGranularity)
            DD_APPEND_PROPERTY_LIMIT(sparseAddressSpaceSize)
            DD_APPEND_PROPERTY_LIMIT(maxBoundDescriptorSets)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageDescriptorSamplers)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageDescriptorUniformBuffers)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageDescriptorStorageBuffers)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageDescriptorSampledImages)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageDescriptorStorageImages)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageDescriptorInputAttachments)
            DD_APPEND_PROPERTY_LIMIT(maxPerStageResources)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetSamplers)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetUniformBuffers)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetUniformBuffersDynamic)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetStorageBuffers)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetStorageBuffersDynamic)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetSampledImages)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetStorageImages)
            DD_APPEND_PROPERTY_LIMIT(maxDescriptorSetInputAttachments)
            DD_APPEND_PROPERTY_LIMIT(maxVertexInputAttributes)
            DD_APPEND_PROPERTY_LIMIT(maxVertexInputBindings)
            DD_APPEND_PROPERTY_LIMIT(maxVertexInputAttributeOffset)
            DD_APPEND_PROPERTY_LIMIT(maxVertexInputBindingStride)
            DD_APPEND_PROPERTY_LIMIT(maxVertexOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationGenerationLevel)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationPatchSize)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationControlPerVertexInputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationControlPerVertexOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationControlPerPatchOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationControlTotalOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationEvaluationInputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxTessellationEvaluationOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxGeometryShaderInvocations)
            DD_APPEND_PROPERTY_LIMIT(maxGeometryInputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxGeometryOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxGeometryOutputVertices)
            DD_APPEND_PROPERTY_LIMIT(maxGeometryTotalOutputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxFragmentInputComponents)
            DD_APPEND_PROPERTY_LIMIT(maxFragmentOutputAttachments)
            DD_APPEND_PROPERTY_LIMIT(maxFragmentDualSrcAttachments)
            DD_APPEND_PROPERTY_LIMIT(maxFragmentCombinedOutputResources)
            DD_APPEND_PROPERTY_LIMIT(maxComputeSharedMemorySize)
            DD_APPEND_PROPERTY_LIMIT_3(maxComputeWorkGroupCount)
            DD_APPEND_PROPERTY_LIMIT(maxComputeWorkGroupInvocations)
            DD_APPEND_PROPERTY_LIMIT_3(maxComputeWorkGroupSize)
            DD_APPEND_PROPERTY_LIMIT(subPixelPrecisionBits)
            DD_APPEND_PROPERTY_LIMIT(subTexelPrecisionBits)
            DD_APPEND_PROPERTY_LIMIT(mipmapPrecisionBits)
            DD_APPEND_PROPERTY_LIMIT(maxDrawIndexedIndexValue)
            DD_APPEND_PROPERTY_LIMIT(maxDrawIndirectCount)
            DD_APPEND_PROPERTY_LIMIT(maxSamplerLodBias)
            DD_APPEND_PROPERTY_LIMIT(maxSamplerAnisotropy)
            DD_APPEND_PROPERTY_LIMIT(maxViewports)
            DD_APPEND_PROPERTY_LIMIT_2(maxViewportDimensions)
            DD_APPEND_PROPERTY_LIMIT_2(viewportBoundsRange)
            DD_APPEND_PROPERTY_LIMIT(viewportSubPixelBits)
            device_defines.append("minMemoryMapAlignment", (uint32_t)properties.limits.minMemoryMapAlignment);
            DD_APPEND_PROPERTY_LIMIT(minTexelBufferOffsetAlignment)
            DD_APPEND_PROPERTY_LIMIT(minUniformBufferOffsetAlignment)
            DD_APPEND_PROPERTY_LIMIT(minStorageBufferOffsetAlignment)
            DD_APPEND_PROPERTY_LIMIT(minTexelOffset)
            DD_APPEND_PROPERTY_LIMIT(maxTexelOffset)
            DD_APPEND_PROPERTY_LIMIT(minTexelGatherOffset)
            DD_APPEND_PROPERTY_LIMIT(maxTexelGatherOffset)
            DD_APPEND_PROPERTY_LIMIT(minInterpolationOffset)
            DD_APPEND_PROPERTY_LIMIT(maxInterpolationOffset)
            DD_APPEND_PROPERTY_LIMIT(subPixelInterpolationOffsetBits)
            DD_APPEND_PROPERTY_LIMIT(maxFramebufferWidth)
            DD_APPEND_PROPERTY_LIMIT(maxFramebufferHeight)
            DD_APPEND_PROPERTY_LIMIT(maxFramebufferLayers)
            DD_APPEND_PROPERTY_LIMIT(framebufferColorSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(framebufferDepthSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(framebufferStencilSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(framebufferNoAttachmentsSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(maxColorAttachments)
            DD_APPEND_PROPERTY_LIMIT(sampledImageColorSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(sampledImageIntegerSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(sampledImageDepthSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(sampledImageStencilSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(storageImageSampleCounts)
            DD_APPEND_PROPERTY_LIMIT(maxSampleMaskWords)
            DD_APPEND_PROPERTY_LIMIT(timestampComputeAndGraphics)
            DD_APPEND_PROPERTY_LIMIT(timestampPeriod)
            DD_APPEND_PROPERTY_LIMIT(maxClipDistances)
            DD_APPEND_PROPERTY_LIMIT(maxCullDistances)
            DD_APPEND_PROPERTY_LIMIT(maxCombinedClipAndCullDistances)
            DD_APPEND_PROPERTY_LIMIT(discreteQueuePriorities)
            DD_APPEND_PROPERTY_LIMIT_2(pointSizeRange)
            DD_APPEND_PROPERTY_LIMIT_2(lineWidthRange)
            DD_APPEND_PROPERTY_LIMIT(pointSizeGranularity)
            DD_APPEND_PROPERTY_LIMIT(lineWidthGranularity)
            DD_APPEND_PROPERTY_LIMIT(strictLines)
            DD_APPEND_PROPERTY_LIMIT(standardSampleLocations)
            DD_APPEND_PROPERTY_LIMIT(optimalBufferCopyOffsetAlignment)
            DD_APPEND_PROPERTY_LIMIT(optimalBufferCopyRowPitchAlignment)
            DD_APPEND_PROPERTY_LIMIT(nonCoherentAtomSize)

#undef DD_APPEND_PROPERTY_LIMIT
#undef DD_APPEND_PROPERTY_LIMIT_2
#undef DD_APPEND_PROPERTY_LIMIT_3

#define DD_APPEND_PROPERTY_SPARSE(X) device_defines.append(#X, properties.sparseProperties.X);

            DD_APPEND_PROPERTY_SPARSE(residencyStandard2DBlockShape)
            DD_APPEND_PROPERTY_SPARSE(residencyStandard2DMultisampleBlockShape)
            DD_APPEND_PROPERTY_SPARSE(residencyStandard3DBlockShape)
            DD_APPEND_PROPERTY_SPARSE(residencyAlignedMipSize)
            DD_APPEND_PROPERTY_SPARSE(residencyNonResidentStrict)

#undef DD_APPEND_PROPERTY_SPARSE
        }
        {
            const VkPhysicalDeviceSubgroupProperties& properties = info.querySubgroupProperties();
            DD_APPEND_PROPERTY(subgroupSize)
            DD_APPEND_PROPERTY(supportedStages)
            DD_APPEND_PROPERTY(supportedOperations)
            DD_APPEND_PROPERTY(quadOperationsInAllStages)

            // append subgroup ops
            device_defines.append("subgroup_basic", (properties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) ? 1 : 0);
            device_defines.append("subgroup_vote", (properties.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) ? 1 : 0);
            device_defines.append("subgroup_arithmetic", (properties.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) ? 1 : 0);
            device_defines.append("subgroup_ballot", (properties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) ? 1 : 0);
            device_defines.append("subgroup_shuffle", (properties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) ? 1 : 0);
            device_defines.append("subgroup_shuffle_relative", (properties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) ? 1 : 0);
            device_defines.append("subgroup_clustered", (properties.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) ? 1 : 0);
            device_defines.append("subgroup_quad", (properties.supportedOperations & VK_SUBGROUP_FEATURE_QUAD_BIT) ? 1 : 0);
            device_defines.append("subgroup_rotate", (properties.supportedOperations & VK_SUBGROUP_FEATURE_ROTATE_BIT) ? 1 : 0);
            device_defines.append("subgroup_rotate_relative", (properties.supportedOperations & VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT) ? 1 : 0);
            device_defines.append("subgroup_partitioned", (properties.supportedOperations & VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV) ? 1 : 0);
        }
        if (info.support_VK_NV_cooperative_matrix2())
        {
            const VkPhysicalDeviceCooperativeMatrix2PropertiesNV& properties = info.queryCooperativeMatrix2PropertiesNV();
            DD_APPEND_PROPERTY(cooperativeMatrixWorkgroupScopeMaxWorkgroupSize)
            DD_APPEND_PROPERTY(cooperativeMatrixFlexibleDimensionsMaxDimension)
            DD_APPEND_PROPERTY(cooperativeMatrixWorkgroupScopeReservedSharedMemory)
        }
        if (info.support_VK_NV_cooperative_vector())
        {
            const VkPhysicalDeviceCooperativeVectorPropertiesNV& properties = info.queryCooperativeVectorPropertiesNV();
            DD_APPEND_PROPERTY(cooperativeVectorSupportedStages)
            DD_APPEND_PROPERTY(cooperativeVectorTrainingFloat16Accumulation)
            DD_APPEND_PROPERTY(cooperativeVectorTrainingFloat32Accumulation)
            DD_APPEND_PROPERTY(maxCooperativeVectorComponents)
        }
        if (info.support_VK_KHR_driver_properties())
        {
            const VkPhysicalDeviceDriverPropertiesKHR& properties = info.queryDriverProperties();
            DD_APPEND_PROPERTY(driverID)
            // DD_APPEND_PROPERTY(driverName)
            // DD_APPEND_PROPERTY(driverInfo)
            device_defines.append("conformanceVersion_major", properties.conformanceVersion.major);
            device_defines.append("conformanceVersion_minor", properties.conformanceVersion.minor);
            device_defines.append("conformanceVersion_subminor", properties.conformanceVersion.subminor);
            device_defines.append("conformanceVersion_patch", properties.conformanceVersion.patch);
        }
        if (info.support_VK_KHR_robustness2() || info.support_VK_EXT_robustness2())
        {
            const VkPhysicalDeviceRobustness2PropertiesKHR& properties = info.queryRobustness2Properties();
            DD_APPEND_PROPERTY(robustStorageBufferAccessSizeAlignment)
            DD_APPEND_PROPERTY(robustUniformBufferAccessSizeAlignment)
        }
        if (info.support_VK_KHR_shader_integer_dot_product())
        {
            const VkPhysicalDeviceShaderIntegerDotProductProperties& properties = info.queryShaderIntegerDotProductProperties();
            DD_APPEND_PROPERTY(integerDotProduct8BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct8BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct8BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct4x8BitPackedUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct4x8BitPackedSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct4x8BitPackedMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct16BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct16BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct16BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct32BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct32BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct32BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct64BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct64BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProduct64BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating8BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating8BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating16BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating16BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating32BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating32BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating64BitUnsignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating64BitSignedAccelerated)
            DD_APPEND_PROPERTY(integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated)
        }
        if (info.support_VK_EXT_subgroup_size_control())
        {
            const VkPhysicalDeviceSubgroupSizeControlPropertiesEXT& properties = info.querySubgroupSizeControlProperties();
            DD_APPEND_PROPERTY(minSubgroupSize)
            DD_APPEND_PROPERTY(maxSubgroupSize)
            DD_APPEND_PROPERTY(maxComputeWorkgroupSubgroups)
            DD_APPEND_PROPERTY(requiredSubgroupSizeStages)
        }

#if ENABLE_VALIDATION_LAYER
        if (info.support_VK_KHR_shader_non_semantic_info())
        {
            device_defines.append("enable_validation_layer", VK_TRUE);
            custom_defines.append("NCNN_LOGE", "debugPrintfEXT");
        }
#endif

#undef DD_APPEND_PROPERTY
    }

    std::string define_macro_data;

    for (size_t i = 0; i < custom_defines.definitions.size(); i++)
    {
        const char* key = custom_defines.definitions[i].first;
        const DefinitionCollector::typed_value& def = custom_defines.definitions[i].second;

        if (def.type == 0)
        {
            define_macro_data += std::string("#define ") + key + " " + def.s + "\n";
        }
        else
        {
            char defstr[256];
            if (def.type == 1)
            {
                sprintf(defstr, "%u", def.u8);
            }
            if (def.type == 2)
            {
                sprintf(defstr, "%u", def.u32);
            }
            if (def.type == 3)
            {
                sprintf(defstr, "%d", def.i32);
            }
            if (def.type == 4)
            {
                if (support_shader_int64)
                {
                    sprintf(defstr, "%luull", def.u64);
                }
                else
                {
                    uint32_t u32 = def.u64 > UINT_MAX ? UINT_MAX : (uint32_t)def.u64;
                    sprintf(defstr, "%u", u32);
                }
            }
            if (def.type == 5)
            {
                sprintf(defstr, "%e", def.f32);
            }

            define_macro_data += std::string("#define ") + key + " " + defstr + "\n";
        }
    }
    for (size_t i = 0; i < device_defines.definitions.size(); i++)
    {
        const char* key = device_defines.definitions[i].first;
        const DefinitionCollector::typed_value& def = device_defines.definitions[i].second;

        if (def.type == 0)
        {
            define_macro_data += std::string("#define ncnn_") + key + " \"" + def.s + "\"\n";
        }
        else
        {
            char defstr[256];
            if (def.type == 1)
            {
                sprintf(defstr, "%u", def.u8);
            }
            if (def.type == 2)
            {
                sprintf(defstr, "%u", def.u32);
            }
            if (def.type == 3)
            {
                sprintf(defstr, "%d", def.i32);
            }
            if (def.type == 4)
            {
                if (support_shader_int64)
                {
                    sprintf(defstr, "%luull", def.u64);
                }
                else
                {
                    uint32_t u32 = def.u64 > UINT_MAX ? UINT_MAX : (uint32_t)def.u64;
                    sprintf(defstr, "%u", u32);
                }
            }
            if (def.type == 5)
            {
                sprintf(defstr, "%e", def.f32);
            }

            define_macro_data += std::string("#define ncnn_") + key + " " + defstr + "\n";
        }
    }

    // enable extensions
    std::string custom_exts;
    if (support_shader_int64)
    {
        custom_exts += "#extension GL_EXT_shader_explicit_arithmetic_types_int64: require\n";
    }
    if (opt.use_fp16_storage)
    {
        custom_exts += "#extension GL_EXT_shader_16bit_storage: require\n";
    }
    if (opt.use_fp16_arithmetic)
    {
        custom_exts += "#extension GL_EXT_shader_explicit_arithmetic_types_float16: require\n";
    }
    if (opt.use_int8_storage)
    {
        custom_exts += "#extension GL_EXT_shader_8bit_storage: require\n";
    }
    if (opt.use_int8_arithmetic)
    {
        custom_exts += "#extension GL_EXT_shader_explicit_arithmetic_types_int8: require\n";
    }
#if ENABLE_VALIDATION_LAYER
    {
        custom_exts += "#extension GL_EXT_debug_printf : require\n";
    }
#endif

    // debug
    // NCNN_LOGE("%s", define_macro_data.c_str());

    bool compile_success = true;

    {
        glslang::TShader s(EShLangCompute);

        // split shader source by token "#version 450\n"
        int version_end_pos = -1;
        {
            for (int i = 0; i < comp_data_size - 8; i++)
            {
                if (strncmp(comp_data + i, "#version", 8) != 0)
                    continue;

                // #version shall be the very beginning or after newline
                if (i != 0 && comp_data[i - 1] != '\n')
                    continue;

                int nversion = 0;
                sscanf(comp_data + i, "#version %*d\n%n", &nversion);
                if (nversion == 0)
                    continue;

                version_end_pos = i + nversion;
                break;
            }

            if (version_end_pos == -1)
            {
                NCNN_LOGE("shader source has no #version token");
                return -1;
            }

            // NCNN_LOGE("version_end_pos = %d", version_end_pos);
        }

        const char* comp_data_2 = comp_data + version_end_pos;
        int comp_data_size_1 = version_end_pos;
        int comp_data_size_2 = comp_data_size - comp_data_size_1;

        const char* comp_datas[4] = {comp_data, custom_exts.c_str(), define_macro_data.c_str(), comp_data_2};
        const int comp_data_sizes[4] = {comp_data_size_1, (int)custom_exts.size(), (int)define_macro_data.size(), comp_data_size_2};

        s.setStringsWithLengths(comp_datas, comp_data_sizes, 4);

        s.setEntryPoint("main");
        s.setSourceEntryPoint("main");

        s.setEnvInput(glslang::EShSourceGlsl, EShLangCompute, glslang::EShClientVulkan, 1);

        if (opt.use_subgroup_ops || opt.use_cooperative_matrix)
        {
            // subgroup / cooperative_matrix need vulkan-1.1 and spirv-1.3
            s.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
            s.setEnvTarget(glslang::EshTargetSpv, glslang::EShTargetSpv_1_3);
        }
        else
        {
            s.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
            s.setEnvTarget(glslang::EshTargetSpv, glslang::EShTargetSpv_1_0);
        }

        TBuiltInResource resources = get_default_TBuiltInResource();

        VulkanShaderIncluder includer;

        bool pr = s.parse(&resources, 100, ENoProfile, false, false, EShMsgDefault, includer);
        if (!pr)
        {
            NCNN_LOGE("compile spir-v module failed");
            NCNN_LOGE("%s", s.getInfoLog());
            NCNN_LOGE("%s", s.getInfoDebugLog());

            // print as line_number: code
            {
                const char* p = comp_datas[3];
                const char* line_end;
                int line_number = 1;

                while ((line_end = strchr(p, '\n')) != NULL)
                {
                    NCNN_LOGE("%d:\t%.*s", line_number++, (int)(line_end - p), p);
                    p = line_end + 1;
                }

                if (*p != '\0')
                {
                    NCNN_LOGE("%d:\t%s", line_number, p);
                }
            }

            compile_success = false;
        }
        else
        {
            glslang::TIntermediate* ir = s.getIntermediate();
            glslang::GlslangToSpv(*ir, spirv);
        }
    }

    return compile_success ? 0 : -1;
}

int compile_spirv_module(int shader_type_index, const Option& opt, std::vector<uint32_t>& spirv)
{
    if (shader_type_index < 0 || shader_type_index >= layer_shader_registry_entry_count)
    {
        NCNN_LOGE("no such shader module %d", shader_type_index);
        return -1;
    }

    const char* comp_data = layer_shader_registry[shader_type_index].comp_data;
    int comp_data_size = layer_shader_registry[shader_type_index].comp_data_size;

    return compile_spirv_module(comp_data, comp_data_size, opt, spirv);
}

int resolve_shader_info(const uint32_t* spv_data, size_t spv_data_size, ShaderInfo& shader_info)
{
    shader_info.specialization_count = 0;
    shader_info.binding_count = 0;
    shader_info.push_constant_count = 0;

    uint32_t parameter_id = -233;

    int specialization_count = 0;
    int binding_count = 0;
    int push_constant_count = 0;

    // id -> binding_type
    std::vector<int> id_types;

    // binding_id -> binding_type
    std::vector<int> binding_types;

    const uint32_t* p = spv_data;

    int bound = p[3];

    id_types.resize(bound);

    // skip magic version generator bound schema
    p += 5;

    // foreach op
    while ((const unsigned char*)p < (const unsigned char*)spv_data + spv_data_size)
    {
        uint32_t opcode = p[0];

        uint16_t wordcount = opcode >> 16;
        uint16_t op = opcode & 0xffff;

        if (op == 5) // OpName
        {
            uint32_t id = p[1];
            const char* name = (const char*)&p[2];
            if (strcmp(name, "parameter") == 0)
            {
                parameter_id = id;
            }
        }
        else if (op == 6) // OpMemberName
        {
            uint32_t id = p[1];
            if (id == parameter_id)
            {
                push_constant_count++;
            }
        }
        else if (op == 25) // OpTypeImage
        {
            uint32_t id = p[1];
            id_types[id] = 2;
        }
        else if (op == 27) // OpTypeSampledImage
        {
            uint32_t id = p[1];
            id_types[id] = 3;
        }
        else if (op == 32) // OpTypePointer
        {
            uint32_t id = p[1];
            uint32_t storage_class = p[2];
            uint32_t type = p[3];
            if (storage_class == 0) // UniformConstant
            {
                id_types[id] = id_types[type];
            }
            if (storage_class == 2) // Uniform
            {
                id_types[id] = id_types[type];
            }
            if (storage_class == 12) // StorageBuffer
            {
                id_types[type] = 1;
                id_types[id] = id_types[type];
            }
        }
        else if (op == 59) // OpVariable
        {
            uint32_t id = p[1];
            uint32_t var_id = p[2];
            uint32_t storage_class = p[3];
            if (storage_class == 0) // UniformConstant
            {
                id_types[var_id] = id_types[id];
            }
            if (storage_class == 2) // Uniform
            {
                id_types[var_id] = id_types[id];
            }
            if (storage_class == 12) // StorageBuffer
            {
                id_types[var_id] = id_types[id];
            }
        }
        else if (op == 71) // OpDecorate
        {
            uint32_t id = p[1];
            uint32_t decoration = p[2];
            uint32_t binding_id = p[3];
            if (decoration == 1) // SpecId
            {
                specialization_count++;
            }
            if (decoration == 3) // BufferBlock
            {
                id_types[id] = 1;
            }
            else if (decoration == 33) // Binding
            {
                binding_count = std::max(binding_count, (int)binding_id + 1);

                binding_types.resize(binding_count);
                binding_types[binding_id] = id;
            }
        }

        p += wordcount;
    }

    if (binding_count > 16)
    {
        NCNN_LOGE("too many binding %d", binding_count);
        return -1;
    }

    shader_info.specialization_count = specialization_count;
    shader_info.binding_count = binding_count;
    shader_info.push_constant_count = push_constant_count;

    // resolve binding_types
    for (int i = 0; i < binding_count; i++)
    {
        shader_info.binding_types[i] = id_types[binding_types[i]];
    }

    return 0;
}

} // namespace ncnn

#endif // NCNN_VULKAN
