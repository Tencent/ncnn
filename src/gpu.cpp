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

#include "gpu.h"

#if NCNN_VULKAN

#include <math.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "glslang/SPIRV/GlslangToSpv.h"
#include "glslang/glslang/Public/ShaderLang.h"

#include "command.h"
#include "layer.h"
#include "layer/vulkan/packing_vulkan.h"
#include "layer_type.h"
#include "mat.h"
#include "pipelinecache.h"

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
};
static __ncnn_vulkan_instance_holder g_instance;

static int g_gpu_count = 0;
static int g_default_gpu_index = -1;

// NOTE 8 is large enough i think ...
#define NCNN_MAX_GPU_COUNT 8
static GpuInfo g_gpu_infos[NCNN_MAX_GPU_COUNT];

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

int support_VK_KHR_external_memory_capabilities = 0;
int support_VK_KHR_get_physical_device_properties2 = 0;
int support_VK_KHR_get_surface_capabilities2 = 0;
int support_VK_KHR_surface = 0;
int support_VK_EXT_debug_utils = 0;
#if __ANDROID_API__ >= 26
int support_VK_KHR_android_surface = 0;
#endif // __ANDROID_API__ >= 26

// VK_KHR_external_memory_capabilities
PFN_vkGetPhysicalDeviceExternalBufferPropertiesKHR vkGetPhysicalDeviceExternalBufferPropertiesKHR = 0;

// VK_KHR_get_physical_device_properties2
PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR = 0;
PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR = 0;
PFN_vkGetPhysicalDeviceFormatProperties2KHR vkGetPhysicalDeviceFormatProperties2KHR = 0;
PFN_vkGetPhysicalDeviceImageFormatProperties2KHR vkGetPhysicalDeviceImageFormatProperties2KHR = 0;
PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR vkGetPhysicalDeviceQueueFamilyProperties2KHR = 0;
PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR = 0;
PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR vkGetPhysicalDeviceSparseImageFormatProperties2KHR = 0;

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

// compile with old vulkan sdk
#if VK_HEADER_VERSION < 70
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
#endif // VK_HEADER_VERSION < 70
#if VK_HEADER_VERSION < 80
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR (VkStructureType)1000177000
typedef struct VkPhysicalDevice8BitStorageFeaturesKHR
{
    VkStructureType sType;
    void* pNext;
    VkBool32 storageBuffer8BitAccess;
    VkBool32 uniformAndStorageBuffer8BitAccess;
    VkBool32 storagePushConstant8;
} VkPhysicalDevice8BitStorageFeaturesKHR;
#endif // VK_HEADER_VERSION < 80
#if VK_HEADER_VERSION < 95
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR (VkStructureType)1000082000
typedef struct VkPhysicalDeviceFloat16Int8FeaturesKHR
{
    VkStructureType sType;
    void* pNext;
    VkBool32 shaderFloat16;
    VkBool32 shaderInt8;
} VkPhysicalDeviceFloat16Int8FeaturesKHR;
#endif // VK_HEADER_VERSION < 95
#if VK_HEADER_VERSION < 97
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT (VkStructureType)1000237000
typedef struct VkPhysicalDeviceMemoryBudgetPropertiesEXT
{
    VkStructureType sType;
    void* pNext;
    VkDeviceSize heapBudget[VK_MAX_MEMORY_HEAPS];
    VkDeviceSize heapUsage[VK_MAX_MEMORY_HEAPS];
} VkPhysicalDeviceMemoryBudgetPropertiesEXT;
#endif // VK_HEADER_VERSION < 97

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
        vkGetPhysicalDeviceSparseImageFormatProperties2KHR = (PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSparseImageFormatProperties2KHR");
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

    return 0;
}

#if ENABLE_VALIDATION_LAYER
static VkDebugUtilsMessengerEXT callback;

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
    VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* /*pUserData*/)
{
    NCNN_LOGE("validation layer: %s", pCallbackData->pMessage);

    return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback)
{
    PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func)
        return func(instance, pCreateInfo, pAllocator, pCallback);

    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator)
{
    PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func)
        func(instance, callback, pAllocator);
}
#endif // ENABLE_VALIDATION_LAYER

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

static uint32_t find_device_graphics_queue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties)
{
    // first try, graphics only queue
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if ((queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                && !(queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT))
        {
            return i;
        }
    }

    // second try, any queue with graphics and compute
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if ((queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                && (queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT))
        {
            return i;
        }
    }

    // third try, any queue with graphics
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if (queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            return i;
        }
    }

    //     NCNN_LOGE("no graphics queue");
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

    // fourth try, use graphics queue
    uint32_t graphics_queue_index = find_device_graphics_queue(queueFamilyProperties);
    if (graphics_queue_index != (uint32_t)-1)
    {
        return graphics_queue_index;
    }

    //     NCNN_LOGE("no transfer queue");
    return -1;
}

static int find_default_vulkan_device_index()
{
    // first try, discrete gpu
    for (int i = 0; i < g_gpu_count; i++)
    {
        if (g_gpu_infos[i].type == 0)
            return i;
    }

    // second try, integrated gpu
    for (int i = 0; i < g_gpu_count; i++)
    {
        if (g_gpu_infos[i].type == 1)
            return i;
    }

    // third try, any probed device
    if (g_gpu_count > 0)
        return 0;

    NCNN_LOGE("no vulkan device");
    return -1;
}

int create_gpu_instance()
{
    MutexLockGuard lock(g_instance_lock);

    if ((VkInstance)g_instance != 0)
        return 0;

    // NCNN_LOGE("create_gpu_instance");

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
    support_VK_KHR_surface = 0;
    support_VK_EXT_debug_utils = 0;
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
        else if (strcmp(exp.extensionName, "VK_KHR_surface") == 0)
            support_VK_KHR_surface = exp.specVersion;
        else if (strcmp(exp.extensionName, "VK_EXT_debug_utils") == 0)
            support_VK_EXT_debug_utils = exp.specVersion;
#if __ANDROID_API__ >= 26
        else if (strcmp(exp.extensionName, "VK_KHR_android_surface") == 0)
            support_VK_KHR_android_surface = exp.specVersion;
#endif // __ANDROID_API__ >= 26
    }

    if (support_VK_KHR_external_memory_capabilities)
        enabledExtensions.push_back("VK_KHR_external_memory_capabilities");
    if (support_VK_KHR_get_physical_device_properties2)
        enabledExtensions.push_back("VK_KHR_get_physical_device_properties2");
    if (support_VK_KHR_get_surface_capabilities2)
        enabledExtensions.push_back("VK_KHR_get_surface_capabilities2");
    if (support_VK_KHR_surface)
        enabledExtensions.push_back("VK_KHR_surface");
#if ENABLE_VALIDATION_LAYER
    if (support_VK_EXT_debug_utils)
        enabledExtensions.push_back("VK_EXT_debug_utils");
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
    applicationInfo.engineVersion = 20201010;
    applicationInfo.apiVersion = instance_api_version;

    VkInstanceCreateInfo instanceCreateInfo;
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pNext = 0;
    instanceCreateInfo.flags = 0;
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

#if ENABLE_VALIDATION_LAYER
    if (support_VK_EXT_debug_utils)
    {
        VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
        createInfo.pUserData = 0;
        ret = CreateDebugUtilsMessengerEXT(g_instance, &createInfo, NULL, &callback);
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
        GpuInfo& gpu_info = g_gpu_infos[gpu_info_index];

        // device type
        VkPhysicalDeviceProperties physicalDeviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

        //         NCNN_LOGE("[%u] apiVersion = %u.%u.%u", i, VK_VERSION_MAJOR(physicalDeviceProperties.apiVersion),
        //             VK_VERSION_MINOR(physicalDeviceProperties.apiVersion), VK_VERSION_PATCH(physicalDeviceProperties.apiVersion));
        //         NCNN_LOGE("[%u] driverVersion = %u.%u.%u", i, VK_VERSION_MAJOR(physicalDeviceProperties.driverVersion),
        //             VK_VERSION_MINOR(physicalDeviceProperties.driverVersion), VK_VERSION_PATCH(physicalDeviceProperties.driverVersion));
        //         NCNN_LOGE("[%u] vendorID = %x", i, physicalDeviceProperties.vendorID);
        //         NCNN_LOGE("[%u] deviceID = %x", i, physicalDeviceProperties.deviceID);
        //         NCNN_LOGE("[%u] deviceType = %x", i, physicalDeviceProperties.deviceType);
        //         NCNN_LOGE("[%u] deviceName = %s", i, physicalDeviceProperties.deviceName);
        //         NCNN_LOGE("[%u] pipelineCacheUUID = %u", i, physicalDeviceProperties.pipelineCacheUUID);

        // mali
        // t760 = 0x13b5 0x7500001
        // t860 = 0x13b5 0x8602000
        // t880 = 0x13b5 0x8800020
        // g51  = 0x13b5 0x70901010
        // g52  = 0x13b5 0x74021000
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

        gpu_info.bug_storage_buffer_no_l1 = false;
        gpu_info.bug_layout_binding_id_alias = false;
        gpu_info.bug_corrupted_online_pipeline_cache = false;
        gpu_info.bug_implicit_fp16_arithmetic = false;

        if (physicalDeviceProperties.vendorID == 0x5143 && physicalDeviceProperties.apiVersion < VK_MAKE_VERSION(1, 0, 49))
        {
            // qcom adreno with old buggy driver cannot handle binding id alias
            gpu_info.bug_layout_binding_id_alias = true;
        }

        if (physicalDeviceProperties.vendorID == 0x5143 && physicalDeviceProperties.apiVersion < VK_MAKE_VERSION(1, 0, 66))
        {
            // qcom adreno with old buggy driver cannot share created pipeline properly
            gpu_info.bug_corrupted_online_pipeline_cache = true;
        }

        if (physicalDeviceProperties.vendorID == 0x5143 && !(physicalDeviceProperties.deviceID == 0x6040001 || physicalDeviceProperties.deviceID == 0x6050002))
        {
            // NOTE but qcom855/qcom855plus/qcom865 are known exceptions
            // qcom adreno storage buffer without L1 cache

            // HACK buffer2image before image-read dependency does not work properly
            // even promised with full image memory barrier on old adreno driver
            // TODO figure out a proper workaround without hurt speed too much
            //             gpu_info.bug_storage_buffer_no_l1 = true;
        }

        if (physicalDeviceProperties.vendorID == 0x13b5
                && (physicalDeviceProperties.deviceID == 0x7500001
                    || physicalDeviceProperties.deviceID == 0x8602000
                    || physicalDeviceProperties.deviceID == 0x8800020))
        {
            // these arm mali midgard era driver cannot handle binding id alias
            gpu_info.bug_layout_binding_id_alias = true;
        }

#if __APPLE__
        {
            // metal shader never accept binding id alias
            gpu_info.bug_layout_binding_id_alias = true;
        }
#endif

        if (physicalDeviceProperties.vendorID == 0x13b5
                && (physicalDeviceProperties.deviceID == 0x7500001
                    || physicalDeviceProperties.deviceID == 0x8602000
                    || physicalDeviceProperties.deviceID == 0x8800020
                    || physicalDeviceProperties.deviceID == 0x70901010
                    || physicalDeviceProperties.deviceID == 0x74021000
                    || physicalDeviceProperties.deviceID == 0x60a00002
                    || physicalDeviceProperties.deviceID == 0x62210001))
        {
            // NOTE rk3288/rk3399/t880/g51/g52/g71/g72
            // however, g76/g77 has explicit fp16 arithmetic
            // arm mali driver accept spirv with fp16 arithmetic
            gpu_info.bug_implicit_fp16_arithmetic = true;
        }

        if (physicalDeviceProperties.vendorID == 0x5143
                && (physicalDeviceProperties.deviceID == 0x6030001
                    || physicalDeviceProperties.deviceID == 0x6040001
                    || physicalDeviceProperties.deviceID == 0x6050002))
        {
            // TODO enable devices other than qcom845/qcom855/qcom855plus/qcom865
            // qcom adreno driver accept spirv with fp16 arithmetic
            gpu_info.bug_implicit_fp16_arithmetic = true;
        }

        gpu_info.physical_device = physicalDevice;

        // info
        gpu_info.api_version = physicalDeviceProperties.apiVersion;
        gpu_info.driver_version = physicalDeviceProperties.driverVersion;
        gpu_info.vendor_id = physicalDeviceProperties.vendorID;
        gpu_info.device_id = physicalDeviceProperties.deviceID;
        gpu_info.device_name = std::string(physicalDeviceProperties.deviceName);
        memcpy(gpu_info.pipeline_cache_uuid, physicalDeviceProperties.pipelineCacheUUID, VK_UUID_SIZE);

        if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            gpu_info.type = 0;
        else if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            gpu_info.type = 1;
        else if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
            gpu_info.type = 2;
        else if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
            gpu_info.type = 3;
        else
            gpu_info.type = -1;

        // device capability
        gpu_info.max_shared_memory_size = physicalDeviceProperties.limits.maxComputeSharedMemorySize;

        gpu_info.max_workgroup_count[0] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[0];
        gpu_info.max_workgroup_count[1] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[1];
        gpu_info.max_workgroup_count[2] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[2];

        gpu_info.max_workgroup_invocations = physicalDeviceProperties.limits.maxComputeWorkGroupInvocations;

        gpu_info.max_workgroup_size[0] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[0];
        gpu_info.max_workgroup_size[1] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[1];
        gpu_info.max_workgroup_size[2] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[2];

        gpu_info.memory_map_alignment = physicalDeviceProperties.limits.minMemoryMapAlignment;
        gpu_info.buffer_offset_alignment = physicalDeviceProperties.limits.minStorageBufferOffsetAlignment;
        gpu_info.non_coherent_atom_size = physicalDeviceProperties.limits.nonCoherentAtomSize;
        gpu_info.buffer_image_granularity = physicalDeviceProperties.limits.bufferImageGranularity;
        gpu_info.max_image_dimension_1d = physicalDeviceProperties.limits.maxImageDimension1D;
        gpu_info.max_image_dimension_2d = physicalDeviceProperties.limits.maxImageDimension2D;
        gpu_info.max_image_dimension_3d = physicalDeviceProperties.limits.maxImageDimension3D;

        gpu_info.timestamp_period = physicalDeviceProperties.limits.timestampPeriod;

        //         NCNN_LOGE("[%u] max_shared_memory_size = %u", i, gpu_info.max_shared_memory_size);
        //         NCNN_LOGE("[%u] max_workgroup_count = %u %u %u", i, gpu_info.max_workgroup_count[0], gpu_info.max_workgroup_count[1], gpu_info.max_workgroup_count[2]);
        //         NCNN_LOGE("[%u] max_workgroup_invocations = %u", i, gpu_info.max_workgroup_invocations);
        //         NCNN_LOGE("[%u] max_workgroup_size = %u %u %u", i, gpu_info.max_workgroup_size[0], gpu_info.max_workgroup_size[1], gpu_info.max_workgroup_size[2]);
        //         NCNN_LOGE("[%u] memory_map_alignment = %lu", i, gpu_info.memory_map_alignment);
        //         NCNN_LOGE("[%u] buffer_offset_alignment = %lu", i, gpu_info.buffer_offset_alignment);

        // find compute queue
        uint32_t queueFamilyPropertiesCount;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, 0);

        std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertiesCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, queueFamilyProperties.data());

        gpu_info.compute_queue_family_index = find_device_compute_queue(queueFamilyProperties);
        gpu_info.graphics_queue_family_index = find_device_graphics_queue(queueFamilyProperties);
        gpu_info.transfer_queue_family_index = find_device_transfer_queue(queueFamilyProperties);

        gpu_info.compute_queue_count = queueFamilyProperties[gpu_info.compute_queue_family_index].queueCount;
        gpu_info.graphics_queue_count = queueFamilyProperties[gpu_info.graphics_queue_family_index].queueCount;
        gpu_info.transfer_queue_count = queueFamilyProperties[gpu_info.transfer_queue_family_index].queueCount;

        gpu_info.unified_compute_transfer_queue = gpu_info.compute_queue_family_index == gpu_info.transfer_queue_family_index;

        // additional device properties
        gpu_info.subgroup_size = 64;
        gpu_info.support_subgroup_basic = false;
        gpu_info.support_subgroup_vote = false;
        gpu_info.support_subgroup_ballot = false;
        gpu_info.support_subgroup_shuffle = false;
        if (support_VK_KHR_get_physical_device_properties2)
        {
            void* queryDeviceProperties = 0;

            // query subgroup
            VkPhysicalDeviceSubgroupProperties physicalDeviceSubgroupProperties;
            physicalDeviceSubgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
            physicalDeviceSubgroupProperties.pNext = queryDeviceProperties;
            queryDeviceProperties = &physicalDeviceSubgroupProperties;

            VkPhysicalDeviceProperties2KHR queryProperties;
            queryProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
            queryProperties.pNext = queryDeviceProperties;

            vkGetPhysicalDeviceProperties2KHR(physicalDevice, &queryProperties);

            gpu_info.subgroup_size = physicalDeviceSubgroupProperties.subgroupSize;
            if (physicalDeviceSubgroupProperties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)
            {
                gpu_info.support_subgroup_basic = physicalDeviceSubgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT;
                gpu_info.support_subgroup_vote = physicalDeviceSubgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT;
                gpu_info.support_subgroup_ballot = physicalDeviceSubgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT;
                gpu_info.support_subgroup_shuffle = physicalDeviceSubgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
            }
        }

        // cache memory properties
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &gpu_info.physicalDeviceMemoryProperties);

        // get device extension
        uint32_t deviceExtensionPropertyCount = 0;
        ret = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionPropertyCount, NULL);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkEnumerateDeviceExtensionProperties failed %d", ret);
            return -1;
        }

        std::vector<VkExtensionProperties> deviceExtensionProperties(deviceExtensionPropertyCount);
        ret = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionPropertyCount, deviceExtensionProperties.data());
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkEnumerateDeviceExtensionProperties failed %d", ret);
            return -1;
        }

        // extension capability
        gpu_info.support_VK_KHR_8bit_storage = 0;
        gpu_info.support_VK_KHR_16bit_storage = 0;
        gpu_info.support_VK_KHR_bind_memory2 = 0;
        gpu_info.support_VK_KHR_dedicated_allocation = 0;
        gpu_info.support_VK_KHR_descriptor_update_template = 0;
        gpu_info.support_VK_KHR_external_memory = 0;
        gpu_info.support_VK_KHR_get_memory_requirements2 = 0;
        gpu_info.support_VK_KHR_maintenance1 = 0;
        gpu_info.support_VK_KHR_push_descriptor = 0;
        gpu_info.support_VK_KHR_sampler_ycbcr_conversion = 0;
        gpu_info.support_VK_KHR_shader_float16_int8 = 0;
        gpu_info.support_VK_KHR_shader_float_controls = 0;
        gpu_info.support_VK_KHR_storage_buffer_storage_class = 0;
        gpu_info.support_VK_KHR_swapchain = 0;
        gpu_info.support_VK_EXT_queue_family_foreign = 0;
#if __ANDROID_API__ >= 26
        gpu_info.support_VK_ANDROID_external_memory_android_hardware_buffer = 0;
#endif // __ANDROID_API__ >= 26
        for (uint32_t j = 0; j < deviceExtensionPropertyCount; j++)
        {
            const VkExtensionProperties& exp = deviceExtensionProperties[j];
            //             NCNN_LOGE("device extension %s = %u", exp.extensionName, exp.specVersion);

            if (strcmp(exp.extensionName, "VK_KHR_8bit_storage") == 0)
                gpu_info.support_VK_KHR_8bit_storage = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_16bit_storage") == 0)
                gpu_info.support_VK_KHR_16bit_storage = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_bind_memory2") == 0)
                gpu_info.support_VK_KHR_bind_memory2 = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_dedicated_allocation") == 0)
                gpu_info.support_VK_KHR_dedicated_allocation = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_descriptor_update_template") == 0)
                gpu_info.support_VK_KHR_descriptor_update_template = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_external_memory") == 0)
                gpu_info.support_VK_KHR_external_memory = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_get_memory_requirements2") == 0)
                gpu_info.support_VK_KHR_get_memory_requirements2 = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_maintenance1") == 0)
                gpu_info.support_VK_KHR_maintenance1 = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_push_descriptor") == 0)
                gpu_info.support_VK_KHR_push_descriptor = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_sampler_ycbcr_conversion") == 0)
                gpu_info.support_VK_KHR_sampler_ycbcr_conversion = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_shader_float16_int8") == 0)
                gpu_info.support_VK_KHR_shader_float16_int8 = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_shader_float_controls") == 0)
                gpu_info.support_VK_KHR_shader_float_controls = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_storage_buffer_storage_class") == 0)
                gpu_info.support_VK_KHR_storage_buffer_storage_class = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_swapchain") == 0)
                gpu_info.support_VK_KHR_swapchain = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_EXT_memory_budget") == 0)
                gpu_info.support_VK_EXT_memory_budget = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_EXT_queue_family_foreign") == 0)
                gpu_info.support_VK_EXT_queue_family_foreign = exp.specVersion;
#if __ANDROID_API__ >= 26
            else if (strcmp(exp.extensionName, "VK_ANDROID_external_memory_android_hardware_buffer") == 0)
                gpu_info.support_VK_ANDROID_external_memory_android_hardware_buffer = exp.specVersion;
#endif // __ANDROID_API__ >= 26
        }

        // check features
        gpu_info.support_fp16_packed = true;
        gpu_info.support_fp16_storage = false;
        gpu_info.support_fp16_arithmetic = false;
        gpu_info.support_int8_storage = false;
        gpu_info.support_int8_arithmetic = false;
        gpu_info.support_ycbcr_conversion = false;
        if (support_VK_KHR_get_physical_device_properties2)
        {
            void* queryExtensionFeatures = 0;

            // query int8 storage
            VkPhysicalDevice8BitStorageFeaturesKHR query8BitStorageFeatures;
            query8BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR;
            query8BitStorageFeatures.pNext = 0;
            if (gpu_info.support_VK_KHR_8bit_storage)
            {
                query8BitStorageFeatures.pNext = queryExtensionFeatures;
                queryExtensionFeatures = &query8BitStorageFeatures;
            }

            // query fp16/int16 storage
            VkPhysicalDevice16BitStorageFeaturesKHR query16BitStorageFeatures;
            query16BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;
            query16BitStorageFeatures.pNext = 0;
            if (gpu_info.support_VK_KHR_16bit_storage)
            {
                query16BitStorageFeatures.pNext = queryExtensionFeatures;
                queryExtensionFeatures = &query16BitStorageFeatures;
            }

            // query fp16/int8 arithmetic
            VkPhysicalDeviceFloat16Int8FeaturesKHR queryFloat16Int8Features;
            queryFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
            queryFloat16Int8Features.pNext = 0;
            if (gpu_info.support_VK_KHR_shader_float16_int8)
            {
                queryFloat16Int8Features.pNext = queryExtensionFeatures;
                queryExtensionFeatures = &queryFloat16Int8Features;
            }

            // query ycbcr_conversion
            VkPhysicalDeviceSamplerYcbcrConversionFeaturesKHR querySamplerYcbcrConversionFeatures;
            querySamplerYcbcrConversionFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES_KHR;
            querySamplerYcbcrConversionFeatures.pNext = 0;
            if (gpu_info.support_VK_KHR_sampler_ycbcr_conversion)
            {
                querySamplerYcbcrConversionFeatures.pNext = queryExtensionFeatures;
                queryExtensionFeatures = &querySamplerYcbcrConversionFeatures;
            }

            VkPhysicalDeviceFeatures2KHR queryFeatures;
            queryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR;
            queryFeatures.pNext = queryExtensionFeatures;

            vkGetPhysicalDeviceFeatures2KHR(physicalDevice, &queryFeatures);

            if (gpu_info.support_VK_KHR_8bit_storage)
            {
                gpu_info.support_int8_storage = query8BitStorageFeatures.storageBuffer8BitAccess && query8BitStorageFeatures.uniformAndStorageBuffer8BitAccess;
            }
            if (gpu_info.support_VK_KHR_16bit_storage && queryFeatures.features.shaderStorageImageExtendedFormats)
            {
                // shaderStorageImageExtendedFormats enables r16f format in storage image
                gpu_info.support_fp16_storage = query16BitStorageFeatures.storageBuffer16BitAccess && query16BitStorageFeatures.uniformAndStorageBuffer16BitAccess;
            }
            if (gpu_info.support_VK_KHR_shader_float16_int8)
            {
                gpu_info.support_fp16_arithmetic = queryFloat16Int8Features.shaderFloat16;
                gpu_info.support_int8_arithmetic = queryFloat16Int8Features.shaderInt8;
            }
            if (gpu_info.support_VK_KHR_sampler_ycbcr_conversion)
            {
                gpu_info.support_ycbcr_conversion = querySamplerYcbcrConversionFeatures.samplerYcbcrConversion;
            }
        }
        else
        {
            //             // TODO
            //             VkPhysicalDeviceFeatures features;
            //             vkGetPhysicalDeviceFeatures(physicalDevice, &features);
        }

        if (physicalDeviceProperties.vendorID == 0x13b5)
        {
            // the 16bit_storage implementation of arm mali driver is buggy :[
            gpu_info.support_fp16_storage = false;
        }

        if (physicalDeviceProperties.vendorID == 0x10002 && physicalDeviceProperties.deviceID == 0x70006214 && physicalDeviceProperties.apiVersion == VK_MAKE_VERSION(1, 1, 82))
        {
            // the 16bit_storage implementation of vivante gc1700 driver is buggy :[
            gpu_info.support_fp16_storage = false;
        }

        if (gpu_info.bug_implicit_fp16_arithmetic)
        {
            // force capability on as long as the driver accept spirv with fp16 arithmetic :D
            gpu_info.support_fp16_arithmetic = true;
        }

        NCNN_LOGE("[%u %s]  queueC=%u[%u]  queueG=%u[%u]  queueT=%u[%u]", i, physicalDeviceProperties.deviceName,
                  gpu_info.compute_queue_family_index, gpu_info.compute_queue_count,
                  gpu_info.graphics_queue_family_index, gpu_info.graphics_queue_count,
                  gpu_info.transfer_queue_family_index, gpu_info.transfer_queue_count);

        NCNN_LOGE("[%u %s]  bugsbn1=%d  buglbia=%d  bugcopc=%d  bugihfa=%d", i, physicalDeviceProperties.deviceName,
                  gpu_info.bug_storage_buffer_no_l1, gpu_info.bug_layout_binding_id_alias, gpu_info.bug_corrupted_online_pipeline_cache, gpu_info.bug_implicit_fp16_arithmetic);

        NCNN_LOGE("[%u %s]  fp16p=%d  fp16s=%d  fp16a=%d  int8s=%d  int8a=%d", i, physicalDeviceProperties.deviceName,
                  gpu_info.support_fp16_packed, gpu_info.support_fp16_storage, gpu_info.support_fp16_arithmetic,
                  gpu_info.support_int8_storage, gpu_info.support_int8_arithmetic);

        NCNN_LOGE("[%u %s]  subgroup=%u  basic=%d  vote=%d  ballot=%d  shuffle=%d", i, physicalDeviceProperties.deviceName,
                  gpu_info.subgroup_size, gpu_info.support_subgroup_basic, gpu_info.support_subgroup_vote,
                  gpu_info.support_subgroup_ballot, gpu_info.support_subgroup_shuffle);

        gpu_info_index++;
    }

    g_gpu_count = gpu_info_index;

    // the default gpu device
    g_default_gpu_index = find_default_vulkan_device_index();

    glslang::InitializeProcess();

    return 0;
}

void destroy_gpu_instance()
{
    MutexLockGuard lock(g_instance_lock);

    if ((VkInstance)g_instance == 0)
        return;

    // NCNN_LOGE("destroy_gpu_instance");

    glslang::FinalizeProcess();

    for (int i = 0; i < NCNN_MAX_GPU_COUNT; i++)
    {
        delete g_default_vkdev[i];
        g_default_vkdev[i] = 0;
    }

#if ENABLE_VALIDATION_LAYER
    if (support_VK_EXT_debug_utils)
    {
        DestroyDebugUtilsMessengerEXT(g_instance, callback, NULL);
    }
#endif // ENABLE_VALIDATION_LAYER

    vkDestroyInstance(g_instance, 0);

    g_instance.instance = 0;
}

static bool is_gpu_instance_ready()
{
    MutexLockGuard lock(g_instance_lock);

    return (VkInstance)g_instance != 0;
}

static void try_create_gpu_instance()
{
    if (!is_gpu_instance_ready())
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

    return g_gpu_infos[device_index];
}

VulkanDevice::VulkanDevice(int device_index)
    : info(g_gpu_infos[device_index])
{
    try_create_gpu_instance();

    std::vector<const char*> enabledExtensions;
    if (info.support_VK_KHR_8bit_storage)
        enabledExtensions.push_back("VK_KHR_8bit_storage");
    if (info.support_VK_KHR_16bit_storage)
        enabledExtensions.push_back("VK_KHR_16bit_storage");
    if (info.support_VK_KHR_bind_memory2)
        enabledExtensions.push_back("VK_KHR_bind_memory2");
    if (info.support_VK_KHR_dedicated_allocation)
        enabledExtensions.push_back("VK_KHR_dedicated_allocation");
    if (info.support_VK_KHR_descriptor_update_template)
        enabledExtensions.push_back("VK_KHR_descriptor_update_template");
    if (info.support_VK_KHR_external_memory)
        enabledExtensions.push_back("VK_KHR_external_memory");
    if (info.support_VK_KHR_get_memory_requirements2)
        enabledExtensions.push_back("VK_KHR_get_memory_requirements2");
    if (info.support_VK_KHR_maintenance1)
        enabledExtensions.push_back("VK_KHR_maintenance1");
    if (info.support_VK_KHR_push_descriptor)
        enabledExtensions.push_back("VK_KHR_push_descriptor");
    if (info.support_VK_KHR_sampler_ycbcr_conversion)
        enabledExtensions.push_back("VK_KHR_sampler_ycbcr_conversion");
    if (info.support_VK_KHR_shader_float16_int8)
        enabledExtensions.push_back("VK_KHR_shader_float16_int8");
    if (info.support_VK_KHR_shader_float_controls)
        enabledExtensions.push_back("VK_KHR_shader_float_controls");
    if (info.support_VK_KHR_storage_buffer_storage_class)
        enabledExtensions.push_back("VK_KHR_storage_buffer_storage_class");
    if (info.support_VK_KHR_swapchain)
        enabledExtensions.push_back("VK_KHR_swapchain");
    if (info.support_VK_EXT_memory_budget)
        enabledExtensions.push_back("VK_EXT_memory_budget");
    if (info.support_VK_EXT_queue_family_foreign)
        enabledExtensions.push_back("VK_EXT_queue_family_foreign");
#if __ANDROID_API__ >= 26
    if (info.support_VK_ANDROID_external_memory_android_hardware_buffer)
        enabledExtensions.push_back("VK_ANDROID_external_memory_android_hardware_buffer");
#endif // __ANDROID_API__ >= 26

    void* enabledExtensionFeatures = 0;

    // enable int8 storage
    VkPhysicalDevice8BitStorageFeaturesKHR enabled8BitStorageFeatures;
    enabled8BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR;
    enabled8BitStorageFeatures.pNext = 0;
    enabled8BitStorageFeatures.storageBuffer8BitAccess = info.support_int8_storage;
    enabled8BitStorageFeatures.uniformAndStorageBuffer8BitAccess = info.support_int8_storage;
    enabled8BitStorageFeatures.storagePushConstant8 = VK_FALSE;
    if (support_VK_KHR_get_physical_device_properties2 && info.support_VK_KHR_8bit_storage)
    {
        enabled8BitStorageFeatures.pNext = enabledExtensionFeatures;
        enabledExtensionFeatures = &enabled8BitStorageFeatures;
    }

    // enable fp16/int16 storage
    VkPhysicalDevice16BitStorageFeaturesKHR enabled16BitStorageFeatures;
    enabled16BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;
    enabled16BitStorageFeatures.pNext = 0;
    enabled16BitStorageFeatures.storageBuffer16BitAccess = info.support_fp16_storage;
    enabled16BitStorageFeatures.uniformAndStorageBuffer16BitAccess = info.support_fp16_storage;
    enabled16BitStorageFeatures.storagePushConstant16 = VK_FALSE;
    enabled16BitStorageFeatures.storageInputOutput16 = VK_FALSE;
    if (support_VK_KHR_get_physical_device_properties2 && info.support_VK_KHR_16bit_storage)
    {
        enabled16BitStorageFeatures.pNext = enabledExtensionFeatures;
        enabledExtensionFeatures = &enabled16BitStorageFeatures;
    }

    // enable fp16/int8 arithmetic
    VkPhysicalDeviceFloat16Int8FeaturesKHR enabledFloat16Int8Features;
    enabledFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
    enabledFloat16Int8Features.pNext = 0;
    enabledFloat16Int8Features.shaderFloat16 = info.support_fp16_arithmetic;
    enabledFloat16Int8Features.shaderInt8 = info.support_int8_arithmetic;
    if (support_VK_KHR_get_physical_device_properties2 && info.support_VK_KHR_shader_float16_int8)
    {
        enabledFloat16Int8Features.pNext = enabledExtensionFeatures;
        enabledExtensionFeatures = &enabledFloat16Int8Features;
    }

    // enable ycbcr conversion
    VkPhysicalDeviceSamplerYcbcrConversionFeaturesKHR querySamplerYcbcrConversionFeatures;
    querySamplerYcbcrConversionFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES_KHR;
    querySamplerYcbcrConversionFeatures.pNext = 0;
    querySamplerYcbcrConversionFeatures.samplerYcbcrConversion = info.support_ycbcr_conversion;
    if (support_VK_KHR_get_physical_device_properties2 && info.support_ycbcr_conversion)
    {
        querySamplerYcbcrConversionFeatures.pNext = enabledExtensionFeatures;
        enabledExtensionFeatures = &querySamplerYcbcrConversionFeatures;
    }

    std::vector<float> compute_queue_priorities(info.compute_queue_count, 1.f);   // 0.f ~ 1.f
    std::vector<float> graphics_queue_priorities(info.graphics_queue_count, 1.f); // 0.f ~ 1.f
    std::vector<float> transfer_queue_priorities(info.transfer_queue_count, 1.f); // 0.f ~ 1.f

    VkDeviceQueueCreateInfo deviceQueueCreateInfos[3];

    VkDeviceQueueCreateInfo deviceComputeQueueCreateInfo;
    deviceComputeQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    deviceComputeQueueCreateInfo.pNext = 0;
    deviceComputeQueueCreateInfo.flags = 0;
    deviceComputeQueueCreateInfo.queueFamilyIndex = info.compute_queue_family_index;
    deviceComputeQueueCreateInfo.queueCount = info.compute_queue_count;
    deviceComputeQueueCreateInfo.pQueuePriorities = compute_queue_priorities.data();

    VkDeviceQueueCreateInfo deviceGraphicsQueueCreateInfo;
    deviceGraphicsQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    deviceGraphicsQueueCreateInfo.pNext = 0;
    deviceGraphicsQueueCreateInfo.flags = 0;
    deviceGraphicsQueueCreateInfo.queueFamilyIndex = info.graphics_queue_family_index;
    deviceGraphicsQueueCreateInfo.queueCount = info.graphics_queue_count;
    deviceGraphicsQueueCreateInfo.pQueuePriorities = graphics_queue_priorities.data();

    VkDeviceQueueCreateInfo deviceTransferQueueCreateInfo;
    deviceTransferQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    deviceTransferQueueCreateInfo.pNext = 0;
    deviceTransferQueueCreateInfo.flags = 0;
    deviceTransferQueueCreateInfo.queueFamilyIndex = info.transfer_queue_family_index;
    deviceTransferQueueCreateInfo.queueCount = info.transfer_queue_count;
    deviceTransferQueueCreateInfo.pQueuePriorities = transfer_queue_priorities.data();

    VkDeviceCreateInfo deviceCreateInfo;
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pNext = enabledExtensionFeatures;
    deviceCreateInfo.flags = 0;
    if (info.compute_queue_family_index == info.graphics_queue_family_index && info.compute_queue_family_index == info.transfer_queue_family_index)
    {
        deviceQueueCreateInfos[0] = deviceComputeQueueCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 1;
    }
    else if (info.compute_queue_family_index == info.graphics_queue_family_index && info.compute_queue_family_index != info.transfer_queue_family_index)
    {
        deviceQueueCreateInfos[0] = deviceComputeQueueCreateInfo;
        deviceQueueCreateInfos[1] = deviceTransferQueueCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 2;
    }
    else if (info.compute_queue_family_index != info.graphics_queue_family_index && info.graphics_queue_family_index == info.transfer_queue_family_index)
    {
        deviceQueueCreateInfos[0] = deviceComputeQueueCreateInfo;
        deviceQueueCreateInfos[1] = deviceGraphicsQueueCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 2;
    }
    else // if (info.compute_queue_family_index != info.graphics_queue_family_index && info.graphics_queue_family_index != info.transfer_queue_family_index)
    {
        deviceQueueCreateInfos[0] = deviceComputeQueueCreateInfo;
        deviceQueueCreateInfos[1] = deviceGraphicsQueueCreateInfo;
        deviceQueueCreateInfos[2] = deviceTransferQueueCreateInfo;
        deviceCreateInfo.queueCreateInfoCount = 3;
    }
    deviceCreateInfo.pQueueCreateInfos = deviceQueueCreateInfos;
    deviceCreateInfo.enabledLayerCount = 0;
    deviceCreateInfo.ppEnabledLayerNames = 0;
    deviceCreateInfo.enabledExtensionCount = enabledExtensions.size();
    deviceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();
    deviceCreateInfo.pEnabledFeatures = 0; // VkPhysicalDeviceFeatures pointer

    VkResult ret = vkCreateDevice(info.physical_device, &deviceCreateInfo, 0, &device);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateDevice failed %d", ret);
    }

    init_device_extension();

    compute_queues.resize(info.compute_queue_count);
    blob_allocators.resize(info.compute_queue_count);
    staging_allocators.resize(info.compute_queue_count);
    for (uint32_t i = 0; i < info.compute_queue_count; i++)
    {
        vkGetDeviceQueue(device, info.compute_queue_family_index, i, &compute_queues[i]);
        blob_allocators[i] = new VkBlobAllocator(this);
        staging_allocators[i] = new VkStagingAllocator(this);
    }
    if (info.compute_queue_family_index != info.graphics_queue_family_index)
    {
        graphics_queues.resize(info.graphics_queue_count);
        for (uint32_t i = 0; i < info.graphics_queue_count; i++)
        {
            vkGetDeviceQueue(device, info.graphics_queue_family_index, i, &graphics_queues[i]);
        }
    }
    if (info.compute_queue_family_index != info.transfer_queue_family_index && info.graphics_queue_family_index != info.transfer_queue_family_index)
    {
        transfer_queues.resize(info.transfer_queue_count);
        for (uint32_t i = 0; i < info.transfer_queue_count; i++)
        {
            vkGetDeviceQueue(device, info.transfer_queue_family_index, i, &transfer_queues[i]);
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

        texelfetch_sampler = 0;
        ret = vkCreateSampler(device, &samplerCreateInfo, 0, &texelfetch_sampler);
        if (ret != VK_SUCCESS)
        {
            NCNN_LOGE("vkCreateSampler failed %d", ret);
        }
    }

    create_dummy_buffer_image();

    pipeline_cache = new PipelineCache(this);

    memset(uop_packing, 0, sizeof(uop_packing));
}

VulkanDevice::~VulkanDevice()
{
    destroy_utility_operator();

    destroy_dummy_buffer_image();

    if (texelfetch_sampler)
    {
        vkDestroySampler(device, texelfetch_sampler, 0);
    }

    for (size_t i = 0; i < blob_allocators.size(); i++)
    {
        delete blob_allocators[i];
    }
    blob_allocators.clear();
    for (size_t i = 0; i < staging_allocators.size(); i++)
    {
        delete staging_allocators[i];
    }
    staging_allocators.clear();

    delete pipeline_cache;

    vkDestroyDevice(device, 0);
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
    VkResult ret = vkCreateShaderModule(device, &shaderModuleCreateInfo, 0, &shader_module);
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

    if (info.support_VK_KHR_push_descriptor)
    {
        descriptorSetLayoutCreateInfo.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    }

    VkResult ret = vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, 0, descriptorset_layout);
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

    VkResult ret = vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, 0, pipeline_layout);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreatePipelineLayout failed %d", ret);
        return -1;
    }

    return 0;
}

int VulkanDevice::create_pipeline(VkShaderModule shader_module, VkPipelineLayout pipeline_layout, const std::vector<vk_specialization_type>& specializations, VkPipeline* pipeline) const
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

    VkComputePipelineCreateInfo computePipelineCreateInfo;
    computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    computePipelineCreateInfo.pNext = 0;
    computePipelineCreateInfo.flags = 0;
    computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
    computePipelineCreateInfo.layout = pipeline_layout;
    computePipelineCreateInfo.basePipelineHandle = 0;
    computePipelineCreateInfo.basePipelineIndex = 0;

    VkResult ret = vkCreateComputePipelines(device, 0, 1, &computePipelineCreateInfo, 0, pipeline);
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
    if (info.support_VK_KHR_push_descriptor)
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

    VkResult ret = vkCreateDescriptorUpdateTemplateKHR(device, &descriptorUpdateTemplateCreateInfo, 0, descriptor_update_template);
    if (ret != VK_SUCCESS)
    {
        NCNN_LOGE("vkCreateDescriptorUpdateTemplateKHR failed %d", ret);
        return -1;
    }

    return 0;
}

uint32_t VulkanDevice::find_memory_index(uint32_t memory_type_bits, VkFlags required, VkFlags preferred, VkFlags preferred_not) const
{
    // first try, find required and with preferred and without preferred_not
    for (uint32_t i = 0; i < info.physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        bool is_required = (1 << i) & memory_type_bits;
        if (is_required)
        {
            const VkMemoryType& memoryType = info.physicalDeviceMemoryProperties.memoryTypes[i];
            if ((memoryType.propertyFlags & required) == required
                    && (preferred && (memoryType.propertyFlags & preferred))
                    && (preferred_not && !(memoryType.propertyFlags & preferred_not)))
            {
                return i;
            }
        }
    }

    // second try, find required and with preferred
    for (uint32_t i = 0; i < info.physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        bool is_required = (1 << i) & memory_type_bits;
        if (is_required)
        {
            const VkMemoryType& memoryType = info.physicalDeviceMemoryProperties.memoryTypes[i];
            if ((memoryType.propertyFlags & required) == required
                    && (preferred && (memoryType.propertyFlags & preferred)))
            {
                return i;
            }
        }
    }

    // third try, find required and without preferred_not
    for (uint32_t i = 0; i < info.physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        bool is_required = (1 << i) & memory_type_bits;
        if (is_required)
        {
            const VkMemoryType& memoryType = info.physicalDeviceMemoryProperties.memoryTypes[i];
            if ((memoryType.propertyFlags & required) == required
                    && (preferred_not && !(memoryType.propertyFlags & preferred_not)))
            {
                return i;
            }
        }
    }

    // fourth try, find any required
    for (uint32_t i = 0; i < info.physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        bool is_required = (1 << i) & memory_type_bits;
        if (is_required)
        {
            const VkMemoryType& memoryType = info.physicalDeviceMemoryProperties.memoryTypes[i];
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
    const VkMemoryType& memoryType = info.physicalDeviceMemoryProperties.memoryTypes[memory_type_index];

    return memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
}

bool VulkanDevice::is_coherent(uint32_t memory_type_index) const
{
    const VkMemoryType& memoryType = info.physicalDeviceMemoryProperties.memoryTypes[memory_type_index];

    return memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
}

VkQueue VulkanDevice::acquire_queue(uint32_t queue_family_index) const
{
    if (queue_family_index != info.compute_queue_family_index
            && queue_family_index != info.graphics_queue_family_index
            && queue_family_index != info.transfer_queue_family_index)
    {
        NCNN_LOGE("invalid queue_family_index %u", queue_family_index);
        return 0;
    }

    MutexLockGuard lock(queue_lock);

    std::vector<VkQueue>& queues = queue_family_index == info.compute_queue_family_index ? compute_queues
                                   : queue_family_index == info.graphics_queue_family_index ? graphics_queues : transfer_queues;
    for (int i = 0; i < (int)queues.size(); i++)
    {
        VkQueue queue = queues[i];
        if (queue)
        {
            queues[i] = 0;
            return queue;
        }
    }

    NCNN_LOGE("out of hardware queue %u", queue_family_index);
    return 0;
}

void VulkanDevice::reclaim_queue(uint32_t queue_family_index, VkQueue queue) const
{
    if (queue_family_index != info.compute_queue_family_index
            && queue_family_index != info.graphics_queue_family_index
            && queue_family_index != info.transfer_queue_family_index)
    {
        NCNN_LOGE("invalid queue_family_index %u", queue_family_index);
        return;
    }

    MutexLockGuard lock(queue_lock);

    std::vector<VkQueue>& queues = queue_family_index == info.compute_queue_family_index ? compute_queues
                                   : queue_family_index == info.graphics_queue_family_index ? graphics_queues : transfer_queues;
    for (int i = 0; i < (int)queues.size(); i++)
    {
        if (!queues[i])
        {
            queues[i] = queue;
            return;
        }
    }

    NCNN_LOGE("FATAL ERROR! reclaim_queue get wild queue %u %p", queue_family_index, queue);
}

VkAllocator* VulkanDevice::acquire_blob_allocator() const
{
    MutexLockGuard lock(blob_allocator_lock);

    for (int i = 0; i < (int)blob_allocators.size(); i++)
    {
        VkAllocator* allocator = blob_allocators[i];
        if (allocator)
        {
            blob_allocators[i] = 0;
            return allocator;
        }
    }

    // pre-allocated allcator exhausted, create new
    VkAllocator* allocator = new VkBlobAllocator(this);
    blob_allocators.push_back(allocator);
    blob_allocators[blob_allocators.size() - 1] = 0;
    return allocator;
}

void VulkanDevice::reclaim_blob_allocator(VkAllocator* allocator) const
{
    MutexLockGuard lock(blob_allocator_lock);

    for (int i = 0; i < (int)blob_allocators.size(); i++)
    {
        if (!blob_allocators[i])
        {
            blob_allocators[i] = allocator;
            return;
        }
    }

    NCNN_LOGE("FATAL ERROR! reclaim_blob_allocator get wild allocator %p", allocator);
}

VkAllocator* VulkanDevice::acquire_staging_allocator() const
{
    MutexLockGuard lock(staging_allocator_lock);

    for (int i = 0; i < (int)staging_allocators.size(); i++)
    {
        VkAllocator* allocator = staging_allocators[i];
        if (allocator)
        {
            staging_allocators[i] = 0;
            return allocator;
        }
    }

    // pre-allocated allcator exhausted, create new
    VkAllocator* allocator = new VkStagingAllocator(this);
    staging_allocators.push_back(allocator);
    staging_allocators[staging_allocators.size() - 1] = 0;
    return allocator;
}

void VulkanDevice::reclaim_staging_allocator(VkAllocator* allocator) const
{
    MutexLockGuard lock(staging_allocator_lock);

    for (int i = 0; i < (int)staging_allocators.size(); i++)
    {
        if (!staging_allocators[i])
        {
            staging_allocators[i] = allocator;
            return;
        }
    }

    NCNN_LOGE("FATAL ERROR! reclaim_staging_allocator get wild allocator %p", allocator);
}

const VkSampler* VulkanDevice::immutable_texelfetch_sampler() const
{
    return &texelfetch_sampler;
}

VkMat VulkanDevice::get_dummy_buffer() const
{
    return dummy_buffer;
}

VkImageMat VulkanDevice::get_dummy_image() const
{
    return dummy_image;
}

const PipelineCache* VulkanDevice::get_pipeline_cache() const
{
    return pipeline_cache;
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
        if (width > (int)info.max_image_dimension_1d)
        {
            return false;
        }
    }
    else if (dims == 2)
    {
        if (width > (int)info.max_image_dimension_2d || height > (int)info.max_image_dimension_2d)
        {
            return false;
        }
    }
    else // if (dims == 3)
    {
        if (width > (int)info.max_image_dimension_3d || height > (int)info.max_image_dimension_3d || depth > (int)info.max_image_dimension_3d)
        {
            return false;
        }
    }

    return true;
}

uint32_t VulkanDevice::get_heap_budget() const
{
    // the first device local heap
    uint32_t device_local_heap_index = 0;
    uint32_t device_local_heap_size = 0;
    for (uint32_t i = 0; i < info.physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        const VkMemoryHeap& memoryHeap = info.physicalDeviceMemoryProperties.memoryHeaps[i];
        if (memoryHeap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
        {
            device_local_heap_index = i;
            device_local_heap_size = memoryHeap.size / 1024 / 1024;
            break;
        }
    }

    if (!info.support_VK_EXT_memory_budget)
    {
        //         NCNN_LOGE("heap budget from assumption\n");

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

    vkGetPhysicalDeviceMemoryProperties2KHR(info.physical_device, &memoryProperties);

    return memoryBudgetProperties.heapBudget[device_local_heap_index] / 1024 / 1024;
}

void VulkanDevice::convert_packing(const VkMat& src, VkMat& dst, int dst_elempack, VkCompute& cmd, const Option& _opt) const
{
    // buffer2buffer uop is created with use_image_storage disabled
    Option opt = _opt;
    opt.use_image_storage = false;

    int cast_type_to_index = opt.use_fp16_storage ? 2 : opt.use_fp16_packed ? 1 : 0;
    int packing_type_to_index = dst_elempack == 1 ? 0 : dst_elempack == 4 ? 1 : 2;

    int cast_type_from_index;
    if (src.elembits() == 32)
    {
        cast_type_from_index = 0;
    }
    else // if (src.elembits() == 16)
    {
        if (cast_type_to_index != 0)
        {
            cast_type_from_index = cast_type_to_index;
        }
        else if (info.support_fp16_storage)
        {
            cast_type_from_index = 2;
        }
        else // if (info.support_fp16_packed)
        {
            cast_type_from_index = 1;
        }
    }

    // NCNN_LOGE("convert_packing b2b %d %d %d", cast_type_from_index, cast_type_to_index, packing_type_to_index);

    const ncnn::Packing_vulkan* uop = get_utility_operator(0, 0, cast_type_from_index, cast_type_to_index, packing_type_to_index);
    uop->forward(src, dst, cmd, opt);
}

void VulkanDevice::convert_packing(const VkImageMat& src, VkImageMat& dst, int dst_elempack, VkCompute& cmd, const Option& opt) const
{
    if (info.bug_layout_binding_id_alias)
    {
        NCNN_LOGE("cannot convert_packing i2i");
        return;
    }

    int cast_type_to_index = opt.use_fp16_storage ? 2 : opt.use_fp16_packed ? 1 : 0;
    int packing_type_to_index = dst_elempack == 1 ? 0 : dst_elempack == 4 ? 1 : 2;

    int cast_type_from_index;
    if (src.elembits() == 32)
    {
        cast_type_from_index = 0;
    }
    else // if (src.elembits() == 16)
    {
        if (cast_type_to_index != 0)
        {
            cast_type_from_index = cast_type_to_index;
        }
        else if (info.support_fp16_storage)
        {
            cast_type_from_index = 2;
        }
        else // if (info.support_fp16_packed)
        {
            cast_type_from_index = 1;
        }
    }

    // NCNN_LOGE("convert_packing i2i %d %d %d", cast_type_from_index, cast_type_to_index, packing_type_to_index);

    const ncnn::Packing_vulkan* uop = get_utility_operator(1, 1, cast_type_from_index, cast_type_to_index, packing_type_to_index);
    uop->forward(src, dst, cmd, opt);
}

void VulkanDevice::convert_packing(const VkMat& src, VkImageMat& dst, int dst_elempack, VkCompute& cmd, const Option& opt) const
{
    if (info.bug_layout_binding_id_alias)
    {
        NCNN_LOGE("cannot convert_packing b2i");
        return;
    }

    int cast_type_to_index = opt.use_fp16_storage ? 2 : opt.use_fp16_packed ? 1 : 0;
    int packing_type_to_index = dst_elempack == 1 ? 0 : dst_elempack == 4 ? 1 : 2;

    int cast_type_from_index;
    if (src.elembits() == 32)
    {
        cast_type_from_index = 0;
    }
    else // if (src.elembits() == 16)
    {
        if (cast_type_to_index != 0)
        {
            cast_type_from_index = cast_type_to_index;
        }
        else if (info.support_fp16_storage)
        {
            cast_type_from_index = 2;
        }
        else // if (info.support_fp16_packed)
        {
            cast_type_from_index = 1;
        }
    }

    // NCNN_LOGE("convert_packing b2i %d %d %d", cast_type_from_index, cast_type_to_index, packing_type_to_index);

    const ncnn::Packing_vulkan* uop = get_utility_operator(0, 1, cast_type_from_index, cast_type_to_index, packing_type_to_index);
    uop->forward(src, dst, cmd, opt);
}

void VulkanDevice::convert_packing(const VkImageMat& src, VkMat& dst, int dst_elempack, VkCompute& cmd, const Option& opt) const
{
    if (info.bug_layout_binding_id_alias)
    {
        NCNN_LOGE("cannot convert_packing i2b");
        return;
    }

    int cast_type_to_index = opt.use_fp16_storage ? 2 : opt.use_fp16_packed ? 1 : 0;
    int packing_type_to_index = dst_elempack == 1 ? 0 : dst_elempack == 4 ? 1 : 2;

    int cast_type_from_index;
    if (src.elembits() == 32)
    {
        cast_type_from_index = 0;
    }
    else // if (src.elembits() == 16)
    {
        if (cast_type_to_index != 0)
        {
            cast_type_from_index = cast_type_to_index;
        }
        else if (info.support_fp16_storage)
        {
            cast_type_from_index = 2;
        }
        else // if (info.support_fp16_packed)
        {
            cast_type_from_index = 1;
        }
    }

    // NCNN_LOGE("convert_packing i2b %d %d %d", cast_type_from_index, cast_type_to_index, packing_type_to_index);

    const ncnn::Packing_vulkan* uop = get_utility_operator(1, 0, cast_type_from_index, cast_type_to_index, packing_type_to_index);
    uop->forward(src, dst, cmd, opt);
}

int VulkanDevice::init_device_extension()
{
    if (info.support_VK_KHR_bind_memory2)
    {
        vkBindBufferMemory2KHR = (PFN_vkBindBufferMemory2KHR)vkGetDeviceProcAddr(device, "vkBindBufferMemory2KHR");
        vkBindImageMemory2KHR = (PFN_vkBindImageMemory2KHR)vkGetDeviceProcAddr(device, "vkBindImageMemory2KHR");
    }

    if (info.support_VK_KHR_descriptor_update_template)
    {
        vkCreateDescriptorUpdateTemplateKHR = (PFN_vkCreateDescriptorUpdateTemplateKHR)vkGetDeviceProcAddr(device, "vkCreateDescriptorUpdateTemplateKHR");
        vkDestroyDescriptorUpdateTemplateKHR = (PFN_vkDestroyDescriptorUpdateTemplateKHR)vkGetDeviceProcAddr(device, "vkDestroyDescriptorUpdateTemplateKHR");
        vkUpdateDescriptorSetWithTemplateKHR = (PFN_vkUpdateDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(device, "vkUpdateDescriptorSetWithTemplateKHR");
    }

    if (info.support_VK_KHR_get_memory_requirements2)
    {
        vkGetImageMemoryRequirements2KHR = (PFN_vkGetImageMemoryRequirements2KHR)vkGetDeviceProcAddr(device, "vkGetImageMemoryRequirements2KHR");
        vkGetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2KHR)vkGetDeviceProcAddr(device, "vkGetBufferMemoryRequirements2KHR");
        vkGetImageSparseMemoryRequirements2KHR = (PFN_vkGetImageSparseMemoryRequirements2KHR)vkGetDeviceProcAddr(device, "vkGetImageSparseMemoryRequirements2KHR");
    }

    if (info.support_VK_KHR_maintenance1)
    {
        vkTrimCommandPoolKHR = (PFN_vkTrimCommandPoolKHR)vkGetDeviceProcAddr(device, "vkTrimCommandPoolKHR");
    }

    if (info.support_VK_KHR_push_descriptor)
    {
        if (info.support_VK_KHR_descriptor_update_template)
        {
            vkCmdPushDescriptorSetWithTemplateKHR = (PFN_vkCmdPushDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetWithTemplateKHR");
        }

        vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");
    }

    if (info.support_VK_KHR_sampler_ycbcr_conversion)
    {
        vkCreateSamplerYcbcrConversionKHR = (PFN_vkCreateSamplerYcbcrConversionKHR)vkGetDeviceProcAddr(device, "vkCreateSamplerYcbcrConversionKHR");
        vkDestroySamplerYcbcrConversionKHR = (PFN_vkDestroySamplerYcbcrConversionKHR)vkGetDeviceProcAddr(device, "vkDestroySamplerYcbcrConversionKHR");
    }

    if (info.support_VK_KHR_swapchain)
    {
        vkCreateSwapchainKHR = (PFN_vkCreateSwapchainKHR)vkGetDeviceProcAddr(device, "vkCreateSwapchainKHR");
        vkDestroySwapchainKHR = (PFN_vkDestroySwapchainKHR)vkGetDeviceProcAddr(device, "vkDestroySwapchainKHR");
        vkGetSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR)vkGetDeviceProcAddr(device, "vkGetSwapchainImagesKHR");
        vkAcquireNextImageKHR = (PFN_vkAcquireNextImageKHR)vkGetDeviceProcAddr(device, "vkAcquireNextImageKHR");
        vkQueuePresentKHR = (PFN_vkQueuePresentKHR)vkGetDeviceProcAddr(device, "vkQueuePresentKHR");
    }

#if __ANDROID_API__ >= 26
    if (info.support_VK_ANDROID_external_memory_android_hardware_buffer)
    {
        vkGetAndroidHardwareBufferPropertiesANDROID = (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)vkGetDeviceProcAddr(device, "vkGetAndroidHardwareBufferPropertiesANDROID");
        vkGetMemoryAndroidHardwareBufferANDROID = (PFN_vkGetMemoryAndroidHardwareBufferANDROID)vkGetDeviceProcAddr(device, "vkGetMemoryAndroidHardwareBufferANDROID");
    }
#endif // __ANDROID_API__ >= 26

    return 0;
}

class VkDummyAllocator : public VkBlobAllocator
{
public:
    VkDummyAllocator(const VulkanDevice* _vkdev)
        : VkBlobAllocator(_vkdev)
    {
        // NOTE 16k is large enough I think ...
        block_size = alignSize(16 * 1024, buffer_offset_alignment);
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
        //         NCNN_LOGE("xxx barrier buffer %p +%d ~%d", buffer.buffer(), buffer.buffer_offset(), buffer.buffer_capacity());

        // barrier device any @ compute/null to shader-readwrite @ compute
        VkBufferMemoryBarrier* barriers = new VkBufferMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = buffer.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].buffer = buffer.buffer();
        barriers[0].offset = buffer.buffer_offset();
        barriers[0].size = buffer.buffer_capacity();

        VkPipelineStageFlags src_stage = buffer.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 1, barriers, 0, 0);
            delete[] barriers;
        }
        else
        {
            record r;
            r.type = record::TYPE_buffer_barrers;
            r.command_buffer = compute_command_buffer;
            r.buffer_barrers.src_stage = src_stage;
            r.buffer_barrers.dst_stage = dst_stage;
            r.buffer_barrers.barrier_count = 1;
            r.buffer_barrers.barriers = barriers;
            delayed_records.push_back(r);
        }

        // mark device shader-readwrite @ compute
        buffer.data->access_flags = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        buffer.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }

    void record_dummy(const VkImageMat& image)
    {
        //         NCNN_LOGE("xxx barrier image %p +%d ~%d %p", image.image(), image.data->bind_offset, image.data->bind_capacity, image.imageview());

        // image layout transform any @ any to shader-write @ compute
        VkImageMemoryBarrier* barriers = new VkImageMemoryBarrier[1];
        barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barriers[0].pNext = 0;
        barriers[0].srcAccessMask = image.data->access_flags;
        barriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barriers[0].oldLayout = image.data->image_layout;
        barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[0].image = image.image();
        barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barriers[0].subresourceRange.baseMipLevel = 0;
        barriers[0].subresourceRange.levelCount = 1;
        barriers[0].subresourceRange.baseArrayLayer = 0;
        barriers[0].subresourceRange.layerCount = 1;

        VkPipelineStageFlags src_stage = image.data->stage_flags;
        VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

        if (vkdev->info.support_VK_KHR_push_descriptor)
        {
            vkCmdPipelineBarrier(compute_command_buffer, src_stage, dst_stage, 0, 0, 0, 0, 0, 1, barriers);
            delete[] barriers;
        }
        else
        {
            record r;
            r.type = record::TYPE_image_barrers;
            r.command_buffer = compute_command_buffer;
            r.image_barrers.src_stage = src_stage;
            r.image_barrers.dst_stage = dst_stage;
            r.image_barrers.barrier_count = 1;
            r.image_barrers.barriers = barriers;
            delayed_records.push_back(r);
        }

        // mark image shader-write @ compute
        image.data->access_flags = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        image.data->image_layout = VK_IMAGE_LAYOUT_GENERAL;
        image.data->stage_flags = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
};

int VulkanDevice::create_dummy_buffer_image()
{
    dummy_allocator = new VkDummyAllocator(this);

    dummy_buffer.create(1, 4u, dummy_allocator);
    dummy_image.create(1, 4u, dummy_allocator);

    VkDummyCompute cmd(this);

    cmd.record_dummy(dummy_buffer);
    cmd.record_dummy(dummy_image);

    cmd.submit_and_wait();

    return 0;
}

void VulkanDevice::destroy_dummy_buffer_image()
{
    dummy_buffer.release();
    dummy_image.release();

    delete dummy_allocator;
}

const ncnn::Packing_vulkan* VulkanDevice::get_utility_operator(int storage_type_from, int storage_type_to, int cast_type_from_index, int cast_type_to_index, int packing_type_to_index) const
{
    MutexLockGuard lock(uop_lock);

    const ncnn::Packing_vulkan* cached_uop = uop_packing[storage_type_from][storage_type_to][cast_type_from_index][cast_type_to_index][packing_type_to_index];
    if (cached_uop)
        return cached_uop;

    if ((cast_type_from_index == 1 && cast_type_to_index == 2) || (cast_type_from_index == 2 && cast_type_to_index == 1))
    {
        NCNN_LOGE("no fp16p to/from fp16s conversion");
        return 0;
    }

    // create uop
    Option opt;
    opt.use_image_storage = (storage_type_from == 1 || storage_type_to == 1);
    opt.use_fp16_packed = (cast_type_from_index == 1 || cast_type_to_index == 1);
    opt.use_fp16_storage = (cast_type_from_index == 2 || cast_type_to_index == 2);

    if (info.bug_layout_binding_id_alias && opt.use_image_storage)
    {
        NCNN_LOGE("cannot create uop with use_image_storage if bug_layout_binding_id_alias");
        return 0;
    }

    if (!info.support_fp16_packed && opt.use_fp16_packed)
    {
        NCNN_LOGE("cannot create uop with use_fp16_packed if not support_fp16_packed");
        return 0;
    }

    if (!info.support_fp16_storage && opt.use_fp16_storage)
    {
        NCNN_LOGE("cannot create uop with use_fp16_storage if not support_fp16_storage");
        return 0;
    }

    // fp16/int8 arithmetic are not necessary for packing
    // and may conflict with storage options
    opt.use_fp16_arithmetic = false;
    opt.use_int8_arithmetic = false;

    // enable pack8 for pack8to1/pack8to4
    opt.use_shader_pack8 = true;

    opt.use_vulkan_compute = true;

    // cache uop pipeline as device member explicitly
    opt.pipeline_cache = 0;

    ncnn::Packing_vulkan* uop = new ncnn::Packing_vulkan;
    uop->vkdev = this;

    ncnn::ParamDict pd;
    pd.set(0, packing_type_to_index == 0 ? 1 : packing_type_to_index == 1 ? 4 : 8); // out_elempack
    pd.set(2, cast_type_from_index + 1);                                            // 0=auto 1=fp32 2=fp16p 3=fp16s
    pd.set(3, cast_type_to_index + 1);
    pd.set(4, storage_type_from); // 0=buffer 1=image
    pd.set(5, storage_type_to);

    uop->load_param(pd);

    uop->create_pipeline(opt);

    uop_packing[storage_type_from][storage_type_to][cast_type_from_index][cast_type_to_index][packing_type_to_index] = uop;

    return uop;
}

void VulkanDevice::destroy_utility_operator()
{
    Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_arithmetic = false;
    opt.pipeline_cache = 0;

    // from buffer | image
    // to buffer | image
    for (int i0 = 0; i0 < 2; i0++)
    {
        for (int i1 = 0; i1 < 2; i1++)
        {
            opt.use_image_storage = (i0 == 1 || i1 == 1);
            if (info.bug_layout_binding_id_alias && opt.use_image_storage)
                continue;

            // from fp32-b/i | fp16p-b/i | fp16s-b/i
            // to fp32-b/i | fp16p-b/i | fp16s-b/i
            for (int j0 = 0; j0 < 3; j0++)
            {
                for (int j1 = 0; j1 < 3; j1++)
                {
                    if ((j0 == 1 && j1 == 2) || (j0 == 2 && j1 == 1))
                    {
                        // no fp16p to/from fp16s conversion
                        continue;
                    }

                    opt.use_fp16_packed = (j0 == 1 || j1 == 1);
                    opt.use_fp16_storage = (j0 == 2 || j1 == 2);

                    if (!info.support_fp16_packed && opt.use_fp16_packed)
                        continue;

                    if (!info.support_fp16_storage && opt.use_fp16_storage)
                        continue;

                    // to pack1 | pack4 | pack8
                    for (int k = 0; k < 3; k++)
                    {
                        // enable pack8 for pack8to1/pack8to4
                        opt.use_shader_pack8 = true;

                        ncnn::Layer* uop = uop_packing[i0][i1][j0][j1][k];
                        if (!uop)
                            continue;

                        uop->destroy_pipeline(opt);

                        delete uop;

                        uop_packing[i0][i1][j0][j1][k] = 0;
                    }
                }
            }
        }
    }
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

int compile_spirv_module(const char* comp_data, int comp_data_size, const Option& opt, std::vector<uint32_t>& spirv)
{
    std::vector<std::pair<const char*, const char*> > custom_defines;

    if (opt.use_fp16_storage)
    {
        custom_defines.push_back(std::make_pair("sfp", "float16_t"));
        custom_defines.push_back(std::make_pair("sfpvec2", "f16vec2"));
        custom_defines.push_back(std::make_pair("sfpvec4", "f16vec4"));

        if (opt.use_fp16_arithmetic)
        {
            custom_defines.push_back(std::make_pair("sfpvec8", "f16mat2x4"));
            custom_defines.push_back(std::make_pair("sfpmat4", "f16mat4"));
        }
    }
    else if (opt.use_fp16_packed)
    {
        custom_defines.push_back(std::make_pair("sfp", "float"));
        custom_defines.push_back(std::make_pair("sfpvec2", "uint"));
        custom_defines.push_back(std::make_pair("sfpvec4", "uvec2"));
        custom_defines.push_back(std::make_pair("sfpvec8", "uvec4"));
    }
    else
    {
        custom_defines.push_back(std::make_pair("sfp", "float"));
        custom_defines.push_back(std::make_pair("sfpvec2", "vec2"));
        custom_defines.push_back(std::make_pair("sfpvec4", "vec4"));
        custom_defines.push_back(std::make_pair("sfpvec8", "mat2x4"));
        custom_defines.push_back(std::make_pair("sfpmat4", "mat4"));
    }

    if (opt.use_fp16_arithmetic)
    {
        custom_defines.push_back(std::make_pair("afp", "float16_t"));
        custom_defines.push_back(std::make_pair("afpvec2", "f16vec2"));
        custom_defines.push_back(std::make_pair("afpvec4", "f16vec4"));
        custom_defines.push_back(std::make_pair("afpvec8", "f16mat2x4"));
        custom_defines.push_back(std::make_pair("afpmat4", "f16mat4"));
    }
    else
    {
        custom_defines.push_back(std::make_pair("afp", "float"));
        custom_defines.push_back(std::make_pair("afpvec2", "vec2"));
        custom_defines.push_back(std::make_pair("afpvec4", "vec4"));
        custom_defines.push_back(std::make_pair("afpvec8", "mat2x4"));
        custom_defines.push_back(std::make_pair("afpmat4", "mat4"));
    }

    if (opt.use_fp16_storage && opt.use_fp16_arithmetic)
    {
        custom_defines.push_back(std::make_pair("buffer_ld1(buf,i)", "buf[i]"));
        custom_defines.push_back(std::make_pair("buffer_st1(buf,i,v)", "{buf[i]=v;}"));
        custom_defines.push_back(std::make_pair("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i]=f16vec4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a]);}"));
        custom_defines.push_back(std::make_pair("buffer_cp1to8(buf,i,sbuf,si4,sii4)", "{buf[i]=f16mat2x4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a],sbuf[sii4.r],sbuf[sii4.g],sbuf[sii4.b],sbuf[sii4.a]);}"));
        custom_defines.push_back(std::make_pair("buffer_ld2(buf,i)", "buf[i]"));
        custom_defines.push_back(std::make_pair("buffer_st2(buf,i,v)", "{buf[i]=v;}"));
        custom_defines.push_back(std::make_pair("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_ld4(buf,i)", "buf[i]"));
        custom_defines.push_back(std::make_pair("buffer_st4(buf,i,v)", "{buf[i]=v;}"));
        custom_defines.push_back(std::make_pair("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp4to1(buf,i4,sbuf,si)", "{buf[i4.r]=sbuf[si].r;buf[i4.g]=sbuf[si].g;buf[i4.b]=sbuf[si].b;buf[i4.a]=sbuf[si].a;}"));
        custom_defines.push_back(std::make_pair("buffer_cp4to8(buf,i,sbuf,si2)", "{buf[i]=f16mat2x4(sbuf[si2.r],sbuf[si2.g]);}"));
        custom_defines.push_back(std::make_pair("buffer_ld8(buf,i)", "buf[i]"));
        custom_defines.push_back(std::make_pair("buffer_st8(buf,i,v)", "{buf[i]=v;}"));
        custom_defines.push_back(std::make_pair("buffer_cp8(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp8to1(buf,i4,ii4,sbuf,si)", "{f16mat2x4 _v=sbuf[si]; buf[i4.r]=_v[0].r;buf[i4.g]=_v[0].g;buf[i4.b]=_v[0].b;buf[i4.a]=_v[0].a; buf[ii4.r]=_v[1].r;buf[ii4.g]=_v[1].g;buf[ii4.b]=_v[1].b;buf[ii4.a]=_v[1].a;}"));
        custom_defines.push_back(std::make_pair("buffer_cp8to4(buf,i2,sbuf,si)", "{f16mat2x4 _v=sbuf[si]; buf[i2.r]=_v[0];buf[i2.g]=_v[1];}"));
        custom_defines.push_back(std::make_pair("sfp2afpmat4(v)", "v"));
        custom_defines.push_back(std::make_pair("afp2sfpmat4(v)", "v"));
    }
    else if (opt.use_fp16_packed && opt.use_fp16_arithmetic)
    {
        custom_defines.push_back(std::make_pair("buffer_ld1(buf,i)", "float16_t(buf[i])"));
        custom_defines.push_back(std::make_pair("buffer_st1(buf,i,v)", "{buf[i]=float(v);}"));
        custom_defines.push_back(std::make_pair("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i]=uvec2(packHalf2x16(vec2(f16vec2(sbuf[si4.r],sbuf[si4.g]))),packHalf2x16(vec2(f16vec2(sbuf[si4.b],sbuf[si4.a]))));}"));
        custom_defines.push_back(std::make_pair("buffer_cp1to8(buf,i,sbuf,si4,sii4)", "{buf[i]=uvec4(packHalf2x16(vec2(f16vec2(sbuf[si4.r],sbuf[si4.g]))),packHalf2x16(vec2(f16vec2(sbuf[si4.b],sbuf[si4.a]))),packHalf2x16(vec2(f16vec2(sbuf[sii4.r],sbuf[sii4.g]))),packHalf2x16(vec2(f16vec2(sbuf[sii4.b],sbuf[sii4.a]))));}"));
        custom_defines.push_back(std::make_pair("buffer_ld2(buf,i)", "f16vec2(unpackHalf2x16(buf[i]))"));
        custom_defines.push_back(std::make_pair("buffer_st2(buf,i,v)", "{buf[i]=packHalf2x16(vec2(v))}"));
        custom_defines.push_back(std::make_pair("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_ld4(buf,i)", "f16vec4(vec4(unpackHalf2x16(buf[i].x),unpackHalf2x16(buf[i].y)))"));
        custom_defines.push_back(std::make_pair("buffer_st4(buf,i,v)", "{buf[i]=uvec2(packHalf2x16(vec2(v.rg)),packHalf2x16(vec2(v.ba)));}"));
        custom_defines.push_back(std::make_pair("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp4to1(buf,i4,sbuf,si)", "{uvec2 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g;}"));
        custom_defines.push_back(std::make_pair("buffer_cp4to8(buf,i,sbuf,si2)", "{buf[i]=uvec4(sbuf[si2.r],sbuf[si2.g]);}"));
        custom_defines.push_back(std::make_pair("buffer_ld8(buf,i)", "f16mat2x4(f16vec4(vec4(unpackHalf2x16(buf[i].r),unpackHalf2x16(buf[i].g))),f16vec4(vec4(unpackHalf2x16(buf[i].b),unpackHalf2x16(buf[i].a))))"));
        custom_defines.push_back(std::make_pair("buffer_st8(buf,i,v)", "{buf[i]=uvec4(uvec2(packHalf2x16(vec2(v[0].rg)),packHalf2x16(vec2(v[0].ba))),uvec2(packHalf2x16(vec2(v[1].rg)),packHalf2x16(vec2(v[1].ba))));}"));
        custom_defines.push_back(std::make_pair("buffer_cp8(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp8to1(buf,i4,ii4,sbuf,si)", "{uvec4 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.r);vec2 _v1=unpackHalf2x16(_v.g);vec2 _v2=unpackHalf2x16(_v.b);vec2 _v3=unpackHalf2x16(_v.a); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g; buf[ii4.r]=_v2.r;buf[ii4.g]=_v2.g;buf[ii4.b]=_v3.r;buf[ii4.a]=_v3.g;}"));
        custom_defines.push_back(std::make_pair("buffer_cp8to4(buf,i2,sbuf,si)", "{uvec4 _v=sbuf[si]; buf[i2.r]=_v.rg;buf[i2.g]=_v.ba;}"));
    }
    else if (opt.use_fp16_storage)
    {
        custom_defines.push_back(std::make_pair("buffer_ld1(buf,i)", "float(buf[i])"));
        custom_defines.push_back(std::make_pair("buffer_st1(buf,i,v)", "{buf[i]=float16_t(v);}"));
        custom_defines.push_back(std::make_pair("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i].r=sbuf[si4.r];buf[i].g=sbuf[si4.g];buf[i].b=sbuf[si4.b];buf[i].a=sbuf[si4.a];}"));
        custom_defines.push_back(std::make_pair("buffer_cp1to8(buf,i,sbuf,si4,sii4)", "{buf[i].abcd.r=sbuf[si4.r];buf[i].abcd.g=sbuf[si4.g];buf[i].abcd.b=sbuf[si4.b];buf[i].abcd.a=sbuf[si4.a];buf[i].efgh.r=sbuf[sii4.r];buf[i].efgh.g=sbuf[sii4.g];buf[i].efgh.b=sbuf[sii4.b];buf[i].efgh.a=sbuf[sii4.a];}"));
        custom_defines.push_back(std::make_pair("buffer_ld2(buf,i)", "vec2(buf[i])"));
        custom_defines.push_back(std::make_pair("buffer_st2(buf,i,v)", "{buf[i]=f16vec2(v);}"));
        custom_defines.push_back(std::make_pair("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_ld4(buf,i)", "vec4(buf[i])"));
        custom_defines.push_back(std::make_pair("buffer_st4(buf,i,v)", "{buf[i]=f16vec4(v);}"));
        custom_defines.push_back(std::make_pair("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp4to1(buf,i4,sbuf,si)", "{buf[i4.r]=sbuf[si].r;buf[i4.g]=sbuf[si].g;buf[i4.b]=sbuf[si].b;buf[i4.a]=sbuf[si].a;}"));
        custom_defines.push_back(std::make_pair("buffer_cp4to8(buf,i,sbuf,si2)", "{buf[i].abcd=sbuf[si2.r];buf[i].efgh=sbuf[si2.g];}"));
        custom_defines.push_back(std::make_pair("buffer_ld8(buf,i)", "mat2x4(vec4(buf[i].abcd),vec4(buf[i].efgh))"));
        custom_defines.push_back(std::make_pair("buffer_st8(buf,i,v)", "{buf[i].abcd=f16vec4(v[0]);buf[i].efgh=f16vec4(v[1]);}"));
        custom_defines.push_back(std::make_pair("buffer_cp8(buf,i,sbuf,si)", "{buf[i].abcd=sbuf[si].abcd;buf[i].efgh=sbuf[si].efgh;}"));
        custom_defines.push_back(std::make_pair("buffer_cp8to1(buf,i4,ii4,sbuf,si)", "{buf[i4.r]=sbuf[si].abcd.r;buf[i4.g]=sbuf[si].abcd.g;buf[i4.b]=sbuf[si].abcd.b;buf[i4.a]=sbuf[si].abcd.a; buf[ii4.r]=sbuf[si].efgh.r;buf[ii4.g]=sbuf[si].efgh.g;buf[ii4.b]=sbuf[si].efgh.b;buf[ii4.a]=sbuf[si].efgh.a;}"));
        custom_defines.push_back(std::make_pair("buffer_cp8to4(buf,i2,sbuf,si)", "{buf[i2.r]=sbuf[si].abcd;buf[i2.g]=sbuf[si].efgh;}"));
    }
    else if (opt.use_fp16_packed)
    {
        custom_defines.push_back(std::make_pair("buffer_ld1(buf,i)", "buf[i]"));
        custom_defines.push_back(std::make_pair("buffer_st1(buf,i,v)", "{buf[i]=v;}"));
        custom_defines.push_back(std::make_pair("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i]=uvec2(packHalf2x16(vec2(sbuf[si4.r],sbuf[si4.g])),packHalf2x16(vec2(sbuf[si4.b],sbuf[si4.a])));}"));
        custom_defines.push_back(std::make_pair("buffer_cp1to8(buf,i,sbuf,si4,sii4)", "{buf[i]=uvec4(packHalf2x16(vec2(sbuf[si4.r],sbuf[si4.g])),packHalf2x16(vec2(sbuf[si4.b],sbuf[si4.a])),packHalf2x16(vec2(sbuf[sii4.r],sbuf[sii4.g])),packHalf2x16(vec2(sbuf[sii4.b],sbuf[sii4.a])));}"));
        custom_defines.push_back(std::make_pair("buffer_ld2(buf,i)", "unpackHalf2x16(buf[i])"));
        custom_defines.push_back(std::make_pair("buffer_st2(buf,i,v)", "{buf[i]=packHalf2x16(v)}"));
        custom_defines.push_back(std::make_pair("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_ld4(buf,i)", "vec4(unpackHalf2x16(buf[i].x),unpackHalf2x16(buf[i].y))"));
        custom_defines.push_back(std::make_pair("buffer_st4(buf,i,v)", "{buf[i]=uvec2(packHalf2x16(v.rg),packHalf2x16(v.ba));}"));
        custom_defines.push_back(std::make_pair("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp4to1(buf,i4,sbuf,si)", "{uvec2 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g;}"));
        custom_defines.push_back(std::make_pair("buffer_cp4to8(buf,i,sbuf,si2)", "{buf[i]=uvec4(sbuf[si2.r],sbuf[si2.g]);}"));
        custom_defines.push_back(std::make_pair("buffer_ld8(buf,i)", "mat2x4(vec4(unpackHalf2x16(buf[i].r),unpackHalf2x16(buf[i].g)),vec4(unpackHalf2x16(buf[i].b),unpackHalf2x16(buf[i].a)))"));
        custom_defines.push_back(std::make_pair("buffer_st8(buf,i,v)", "{buf[i]=uvec4(uvec2(packHalf2x16(v[0].rg),packHalf2x16(v[0].ba)),uvec2(packHalf2x16(v[1].rg),packHalf2x16(v[1].ba)));}"));
        custom_defines.push_back(std::make_pair("buffer_cp8(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp8to1(buf,i4,ii4,sbuf,si)", "{uvec4 _v=sbuf[si]; vec2 _v0=unpackHalf2x16(_v.r);vec2 _v1=unpackHalf2x16(_v.g);vec2 _v2=unpackHalf2x16(_v.b);vec2 _v3=unpackHalf2x16(_v.a); buf[i4.r]=_v0.r;buf[i4.g]=_v0.g;buf[i4.b]=_v1.r;buf[i4.a]=_v1.g; buf[ii4.r]=_v2.r;buf[ii4.g]=_v2.g;buf[ii4.b]=_v3.r;buf[ii4.a]=_v3.g;}"));
        custom_defines.push_back(std::make_pair("buffer_cp8to4(buf,i2,sbuf,si)", "{uvec4 _v=sbuf[si]; buf[i2.r]=_v.rg;buf[i2.g]=_v.ba;}"));
    }
    else
    {
        custom_defines.push_back(std::make_pair("buffer_ld1(buf,i)", "buf[i]"));
        custom_defines.push_back(std::make_pair("buffer_st1(buf,i,v)", "{buf[i]=v;}"));
        custom_defines.push_back(std::make_pair("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i]=vec4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a]);}"));
        custom_defines.push_back(std::make_pair("buffer_cp1to8(buf,i,sbuf,si4,sii4)", "{buf[i]=mat2x4(sbuf[si4.r],sbuf[si4.g],sbuf[si4.b],sbuf[si4.a],sbuf[sii4.r],sbuf[sii4.g],sbuf[sii4.b],sbuf[sii4.a]);}"));
        custom_defines.push_back(std::make_pair("buffer_ld2(buf,i)", "buf[i]"));
        custom_defines.push_back(std::make_pair("buffer_st2(buf,i,v)", "{buf[i]=v;}"));
        custom_defines.push_back(std::make_pair("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_ld4(buf,i)", "buf[i]"));
        custom_defines.push_back(std::make_pair("buffer_st4(buf,i,v)", "{buf[i]=v;}"));
        custom_defines.push_back(std::make_pair("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp4to1(buf,i4,sbuf,si)", "{vec4 _v=sbuf[si]; buf[i4.r]=_v.r;buf[i4.g]=_v.g;buf[i4.b]=_v.b;buf[i4.a]=_v.a;}"));
        custom_defines.push_back(std::make_pair("buffer_cp4to8(buf,i,sbuf,si2)", "{buf[i]=mat2x4(sbuf[si2.r],sbuf[si2.g]);}"));
        custom_defines.push_back(std::make_pair("buffer_ld8(buf,i)", "buf[i]"));
        custom_defines.push_back(std::make_pair("buffer_st8(buf,i,v)", "{buf[i]=v;}"));
        custom_defines.push_back(std::make_pair("buffer_cp8(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}"));
        custom_defines.push_back(std::make_pair("buffer_cp8to1(buf,i4,ii4,sbuf,si)", "{mat2x4 _v=sbuf[si]; buf[i4.r]=_v[0].r;buf[i4.g]=_v[0].g;buf[i4.b]=_v[0].b;buf[i4.a]=_v[0].a; buf[ii4.r]=_v[1].r;buf[ii4.g]=_v[1].g;buf[ii4.b]=_v[1].b;buf[ii4.a]=_v[1].a;}"));
        custom_defines.push_back(std::make_pair("buffer_cp8to4(buf,i2,sbuf,si)", "{mat2x4 _v=sbuf[si]; buf[i2.r]=_v[0];buf[i2.g]=_v[1];}"));
        custom_defines.push_back(std::make_pair("sfp2afpmat4(v)", "v"));
        custom_defines.push_back(std::make_pair("afp2sfpmat4(v)", "v"));
    }

    if (opt.use_image_storage)
    {
        if (opt.use_fp16_storage)
        {
            custom_defines.push_back(std::make_pair("imfmtc1", "r16f"));
            custom_defines.push_back(std::make_pair("imfmtc4", "rgba16f"));
            custom_defines.push_back(std::make_pair("unfp", "mediump"));
        }
        else if (opt.use_fp16_packed)
        {
            custom_defines.push_back(std::make_pair("imfmtc1", "r32f"));
            custom_defines.push_back(std::make_pair("imfmtc4", "rgba16f"));
            custom_defines.push_back(std::make_pair("unfp", "mediump"));
        }
        else
        {
            custom_defines.push_back(std::make_pair("imfmtc1", "r32f"));
            custom_defines.push_back(std::make_pair("imfmtc4", "rgba32f"));
            custom_defines.push_back(std::make_pair("unfp", "highp"));
        }

        if (opt.use_fp16_storage && opt.use_fp16_arithmetic)
        {
            custom_defines.push_back(std::make_pair("image1d_ld1(tex,p)", "float16_t(texelFetch(tex,p,0).r)"));
            custom_defines.push_back(std::make_pair("image2d_ld1(tex,p)", "float16_t(texelFetch(tex,p,0).r)"));
            custom_defines.push_back(std::make_pair("image3d_ld1(tex,p)", "float16_t(texelFetch(tex,p,0).r)"));
            custom_defines.push_back(std::make_pair("image1d_st1(img,p,v)", "{f16vec4 _v;_v.r=float16_t(v);imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image2d_st1(img,p,v)", "{f16vec4 _v;_v.r=float16_t(v);imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image3d_st1(img,p,v)", "{f16vec4 _v;_v.r=float16_t(v);imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image1d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image1d_ld4(tex,p)", "f16vec4(texelFetch(tex,p,0))"));
            custom_defines.push_back(std::make_pair("image2d_ld4(tex,p)", "f16vec4(texelFetch(tex,p,0))"));
            custom_defines.push_back(std::make_pair("image3d_ld4(tex,p)", "f16vec4(texelFetch(tex,p,0))"));
            custom_defines.push_back(std::make_pair("image1d_st4(img,p,v)", "{imageStore(img,p,vec4(v));}"));
            custom_defines.push_back(std::make_pair("image2d_st4(img,p,v)", "{imageStore(img,p,vec4(v));}"));
            custom_defines.push_back(std::make_pair("image3d_st4(img,p,v)", "{imageStore(img,p,vec4(v));}"));
            custom_defines.push_back(std::make_pair("image1d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image1d_ld8(tex,p)", "f16mat2x4(texelFetch(tex,(p)*2,0),texelFetch(tex,(p)*2+1,0))"));
            custom_defines.push_back(std::make_pair("image2d_ld8(tex,p)", "f16mat2x4(texelFetch(tex,ivec2(p.x*2,p.y),0),texelFetch(tex,ivec2(p.x*2+1,p.y),0))"));
            custom_defines.push_back(std::make_pair("image3d_ld8(tex,p)", "f16mat2x4(texelFetch(tex,ivec3(p.x*2,p.y,p.z),0),texelFetch(tex,ivec3(p.x*2+1,p.y,p.z),0))"));
            custom_defines.push_back(std::make_pair("image1d_st8(img,p,v)", "{imageStore(img,(p)*2,vec4(v[0]));imageStore(img,(p)*2+1,vec4(v[1]));}"));
            custom_defines.push_back(std::make_pair("image2d_st8(img,p,v)", "{imageStore(img,ivec2(p.x*2,p.y),vec4(v[0]));imageStore(img,ivec2(p.x*2+1,p.y),vec4(v[1]));}"));
            custom_defines.push_back(std::make_pair("image3d_st8(img,p,v)", "{imageStore(img,ivec3(p.x*2,p.y,p.z),vec4(v[0]));imageStore(img,ivec3(p.x*2+1,p.y,p.z),vec4(v[1]));}"));
            custom_defines.push_back(std::make_pair("image1d_cp8(img,p,tex,sp)", "{imageStore(img,(p)*2,texelFetch(tex,sp*2,0));imageStore(img,(p)*2+1,texelFetch(tex,sp*2+1,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp8(img,p,tex,sp)", "{imageStore(img,ivec2(p.x*2,p.y),texelFetch(tex,ivec2(sp.x*2,sp.y),0));imageStore(img,ivec2(p.x*2+1,p.y),texelFetch(tex,ivec2(sp.x*2+1,sp.y),0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp8(img,p,tex,sp)", "{imageStore(img,ivec3(p.x*2,p.y,p.z),texelFetch(tex,ivec3(sp.x*2,sp.y,sp.z),0));imageStore(img,ivec3(p.x*2+1,p.y,p.z),texelFetch(tex,ivec3(sp.x*2+1,sp.y,sp.z),0));}"));
        }
        else if (opt.use_fp16_packed && opt.use_fp16_arithmetic)
        {
            custom_defines.push_back(std::make_pair("image1d_ld1(tex,p)", "float16_t(texelFetch(tex,p,0).r)"));
            custom_defines.push_back(std::make_pair("image2d_ld1(tex,p)", "float16_t(texelFetch(tex,p,0).r)"));
            custom_defines.push_back(std::make_pair("image3d_ld1(tex,p)", "float16_t(texelFetch(tex,p,0).r)"));
            custom_defines.push_back(std::make_pair("image1d_st1(img,p,v)", "{vec4 _v;_v.r=v;imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image2d_st1(img,p,v)", "{vec4 _v;_v.r=v;imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image3d_st1(img,p,v)", "{vec4 _v;_v.r=v;imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image1d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image1d_ld4(tex,p)", "f16vec4(texelFetch(tex,p,0))"));
            custom_defines.push_back(std::make_pair("image2d_ld4(tex,p)", "f16vec4(texelFetch(tex,p,0))"));
            custom_defines.push_back(std::make_pair("image3d_ld4(tex,p)", "f16vec4(texelFetch(tex,p,0))"));
            custom_defines.push_back(std::make_pair("image1d_st4(img,p,v)", "{imageStore(img,p,v);}"));
            custom_defines.push_back(std::make_pair("image2d_st4(img,p,v)", "{imageStore(img,p,v);}"));
            custom_defines.push_back(std::make_pair("image3d_st4(img,p,v)", "{imageStore(img,p,v);}"));
            custom_defines.push_back(std::make_pair("image1d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image1d_ld8(tex,p)", "f16mat2x4(texelFetch(tex,(p)*2,0),texelFetch(tex,(p)*2+1,0))"));
            custom_defines.push_back(std::make_pair("image2d_ld8(tex,p)", "f16mat2x4(texelFetch(tex,ivec2(p.x*2,p.y),0),texelFetch(tex,ivec2(p.x*2+1,p.y),0))"));
            custom_defines.push_back(std::make_pair("image3d_ld8(tex,p)", "f16mat2x4(texelFetch(tex,ivec3(p.x*2,p.y,p.z),0),texelFetch(tex,ivec3(p.x*2+1,p.y,p.z),0))"));
            custom_defines.push_back(std::make_pair("image1d_st8(img,p,v)", "{imageStore(img,(p)*2,v[0]);imageStore(img,(p)*2+1,v[1]);}"));
            custom_defines.push_back(std::make_pair("image2d_st8(img,p,v)", "{imageStore(img,ivec2(p.x*2,p.y),v[0]);imageStore(img,ivec2(p.x*2+1,p.y),v[1]);}"));
            custom_defines.push_back(std::make_pair("image3d_st8(img,p,v)", "{imageStore(img,ivec3(p.x*2,p.y,p.z),v[0]);imageStore(img,ivec3(p.x*2+1,p.y,p.z),v[1]);}"));
            custom_defines.push_back(std::make_pair("image1d_cp8(img,p,tex,sp)", "{imageStore(img,(p)*2,texelFetch(tex,sp*2,0));imageStore(img,(p)*2+1,texelFetch(tex,sp*2+1,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp8(img,p,tex,sp)", "{imageStore(img,ivec2(p.x*2,p.y),texelFetch(tex,ivec2(sp.x*2,sp.y),0));imageStore(img,ivec2(p.x*2+1,p.y),texelFetch(tex,ivec2(sp.x*2+1,sp.y),0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp8(img,p,tex,sp)", "{imageStore(img,ivec3(p.x*2,p.y,p.z),texelFetch(tex,ivec3(sp.x*2,sp.y,sp.z),0));imageStore(img,ivec3(p.x*2+1,p.y,p.z),texelFetch(tex,ivec3(sp.x*2+1,sp.y,sp.z),0));}"));
        }
        else if (opt.use_fp16_storage)
        {
            custom_defines.push_back(std::make_pair("image1d_ld1(tex,p)", "texelFetch(tex,p,0).r"));
            custom_defines.push_back(std::make_pair("image2d_ld1(tex,p)", "texelFetch(tex,p,0).r"));
            custom_defines.push_back(std::make_pair("image3d_ld1(tex,p)", "texelFetch(tex,p,0).r"));
            custom_defines.push_back(std::make_pair("image1d_st1(img,p,v)", "{vec4 _v;_v.r=v;imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image2d_st1(img,p,v)", "{vec4 _v;_v.r=v;imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image3d_st1(img,p,v)", "{vec4 _v;_v.r=v;imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image1d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image1d_ld4(tex,p)", "texelFetch(tex,p,0)"));
            custom_defines.push_back(std::make_pair("image2d_ld4(tex,p)", "texelFetch(tex,p,0)"));
            custom_defines.push_back(std::make_pair("image3d_ld4(tex,p)", "texelFetch(tex,p,0)"));
            custom_defines.push_back(std::make_pair("image1d_st4(img,p,v)", "{imageStore(img,p,v);}"));
            custom_defines.push_back(std::make_pair("image2d_st4(img,p,v)", "{imageStore(img,p,v);}"));
            custom_defines.push_back(std::make_pair("image3d_st4(img,p,v)", "{imageStore(img,p,v);}"));
            custom_defines.push_back(std::make_pair("image1d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image1d_ld8(tex,p)", "mat2x4(texelFetch(tex,(p)*2,0),texelFetch(tex,(p)*2+1,0))"));
            custom_defines.push_back(std::make_pair("image2d_ld8(tex,p)", "mat2x4(texelFetch(tex,ivec2(p.x*2,p.y),0),texelFetch(tex,ivec2(p.x*2+1,p.y),0))"));
            custom_defines.push_back(std::make_pair("image3d_ld8(tex,p)", "mat2x4(texelFetch(tex,ivec3(p.x*2,p.y,p.z),0),texelFetch(tex,ivec3(p.x*2+1,p.y,p.z),0))"));
            custom_defines.push_back(std::make_pair("image1d_st8(img,p,v)", "{imageStore(img,(p)*2,v[0]);imageStore(img,(p)*2+1,v[1]);}"));
            custom_defines.push_back(std::make_pair("image2d_st8(img,p,v)", "{imageStore(img,ivec2(p.x*2,p.y),v[0]);imageStore(img,ivec2(p.x*2+1,p.y),v[1]);}"));
            custom_defines.push_back(std::make_pair("image3d_st8(img,p,v)", "{imageStore(img,ivec3(p.x*2,p.y,p.z),v[0]);imageStore(img,ivec3(p.x*2+1,p.y,p.z),v[1]);}"));
            custom_defines.push_back(std::make_pair("image1d_cp8(img,p,tex,sp)", "{imageStore(img,(p)*2,texelFetch(tex,sp*2,0));imageStore(img,(p)*2+1,texelFetch(tex,sp*2+1,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp8(img,p,tex,sp)", "{imageStore(img,ivec2(p.x*2,p.y),texelFetch(tex,ivec2(sp.x*2,sp.y),0));imageStore(img,ivec2(p.x*2+1,p.y),texelFetch(tex,ivec2(sp.x*2+1,sp.y),0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp8(img,p,tex,sp)", "{imageStore(img,ivec3(p.x*2,p.y,p.z),texelFetch(tex,ivec3(sp.x*2,sp.y,sp.z),0));imageStore(img,ivec3(p.x*2+1,p.y,p.z),texelFetch(tex,ivec3(sp.x*2+1,sp.y,sp.z),0));}"));
        }
        else if (opt.use_fp16_packed)
        {
            custom_defines.push_back(std::make_pair("image1d_ld1(tex,p)", "texelFetch(tex,p,0).r"));
            custom_defines.push_back(std::make_pair("image2d_ld1(tex,p)", "texelFetch(tex,p,0).r"));
            custom_defines.push_back(std::make_pair("image3d_ld1(tex,p)", "texelFetch(tex,p,0).r"));
            custom_defines.push_back(std::make_pair("image1d_st1(img,p,v)", "{vec4 _v;_v.r=v;imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image2d_st1(img,p,v)", "{vec4 _v;_v.r=v;imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image3d_st1(img,p,v)", "{vec4 _v;_v.r=v;imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image1d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image1d_ld4(tex,p)", "texelFetch(tex,p,0)"));
            custom_defines.push_back(std::make_pair("image2d_ld4(tex,p)", "texelFetch(tex,p,0)"));
            custom_defines.push_back(std::make_pair("image3d_ld4(tex,p)", "texelFetch(tex,p,0)"));
            custom_defines.push_back(std::make_pair("image1d_st4(img,p,v)", "{imageStore(img,p,v);}"));
            custom_defines.push_back(std::make_pair("image2d_st4(img,p,v)", "{imageStore(img,p,v);}"));
            custom_defines.push_back(std::make_pair("image3d_st4(img,p,v)", "{imageStore(img,p,v);}"));
            custom_defines.push_back(std::make_pair("image1d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image1d_ld8(tex,p)", "mat2x4(texelFetch(tex,(p)*2,0),texelFetch(tex,(p)*2+1,0))"));
            custom_defines.push_back(std::make_pair("image2d_ld8(tex,p)", "mat2x4(texelFetch(tex,ivec2(p.x*2,p.y),0),texelFetch(tex,ivec2(p.x*2+1,p.y),0))"));
            custom_defines.push_back(std::make_pair("image3d_ld8(tex,p)", "mat2x4(texelFetch(tex,ivec3(p.x*2,p.y,p.z),0),texelFetch(tex,ivec3(p.x*2+1,p.y,p.z),0))"));
            custom_defines.push_back(std::make_pair("image1d_st8(img,p,v)", "{imageStore(img,(p)*2,v[0]);imageStore(img,(p)*2+1,v[1]);}"));
            custom_defines.push_back(std::make_pair("image2d_st8(img,p,v)", "{imageStore(img,ivec2(p.x*2,p.y),v[0]);imageStore(img,ivec2(p.x*2+1,p.y),v[1]);}"));
            custom_defines.push_back(std::make_pair("image3d_st8(img,p,v)", "{imageStore(img,ivec3(p.x*2,p.y,p.z),v[0]);imageStore(img,ivec3(p.x*2+1,p.y,p.z),v[1]);}"));
            custom_defines.push_back(std::make_pair("image1d_cp8(img,p,tex,sp)", "{imageStore(img,(p)*2,texelFetch(tex,sp*2,0));imageStore(img,(p)*2+1,texelFetch(tex,sp*2+1,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp8(img,p,tex,sp)", "{imageStore(img,ivec2(p.x*2,p.y),texelFetch(tex,ivec2(sp.x*2,sp.y),0));imageStore(img,ivec2(p.x*2+1,p.y),texelFetch(tex,ivec2(sp.x*2+1,sp.y),0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp8(img,p,tex,sp)", "{imageStore(img,ivec3(p.x*2,p.y,p.z),texelFetch(tex,ivec3(sp.x*2,sp.y,sp.z),0));imageStore(img,ivec3(p.x*2+1,p.y,p.z),texelFetch(tex,ivec3(sp.x*2+1,sp.y,sp.z),0));}"));
        }
        else
        {
            custom_defines.push_back(std::make_pair("image1d_ld1(tex,p)", "texelFetch(tex,p,0).r"));
            custom_defines.push_back(std::make_pair("image2d_ld1(tex,p)", "texelFetch(tex,p,0).r"));
            custom_defines.push_back(std::make_pair("image3d_ld1(tex,p)", "texelFetch(tex,p,0).r"));
            custom_defines.push_back(std::make_pair("image1d_st1(img,p,v)", "{vec4 _v;_v.r=v;imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image2d_st1(img,p,v)", "{vec4 _v;_v.r=v;imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image3d_st1(img,p,v)", "{vec4 _v;_v.r=v;imageStore(img,p,_v);}"));
            custom_defines.push_back(std::make_pair("image1d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp1(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image1d_ld4(tex,p)", "texelFetch(tex,p,0)"));
            custom_defines.push_back(std::make_pair("image2d_ld4(tex,p)", "texelFetch(tex,p,0)"));
            custom_defines.push_back(std::make_pair("image3d_ld4(tex,p)", "texelFetch(tex,p,0)"));
            custom_defines.push_back(std::make_pair("image1d_st4(img,p,v)", "{imageStore(img,p,v);}"));
            custom_defines.push_back(std::make_pair("image2d_st4(img,p,v)", "{imageStore(img,p,v);}"));
            custom_defines.push_back(std::make_pair("image3d_st4(img,p,v)", "{imageStore(img,p,v);}"));
            custom_defines.push_back(std::make_pair("image1d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp4(img,p,tex,sp)", "{imageStore(img,p,texelFetch(tex,sp,0));}"));
            custom_defines.push_back(std::make_pair("image1d_ld8(tex,p)", "mat2x4(texelFetch(tex,(p)*2,0),texelFetch(tex,(p)*2+1,0))"));
            custom_defines.push_back(std::make_pair("image2d_ld8(tex,p)", "mat2x4(texelFetch(tex,ivec2(p.x*2,p.y),0),texelFetch(tex,ivec2(p.x*2+1,p.y),0))"));
            custom_defines.push_back(std::make_pair("image3d_ld8(tex,p)", "mat2x4(texelFetch(tex,ivec3(p.x*2,p.y,p.z),0),texelFetch(tex,ivec3(p.x*2+1,p.y,p.z),0))"));
            custom_defines.push_back(std::make_pair("image1d_st8(img,p,v)", "{imageStore(img,(p)*2,v[0]);imageStore(img,(p)*2+1,v[1]);}"));
            custom_defines.push_back(std::make_pair("image2d_st8(img,p,v)", "{imageStore(img,ivec2(p.x*2,p.y),v[0]);imageStore(img,ivec2(p.x*2+1,p.y),v[1]);}"));
            custom_defines.push_back(std::make_pair("image3d_st8(img,p,v)", "{imageStore(img,ivec3(p.x*2,p.y,p.z),v[0]);imageStore(img,ivec3(p.x*2+1,p.y,p.z),v[1]);}"));
            custom_defines.push_back(std::make_pair("image1d_cp8(img,p,tex,sp)", "{imageStore(img,(p)*2,texelFetch(tex,sp*2,0));imageStore(img,(p)*2+1,texelFetch(tex,sp*2+1,0));}"));
            custom_defines.push_back(std::make_pair("image2d_cp8(img,p,tex,sp)", "{imageStore(img,ivec2(p.x*2,p.y),texelFetch(tex,ivec2(sp.x*2,sp.y),0));imageStore(img,ivec2(p.x*2+1,p.y),texelFetch(tex,ivec2(sp.x*2+1,sp.y),0));}"));
            custom_defines.push_back(std::make_pair("image3d_cp8(img,p,tex,sp)", "{imageStore(img,ivec3(p.x*2,p.y,p.z),texelFetch(tex,ivec3(sp.x*2,sp.y,sp.z),0));imageStore(img,ivec3(p.x*2+1,p.y,p.z),texelFetch(tex,ivec3(sp.x*2+1,sp.y,sp.z),0));}"));
        }
    }

    custom_defines.push_back(std::make_pair("psc(x)", "(x==0?p.x:x)"));

    if (opt.use_fp16_storage)
    {
        custom_defines.push_back(std::make_pair("NCNN_fp16_storage", "1"));
    }
    else if (opt.use_fp16_packed)
    {
        custom_defines.push_back(std::make_pair("NCNN_fp16_packed", "1"));
    }

    if (opt.use_fp16_arithmetic)
    {
        custom_defines.push_back(std::make_pair("NCNN_fp16_arithmetic", "1"));
    }

    if (opt.use_int8_storage)
    {
        custom_defines.push_back(std::make_pair("NCNN_int8_storage", "1"));
    }

    if (opt.use_int8_arithmetic)
    {
        custom_defines.push_back(std::make_pair("NCNN_int8_arithmetic", "1"));
    }

    if (opt.use_image_storage)
    {
        custom_defines.push_back(std::make_pair("NCNN_image_shader", "1"));
    }

    if (opt.use_subgroup_basic)
    {
        custom_defines.push_back(std::make_pair("NCNN_subgroup_basic", "1"));

        if (opt.use_subgroup_vote)
        {
            custom_defines.push_back(std::make_pair("NCNN_subgroup_vote", "1"));
        }
        if (opt.use_subgroup_ballot)
        {
            custom_defines.push_back(std::make_pair("NCNN_subgroup_ballot", "1"));
        }
        if (opt.use_subgroup_shuffle)
        {
            custom_defines.push_back(std::make_pair("NCNN_subgroup_shuffle", "1"));
        }
    }

    std::string preamble;
    std::vector<std::string> processes;

    processes.resize(custom_defines.size());
    for (size_t i = 0; i < custom_defines.size(); i++)
    {
        const char* key = custom_defines[i].first;
        const char* def = custom_defines[i].second;

        preamble += std::string("#define ") + key + " " + def + "\n";
        processes[i] = std::string("define-macro ") + key + "=" + def;
    }

    bool compile_success = true;

    {
        glslang::TShader s(EShLangCompute);

        s.setStringsWithLengths(&comp_data, &comp_data_size, 1);

        s.setPreamble(preamble.c_str());
        s.addProcesses(processes);
        s.setEntryPoint("main");
        s.setSourceEntryPoint("main");

        s.setEnvInput(glslang::EShSourceGlsl, EShLangCompute, glslang::EShClientVulkan, 1);

        if (opt.use_subgroup_basic)
        {
            // subgroup need vulkan-1.1 and spirv-1.3
            s.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
            s.setEnvTarget(glslang::EshTargetSpv, glslang::EShTargetSpv_1_3);
        }
        else
        {
            s.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
            s.setEnvTarget(glslang::EshTargetSpv, glslang::EShTargetSpv_1_0);
        }

        TBuiltInResource resources = get_default_TBuiltInResource();

        bool pr = s.parse(&resources, 100, false, EShMsgDefault);
        if (!pr)
        {
            NCNN_LOGE("compile spir-v module failed");
            NCNN_LOGE("%s", s.getInfoLog());
            NCNN_LOGE("%s", s.getInfoDebugLog());

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
