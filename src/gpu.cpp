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

#include <vulkan/vulkan.h>

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

#include "mat.h"
#include "command.h"
#include "layer_type.h"
#include "layer.h"

#if __ANDROID__
#define ENABLE_VALIDATION_LAYER 0
#else
#define ENABLE_VALIDATION_LAYER 0
#endif

namespace ncnn {

// global
static VkInstance g_instance = 0;
static int g_gpu_count = 0;
static int g_default_gpu_index = -1;

// NOTE 8 is large enough i think ...
#define NCNN_MAX_GPU_COUNT 8
static GpuInfo g_gpu_infos[NCNN_MAX_GPU_COUNT];

// default vulkan device
static Mutex g_default_vkdev_lock;
static VulkanDevice* g_default_vkdev[NCNN_MAX_GPU_COUNT] = {0};

// precompiled spirv
struct layer_shader_registry_entry
{
    const uint32_t* spv_data;
    size_t spv_data_size;
};

#include "layer_shader_spv_data.h"

static const layer_shader_registry_entry layer_shader_registry[] =
{
#include "layer_shader_registry.h"
};

static ShaderInfo layer_shader_infos[sizeof(layer_shader_registry) / sizeof(layer_shader_registry_entry)];

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
#if VK_HEADER_VERSION < 80
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR (VkStructureType)1000177000
typedef struct VkPhysicalDevice8BitStorageFeaturesKHR {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           storageBuffer8BitAccess;
    VkBool32           uniformAndStorageBuffer8BitAccess;
    VkBool32           storagePushConstant8;
} VkPhysicalDevice8BitStorageFeaturesKHR;
#endif // VK_HEADER_VERSION < 80
#if VK_HEADER_VERSION < 95
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR (VkStructureType)1000082000
typedef struct VkPhysicalDeviceFloat16Int8FeaturesKHR {
    VkStructureType    sType;
    void*              pNext;
    VkBool32           shaderFloat16;
    VkBool32           shaderInt8;
} VkPhysicalDeviceFloat16Int8FeaturesKHR;
#endif // VK_HEADER_VERSION < 95

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
        vkGetPhysicalDeviceSurfaceCapabilities2KHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilities2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceCapabilities2KHR");;
        vkGetPhysicalDeviceSurfaceFormats2KHR = (PFN_vkGetPhysicalDeviceSurfaceFormats2KHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceFormats2KHR");;
    }

    if (support_VK_KHR_surface)
    {
        vkDestroySurfaceKHR = (PFN_vkDestroySurfaceKHR)vkGetInstanceProcAddr(g_instance, "vkDestroySurfaceKHR");;
        vkGetPhysicalDeviceSurfaceSupportKHR = (PFN_vkGetPhysicalDeviceSurfaceSupportKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceSupportKHR");;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR = (PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");;
        vkGetPhysicalDeviceSurfaceFormatsKHR = (PFN_vkGetPhysicalDeviceSurfaceFormatsKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfaceFormatsKHR");;
        vkGetPhysicalDeviceSurfacePresentModesKHR = (PFN_vkGetPhysicalDeviceSurfacePresentModesKHR)vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceSurfacePresentModesKHR");;
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
    fprintf(stderr, "validation layer: %s\n", pCallbackData->pMessage);

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
    for (uint32_t i=0; i<queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if ((queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
            && !(queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
        {
            return i;
        }
    }

    // second try, any queue with compute and graphics
    for (uint32_t i=0; i<queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if ((queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
            && (queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
        {
            return i;
        }
    }

    // third try, any queue with compute
    for (uint32_t i=0; i<queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if (queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT)
        {
            return i;
        }
    }

//     fprintf(stderr, "no compute queue\n");
    return -1;
}

static uint32_t find_device_graphics_queue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties)
{
    // first try, graphics only queue
    for (uint32_t i=0; i<queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if ((queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            && !(queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT))
        {
            return i;
        }
    }

    // second try, any queue with graphics and compute
    for (uint32_t i=0; i<queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if ((queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            && (queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT))
        {
            return i;
        }
    }

    // third try, any queue with graphics
    for (uint32_t i=0; i<queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if (queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            return i;
        }
    }

//     fprintf(stderr, "no graphics queue\n");
    return -1;
}

static uint32_t find_device_transfer_queue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties)
{
    // first try, transfer only queue
    for (uint32_t i=0; i<queueFamilyProperties.size(); i++)
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
    for (uint32_t i=0; i<queueFamilyProperties.size(); i++)
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

//     fprintf(stderr, "no transfer queue\n");
    return -1;
}

static int find_default_vulkan_device_index()
{
    // first try, discrete gpu
    for (int i=0; i<g_gpu_count; i++)
    {
        if (g_gpu_infos[i].type == 0)
            return i;
    }

    // second try, integrated gpu
    for (int i=0; i<g_gpu_count; i++)
    {
        if (g_gpu_infos[i].type == 1)
            return i;
    }

    // third try, any probed device
    if (g_gpu_count > 0)
        return 0;

    fprintf(stderr, "no vulkan device\n");
    return -1;
}

int create_gpu_instance()
{
    VkResult ret;

    std::vector<const char*> enabledLayers;

#if ENABLE_VALIDATION_LAYER
    uint32_t instanceLayerPropertyCount;
    ret = vkEnumerateInstanceLayerProperties(&instanceLayerPropertyCount, NULL);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkEnumerateInstanceLayerProperties failed %d\n", ret);
        return -1;
    }

    std::vector<VkLayerProperties> instanceLayerProperties(instanceLayerPropertyCount);
    ret = vkEnumerateInstanceLayerProperties(&instanceLayerPropertyCount, instanceLayerProperties.data());
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkEnumerateInstanceLayerProperties failed %d\n", ret);
        return -1;
    }

    for (uint32_t i=0; i<instanceLayerPropertyCount; i++)
    {
        const VkLayerProperties& lp = instanceLayerProperties[i];
//         fprintf(stderr, "instance layer %s = %u\n", lp.layerName, lp.implementationVersion);

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
        fprintf(stderr, "vkEnumerateInstanceExtensionProperties failed %d\n", ret);
        return -1;
    }

    std::vector<VkExtensionProperties> instanceExtensionProperties(instanceExtensionPropertyCount);
    ret = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionPropertyCount, instanceExtensionProperties.data());
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkEnumerateInstanceExtensionProperties failed %d\n", ret);
        return -1;
    }

    support_VK_KHR_get_physical_device_properties2 = 0;
    support_VK_KHR_get_surface_capabilities2 = 0;
    support_VK_KHR_surface = 0;
    support_VK_EXT_debug_utils = 0;
#if __ANDROID_API__ >= 26
    support_VK_KHR_android_surface = 0;
#endif // __ANDROID_API__ >= 26
    for (uint32_t j=0; j<instanceExtensionPropertyCount; j++)
    {
        const VkExtensionProperties& exp = instanceExtensionProperties[j];
//         fprintf(stderr, "instance extension %s = %u\n", exp.extensionName, exp.specVersion);

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

    VkApplicationInfo applicationInfo;
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pNext = 0;
    applicationInfo.pApplicationName = "ncnn";
    applicationInfo.applicationVersion = 0;
    applicationInfo.pEngineName = "ncnn";
    applicationInfo.engineVersion = 20200413;
    applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

    VkInstanceCreateInfo instanceCreateInfo;
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pNext = 0;
    instanceCreateInfo.flags = 0;
    instanceCreateInfo.pApplicationInfo = &applicationInfo;
    instanceCreateInfo.enabledLayerCount = enabledLayers.size();
    instanceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
    instanceCreateInfo.enabledExtensionCount = enabledExtensions.size();
    instanceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();

    ret = vkCreateInstance(&instanceCreateInfo, 0, &g_instance);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateInstance failed %d\n", ret);
        return -1;
    }

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
            fprintf(stderr, "CreateDebugUtilsMessengerEXT failed %d\n", ret);
            return -1;
        }
    }
#endif // ENABLE_VALIDATION_LAYER

    init_instance_extension();

    uint32_t physicalDeviceCount = 0;
    ret = vkEnumeratePhysicalDevices(g_instance, &physicalDeviceCount, 0);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkEnumeratePhysicalDevices failed %d\n", ret);
        return -1;
    }

    if (physicalDeviceCount > NCNN_MAX_GPU_COUNT)
        physicalDeviceCount = NCNN_MAX_GPU_COUNT;

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);

    ret = vkEnumeratePhysicalDevices(g_instance, &physicalDeviceCount, physicalDevices.data());
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkEnumeratePhysicalDevices failed %d\n", ret);
        return -1;
    }

    // find proper device and queue
    int gpu_info_index = 0;
    for (uint32_t i=0; i<physicalDeviceCount; i++)
    {
        const VkPhysicalDevice& physicalDevice = physicalDevices[i];
        GpuInfo& gpu_info = g_gpu_infos[gpu_info_index];

        // device type
        VkPhysicalDeviceProperties physicalDeviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

//         fprintf(stderr, "[%u] apiVersion = %u.%u.%u\n", i, VK_VERSION_MAJOR(physicalDeviceProperties.apiVersion),
//             VK_VERSION_MINOR(physicalDeviceProperties.apiVersion), VK_VERSION_PATCH(physicalDeviceProperties.apiVersion));
//         fprintf(stderr, "[%u] driverVersion = %u.%u.%u\n", i, VK_VERSION_MAJOR(physicalDeviceProperties.driverVersion),
//             VK_VERSION_MINOR(physicalDeviceProperties.driverVersion), VK_VERSION_PATCH(physicalDeviceProperties.driverVersion));
//         fprintf(stderr, "[%u] vendorID = %x\n", i, physicalDeviceProperties.vendorID);
//         fprintf(stderr, "[%u] deviceID = %x\n", i, physicalDeviceProperties.deviceID);
//         fprintf(stderr, "[%u] deviceType = %x\n", i, physicalDeviceProperties.deviceType);
//         fprintf(stderr, "[%u] deviceName = %s\n", i, physicalDeviceProperties.deviceName);
//         fprintf(stderr, "[%u] pipelineCacheUUID = %u\n", i, physicalDeviceProperties.pipelineCacheUUID);

        gpu_info.bug_local_size_spec_const = false;
        gpu_info.bug_implicit_fp16_arithmetic = false;

        if (physicalDeviceProperties.vendorID == 0x13b5 && physicalDeviceProperties.apiVersion < VK_MAKE_VERSION(1, 0, 66))
        {
            // arm mali with old buggy driver
            gpu_info.bug_local_size_spec_const = true;
        }

        if (physicalDeviceProperties.vendorID == 0x5143 && physicalDeviceProperties.apiVersion < VK_MAKE_VERSION(1, 0, 49))
        {
            // qcom adreno with old buggy driver
            gpu_info.bug_local_size_spec_const = true;
        }

        if (physicalDeviceProperties.vendorID == 0x13b5 && (physicalDeviceProperties.deviceID == 0x7500001 || physicalDeviceProperties.deviceID == 0x8602000))
        {
            // TODO enable devices other than rk3288/rk3399
            // arm mali driver accept spirv with fp16 arithmetic
            gpu_info.bug_implicit_fp16_arithmetic = true;
        }

        if (physicalDeviceProperties.vendorID == 0x5143 && (physicalDeviceProperties.deviceID == 0x6030001 || physicalDeviceProperties.deviceID == 0x6040001))
        {
            // TODO enable devices other than qcom855/qcom855plus
            // qcom adreno driver accept spirv with fp16 arithmetic
            gpu_info.bug_implicit_fp16_arithmetic = true;
        }

        gpu_info.physical_device = physicalDevice;

        // info
        gpu_info.api_version = physicalDeviceProperties.apiVersion;
        gpu_info.driver_version = physicalDeviceProperties.driverVersion;
        gpu_info.vendor_id = physicalDeviceProperties.vendorID;
        gpu_info.device_id = physicalDeviceProperties.deviceID;
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

//         fprintf(stderr, "[%u] max_shared_memory_size = %u\n", i, gpu_info.max_shared_memory_size);
//         fprintf(stderr, "[%u] max_workgroup_count = %u %u %u\n", i, gpu_info.max_workgroup_count[0], gpu_info.max_workgroup_count[1], gpu_info.max_workgroup_count[2]);
//         fprintf(stderr, "[%u] max_workgroup_invocations = %u\n", i, gpu_info.max_workgroup_invocations);
//         fprintf(stderr, "[%u] max_workgroup_size = %u %u %u\n", i, gpu_info.max_workgroup_size[0], gpu_info.max_workgroup_size[1], gpu_info.max_workgroup_size[2]);
//         fprintf(stderr, "[%u] memory_map_alignment = %lu\n", i, gpu_info.memory_map_alignment);
//         fprintf(stderr, "[%u] buffer_offset_alignment = %lu\n", i, gpu_info.buffer_offset_alignment);

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

        // cache memory properties
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &gpu_info.physicalDeviceMemoryProperties);

        // get device extension
        uint32_t deviceExtensionPropertyCount = 0;
        ret = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionPropertyCount, NULL);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkEnumerateDeviceExtensionProperties failed %d\n", ret);
            return -1;
        }

        std::vector<VkExtensionProperties> deviceExtensionProperties(deviceExtensionPropertyCount);
        ret = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionPropertyCount, deviceExtensionProperties.data());
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkEnumerateDeviceExtensionProperties failed %d\n", ret);
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
        for (uint32_t j=0; j<deviceExtensionPropertyCount; j++)
        {
            const VkExtensionProperties& exp = deviceExtensionProperties[j];
//             fprintf(stderr, "device extension %s = %u\n", exp.extensionName, exp.specVersion);

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
            queryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR,
            queryFeatures.pNext = queryExtensionFeatures;

            vkGetPhysicalDeviceFeatures2KHR(physicalDevice, &queryFeatures);

            if (gpu_info.support_VK_KHR_8bit_storage)
            {
                gpu_info.support_int8_storage = query8BitStorageFeatures.storageBuffer8BitAccess && query8BitStorageFeatures.uniformAndStorageBuffer8BitAccess;
            }
            if (gpu_info.support_VK_KHR_16bit_storage)
            {
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

        // check format
        gpu_info.support_image_storage = false;
        gpu_info.support_image_fp16_packed = false;
        gpu_info.support_image_fp16_storage = false;
        gpu_info.support_image_fp16_arithmetic = false;
        {
            VkFormatProperties r32f_formatProperties;
            VkFormatProperties rgba32f_formatProperties;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R32_SFLOAT, &r32f_formatProperties);
            vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R32G32B32A32_SFLOAT, &rgba32f_formatProperties);

            if ((r32f_formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
                && (r32f_formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
                && (rgba32f_formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
                && (rgba32f_formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
                gpu_info.support_image_storage = true;
        }
        {
            VkFormatProperties rgba16f_formatProperties;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R16G16B16A16_SFLOAT, &rgba16f_formatProperties);

            if ((rgba16f_formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
                && (rgba16f_formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
                gpu_info.support_image_fp16_packed = true;
        }
        {
            VkFormatProperties r16f_formatProperties;
            VkFormatProperties rgba16f_formatProperties;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R16_SFLOAT, &r16f_formatProperties);
            vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R16G16B16A16_SFLOAT, &rgba16f_formatProperties);

            if ((r16f_formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
                && (r16f_formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
                && (rgba16f_formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
                && (rgba16f_formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
                gpu_info.support_image_fp16_storage = true;
        }
        if (gpu_info.support_fp16_arithmetic)
        {
            gpu_info.support_image_fp16_arithmetic = true;
        }

        if (physicalDeviceProperties.vendorID == 0x1ae0 && physicalDeviceProperties.deviceID == 0xc0de)
        {
            // swiftshader image r16f is not supported
            gpu_info.support_image_fp16_storage = false;
        }

        fprintf(stderr, "[%u %s]  queueC=%u[%u]  queueG=%u[%u]  queueT=%u[%u]\n", i, physicalDeviceProperties.deviceName,
                gpu_info.compute_queue_family_index, gpu_info.compute_queue_count,
                gpu_info.graphics_queue_family_index, gpu_info.graphics_queue_count,
                gpu_info.transfer_queue_family_index, gpu_info.transfer_queue_count);

        fprintf(stderr, "[%u %s]  buglssc=%d  bugihfa=%d\n", i, physicalDeviceProperties.deviceName,
                gpu_info.bug_local_size_spec_const, gpu_info.bug_implicit_fp16_arithmetic);

        fprintf(stderr, "[%u %s]  fp16p=%d  fp16s=%d  fp16a=%d  int8s=%d  int8a=%d\n", i, physicalDeviceProperties.deviceName,
                gpu_info.support_fp16_packed, gpu_info.support_fp16_storage, gpu_info.support_fp16_arithmetic,
                gpu_info.support_int8_storage, gpu_info.support_int8_arithmetic);

        fprintf(stderr, "[%u %s]  imgfp32=%d  imgfp16p=%d  imgfp16s=%d  imgfp16a=%d\n", i, physicalDeviceProperties.deviceName,
                gpu_info.support_image_storage, gpu_info.support_image_fp16_packed,
                gpu_info.support_image_fp16_storage, gpu_info.support_image_fp16_arithmetic);

        gpu_info_index++;
    }

    g_gpu_count = gpu_info_index;

    // the default gpu device
    g_default_gpu_index = find_default_vulkan_device_index();

    // resolve shader info
    for (int i=0; i<layer_shader_registry_entry_count; i++)
    {
        resolve_shader_info(layer_shader_registry[i].spv_data, layer_shader_registry[i].spv_data_size, layer_shader_infos[i]);
    }

    return 0;
}

void destroy_gpu_instance()
{
    for (int i=0; i<NCNN_MAX_GPU_COUNT; i++)
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
}

int get_gpu_count()
{
    return g_gpu_count;
}

int get_default_gpu_index()
{
    return g_default_gpu_index;
}

const GpuInfo& get_gpu_info(int device_index)
{
    return g_gpu_infos[device_index];
}

VulkanDevice::VulkanDevice(int device_index) : info(g_gpu_infos[device_index])
{
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

    std::vector<float> compute_queue_priorities(info.compute_queue_count, 1.f);// 0.f ~ 1.f
    std::vector<float> graphics_queue_priorities(info.graphics_queue_count, 1.f);// 0.f ~ 1.f
    std::vector<float> transfer_queue_priorities(info.transfer_queue_count, 1.f);// 0.f ~ 1.f

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
    deviceCreateInfo.pEnabledFeatures = 0;// VkPhysicalDeviceFeatures pointer

    VkResult ret = vkCreateDevice(info.physical_device, &deviceCreateInfo, 0, &device);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkCreateDevice failed %d\n", ret);
    }

    init_device_extension();

    create_shader_module();

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
            fprintf(stderr, "vkCreateSampler failed %d\n", ret);
        }
    }

    create_utility_operator();
}

VulkanDevice::~VulkanDevice()
{
    destroy_utility_operator();

    if (texelfetch_sampler)
    {
        vkDestroySampler(device, texelfetch_sampler, 0);
    }

    for (uint32_t i = 0; i < info.compute_queue_count; i++)
    {
        delete blob_allocators[i];
        delete staging_allocators[i];
    }
    blob_allocators.clear();
    staging_allocators.clear();

    destroy_shader_module();

    vkDestroyDevice(device, 0);
}

VkShaderModule VulkanDevice::get_shader_module(int shader_type_index) const
{
    if (shader_type_index < 0 || shader_type_index >= layer_shader_registry_entry_count)
    {
        fprintf(stderr, "no such shader module %d\n", shader_type_index);
        return 0;
    }

    return shader_modules[shader_type_index];
}

VkShaderModule VulkanDevice::create_shader_module(int shader_type_index, uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z) const
{
    if (shader_type_index < 0 || shader_type_index >= layer_shader_registry_entry_count)
    {
        fprintf(stderr, "no such shader module %d\n", shader_type_index);
        return 0;
    }

    const uint32_t* spv_data = layer_shader_registry[shader_type_index].spv_data;
    size_t spv_data_size = layer_shader_registry[shader_type_index].spv_data_size;

    return compile_shader_module(spv_data, spv_data_size, local_size_x, local_size_y, local_size_z);
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
        fprintf(stderr, "vkCreateShaderModule failed %d\n", ret);
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

uint32_t VulkanDevice::find_memory_index(uint32_t memory_type_bits, VkFlags required, VkFlags preferred, VkFlags preferred_not) const
{
    // first try, find required and with preferred and without preferred_not
    for (uint32_t i=0; i<info.physicalDeviceMemoryProperties.memoryTypeCount; i++)
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
    for (uint32_t i=0; i<info.physicalDeviceMemoryProperties.memoryTypeCount; i++)
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
    for (uint32_t i=0; i<info.physicalDeviceMemoryProperties.memoryTypeCount; i++)
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
    for (uint32_t i=0; i<info.physicalDeviceMemoryProperties.memoryTypeCount; i++)
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

    fprintf(stderr, "no such memory type %u %u %u %u\n", memory_type_bits, required, preferred, preferred_not);
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
        fprintf(stderr, "invalid queue_family_index %u\n", queue_family_index);
        return 0;
    }

    MutexLockGuard lock(queue_lock);

    std::vector<VkQueue>& queues = queue_family_index == info.compute_queue_family_index ? compute_queues
                                 : queue_family_index == info.graphics_queue_family_index ? graphics_queues : transfer_queues;
    for (int i=0; i<(int)queues.size(); i++)
    {
        VkQueue queue = queues[i];
        if (queue)
        {
            queues[i] = 0;
            return queue;
        }
    }

    // out of hardware queue
    return 0;
}

void VulkanDevice::reclaim_queue(uint32_t queue_family_index, VkQueue queue) const
{
    if (queue_family_index != info.compute_queue_family_index
        && queue_family_index != info.graphics_queue_family_index
        && queue_family_index != info.transfer_queue_family_index)
    {
        fprintf(stderr, "invalid queue_family_index %u\n", queue_family_index);
        return;
    }

    MutexLockGuard lock(queue_lock);

    std::vector<VkQueue>& queues = queue_family_index == info.compute_queue_family_index ? compute_queues
                                 : queue_family_index == info.graphics_queue_family_index ? graphics_queues : transfer_queues;
    for (int i=0; i<(int)queues.size(); i++)
    {
        if (!queues[i])
        {
            queues[i] = queue;
            return;
        }
    }

    fprintf(stderr, "FATAL ERROR! reclaim_queue get wild queue %u %p\n", queue_family_index, queue);
}

VkAllocator* VulkanDevice::acquire_blob_allocator() const
{
    MutexLockGuard lock(blob_allocator_lock);

    for (int i=0; i<(int)blob_allocators.size(); i++)
    {
        VkAllocator* allocator = blob_allocators[i];
        if (allocator)
        {
            blob_allocators[i] = 0;
            return allocator;
        }
    }

    // out of blob allocator
    return 0;
}

void VulkanDevice::reclaim_blob_allocator(VkAllocator* allocator) const
{
    MutexLockGuard lock(blob_allocator_lock);

    for (int i=0; i<(int)blob_allocators.size(); i++)
    {
        if (!blob_allocators[i])
        {
            blob_allocators[i] = allocator;
            return;
        }
    }

    fprintf(stderr, "FATAL ERROR! reclaim_blob_allocator get wild allocator %p\n", allocator);
}

VkAllocator* VulkanDevice::acquire_staging_allocator() const
{
    MutexLockGuard lock(staging_allocator_lock);

    for (int i=0; i<(int)staging_allocators.size(); i++)
    {
        VkAllocator* allocator = staging_allocators[i];
        if (allocator)
        {
            staging_allocators[i] = 0;
            return allocator;
        }
    }

    // out of staging allocator
    return 0;
}

void VulkanDevice::reclaim_staging_allocator(VkAllocator* allocator) const
{
    MutexLockGuard lock(staging_allocator_lock);

    for (int i=0; i<(int)staging_allocators.size(); i++)
    {
        if (!staging_allocators[i])
        {
            staging_allocators[i] = allocator;
            return;
        }
    }

    fprintf(stderr, "FATAL ERROR! reclaim_staging_allocator get wild allocator %p\n", allocator);
}

const VkSampler* VulkanDevice::immutable_texelfetch_sampler() const
{
    return &texelfetch_sampler;
}

void VulkanDevice::cast_float32_to_float16(const VkMat& src, VkMat& dst, VkCompute& cmd, const Option& opt) const
{
    int uoi = opt.use_fp16_storage ? 2 : opt.use_fp16_packed ? 1 : 0;
    uop_cast_float32_to_float16[uoi]->forward(src, dst, cmd, opt);
}

void VulkanDevice::cast_float32_to_float16(const VkImageMat& src, VkImageMat& dst, VkCompute& cmd, const Option& opt) const
{
    int uoi = opt.use_image_fp16_storage ? 5 : opt.use_image_fp16_packed ? 4 : 3;
    uop_cast_float32_to_float16[uoi]->forward(src, dst, cmd, opt);
}

void VulkanDevice::cast_float16_to_float32(const VkMat& src, VkMat& dst, VkCompute& cmd, const Option& opt) const
{
    int uoi = opt.use_fp16_storage ? 2 : opt.use_fp16_packed ? 1 : 0;
    uop_cast_float16_to_float32[uoi]->forward(src, dst, cmd, opt);
}

void VulkanDevice::cast_float16_to_float32(const VkImageMat& src, VkImageMat& dst, VkCompute& cmd, const Option& opt) const
{
    int uoi = opt.use_image_fp16_storage ? 5 : opt.use_image_fp16_packed ? 4 : 3;
    uop_cast_float16_to_float32[uoi]->forward(src, dst, cmd, opt);
}

void VulkanDevice::packing_pack1(const VkMat& src, VkMat& dst, VkCompute& cmd, const Option& opt) const
{
    int uoi = opt.use_fp16_storage ? 2 : opt.use_fp16_packed ? 1 : 0;
    uop_packing_pack1[uoi]->forward(src, dst, cmd, opt);
}

void VulkanDevice::packing_pack1(const VkImageMat& src, VkImageMat& dst, VkCompute& cmd, const Option& opt) const
{
    int uoi = opt.use_image_fp16_storage ? 5 : opt.use_image_fp16_packed ? 4 : 3;
    uop_packing_pack1[uoi]->forward(src, dst, cmd, opt);
}

void VulkanDevice::packing_pack4(const VkMat& src, VkMat& dst, VkCompute& cmd, const Option& opt) const
{
    int uoi = opt.use_fp16_storage ? 2 : opt.use_fp16_packed ? 1 : 0;
    uop_packing_pack4[uoi]->forward(src, dst, cmd, opt);
}

void VulkanDevice::packing_pack4(const VkImageMat& src, VkImageMat& dst, VkCompute& cmd, const Option& opt) const
{
    int uoi = opt.use_image_fp16_storage ? 5 : opt.use_image_fp16_packed ? 4 : 3;
    uop_packing_pack4[uoi]->forward(src, dst, cmd, opt);
}

void VulkanDevice::packing_pack8(const VkMat& src, VkMat& dst, VkCompute& cmd, const Option& opt) const
{
    int uoi = opt.use_fp16_storage ? 2 : opt.use_fp16_packed ? 1 : 0;
    uop_packing_pack8[uoi]->forward(src, dst, cmd, opt);
}

void VulkanDevice::packing_pack8(const VkImageMat& src, VkImageMat& dst, VkCompute& cmd, const Option& opt) const
{
    int uoi = opt.use_image_fp16_storage ? 5 : opt.use_image_fp16_packed ? 4 : 3;
    uop_packing_pack8[uoi]->forward(src, dst, cmd, opt);
}

int VulkanDevice::create_shader_module()
{
    if (info.bug_local_size_spec_const)
    {
        // do not cache shader module
        return 0;
    }

    shader_modules.resize(layer_shader_registry_entry_count, VK_NULL_HANDLE);

    for (int i=0; i<layer_shader_registry_entry_count; i++)
    {
        // ncnn_add_shader cmake macro
        // 0 = fp32
        // 1 = fp16p
        // 2 = fp16pa
        // 3 = fp16s
        // 4 = fp16sa
        // 5 = image
        // 6 = image_fp16p
        // 7 = image_fp16s
        // 8 = image_fp16a

        if (!info.support_fp16_packed)
        {
            if (i % 9 == 1)
                continue;
        }

        if (!info.support_fp16_packed || !info.support_fp16_arithmetic)
        {
            if (i % 9 == 2)
                continue;
        }

        if (!info.support_fp16_storage)
        {
            if (i % 9 == 3)
                continue;
        }

        if (!info.support_fp16_storage || !info.support_fp16_arithmetic)
        {
            if (i % 9 == 4)
                continue;
        }

        if (!info.support_image_storage)
        {
            if (i % 9 == 5)
                continue;
        }

        if (!info.support_image_storage || !info.support_image_fp16_packed)
        {
            if (i % 9 == 6)
                continue;
        }

        if (!info.support_image_storage || !info.support_image_fp16_storage)
        {
            if (i % 9 == 7)
                continue;
        }

        if (!info.support_image_storage || !info.support_image_fp16_storage || !info.support_image_fp16_arithmetic)
        {
            if (i % 9 == 8)
                continue;
        }

        const uint32_t* spv_data = layer_shader_registry[i].spv_data;
        size_t spv_data_size = layer_shader_registry[i].spv_data_size;

        VkShaderModule shader_module = compile_shader_module(spv_data, spv_data_size);
        if (shader_module == 0)
        {
            fprintf(stderr, "compile_shader_module %d failed\n", i);
            return -1;
        }

        shader_modules[i] = shader_module;

//         fprintf(stderr, "shader_module %d created\n", i);
    }

    return 0;
}

void VulkanDevice::destroy_shader_module()
{
    for (int i=0; i<(int)shader_modules.size(); i++)
    {
        vkDestroyShaderModule(device, shader_modules[i], 0);
    }

    shader_modules.clear();
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

int VulkanDevice::create_utility_operator()
{
    Option opt[6];

    opt[0].use_fp16_packed = false;
    opt[0].use_fp16_storage = false;
    opt[0].use_image_storage = false;
    opt[0].use_image_fp16_packed = false;
    opt[0].use_image_fp16_storage = false;
    opt[0].use_shader_pack8 = true;

    opt[1].use_fp16_packed = true;
    opt[1].use_fp16_storage = false;
    opt[1].use_image_storage = false;
    opt[1].use_image_fp16_packed = false;
    opt[1].use_image_fp16_storage = false;
    opt[1].use_shader_pack8 = true;

    opt[2].use_fp16_packed = true;
    opt[2].use_fp16_storage = true;
    opt[2].use_image_storage = false;
    opt[2].use_image_fp16_packed = false;
    opt[2].use_image_fp16_storage = false;
    opt[2].use_shader_pack8 = true;

    opt[3].use_fp16_packed = false;
    opt[3].use_fp16_storage = false;
    opt[3].use_image_storage = true;
    opt[3].use_image_fp16_packed = false;
    opt[3].use_image_fp16_storage = false;
    opt[3].use_shader_pack8 = true;

    opt[4].use_fp16_packed = false;
    opt[4].use_fp16_storage = false;
    opt[4].use_image_storage = true;
    opt[4].use_image_fp16_packed = true;
    opt[4].use_image_fp16_storage = false;
    opt[4].use_shader_pack8 = true;

    opt[5].use_fp16_packed = false;
    opt[5].use_fp16_storage = false;
    opt[5].use_image_storage = true;
    opt[5].use_image_fp16_packed = true;
    opt[5].use_image_fp16_storage = true;
    opt[5].use_shader_pack8 = true;

    for (int i = 0; i < 6; i++)
    {
        uop_cast_float32_to_float16[i] = 0;
        uop_cast_float16_to_float32[i] = 0;
        uop_packing_pack1[i] = 0;
        uop_packing_pack4[i] = 0;
        uop_packing_pack8[i] = 0;

        if (i == 1 && !info.support_fp16_packed)
            continue;

        if (i == 2 && !info.support_fp16_storage)
            continue;

        if (i == 3 && !info.support_image_storage)
            continue;

        if (i == 4 && (!info.support_image_storage || !info.support_image_fp16_packed))
            continue;

        if (i == 5 && (!info.support_image_storage || !info.support_image_fp16_storage))
            continue;

        {
            uop_cast_float32_to_float16[i] = ncnn::create_layer(ncnn::LayerType::Cast);
            uop_cast_float32_to_float16[i]->vkdev = this;

            ncnn::ParamDict pd;
            pd.set(0, 1);
            pd.set(1, 2);

            uop_cast_float32_to_float16[i]->load_param(pd);
        }

        {
            uop_cast_float16_to_float32[i] = ncnn::create_layer(ncnn::LayerType::Cast);
            uop_cast_float16_to_float32[i]->vkdev = this;

            ncnn::ParamDict pd;
            pd.set(0, 2);
            pd.set(1, 1);

            uop_cast_float16_to_float32[i]->load_param(pd);
        }

        {
            uop_packing_pack1[i] = ncnn::create_layer(ncnn::LayerType::Packing);
            uop_packing_pack1[i]->vkdev = this;

            ncnn::ParamDict pd;
            pd.set(0, 1);

            uop_packing_pack1[i]->load_param(pd);
        }

        {
            uop_packing_pack4[i] = ncnn::create_layer(ncnn::LayerType::Packing);
            uop_packing_pack4[i]->vkdev = this;

            ncnn::ParamDict pd;
            pd.set(0, 4);

            uop_packing_pack4[i]->load_param(pd);
        }

        {
            uop_packing_pack8[i] = ncnn::create_layer(ncnn::LayerType::Packing);
            uop_packing_pack8[i]->vkdev = this;

            ncnn::ParamDict pd;
            pd.set(0, 8);

            uop_packing_pack8[i]->load_param(pd);
        }

        uop_cast_float32_to_float16[i]->create_pipeline(opt[i]);
        uop_cast_float16_to_float32[i]->create_pipeline(opt[i]);
        uop_packing_pack1[i]->create_pipeline(opt[i]);
        uop_packing_pack4[i]->create_pipeline(opt[i]);
        uop_packing_pack8[i]->create_pipeline(opt[i]);
    }

    return 0;
}

void VulkanDevice::destroy_utility_operator()
{
    Option opt[6];

    opt[0].use_fp16_packed = false;
    opt[0].use_fp16_storage = false;
    opt[0].use_image_storage = false;
    opt[0].use_image_fp16_packed = false;
    opt[0].use_image_fp16_storage = false;
    opt[0].use_shader_pack8 = true;

    opt[1].use_fp16_packed = true;
    opt[1].use_fp16_storage = false;
    opt[1].use_image_storage = false;
    opt[1].use_image_fp16_packed = false;
    opt[1].use_image_fp16_storage = false;
    opt[1].use_shader_pack8 = true;

    opt[2].use_fp16_packed = true;
    opt[2].use_fp16_storage = true;
    opt[2].use_image_storage = false;
    opt[2].use_image_fp16_packed = false;
    opt[2].use_image_fp16_storage = false;
    opt[2].use_shader_pack8 = true;

    opt[3].use_fp16_packed = false;
    opt[3].use_fp16_storage = false;
    opt[3].use_image_storage = true;
    opt[3].use_image_fp16_packed = false;
    opt[3].use_image_fp16_storage = false;
    opt[3].use_shader_pack8 = true;

    opt[4].use_fp16_packed = false;
    opt[4].use_fp16_storage = false;
    opt[4].use_image_storage = true;
    opt[4].use_image_fp16_packed = true;
    opt[4].use_image_fp16_storage = false;
    opt[4].use_shader_pack8 = true;

    opt[5].use_fp16_packed = false;
    opt[5].use_fp16_storage = false;
    opt[5].use_image_storage = true;
    opt[5].use_image_fp16_packed = true;
    opt[5].use_image_fp16_storage = true;
    opt[5].use_shader_pack8 = true;

    for (int i = 0; i < 6; i++)
    {
        if (i == 1 && !info.support_fp16_packed)
            continue;

        if (i == 2 && !info.support_fp16_storage)
            continue;

        if (i == 3 && !info.support_image_storage)
            continue;

        if (i == 4 && (!info.support_image_storage || !info.support_image_fp16_packed))
            continue;

        if (i == 5 && (!info.support_image_storage || !info.support_image_fp16_storage))
            continue;

        uop_cast_float32_to_float16[i]->destroy_pipeline(opt[i]);
        uop_cast_float16_to_float32[i]->destroy_pipeline(opt[i]);
        uop_packing_pack1[i]->destroy_pipeline(opt[i]);
        uop_packing_pack4[i]->destroy_pipeline(opt[i]);
        uop_packing_pack8[i]->destroy_pipeline(opt[i]);

        delete uop_cast_float32_to_float16[i];
        delete uop_cast_float16_to_float32[i];
        delete uop_packing_pack1[i];
        delete uop_packing_pack4[i];
        delete uop_packing_pack8[i];
    }
}

VulkanDevice* get_gpu_device(int device_index)
{
    if (device_index < 0 || device_index >= g_gpu_count)
        return 0;

    MutexLockGuard lock(g_default_vkdev_lock);

    if (!g_default_vkdev[device_index])
        g_default_vkdev[device_index] = new VulkanDevice(device_index);

    return g_default_vkdev[device_index];
}

const ShaderInfo& get_shader_info(int shader_type_index)
{
    if (shader_type_index < 0 || shader_type_index >= layer_shader_registry_entry_count)
    {
        fprintf(stderr, "no such shader module %d\n", shader_type_index);
        return layer_shader_infos[0];
    }

    return layer_shader_infos[shader_type_index];
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
        fprintf(stderr, "too many binding %d\n", binding_count);
        return -1;
    }

    shader_info.specialization_count = specialization_count;
    shader_info.binding_count = binding_count;
    shader_info.push_constant_count = push_constant_count;

    // resolve binding_types
    for (int i=0; i<binding_count; i++)
    {
        shader_info.binding_types[i] = id_types[ binding_types[i] ];
    }

    return 0;
}

} // namespace ncnn

#endif // NCNN_VULKAN
