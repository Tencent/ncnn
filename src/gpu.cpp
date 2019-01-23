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
#include <vector>

#include "mat.h"

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
static GpuInfo g_gpu_infos[8];

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

        if ((queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT) && !(queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
        {
            return i;
        }
    }

    // second try, any queue with compute
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

static uint32_t find_device_transfer_queue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties)
{
    // first try, transfer only queue
    for (uint32_t i=0; i<queueFamilyProperties.size(); i++)
    {
        const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];

        if ((queueFamilyProperty.queueFlags & VK_QUEUE_TRANSFER_BIT) && !(queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT) && !(queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT))
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

//     fprintf(stderr, "no transfer queue\n");
    return -1;
}

static uint32_t find_unified_memory(VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties)
{
    // first try, host visible + host coherent + device local
    for (uint32_t i=0; i<physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

        if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            && (memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            && (memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        {
            return i;
        }
    }

    // second try, host visible + device local
    for (uint32_t i=0; i<physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

        if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            && (memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        {
            return i;
        }
    }

//     fprintf(stderr, "no unified memory\n");
    return -1;
}

static uint32_t find_device_local_memory(VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties)
{
    // first try, device local only
    for (uint32_t i=0; i<physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

        if (memoryType.propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        {
            return i;
        }
    }

    // second try, with device local bit
    for (uint32_t i=0; i<physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

        if (memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        {
            return i;
        }
    }

//     fprintf(stderr, "no device local memory\n");
    return -1;
}

static uint32_t find_host_visible_memory(VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties)
{
    // first try, host visible + host coherent, without device local bit
    for (uint32_t i=0; i<physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

        if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            && (memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
            && !(memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        {
            return i;
        }
    }

    // second try, with host visible bit, without device local bit
    for (uint32_t i=0; i<physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

        if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            && !(memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        {
            return i;
        }
    }

    // third try, with host visible bit
    for (uint32_t i=0; i<physicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

        if (memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        {
            return i;
        }
    }

//     fprintf(stderr, "no host visible memory\n");
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

    for (uint32_t j=0; j<instanceExtensionPropertyCount; j++)
    {
        const VkExtensionProperties& exp = instanceExtensionProperties[j];
//         fprintf(stderr, "instance extension %s = %u\n", exp.extensionName, exp.specVersion);

        if (strcmp(exp.extensionName, "VK_KHR_get_physical_device_properties2") == 0)
        {
            enabledExtensions.push_back("VK_KHR_get_physical_device_properties2");
        }
#if ENABLE_VALIDATION_LAYER
        if (strcmp(exp.extensionName, "VK_EXT_debug_utils") == 0)
        {
            enabledExtensions.push_back("VK_EXT_debug_utils");
        }
#endif // ENABLE_VALIDATION_LAYER
    }

    VkApplicationInfo applicationInfo;
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pNext = 0;
    applicationInfo.pApplicationName = "ncnn";
    applicationInfo.applicationVersion = 0;
    applicationInfo.pEngineName = "ncnn";
    applicationInfo.engineVersion = 20181026;
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
    VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    createInfo.pUserData = 0;
    ret = CreateDebugUtilsMessengerEXT(g_instance, &createInfo, nullptr, &callback);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "CreateDebugUtilsMessengerEXT failed %d\n", ret);
        return -1;
    }
#endif // ENABLE_VALIDATION_LAYER

    uint32_t physicalDeviceCount = 0;
    ret = vkEnumeratePhysicalDevices(g_instance, &physicalDeviceCount, 0);
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkEnumeratePhysicalDevices failed %d\n", ret);
        return -1;
    }

    // NOTE 8 is large enough i think ...
    if (physicalDeviceCount > 8)
        physicalDeviceCount = 8;

    std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);

    ret = vkEnumeratePhysicalDevices(g_instance, &physicalDeviceCount, physicalDevices.data());
    if (ret != VK_SUCCESS)
    {
        fprintf(stderr, "vkEnumeratePhysicalDevices failed %d\n", ret);
        return -1;
    }

    g_gpu_count = physicalDeviceCount;

    // find proper device and queue
    for (uint32_t i=0; i<physicalDeviceCount; i++)
    {
        const VkPhysicalDevice& physicalDevice = physicalDevices[i];
        GpuInfo& gpu_info = g_gpu_infos[i];

        gpu_info.physical_device = physicalDevice;

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

//         fprintf(stderr, "[%u] max_shared_memory_size = %d\n", i, gpu_info.max_shared_memory_size);
//         fprintf(stderr, "[%u] max_workgroup_count = %d %d %d\n", i, gpu_info.max_workgroup_count[0], gpu_info.max_workgroup_count[1], gpu_info.max_workgroup_count[2]);
//         fprintf(stderr, "[%u] max_workgroup_invocations = %d\n", i, gpu_info.max_workgroup_invocations);
//         fprintf(stderr, "[%u] max_workgroup_size = %d %d %d\n", i, gpu_info.max_workgroup_size[0], gpu_info.max_workgroup_size[1], gpu_info.max_workgroup_size[2]);
//         fprintf(stderr, "[%u] memory_map_alignment = %lu\n", i, gpu_info.memory_map_alignment);
//         fprintf(stderr, "[%u] buffer_offset_alignment = %lu\n", i, gpu_info.buffer_offset_alignment);

//         // TODO check features
//         VkPhysicalDeviceFeatures features;
//         vkGetPhysicalDeviceFeatures(physicalDevice, &features);
//
//         // TODO check formatProperties
//         VkFormat format = VK_FORMAT_R32_SFLOAT;
//         VkFormatProperties formatProperties;
//         vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProperties);

        // find compute queue
        uint32_t queueFamilyPropertiesCount;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, 0);

        std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertiesCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, queueFamilyProperties.data());

        gpu_info.compute_queue_index = find_device_compute_queue(queueFamilyProperties);
        gpu_info.transfer_queue_index = find_device_transfer_queue(queueFamilyProperties);

        // find memory type index
        VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemoryProperties);

//         // print memory info
//         for (uint32_t j=0; j<physicalDeviceMemoryProperties.memoryTypeCount; j++)
//         {
//             const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[j];
//             fprintf(stderr, "[%u] memoryType %u heapIndex/propertyFlags = %d  %u\n", i, j, memoryType.heapIndex, memoryType.propertyFlags);
//         }
//         for (uint32_t j=0; j<physicalDeviceMemoryProperties.memoryHeapCount; j++)
//         {
//             const VkMemoryHeap& memoryHeap = physicalDeviceMemoryProperties.memoryHeaps[j];
//             fprintf(stderr, "[%u] memoryHeap %u size/flags = %lu  %u\n", i, j, memoryHeap.size, memoryHeap.flags);
//         }

        gpu_info.unified_memory_index = find_unified_memory(physicalDeviceMemoryProperties);
        gpu_info.device_local_memory_index = find_device_local_memory(physicalDeviceMemoryProperties);
        gpu_info.host_visible_memory_index = find_host_visible_memory(physicalDeviceMemoryProperties);

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
        gpu_info.support_VK_KHR_get_memory_requirements2 = 0;
        gpu_info.support_VK_KHR_get_physical_device_properties2 = 0;
        gpu_info.support_VK_KHR_push_descriptor = 0;
        gpu_info.support_VK_KHR_shader_float16_int8 = 0;
        gpu_info.support_VK_KHR_shader_float_controls = 0;
        gpu_info.support_VK_KHR_storage_buffer_storage_class = 0;
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
            else if (strcmp(exp.extensionName, "VK_KHR_get_memory_requirements2") == 0)
                gpu_info.support_VK_KHR_get_memory_requirements2 = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_get_physical_device_properties2") == 0)
                gpu_info.support_VK_KHR_get_physical_device_properties2 = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_push_descriptor") == 0)
                gpu_info.support_VK_KHR_push_descriptor = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_shader_float16_int8") == 0)
                gpu_info.support_VK_KHR_shader_float16_int8 = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_shader_float_controls") == 0)
                gpu_info.support_VK_KHR_shader_float_controls = exp.specVersion;
            else if (strcmp(exp.extensionName, "VK_KHR_storage_buffer_storage_class") == 0)
                gpu_info.support_VK_KHR_storage_buffer_storage_class = exp.specVersion;
        }

//         fprintf(stderr, "[%u] VK_KHR_8bit_storage                       = %d\n", i, gpu_info.support_VK_KHR_8bit_storage);
//         fprintf(stderr, "[%u] VK_KHR_16bit_storage                      = %d\n", i, gpu_info.support_VK_KHR_16bit_storage);
//         fprintf(stderr, "[%u] VK_KHR_bind_memory2                       = %d\n", i, gpu_info.support_VK_KHR_bind_memory2);
//         fprintf(stderr, "[%u] VK_KHR_dedicated_allocation               = %d\n", i, gpu_info.support_VK_KHR_dedicated_allocation);
//         fprintf(stderr, "[%u] VK_KHR_descriptor_update_template         = %d\n", i, gpu_info.support_VK_KHR_descriptor_update_template);
//         fprintf(stderr, "[%u] VK_KHR_get_memory_requirements2           = %d\n", i, gpu_info.support_VK_KHR_get_memory_requirements2);
//         fprintf(stderr, "[%u] VK_KHR_get_physical_device_properties2    = %d\n", i, gpu_info.support_VK_KHR_get_physical_device_properties2);
//         fprintf(stderr, "[%u] VK_KHR_push_descriptor                    = %d\n", i, gpu_info.support_VK_KHR_push_descriptor);
//         fprintf(stderr, "[%u] VK_KHR_shader_float16_int8                = %d\n", i, gpu_info.support_VK_KHR_shader_float16_int8);
//         fprintf(stderr, "[%u] VK_KHR_shader_float_controls              = %d\n", i, gpu_info.support_VK_KHR_shader_float_controls);
//         fprintf(stderr, "[%u] VK_KHR_storage_buffer_storage_class       = %d\n", i, gpu_info.support_VK_KHR_storage_buffer_storage_class);

        fprintf(stderr, "[%u %s]  queueC=%u  queueT=%u  memU=%u  memDL=%u  memHV=%u\n", i, physicalDeviceProperties.deviceName,
                gpu_info.compute_queue_index, gpu_info.transfer_queue_index,
                gpu_info.unified_memory_index, gpu_info.device_local_memory_index, gpu_info.host_visible_memory_index);
    }

    // the default gpu device
    g_default_gpu_index = find_default_vulkan_device_index();

    return 0;
}

void destroy_gpu_instance()
{
#if ENABLE_VALIDATION_LAYER
    DestroyDebugUtilsMessengerEXT(g_instance, callback, NULL);
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

struct layer_shader_registry_entry
{
    const char* name;
    const uint32_t* spv_data;
    size_t spv_data_size;
};

#include "layer_shader_spv_data.h"

static const layer_shader_registry_entry layer_shader_registry[] =
{
#include "layer_shader_registry.h"
};

static const int layer_shader_registry_entry_count = sizeof(layer_shader_registry) / sizeof(layer_shader_registry_entry);

VulkanDevice::VulkanDevice(int device_index) : info(g_gpu_infos[device_index])
{
    const float queuePriorities[1] = { 1.f };// 0.f ~ 1.f

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
    if (info.support_VK_KHR_get_memory_requirements2)
        enabledExtensions.push_back("VK_KHR_get_memory_requirements2");
    if (info.support_VK_KHR_get_physical_device_properties2)
        enabledExtensions.push_back("VK_KHR_get_physical_device_properties2");
    if (info.support_VK_KHR_push_descriptor)
        enabledExtensions.push_back("VK_KHR_push_descriptor");
    if (info.support_VK_KHR_shader_float16_int8)
        enabledExtensions.push_back("VK_KHR_shader_float16_int8");
    if (info.support_VK_KHR_shader_float_controls)
        enabledExtensions.push_back("VK_KHR_shader_float_controls");
    if (info.support_VK_KHR_storage_buffer_storage_class)
        enabledExtensions.push_back("VK_KHR_storage_buffer_storage_class");

    VkDeviceQueueCreateInfo deviceQueueCreateInfos[2];
    deviceQueueCreateInfos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    deviceQueueCreateInfos[0].pNext = 0;
    deviceQueueCreateInfos[0].flags = 0;
    deviceQueueCreateInfos[0].queueFamilyIndex = info.compute_queue_index;
    deviceQueueCreateInfos[0].queueCount = 1;
    deviceQueueCreateInfos[0].pQueuePriorities = queuePriorities;
    deviceQueueCreateInfos[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    deviceQueueCreateInfos[1].pNext = 0;
    deviceQueueCreateInfos[1].flags = 0;
    deviceQueueCreateInfos[1].queueFamilyIndex = info.transfer_queue_index;
    deviceQueueCreateInfos[1].queueCount = 1;
    deviceQueueCreateInfos[1].pQueuePriorities = queuePriorities;

    VkDeviceCreateInfo deviceCreateInfo;
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pNext = 0;
    deviceCreateInfo.flags = 0;
    if (info.compute_queue_index == info.transfer_queue_index)
    {
    deviceCreateInfo.queueCreateInfoCount = 1;
    }
    else
    {
    deviceCreateInfo.queueCreateInfoCount = 2;
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

    blob_buffer_allocator = new VkBlobBufferAllocator(this);
    staging_buffer_allocator = new VkStagingBufferAllocator(this);
}

VulkanDevice::~VulkanDevice()
{
    delete blob_buffer_allocator;
    delete staging_buffer_allocator;

    destroy_shader_module();

    vkDestroyDevice(device, 0);
}

VkShaderModule VulkanDevice::get_shader_module(const char* name) const
{
    for (int i=0; i<layer_shader_registry_entry_count; i++)
    {
        if (strcmp(layer_shader_registry[i].name, name) == 0)
            return shader_modules[i];
    }

    fprintf(stderr, "no such shader module %s\n", name);
    return 0;
}

VkAllocator* VulkanDevice::allocator() const
{
    return blob_buffer_allocator;
}

VkAllocator* VulkanDevice::staging_allocator() const
{
    return staging_buffer_allocator;
}

int VulkanDevice::create_shader_module()
{
    shader_modules.resize(layer_shader_registry_entry_count, VK_NULL_HANDLE);

    for (int i=0; i<layer_shader_registry_entry_count; i++)
    {
        VkShaderModuleCreateInfo shaderModuleCreateInfo;
        shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderModuleCreateInfo.pNext = 0;
        shaderModuleCreateInfo.flags = 0;
        shaderModuleCreateInfo.codeSize = layer_shader_registry[i].spv_data_size;
        shaderModuleCreateInfo.pCode = layer_shader_registry[i].spv_data;

        VkResult ret = vkCreateShaderModule(device, &shaderModuleCreateInfo, 0, &shader_modules[i]);
        if (ret != VK_SUCCESS)
        {
            fprintf(stderr, "vkCreateShaderModule %s failed %d\n", layer_shader_registry[i].name, ret);
            return -1;
        }

//         fprintf(stderr, "shader_module %s created\n", layer_shader_registry[i].name);
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
    if (info.support_VK_KHR_descriptor_update_template)
    {
        vkCreateDescriptorUpdateTemplateKHR = (PFN_vkCreateDescriptorUpdateTemplateKHR)vkGetDeviceProcAddr(device, "vkCreateDescriptorUpdateTemplateKHR");
        vkDestroyDescriptorUpdateTemplateKHR = (PFN_vkDestroyDescriptorUpdateTemplateKHR)vkGetDeviceProcAddr(device, "vkDestroyDescriptorUpdateTemplateKHR");
        vkUpdateDescriptorSetWithTemplateKHR = (PFN_vkUpdateDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(device, "vkUpdateDescriptorSetWithTemplateKHR");

//         fprintf(stderr, "vkCreateDescriptorUpdateTemplateKHR = %p\n", vkCreateDescriptorUpdateTemplateKHR);
//         fprintf(stderr, "vkDestroyDescriptorUpdateTemplateKHR = %p\n", vkDestroyDescriptorUpdateTemplateKHR);
//         fprintf(stderr, "vkUpdateDescriptorSetWithTemplateKHR = %p\n", vkUpdateDescriptorSetWithTemplateKHR);
    }

    if (info.support_VK_KHR_get_memory_requirements2)
    {
        vkGetImageMemoryRequirements2KHR = (PFN_vkGetImageMemoryRequirements2KHR)vkGetDeviceProcAddr(device, "vkGetImageMemoryRequirements2KHR");
        vkGetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2KHR)vkGetDeviceProcAddr(device, "vkGetBufferMemoryRequirements2KHR");
        vkGetImageSparseMemoryRequirements2KHR = (PFN_vkGetImageSparseMemoryRequirements2KHR)vkGetDeviceProcAddr(device, "vkGetImageSparseMemoryRequirements2KHR");

//         fprintf(stderr, "vkGetImageMemoryRequirements2KHR = %p\n", vkGetImageMemoryRequirements2KHR);
//         fprintf(stderr, "vkGetBufferMemoryRequirements2KHR = %p\n", vkGetBufferMemoryRequirements2KHR);
//         fprintf(stderr, "vkGetImageSparseMemoryRequirements2KHR = %p\n", vkGetImageSparseMemoryRequirements2KHR);
    }

    if (info.support_VK_KHR_push_descriptor)
    {
        if (info.support_VK_KHR_descriptor_update_template)
        {
            vkCmdPushDescriptorSetWithTemplateKHR = (PFN_vkCmdPushDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetWithTemplateKHR");
//             fprintf(stderr, "vkCmdPushDescriptorSetWithTemplateKHR = %p\n", vkCmdPushDescriptorSetWithTemplateKHR);
        }

        vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");
//         fprintf(stderr, "vkCmdPushDescriptorSetKHR = %p\n", vkCmdPushDescriptorSetKHR);
    }

    return 0;
}

} // namespace ncnn

#endif // NCNN_VULKAN
