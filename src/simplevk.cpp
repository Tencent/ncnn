// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "platform.h"

#if NCNN_VULKAN
#if NCNN_SIMPLEVK

#include "simplevk.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/types.h>
#include <unistd.h>
#if defined __ANDROID__
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#endif
#endif

#if __APPLE__

// always use static vulkan linkage on apple platform
extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName);

#endif

namespace ncnn {

// vulkan loader entrypoint
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = 0;

// vulkan global functions
PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties = 0;
PFN_vkCreateInstance vkCreateInstance = 0;
PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties = 0;

#if __APPLE__

int load_vulkan_driver(const char* /*driver_path*/)
{
    unload_vulkan_driver();

    vkGetInstanceProcAddr = ::vkGetInstanceProcAddr;
    vkEnumerateInstanceExtensionProperties = (PFN_vkEnumerateInstanceExtensionProperties)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceExtensionProperties");
    vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(NULL, "vkCreateInstance");
    vkEnumerateInstanceLayerProperties = (PFN_vkEnumerateInstanceLayerProperties)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceLayerProperties");
    return 0;
}

void unload_vulkan_driver()
{
    vkGetInstanceProcAddr = 0;
    vkEnumerateInstanceExtensionProperties = 0;
    vkCreateInstance = 0;
    vkEnumerateInstanceLayerProperties = 0;
}

#else // __APPLE__

#if defined _WIN32
static HMODULE g_libvulkan = 0;
#else
static void* g_libvulkan = 0;
#if defined __ANDROID__

struct hw_module_t;
struct hw_module_methods_t;
struct hw_device_t;

struct hw_module_methods_t
{
    /** Open a specific device */
    int (*open)(const hw_module_t* mod, const char* id, hw_device_t** device);
};

struct hw_device_t
{
    /** tag must be initialized to HARDWARE_DEVICE_TAG */
    uint32_t tag;
    uint32_t version;
    /** reference to the module this device belongs to */
    hw_module_t* mod;
    /** padding reserved for future use */
#ifdef __LP64__
    uint64_t reserved[12];
#else
    uint32_t reserved[12];
#endif
    /** Close this device */
    int (*close)(hw_device_t* device);
};

struct hw_module_t
{
    /** tag must be initialized to HARDWARE_MODULE_TAG */
    uint32_t tag;
    uint16_t module_api_version;
    uint16_t hal_api_version;
    const char* id;
    const char* name;
    const char* author;
    hw_module_methods_t* methods;
    void* dso;
#ifdef __LP64__
    uint64_t reserved[32 - 7];
#else
    /** padding to 128 bytes, reserved for future use */
    uint32_t reserved[32 - 7];
#endif
};

struct hwvulkan_module_t : public hw_module_t
{
};

struct hwvulkan_device_t : public hw_device_t
{
    PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties;
    PFN_vkCreateInstance CreateInstance;
    PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
};

// android hal vulkan loader
static hwvulkan_device_t* g_hvkdi = 0;
#endif
#endif

static std::string get_driver_path_from_icd(const char* icd_path)
{
    FILE* fp = fopen(icd_path, "rb");
    if (!fp)
        return std::string();

    std::string driver_path;

    char line[256];
    while (!feof(fp))
    {
        char* s = fgets(line, 256, fp);
        if (!s)
            break;

        // "library_path": "path to driver library",
        char path[256];
        int nscan = sscanf(line, " \"library_path\" : \"%255[^\"]\"", path);
        if (nscan == 1)
        {
            if (path[0] == '.' || (path[0] != '/' && !strchr(path, ':') && (strchr(path, '/') || strchr(path, '\\'))))
            {
                // relative to the icd file path
                std::string icd_dir = icd_path;
                size_t dirpos = icd_dir.find_last_of("/\\");
                if (dirpos != std::string::npos)
                {
                    icd_dir = icd_dir.substr(0, dirpos + 1);
                }
                else
                {
                    icd_dir = "./";
                }

                driver_path = icd_dir + path;
            }
            else
            {
                // filename or absolute path
                driver_path = path;
            }

            break;
        }
    }

    fclose(fp);

    return driver_path;
}

static std::string get_driver_path_from_icd_env()
{
    const char* icd_path = getenv("VK_ICD_FILENAMES");
    if (!icd_path)
        return std::string();

    return get_driver_path_from_icd(icd_path);
}

static std::string get_driver_path_from_ncnn_env()
{
    const char* driver_path = getenv("NCNN_VULKAN_DRIVER");
    if (!driver_path)
        return std::string();

    return std::string(driver_path);
}

#if defined _WIN32
static std::string search_file(const std::string& dirpath, const std::string& needle)
{
    WIN32_FIND_DATA file;
    HANDLE handle = FindFirstFileA((dirpath + "\\*").c_str(), &file);
    if (handle == INVALID_HANDLE_VALUE)
        return std::string();

    int found = 0;
    std::vector<std::string> subdirs;

    do
    {
        std::string name = file.cFileName;

        if (file.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
        {
            if (name != "." && name != "..")
                subdirs.push_back(name);
        }
        else if (name == needle)
        {
            found = 1;
            break;
        }
    } while (FindNextFileA(handle, &file));

    FindClose(handle);

    if (found)
        return dirpath + "\\" + needle;

    // recurse into subdirs
    for (int i = 0; i < subdirs.size(); ++i)
    {
        std::string found_path = search_file(dirpath + "\\" + subdirs[i], needle);
        if (!found_path.empty())
            return found_path;
    }

    return std::string();
}

static int load_vulkan_windows(const char* driver_path)
{
    const char* libpath = driver_path ? driver_path : "vulkan-1.dll";

    HMODULE libvulkan = LoadLibraryA(libpath);
    if (!libvulkan)
    {
        NCNN_LOGE("LoadLibraryA %s failed %d", libpath, GetLastError());
        return -1;
    }

    PFN_vkGetInstanceProcAddr GetInstanceProcAddr = 0;

    GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetProcAddress(libvulkan, "vk_icdGetInstanceProcAddr");
    if (GetInstanceProcAddr)
    {
        // load icd driver
        typedef VkResult(VKAPI_PTR * PFN_icdNegotiateLoaderICDInterfaceVersion)(uint32_t * pSupportedVersion);
        PFN_icdNegotiateLoaderICDInterfaceVersion icdNegotiateLoaderICDInterfaceVersion = (PFN_icdNegotiateLoaderICDInterfaceVersion)GetProcAddress(libvulkan, "vk_icdNegotiateLoaderICDInterfaceVersion");
        if (icdNegotiateLoaderICDInterfaceVersion)
        {
            uint32_t supported_version = 5;
            VkResult ret = icdNegotiateLoaderICDInterfaceVersion(&supported_version);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("icdNegotiateLoaderICDInterfaceVersion failed");
                FreeLibrary(libvulkan);
                return -1;
            }
        }
    }
    else
    {
        GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)GetProcAddress(libvulkan, "vkGetInstanceProcAddr");
        if (!GetInstanceProcAddr)
        {
            NCNN_LOGE("GetProcAddress failed %d", GetLastError());
            FreeLibrary(libvulkan);
            return -1;
        }
    }

    g_libvulkan = libvulkan;
    vkGetInstanceProcAddr = GetInstanceProcAddr;
    vkEnumerateInstanceExtensionProperties = (PFN_vkEnumerateInstanceExtensionProperties)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceExtensionProperties");
    vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(NULL, "vkCreateInstance");
    vkEnumerateInstanceLayerProperties = (PFN_vkEnumerateInstanceLayerProperties)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceLayerProperties");
    return 0;
}
#else
static int load_vulkan_linux(const char* driver_path)
{
#if __APPLE__
    const char* libpath = driver_path ? driver_path : "libvulkan.dylib";
#else
    const char* libpath = driver_path ? driver_path : "libvulkan.so";
#endif

    void* libvulkan = dlopen(libpath, RTLD_LOCAL | RTLD_NOW);
    if (!libvulkan)
    {
        NCNN_LOGE("dlopen failed %s", dlerror());
        return -1;
    }

    PFN_vkGetInstanceProcAddr GetInstanceProcAddr = 0;

    GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(libvulkan, "vk_icdGetInstanceProcAddr");
    if (GetInstanceProcAddr)
    {
        // load icd driver
        typedef VkResult(VKAPI_PTR * PFN_icdNegotiateLoaderICDInterfaceVersion)(uint32_t * pSupportedVersion);
        PFN_icdNegotiateLoaderICDInterfaceVersion icdNegotiateLoaderICDInterfaceVersion = (PFN_icdNegotiateLoaderICDInterfaceVersion)dlsym(libvulkan, "vk_icdNegotiateLoaderICDInterfaceVersion");
        if (icdNegotiateLoaderICDInterfaceVersion)
        {
            uint32_t supported_version = 5;
            VkResult ret = icdNegotiateLoaderICDInterfaceVersion(&supported_version);
            if (ret != VK_SUCCESS)
            {
                NCNN_LOGE("icdNegotiateLoaderICDInterfaceVersion failed");
                dlclose(libvulkan);
                return -1;
            }
        }
    }
    else
    {
        GetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(libvulkan, "vkGetInstanceProcAddr");
        if (!GetInstanceProcAddr)
        {
            NCNN_LOGE("dlsym failed %s", dlerror());
            dlclose(libvulkan);
            return -1;
        }
    }

    g_libvulkan = libvulkan;
    vkGetInstanceProcAddr = GetInstanceProcAddr;
    vkEnumerateInstanceExtensionProperties = (PFN_vkEnumerateInstanceExtensionProperties)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceExtensionProperties");
    vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(NULL, "vkCreateInstance");
    vkEnumerateInstanceLayerProperties = (PFN_vkEnumerateInstanceLayerProperties)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceLayerProperties");
    return 0;
}

#if defined __ANDROID__
static int load_vulkan_android(const char* driver_path)
{
    char hal_driver_path[256];
    if (!driver_path)
    {
        // https://source.android.com/docs/core/graphics/implement-vulkan#driver_emun

        // /vendor/lib/hw/vulkan.<ro.hardware.vulkan>.so
        // /vendor/lib/hw/vulkan.<ro.product.platform>.so
        // /vendor/lib64/hw/vulkan.<ro.hardware.vulkan>.so
        // /vendor/lib64/hw/vulkan.<ro.product.platform>.so

#ifdef __LP64__
        DIR* d = opendir("/vendor/lib64/hw");
#else
        DIR* d = opendir("/vendor/lib/hw");
#endif
        if (!d)
            return -1;

        int hal_driver_found = 0;
        struct dirent* dir;
        while ((dir = readdir(d)) != NULL)
        {
            char platform[256];
            int nscan = sscanf(dir->d_name, "vulkan.%255s.so", platform);
            if (nscan == 1)
            {
#ifdef __LP64__
                snprintf(hal_driver_path, 256, "/vendor/lib64/hw/%s", dir->d_name);
#else
                snprintf(hal_driver_path, 256, "/vendor/lib/hw/%s", dir->d_name);
#endif
                hal_driver_found = 1;
                break;
            }
        }
        closedir(d);

        if (!hal_driver_found)
        {
            NCNN_LOGE("no hal driver found");
            return -1;
        }

        NCNN_LOGE("hal_driver_path = %s", hal_driver_path);
    }

    const char* libpath = driver_path ? driver_path : hal_driver_path;

    void* libvulkan = dlopen(libpath, RTLD_LOCAL | RTLD_NOW);
    if (!libvulkan)
    {
        NCNN_LOGE("dlopen failed %s", dlerror());
        return -1;
    }

    // resolve entrypoint from android hal module
    hw_module_t* hmi = 0;
    hmi = (hw_module_t*)dlsym(libvulkan, "HMI");
    if (!hmi)
    {
        NCNN_LOGE("dlsym failed %s", dlerror());
        dlclose(libvulkan);
        return -1;
    }

    if (strcmp(hmi->id, "vulkan") != 0)
    {
        NCNN_LOGE("hmi->id != vulkan");
        dlclose(libvulkan);
        return -1;
    }

    hwvulkan_module_t* hvkmi = (hwvulkan_module_t*)hmi;

    // NCNN_LOGE("hvkmi name = %s", hvkmi->name);
    // NCNN_LOGE("hvkmi author = %s", hvkmi->author);

    hwvulkan_device_t* hvkdi = 0;
    int result = hvkmi->methods->open(hvkmi, "vk0", (hw_device_t**)&hvkdi);
    if (result != 0)
    {
        NCNN_LOGE("hmi->open failed %d", result);
        dlclose(libvulkan);
        return -1;
    }

    g_libvulkan = libvulkan;
    g_hvkdi = hvkdi;
    vkGetInstanceProcAddr = hvkdi->GetInstanceProcAddr;
    vkEnumerateInstanceExtensionProperties = hvkdi->EnumerateInstanceExtensionProperties;
    vkCreateInstance = hvkdi->CreateInstance;
    vkEnumerateInstanceLayerProperties = (PFN_vkEnumerateInstanceLayerProperties)vkGetInstanceProcAddr(NULL, "vkEnumerateInstanceLayerProperties");
    return 0;
}
#endif // __ANDROID__
#endif // _WIN32

int load_vulkan_driver(const char* driver_path)
{
    unload_vulkan_driver();

    int ret = 0;

    std::string driver_path_from_icd_env;
    std::string driver_path_from_ncnn_env;
    if (driver_path == 0)
    {
        driver_path_from_icd_env = get_driver_path_from_icd_env();
        if (!driver_path_from_icd_env.empty())
        {
            driver_path = driver_path_from_icd_env.c_str();
        }
        else
        {
            driver_path_from_ncnn_env = get_driver_path_from_ncnn_env();
            if (!driver_path_from_ncnn_env.empty())
            {
                driver_path = driver_path_from_ncnn_env.c_str();
            }
        }
    }

    // first try, load from driver_path
#if defined _WIN32
    ret = load_vulkan_windows(driver_path);
#else
    ret = load_vulkan_linux(driver_path);
#if defined __ANDROID__
    if (ret != 0)
    {
        // second try, load from android hal module
        ret = load_vulkan_android(driver_path);
    }
#endif // __ANDROID__
#endif // _WIN32
    if (driver_path != 0 && ret != 0)
    {
        // third try, load from system vulkan
#if defined _WIN32
        ret = load_vulkan_windows(0);
#else
        ret = load_vulkan_linux(0);
#if defined __ANDROID__
        if (ret != 0)
        {
            // fourth try, load from any android hal module found
            ret = load_vulkan_android(0);
        }
#endif // __ANDROID__
#endif // _WIN32
    }
    if (ret != 0)
    {
        // fifth try, load from well-known path
#if defined _WIN32
        const char* well_known_path[] = {
#if defined(__x86_64__) || defined(_M_X64)
            "nvoglv64.dll",
            "amdvlk64.dll",
            "igvk64.dll"
#else
            "nvoglv32.dll",
            "amdvlk32.dll",
            "igvk32.dll"
#endif
        };
#elif defined __ANDROID__
        const char* well_known_path[] = {
#ifdef __LP64__
            "/vendor/lib64/hw/vulkan.adreno.so",
            "/vendor/lib64/egl/libGLES_mali.so"
#else
            "/vendor/lib/hw/vulkan.adreno.so",
            "/vendor/lib/egl/libGLES_mali.so"
#endif
        };
#else
        const char* well_known_path[] = {
            "libGLX_nvidia.so.0",
            "libvulkan_radeon.so",
            "libvulkan_intel.so",
            "libMaliVulkan.so.1",
            "libVK_IMG.so"
        };
#endif

        const int well_known_path_count = sizeof(well_known_path) / sizeof(const char*);
        for (int i = 0; i < well_known_path_count; i++)
        {
#if defined _WIN32
            // find driver dll in %SystemRoot%\System32\DriverStore\FileRepository  (32bit and 64bit both and in here)
            std::string dllpath = search_file("%SystemRoot%\\System32\\DriverStore\\FileRepository", well_known_path[i]);
            if (dllpath.empty())
                continue;

            ret = load_vulkan_windows(well_known_path[i]);
#elif defined __ANDROID__
            ret = load_vulkan_android(well_known_path[i]);
#else
            ret = load_vulkan_linux(well_known_path[i]);
#endif
            if (ret == 0)
                break;
        }
    }

    return ret;
}

void unload_vulkan_driver()
{
    vkGetInstanceProcAddr = 0;
    vkEnumerateInstanceExtensionProperties = 0;
    vkCreateInstance = 0;
    vkEnumerateInstanceLayerProperties = 0;

#if defined _WIN32
    if (g_libvulkan)
    {
        FreeLibrary(g_libvulkan);
        g_libvulkan = 0;
    }
#else
#if defined __ANDROID__
    if (g_hvkdi)
    {
        if (g_hvkdi->close)
        {
            g_hvkdi->close(g_hvkdi);
        }
        g_hvkdi = 0;
    }
#endif // __ANDROID__

    if (g_libvulkan)
    {
        dlclose(g_libvulkan);
        g_libvulkan = 0;
    }
#endif // _WIN32
}

#endif // __APPLE__

} // namespace ncnn

#endif // NCNN_SIMPLEVK
#endif // NCNN_VULKAN
