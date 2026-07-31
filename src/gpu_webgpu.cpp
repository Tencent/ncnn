// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gpu.h"

#if NCNN_WEBGPU

#include <string.h>

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>

#include "glslang/Public/ResourceLimits.h"
#include "glslang/Public/ShaderLang.h"
#include "glslang/SPIRV/GlslangToSpv.h"
#include "SPIRV/spirv.hpp11"
#include "src/tint/api/tint.h"

#include "allocator.h"
#include "command.h"
#include "layer.h"
#include "layer_shader_type.h"
#include "layer_type.h"
#include "layer/vulkan/shader/vulkan_activation.comp.hex.h"
#include "mat.h"
#include "pipelinecache.h"

namespace ncnn {

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

enum WebGpuShaderProfile
{
    WEBGPU_SHADER_PROFILE_FP32 = 0,
    WEBGPU_SHADER_PROFILE_FP16_PACKED = 1
};

struct WebGpuShaderCompileOptions
{
    int shader_type;
    const Option* opt;

    uint32_t local_size_x;
    uint32_t local_size_y;
    uint32_t local_size_z;
};

struct WebGpuCompileKey
{
    int shader_type;
    uint32_t option_bits;
    uint32_t feature_bits;
};

struct WebGpuTranslatedShader
{
    std::vector<uint32_t> spirv;
    std::string wgsl;
    WebGpuShaderInfo shader_info;
    WebGpuCompileKey compile_key;
};

static uint32_t get_webgpu_shader_option_bits(const Option& opt)
{
    uint32_t option_bits = 0;
    if (opt.use_fp16_packed)
        option_bits |= 1;
    if (opt.use_int8_packed)
        option_bits |= 2;
    if (opt.use_shader_local_memory)
        option_bits |= 4;

    return option_bits;
}

static int validate_webgpu_shader_compile_options(const WebGpuShaderCompileOptions& options)
{
    if (!options.opt
            || options.shader_type < 0 || options.shader_type >= layer_shader_registry_entry_count
            || options.local_size_x == 0 || options.local_size_y == 0 || options.local_size_z == 0)
    {
        NCNN_LOGE("WebGPU shader compile options are invalid");
        return -1;
    }

    return 0;
}

class WebGpuCompilerContext
{
public:
    WebGpuCompilerContext()
        : initialized(false)
    {
    }

    ~WebGpuCompilerContext()
    {
        if (!initialized)
            return;

        tint::Shutdown();
        glslang_finalize_process();
    }

    int initialize()
    {
        if (initialized)
            return 0;

        if (!glslang_initialize_process())
        {
            NCNN_LOGE("WebGPU failed to initialize glslang");
            return -1;
        }

        tint::Initialize();
        initialized = true;
        return 0;
    }

    bool initialized;
};

static WebGpuCompilerContext g_webgpu_compiler;

class GpuInfoPrivate
{
public:
    GpuInfoPrivate()
        : device_index(-1), vendor_id(0), device_id(0), adapter_type(WGPUAdapterType_Unknown)
    {
        limits = WGPU_LIMITS_INIT;
    }

    int device_index;
    uint32_t vendor_id;
    uint32_t device_id;
    WGPUAdapterType adapter_type;
    WGPULimits limits;
    std::string device_name;
    std::string driver_name;
};

enum WebGpuInstanceState
{
    WEBGPU_INSTANCE_UNINITIALIZED,
    WEBGPU_INSTANCE_INITIALIZING,
    WEBGPU_INSTANCE_READY,
    WEBGPU_INSTANCE_FAILED,
    WEBGPU_INSTANCE_SHUTTING_DOWN
};

struct WebGpuContext
{
    WebGpuContext()
        : state(WEBGPU_INSTANCE_UNINITIALIZED), instance(0), adapter(0), device(0), queue(0), info(0), vkdev(0), last_error(0), next_operation_id(0), active_operation_id(0), error_operation_id(0), operation_depth(0), transient_error_consumed(false), device_lost(false)
    {
    }

    WebGpuInstanceState state;
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    GpuInfo* info;
    VulkanDevice* vkdev;
    int last_error;
    uint64_t next_operation_id;
    uint64_t active_operation_id;
    uint64_t error_operation_id;
    int operation_depth;
    bool transient_error_consumed;
    bool device_lost;
    std::string first_error_reason;
};

static WebGpuContext g_webgpu;

uint64_t begin_webgpu_sync_operation(const char* operation)
{
    if (!operation || g_webgpu.device_lost)
        return 0;

    if (g_webgpu.active_operation_id != 0)
    {
        g_webgpu.operation_depth++;
        return g_webgpu.active_operation_id;
    }

    if (g_webgpu.last_error != 0 && !g_webgpu.transient_error_consumed)
        return 0;
    if (g_webgpu.transient_error_consumed)
    {
        g_webgpu.last_error = 0;
        g_webgpu.error_operation_id = 0;
        g_webgpu.transient_error_consumed = false;
        g_webgpu.first_error_reason.clear();
    }

    g_webgpu.next_operation_id++;
    if (g_webgpu.next_operation_id == 0)
        g_webgpu.next_operation_id++;
    g_webgpu.active_operation_id = g_webgpu.next_operation_id;
    g_webgpu.operation_depth = 1;

#if NCNN_STDIO
    printf("NCNN_WEBGPU_OPERATION_BEGIN:%s:%llu\n", operation, (unsigned long long)g_webgpu.active_operation_id);
#endif
    return g_webgpu.active_operation_id;
}

static void record_webgpu_background_error(const char* reason, bool device_lost)
{
    if (g_webgpu.last_error == 0)
    {
        g_webgpu.last_error = -1;
        g_webgpu.error_operation_id = g_webgpu.active_operation_id;
        g_webgpu.transient_error_consumed = false;
        g_webgpu.first_error_reason = reason ? reason : "unknown";
    }
    if (device_lost)
        g_webgpu.device_lost = true;
}

int finish_webgpu_sync_operation(uint64_t operation_id, int result)
{
    if (operation_id == 0 || operation_id != g_webgpu.active_operation_id || g_webgpu.operation_depth <= 0)
        return -1;

    if (g_webgpu.last_error != 0
            && (g_webgpu.device_lost || g_webgpu.error_operation_id == 0 || g_webgpu.error_operation_id == operation_id))
        result = -1;

    g_webgpu.operation_depth--;
    if (g_webgpu.operation_depth == 0)
    {
#if NCNN_STDIO
        printf("NCNN_WEBGPU_OPERATION_END:%llu:%d\n", (unsigned long long)operation_id, result);
#endif
        g_webgpu.active_operation_id = 0;
        if (g_webgpu.last_error != 0 && !g_webgpu.device_lost)
            g_webgpu.transient_error_consumed = true;
    }

    return result;
}

static bool g_webgpu_sync_wait_active = false;

class WebGpuSyncWaitGuard
{
public:
    WebGpuSyncWaitGuard()
    {
        g_webgpu_sync_wait_active = true;
    }

    ~WebGpuSyncWaitGuard()
    {
        g_webgpu_sync_wait_active = false;
    }
};

struct AdapterResult
{
    WGPURequestAdapterStatus status;
    WGPUAdapter adapter;
};

struct DeviceResult
{
    WGPURequestDeviceStatus status;
    WGPUDevice device;
};

static std::string string_from_webgpu(WGPUStringView value)
{
    if (!value.data)
        return std::string();

    return std::string(value.data, value.length);
}

static void log_webgpu_limits(const char* source, const WGPULimits& limits)
{
    NCNN_LOGE("NCNN_WEBGPU_LIMITS:%s:maxBufferSize=%llu:maxStorageBufferBindingSize=%llu:minStorageBufferOffsetAlignment=%u:maxStorageBuffersPerShaderStage=%u:maxComputeWorkgroupStorageSize=%u:maxComputeInvocationsPerWorkgroup=%u:maxComputeWorkgroupSize=%u,%u,%u:maxComputeWorkgroupsPerDimension=%u:maxImmediateSize=%u",
              source,
              (unsigned long long)limits.maxBufferSize,
              (unsigned long long)limits.maxStorageBufferBindingSize,
              limits.minStorageBufferOffsetAlignment,
              limits.maxStorageBuffersPerShaderStage,
              limits.maxComputeWorkgroupStorageSize,
              limits.maxComputeInvocationsPerWorkgroup,
              limits.maxComputeWorkgroupSizeX,
              limits.maxComputeWorkgroupSizeY,
              limits.maxComputeWorkgroupSizeZ,
              limits.maxComputeWorkgroupsPerDimension,
              limits.maxImmediateSize);
}

static void request_adapter_callback(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1, void*)
{
    AdapterResult* result = (AdapterResult*)userdata1;
    result->status = status;
    result->adapter = adapter;

    if (status != WGPURequestAdapterStatus_Success)
        NCNN_LOGE("WebGPU request adapter failed %.*s", (int)message.length, message.data ? message.data : "");
}

static void request_device_callback(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1, void*)
{
    DeviceResult* result = (DeviceResult*)userdata1;
    result->status = status;
    result->device = device;

    if (status != WGPURequestDeviceStatus_Success)
        NCNN_LOGE("WebGPU request device failed %.*s", (int)message.length, message.data ? message.data : "");
}

static void device_lost_callback(const WGPUDevice*, WGPUDeviceLostReason reason, WGPUStringView message, void*, void*)
{
    if (reason == WGPUDeviceLostReason_Destroyed)
        return;

    record_webgpu_background_error("device-lost", true);

    NCNN_LOGE("WebGPU device lost reason=%d %.*s", (int)reason, (int)message.length, message.data ? message.data : "");
}

static void uncaptured_error_callback(const WGPUDevice*, WGPUErrorType type, WGPUStringView message, void*, void*)
{
    record_webgpu_background_error("uncaptured-error", false);

    NCNN_LOGE("WebGPU uncaptured error type=%d %.*s", (int)type, (int)message.length, message.data ? message.data : "");
}

static int wait_webgpu_future(WGPUInstance instance, WGPUFuture future, WGPUFutureWaitInfo* wait_info, const char* operation)
{
    if (!instance || !wait_info || !operation)
        return -1;
    if (g_webgpu_sync_wait_active)
    {
        NCNN_LOGE("WebGPU nested synchronous wait is not supported operation=%s", operation);
        return -1;
    }

    wait_info->future = future;

    WebGpuSyncWaitGuard wait_guard;
#if NCNN_STDIO
    printf("NCNN_WEBGPU_WAIT_BEGIN:%s:%llu\n", operation, (unsigned long long)future.id);
#endif
    WGPUWaitStatus status = wgpuInstanceWaitAny(instance, 1, wait_info, UINT64_MAX);
#if NCNN_STDIO
    printf("NCNN_WEBGPU_WAIT_END:%s:%d\n", operation, (int)status);
#endif

    if (status != WGPUWaitStatus_Success || wait_info->completed != WGPU_TRUE)
    {
        NCNN_LOGE("WebGPU wait failed operation=%s status=%d completed=%d", operation, (int)status, (int)wait_info->completed);
        return -1;
    }

    return 0;
}

static void release_webgpu_resources()
{
    delete g_webgpu.vkdev;
    g_webgpu.vkdev = 0;
    delete g_webgpu.info;
    g_webgpu.info = 0;

    if (g_webgpu.device)
        wgpuDeviceDestroy(g_webgpu.device);
    if (g_webgpu.queue)
        wgpuQueueRelease(g_webgpu.queue);
    if (g_webgpu.device)
        wgpuDeviceRelease(g_webgpu.device);
    if (g_webgpu.adapter)
        wgpuAdapterRelease(g_webgpu.adapter);
    if (g_webgpu.instance)
        wgpuInstanceRelease(g_webgpu.instance);

    g_webgpu.queue = 0;
    g_webgpu.device = 0;
    g_webgpu.adapter = 0;
    g_webgpu.instance = 0;
}

GpuInfo::GpuInfo()
    : d(new GpuInfoPrivate)
{
}

GpuInfo::~GpuInfo()
{
    delete d;
}

int GpuInfo::device_index() const
{
    return d->device_index;
}
uint32_t GpuInfo::api_version() const
{
    return 0;
}
uint32_t GpuInfo::driver_version() const
{
    return 0;
}
uint32_t GpuInfo::vendor_id() const
{
    return d->vendor_id;
}
uint32_t GpuInfo::device_id() const
{
    return d->device_id;
}
const char* GpuInfo::device_name() const
{
    return d->device_name.c_str();
}
const char* GpuInfo::driver_name() const
{
    return d->driver_name.c_str();
}

int GpuInfo::type() const
{
    return 3;
}

uint32_t GpuInfo::rough_score() const
{
    return 10;
}

uint32_t GpuInfo::max_shared_memory_size() const
{
    return d->limits.maxComputeWorkgroupStorageSize;
}
uint32_t GpuInfo::max_workgroup_count_x() const
{
    return d->limits.maxComputeWorkgroupsPerDimension;
}
uint32_t GpuInfo::max_workgroup_count_y() const
{
    return d->limits.maxComputeWorkgroupsPerDimension;
}
uint32_t GpuInfo::max_workgroup_count_z() const
{
    return d->limits.maxComputeWorkgroupsPerDimension;
}
uint32_t GpuInfo::max_workgroup_invocations() const
{
    return d->limits.maxComputeInvocationsPerWorkgroup;
}
uint32_t GpuInfo::max_workgroup_size_x() const
{
    return d->limits.maxComputeWorkgroupSizeX;
}
uint32_t GpuInfo::max_workgroup_size_y() const
{
    return d->limits.maxComputeWorkgroupSizeY;
}
uint32_t GpuInfo::max_workgroup_size_z() const
{
    return d->limits.maxComputeWorkgroupSizeZ;
}
size_t GpuInfo::memory_map_alignment() const
{
    return 8;
}
size_t GpuInfo::buffer_offset_alignment() const
{
    return d->limits.minStorageBufferOffsetAlignment;
}
size_t GpuInfo::non_coherent_atom_size() const
{
    return 4;
}
size_t GpuInfo::buffer_image_granularity() const
{
    return 1;
}
uint32_t GpuInfo::max_image_dimension_1d() const
{
    return 0;
}
uint32_t GpuInfo::max_image_dimension_2d() const
{
    return 0;
}
uint32_t GpuInfo::max_image_dimension_3d() const
{
    return 0;
}
float GpuInfo::timestamp_period() const
{
    return 0.f;
}

uint32_t GpuInfo::compute_queue_family_index() const
{
    return 0;
}
uint32_t GpuInfo::transfer_queue_family_index() const
{
    return 0;
}
uint32_t GpuInfo::compute_queue_count() const
{
    return 1;
}
uint32_t GpuInfo::transfer_queue_count() const
{
    return 1;
}
bool GpuInfo::unified_compute_transfer_queue() const
{
    return true;
}
bool GpuInfo::resizable_bar_enabled() const
{
    return false;
}

uint32_t GpuInfo::subgroup_size() const
{
    return 64;
}
uint32_t GpuInfo::min_subgroup_size() const
{
    return 64;
}
uint32_t GpuInfo::max_subgroup_size() const
{
    return 64;
}
uint32_t GpuInfo::max_compute_workgroup_subgroups() const
{
    return 0;
}
bool GpuInfo::support_subgroup_size_control() const
{
    return false;
}
bool GpuInfo::support_compute_full_subgroups() const
{
    return false;
}
uint32_t GpuInfo::support_subgroup_ops() const
{
    return 0;
}

bool GpuInfo::bug_storage_buffer_no_l1() const
{
    return false;
}
bool GpuInfo::bug_corrupted_online_pipeline_cache() const
{
    return false;
}
bool GpuInfo::bug_buffer_image_load_zero() const
{
    return false;
}
bool GpuInfo::bug_implicit_fp16_arithmetic() const
{
    return false;
}

bool GpuInfo::support_fp16_packed() const
{
    return true;
}
bool GpuInfo::support_fp16_storage() const
{
    return false;
}
bool GpuInfo::support_fp16_uniform() const
{
    return false;
}
bool GpuInfo::support_fp16_arithmetic() const
{
    return false;
}
bool GpuInfo::support_int8_packed() const
{
    return false;
}
bool GpuInfo::support_int8_storage() const
{
    return false;
}
bool GpuInfo::support_int8_uniform() const
{
    return false;
}
bool GpuInfo::support_int8_arithmetic() const
{
    return false;
}
bool GpuInfo::support_int16_packed() const
{
    return false;
}
bool GpuInfo::support_int16_storage() const
{
    return false;
}
bool GpuInfo::support_int16_arithmetic() const
{
    return false;
}
bool GpuInfo::support_bf16_packed() const
{
    return false;
}
bool GpuInfo::support_bf16_storage() const
{
    return false;
}
bool GpuInfo::support_fp16_image() const
{
    return false;
}
bool GpuInfo::support_int8_image() const
{
    return false;
}
bool GpuInfo::support_fp_fast_math() const
{
    return false;
}
bool GpuInfo::support_ycbcr_conversion() const
{
    return false;
}
bool GpuInfo::support_cooperative_matrix() const
{
    return false;
}
bool GpuInfo::support_cooperative_matrix_8_8_16() const
{
    return false;
}
bool GpuInfo::support_cooperative_matrix_16_8_8() const
{
    return false;
}
bool GpuInfo::support_cooperative_matrix_16_8_16() const
{
    return false;
}
bool GpuInfo::support_cooperative_matrix_16_16_16() const
{
    return false;
}
bool GpuInfo::support_int8_cooperative_matrix() const
{
    return false;
}
bool GpuInfo::support_bf16_cooperative_matrix() const
{
    return false;
}

int GpuInfo::support_VK_KHR_cooperative_matrix() const
{
    return 0;
}
int GpuInfo::support_VK_NV_cooperative_matrix() const
{
    return 0;
}

void GpuInfo::get_optimal_cooperative_matrix_mnk(int, int, int, VkComponentTypeKHR, VkComponentTypeKHR, VkScopeKHR,
        int& coopmat_M, int& coopmat_N, int& coopmat_K, int& coopmat_subgroup_size) const
{
    coopmat_M = 0;
    coopmat_N = 0;
    coopmat_K = 0;
    coopmat_subgroup_size = 0;
}

int create_gpu_instance(const char* driver_path)
{
    if (driver_path && driver_path[0])
    {
        NCNN_LOGE("WebGPU backend does not accept a Vulkan driver path");
        return -1;
    }

    if (g_webgpu.state == WEBGPU_INSTANCE_READY)
        return 0;
    if (g_webgpu.state != WEBGPU_INSTANCE_UNINITIALIZED)
        return -1;

    const uint64_t operation_id = begin_webgpu_sync_operation("create-gpu-instance");
    if (operation_id == 0)
        return -1;

    g_webgpu.state = WEBGPU_INSTANCE_INITIALIZING;

    WGPUInstanceFeatureName instance_feature = WGPUInstanceFeatureName_TimedWaitAny;
    WGPUInstanceDescriptor instance_descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    instance_descriptor.requiredFeatureCount = 1;
    instance_descriptor.requiredFeatures = &instance_feature;

    g_webgpu.instance = wgpuCreateInstance(&instance_descriptor);
    if (!g_webgpu.instance)
    {
        NCNN_LOGE("WebGPU failed to create TimedWaitAny instance");
        goto fail;
    }

    if (
        wgpuInstanceHasWGSLLanguageFeature(g_webgpu.instance, WGPUWGSLLanguageFeatureName_ImmediateAddressSpace) != WGPU_TRUE)
    {
        NCNN_LOGE("WebGPU immediate_address_space is required");
        goto fail;
    }

    {
        AdapterResult adapter_result;
        adapter_result.status = WGPURequestAdapterStatus_Error;
        adapter_result.adapter = 0;

        WGPURequestAdapterOptions adapter_options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
        WGPURequestAdapterCallbackInfo callback_info = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
        callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
        callback_info.callback = request_adapter_callback;
        callback_info.userdata1 = &adapter_result;

        WGPUFuture future = wgpuInstanceRequestAdapter(g_webgpu.instance, &adapter_options, callback_info);
        WGPUFutureWaitInfo wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
        if (wait_webgpu_future(g_webgpu.instance, future, &wait_info, "request-adapter") != 0
                || adapter_result.status != WGPURequestAdapterStatus_Success
                || !adapter_result.adapter)
            goto fail;

        g_webgpu.adapter = adapter_result.adapter;
    }

    {
        WGPULimits limits = WGPU_LIMITS_INIT;
        if (wgpuAdapterGetLimits(g_webgpu.adapter, &limits) != WGPUStatus_Success)
        {
            NCNN_LOGE("WebGPU failed to query adapter limits");
            goto fail;
        }
        log_webgpu_limits("adapter", limits);

        if (limits.maxImmediateSize < 64
                || limits.maxStorageBuffersPerShaderStage < 8
                || limits.maxComputeWorkgroupSizeX < 64
                || limits.maxComputeInvocationsPerWorkgroup < 64)
        {
            NCNN_LOGE("WebGPU adapter limits are insufficient immediate=%u storage-buffers=%u workgroup-x=%u invocations=%u",
                      limits.maxImmediateSize, limits.maxStorageBuffersPerShaderStage,
                      limits.maxComputeWorkgroupSizeX, limits.maxComputeInvocationsPerWorkgroup);
            goto fail;
        }

        GpuInfo* info = new GpuInfo;
        info->d->device_index = 0;
        info->d->limits = limits;

        WGPUAdapterInfo adapter_info = WGPU_ADAPTER_INFO_INIT;
        if (wgpuAdapterGetInfo(g_webgpu.adapter, &adapter_info) == WGPUStatus_Success)
        {
            std::string vendor = string_from_webgpu(adapter_info.vendor);
            std::string architecture = string_from_webgpu(adapter_info.architecture);
            info->d->vendor_id = adapter_info.vendorID;
            info->d->device_id = adapter_info.deviceID;
            info->d->adapter_type = adapter_info.adapterType;
            info->d->device_name = string_from_webgpu(adapter_info.device);
            info->d->driver_name = string_from_webgpu(adapter_info.description);

            if (info->d->device_name.empty())
                info->d->device_name = !architecture.empty() ? architecture : vendor;
            if (info->d->driver_name.empty())
                info->d->driver_name = !vendor.empty() ? vendor : architecture;

            wgpuAdapterInfoFreeMembers(adapter_info);
        }

        if (info->d->device_name.empty())
            info->d->device_name = "WebGPU";

        g_webgpu.info = info;
    }

    {
        WGPULimits required_limits = WGPU_LIMITS_INIT;
        required_limits.maxBufferSize = g_webgpu.info->d->limits.maxBufferSize;
        required_limits.maxStorageBufferBindingSize = g_webgpu.info->d->limits.maxStorageBufferBindingSize;
        required_limits.maxStorageBuffersPerShaderStage = 8;
        required_limits.minStorageBufferOffsetAlignment = g_webgpu.info->d->limits.minStorageBufferOffsetAlignment;
        required_limits.maxComputeWorkgroupStorageSize = g_webgpu.info->d->limits.maxComputeWorkgroupStorageSize;
        required_limits.maxComputeInvocationsPerWorkgroup = g_webgpu.info->d->limits.maxComputeInvocationsPerWorkgroup;
        required_limits.maxComputeWorkgroupSizeX = g_webgpu.info->d->limits.maxComputeWorkgroupSizeX;
        required_limits.maxComputeWorkgroupSizeY = g_webgpu.info->d->limits.maxComputeWorkgroupSizeY;
        required_limits.maxComputeWorkgroupSizeZ = g_webgpu.info->d->limits.maxComputeWorkgroupSizeZ;
        required_limits.maxComputeWorkgroupsPerDimension = g_webgpu.info->d->limits.maxComputeWorkgroupsPerDimension;
        required_limits.maxImmediateSize = 64;
        log_webgpu_limits("request", required_limits);

        WGPUDeviceDescriptor device_descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
        device_descriptor.requiredLimits = &required_limits;
        device_descriptor.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
        device_descriptor.deviceLostCallbackInfo.callback = device_lost_callback;
        device_descriptor.uncapturedErrorCallbackInfo.callback = uncaptured_error_callback;

        DeviceResult device_result;
        device_result.status = WGPURequestDeviceStatus_Error;
        device_result.device = 0;

        WGPURequestDeviceCallbackInfo callback_info = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
        callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
        callback_info.callback = request_device_callback;
        callback_info.userdata1 = &device_result;

        WGPUFuture future = wgpuAdapterRequestDevice(g_webgpu.adapter, &device_descriptor, callback_info);
        WGPUFutureWaitInfo wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
        if (wait_webgpu_future(g_webgpu.instance, future, &wait_info, "request-device") != 0
                || device_result.status != WGPURequestDeviceStatus_Success
                || !device_result.device)
            goto fail;

        g_webgpu.device = device_result.device;
        g_webgpu.queue = wgpuDeviceGetQueue(g_webgpu.device);
        if (!g_webgpu.queue)
            goto fail;

        WGPULimits device_limits = WGPU_LIMITS_INIT;
        if (wgpuDeviceGetLimits(g_webgpu.device, &device_limits) != WGPUStatus_Success)
        {
            NCNN_LOGE("WebGPU failed to query device limits");
            goto fail;
        }
        log_webgpu_limits("device", device_limits);
        g_webgpu.info->d->limits = device_limits;
    }

    NCNN_LOGE("[0 %s]  queueC=0[1]  queueT=0[1]  rebar=0  r-score=%u", g_webgpu.info->device_name(), g_webgpu.info->rough_score());
    NCNN_LOGE("[0 %s]  fp16-p/s/u/a=%d/%d/%d/%d  int8-p/s/u/a=%d/%d/%d/%d  bf16-p/s=%d/%d", g_webgpu.info->device_name(),
              g_webgpu.info->support_fp16_packed(), g_webgpu.info->support_fp16_storage(), g_webgpu.info->support_fp16_uniform(), g_webgpu.info->support_fp16_arithmetic(),
              g_webgpu.info->support_int8_packed(), g_webgpu.info->support_int8_storage(), g_webgpu.info->support_int8_uniform(), g_webgpu.info->support_int8_arithmetic(),
              g_webgpu.info->support_bf16_packed(), g_webgpu.info->support_bf16_storage());
    NCNN_LOGE("[0 %s]  subgroup=0(0~0)  ops=0/0/0/0/0/0/0/0/0/0", g_webgpu.info->device_name());
    NCNN_LOGE("[0 %s]  fp16-cm=-  int8-cm=-  bf16-cm=-  fp8-cm=-", g_webgpu.info->device_name());

    g_webgpu.state = WEBGPU_INSTANCE_READY;
    return finish_webgpu_sync_operation(operation_id, 0);

fail:
    release_webgpu_resources();
    record_webgpu_background_error("device-initialization", false);
    g_webgpu.state = WEBGPU_INSTANCE_FAILED;
    return finish_webgpu_sync_operation(operation_id, -1);
}

void destroy_gpu_instance()
{
    if (g_webgpu.state == WEBGPU_INSTANCE_UNINITIALIZED)
        return;
    if (g_webgpu_sync_wait_active)
    {
        NCNN_LOGE("WebGPU destroy is not allowed during a synchronous wait");
        return;
    }

    g_webgpu.state = WEBGPU_INSTANCE_SHUTTING_DOWN;

    release_webgpu_resources();
    g_webgpu.last_error = 0;
    g_webgpu.active_operation_id = 0;
    g_webgpu.error_operation_id = 0;
    g_webgpu.operation_depth = 0;
    g_webgpu.transient_error_consumed = false;
    g_webgpu.device_lost = false;
    g_webgpu.first_error_reason.clear();
    g_webgpu.state = WEBGPU_INSTANCE_UNINITIALIZED;
}

int get_gpu_count()
{
    if (g_webgpu.state == WEBGPU_INSTANCE_UNINITIALIZED)
        create_gpu_instance();

    return g_webgpu.state == WEBGPU_INSTANCE_READY ? 1 : 0;
}

int get_default_gpu_index()
{
    return get_gpu_count() == 1 ? 0 : -1;
}

const GpuInfo& get_gpu_info(int device_index)
{
    if (g_webgpu.state == WEBGPU_INSTANCE_UNINITIALIZED)
        create_gpu_instance();

    if (g_webgpu.state == WEBGPU_INSTANCE_READY && device_index == 0)
        return *g_webgpu.info;

    static GpuInfo invalid_info;
    return invalid_info;
}

int get_webgpu_last_error()
{
    return g_webgpu.last_error;
}

class VulkanDevicePrivate
{
public:
    VulkanDevicePrivate(VulkanDevice* _vkdev)
        : vkdev(_vkdev), blob_allocator(0), staging_allocator(0), pipeline_cache(0)
    {
        memset(dummy_allocators, 0, sizeof(dummy_allocators));
        memset(uop_packing, 0, sizeof(uop_packing));
    }

    int create_dummy_buffers();
    void destroy_dummy_buffers();
    const Layer* get_utility_operator(int cast_type_from_index, int cast_type_to_index, int packing_type_to_index) const;
    void destroy_utility_operator();

    VulkanDevice* const vkdev;
    VkBlobAllocator* blob_allocator;
    VkStagingAllocator* staging_allocator;
    PipelineCache* pipeline_cache;
    VkAllocator* dummy_allocators[8];
    VkMat dummy_buffers[8];
    mutable Layer* uop_packing[4][4][2];
};

VulkanDevice::VulkanDevice(int device_index)
    : info(get_gpu_info(device_index)), d(new VulkanDevicePrivate(this))
{
    if (is_valid() && d->create_dummy_buffers() != 0)
        NCNN_LOGE("WebGPU create dummy buffers failed");
}

VulkanDevice::~VulkanDevice()
{
    d->destroy_utility_operator();
    delete d->pipeline_cache;
    delete d->staging_allocator;
    delete d->blob_allocator;
    d->destroy_dummy_buffers();
    delete d;
}

WGPUInstance VulkanDevice::wgpu_instance() const
{
    return g_webgpu.instance;
}
WGPUDevice VulkanDevice::wgpu_device() const
{
    return g_webgpu.device;
}
WGPUQueue VulkanDevice::wgpu_queue() const
{
    return g_webgpu.queue;
}

bool VulkanDevice::is_valid() const
{
    return info.device_index() == 0 && g_webgpu.state == WEBGPU_INSTANCE_READY && g_webgpu.device && g_webgpu.queue;
}

int VulkanDevice::wait_webgpu_future(WGPUFuture future, WGPUFutureWaitInfo* wait_info, const char* operation) const
{
    return ncnn::wait_webgpu_future(g_webgpu.instance, future, wait_info, operation);
}

bool VulkanDevice::is_device_local(uint32_t) const
{
    return true;
}

VkAllocator* VulkanDevice::acquire_blob_allocator() const
{
    if (!d->blob_allocator)
        d->blob_allocator = new VkBlobAllocator(this);

    return d->blob_allocator;
}

void VulkanDevice::reclaim_blob_allocator(VkAllocator*) const
{
}

VkAllocator* VulkanDevice::acquire_staging_allocator() const
{
    if (!d->staging_allocator)
        d->staging_allocator = new VkStagingAllocator(this);

    return d->staging_allocator;
}

void VulkanDevice::reclaim_staging_allocator(VkAllocator*) const
{
}

VkMat VulkanDevice::get_dummy_buffer() const
{
    return d->dummy_buffers[0];
}

VkMat VulkanDevice::get_dummy_buffer(int binding_index) const
{
    if (binding_index < 0 || binding_index >= 8)
        return VkMat();

    return d->dummy_buffers[binding_index];
}
VkImageMat VulkanDevice::get_dummy_image() const
{
    return VkImageMat();
}
VkImageMat VulkanDevice::get_dummy_image_readonly() const
{
    return VkImageMat();
}

const PipelineCache* VulkanDevice::get_pipeline_cache() const
{
    if (!d->pipeline_cache)
        d->pipeline_cache = new PipelineCache(this);

    return d->pipeline_cache;
}

bool VulkanDevice::shape_support_image_storage(const Mat&) const
{
    return false;
}
uint32_t VulkanDevice::get_heap_budget() const
{
    return 0;
}

int VulkanDevicePrivate::create_dummy_buffers()
{
    for (int i = 0; i < 8; i++)
    {
        dummy_allocators[i] = new VkBlobAllocator(vkdev, 256);
        dummy_buffers[i].create(64, 4u, dummy_allocators[i]);
        if (dummy_buffers[i].empty())
        {
            destroy_dummy_buffers();
            return -1;
        }
    }

    return 0;
}

void VulkanDevicePrivate::destroy_dummy_buffers()
{
    for (int i = 0; i < 8; i++)
    {
        dummy_buffers[i].release();
        delete dummy_allocators[i];
        dummy_allocators[i] = 0;
    }
}

const Layer* VulkanDevicePrivate::get_utility_operator(int cast_type_from_index, int cast_type_to_index, int packing_type_to_index) const
{
    if ((cast_type_from_index != 0 && cast_type_from_index != 1 && cast_type_from_index != 3)
            || (cast_type_to_index != 0 && cast_type_to_index != 1 && cast_type_to_index != 3)
            || packing_type_to_index < 0 || packing_type_to_index >= 2)
        return 0;

    Layer*& cached_uop = uop_packing[cast_type_from_index][cast_type_to_index][packing_type_to_index];
    if (cached_uop)
        return cached_uop;

    const bool use_fp16 = cast_type_from_index == 1 || cast_type_to_index == 1;
    const bool use_int8 = cast_type_from_index == 3 || cast_type_to_index == 3;
    Option opt;
    opt.use_fp16_packed = use_fp16;
    opt.use_fp16_storage = false;
    opt.use_fp16_uniform = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_packed = use_int8;
    opt.use_int8_storage = false;
    opt.use_int8_uniform = false;
    opt.use_int8_arithmetic = false;
    opt.use_int16_packed = false;
    opt.use_int16_storage = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_cooperative_matrix = false;
    opt.use_subgroup_ops = false;
    opt.use_shader_local_memory = false;
    opt.use_vulkan_compute = true;
    opt.pipeline_cache = 0;
    opt.vulkan_device_index = vkdev->info.device_index();

    Layer* uop = create_layer_vulkan(LayerType::Packing);
    if (!uop)
        return 0;
    uop->vkdev = vkdev;

    ParamDict pd;
    pd.set(0, packing_type_to_index == 0 ? 1 : 4);
    pd.set(2, cast_type_from_index + 1);
    pd.set(3, cast_type_to_index + 1);
    if (uop->load_param(pd) != 0 || uop->create_pipeline(opt) != 0)
    {
        delete uop;
        return 0;
    }

    cached_uop = uop;
    return cached_uop;
}

void VulkanDevicePrivate::destroy_utility_operator()
{
    Option opt;
    opt.use_vulkan_compute = true;
    opt.pipeline_cache = 0;
    opt.vulkan_device_index = vkdev->info.device_index();

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 2; k++)
            {
                Layer* uop = uop_packing[i][j][k];
                if (!uop)
                    continue;

                uop->destroy_pipeline(opt);
                delete uop;
                uop_packing[i][j][k] = 0;
            }
        }
    }
}

void VulkanDevice::convert_packing(const VkMat& src, VkMat& dst, int dst_elempack, VkCompute& cmd, const Option& opt) const
{
    convert_packing(src, dst, dst_elempack, 0, cmd, opt);
}

void VulkanDevice::convert_packing(const VkMat& src, VkMat& dst, int dst_elempack, int cast_type_to, VkCompute& cmd, const Option& opt) const
{
    const int packing_type_to_index = dst_elempack == 1 ? 0 : dst_elempack == 4 ? 1 : -1;
    int cast_type_from_index = -1;
    if (src.elembits() == 32)
        cast_type_from_index = 0;
    if (src.elembits() == 16)
        cast_type_from_index = 1;
    if (src.elembits() == 8)
        cast_type_from_index = 3;
    const int cast_type_to_index = cast_type_to ? cast_type_to - 1 : cast_type_from_index;
    if (packing_type_to_index < 0
            || (cast_type_from_index != 0 && cast_type_from_index != 1 && cast_type_from_index != 3)
            || (cast_type_to_index != 0 && cast_type_to_index != 1 && cast_type_to_index != 3)
            || ((cast_type_from_index == 3) != (cast_type_to_index == 3)))
    {
        NCNN_LOGE("WebGPU convert_packing unsupported elembits=%d out_elempack=%d cast_type_to=%d", src.elembits(), dst_elempack, cast_type_to);
        return;
    }

    Option opt2 = opt;
    opt2.use_fp16_packed = cast_type_from_index == 1 || cast_type_to_index == 1;
    opt2.use_fp16_storage = false;
    opt2.use_fp16_uniform = false;
    opt2.use_fp16_arithmetic = false;
    opt2.use_int8_packed = cast_type_from_index == 3 || cast_type_to_index == 3;
    opt2.use_int8_storage = false;
    opt2.use_int8_uniform = false;
    opt2.use_int8_arithmetic = false;
    opt2.use_int16_packed = false;
    opt2.use_int16_storage = false;
    opt2.use_bf16_packed = false;
    opt2.use_bf16_storage = false;
    opt2.use_cooperative_matrix = false;
    opt2.use_subgroup_ops = false;
    opt2.use_shader_local_memory = false;
    opt2.pipeline_cache = 0;

    const Layer* uop = d->get_utility_operator(cast_type_from_index, cast_type_to_index, packing_type_to_index);
    if (!uop)
    {
        NCNN_LOGE("WebGPU convert_packing utility operator creation failed");
        return;
    }

    uop->forward(src, dst, cmd, opt2);
}

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

static int build_webgpu_shader_preamble(const Option& opt, std::string& preamble)
{
    if (opt.use_bf16_storage || opt.use_bf16_packed
            || opt.use_fp16_storage || opt.use_fp16_uniform || opt.use_fp16_arithmetic
            || opt.use_int8_storage || opt.use_int8_uniform || opt.use_int8_arithmetic
            || opt.use_int16_storage || opt.use_int16_packed
            || opt.use_subgroup_ops || opt.use_cooperative_matrix)
    {
        NCNN_LOGE("WebGPU M1 shader profile only supports fp32 and fp16-packed bf16s=%d bf16p=%d fp16s=%d fp16u=%d fp16a=%d int8s=%d int8u=%d int8a=%d int16s=%d int16p=%d subgroup=%d coopmat=%d",
                  opt.use_bf16_storage, opt.use_bf16_packed,
                  opt.use_fp16_storage, opt.use_fp16_uniform, opt.use_fp16_arithmetic,
                  opt.use_int8_storage, opt.use_int8_uniform, opt.use_int8_arithmetic,
                  opt.use_int16_storage, opt.use_int16_packed,
                  opt.use_subgroup_ops, opt.use_cooperative_matrix);
        return -1;
    }

    DefinitionCollector custom_defines;
    const bool support_fp16_storage = true;
    const bool support_fp16_uniform = true;
    const bool support_int16_arithmetic = true;

    if (opt.use_bf16_storage)
    {
        custom_defines.append("sfp", "bfloat16_t");
        custom_defines.append("sfpvec2", "bf16vec2");
        custom_defines.append("sfpvec4", "bf16vec4");

        // define pack and unpack macro for bf16s
        custom_defines.append("unpackBFloat2x16(v)", "vec2(uintBitsToBFloat16EXT(unpackUint2x16(v)))");
        custom_defines.append("packBFloat2x16(v)", "packUint2x16(bfloat16BitsToUintEXT(bf16vec2(v)))");
    }
    else if (opt.use_bf16_packed)
    {
        if (support_fp16_storage)
        {
            custom_defines.append("sfp", "uint16_t");
        }
        else
        {
            custom_defines.append("sfp", "uint");
        }
        custom_defines.append("sfpvec2", "uint");
        custom_defines.append("sfpvec4", "uvec2");

        // define pack and unpack macro for bf16p
        custom_defines.append("unpackBFloat2x16(v)", "vec2(uintBitsToFloat(v<<16),uintBitsToFloat(v&0xffff0000u))");
        custom_defines.append("packBFloat2x16(v)", "uint((floatBitsToUint(v.x)>>16)|(floatBitsToUint(v.y)&0xffff0000u))");
    }
    else if (opt.use_fp16_storage)
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

    if (opt.use_bf16_storage || opt.use_bf16_packed)
    {
        // bf16 conflicts with fp16a
        custom_defines.append("afp", "float");
        custom_defines.append("afpvec2", "vec2");
        custom_defines.append("afpvec4", "vec4");
        custom_defines.append("afpmat4", "mat4");
    }
    else if (opt.use_fp16_arithmetic)
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

    if (opt.use_bf16_storage)
    {
        // bf16s implies 16bit uniform
        custom_defines.append("lfp", "bfloat16_t");
        custom_defines.append("lfpvec4", "bf16vec4");
    }
    else if (opt.use_bf16_packed)
    {
        if (support_fp16_uniform)
        {
            custom_defines.append("lfp", "uint16_t");
        }
        else
        {
            custom_defines.append("lfp", "float");
        }
        custom_defines.append("lfpvec4", "uvec2");
    }
    else if (opt.use_fp16_storage && opt.use_fp16_uniform && opt.use_fp16_arithmetic)
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

    if (opt.use_bf16_storage)
    {
        custom_defines.append("buffer_sm1(buf,i)", "buf[i]");
        custom_defines.append("buffer_sm4(buf,i)", "buf[i]");

        custom_defines.append("lfp2afp(v)", "float(v)");
        custom_defines.append("afp2lfp(v)", "bfloat16_t(v)");
        custom_defines.append("lfp2afpvec4(v)", "vec4(v)");
        custom_defines.append("afp2lfpvec4(v)", "bf16vec4(v)");
    }
    else if (opt.use_bf16_packed)
    {
        if (support_fp16_uniform)
        {
            custom_defines.append("buffer_sm1(buf,i)", "buf[i]");
        }
        else if (support_fp16_storage)
        {
            custom_defines.append("buffer_sm1(buf,i)", "uintBitsToFloat(uint(buf[i])<<16)");
        }
        else
        {
            custom_defines.append("buffer_sm1(buf,i)", "unpackBFloat2x16(buf[(i)/2])[(i)%2]");
        }
        custom_defines.append("buffer_sm4(buf,i)", "buf[i]");

        if (support_fp16_uniform)
        {
            custom_defines.append("lfp2afp(v)", "uintBitsToFloat(uint(v)<<16)");
            custom_defines.append("afp2lfp(v)", "uint16_t(floatBitsToUint(v)>>16)");
        }
        else
        {
            custom_defines.append("lfp2afp(v)", "v");
            custom_defines.append("afp2lfp(v)", "v");
        }
        custom_defines.append("lfp2afpvec4(v)", "vec4(unpackBFloat2x16(v.x),unpackBFloat2x16(v.y))");
        custom_defines.append("afp2lfpvec4(v)", "uvec2(packBFloat2x16(v.rg),packBFloat2x16(v.ba))");
    }
    else if (opt.use_fp16_storage && opt.use_fp16_uniform && opt.use_fp16_arithmetic)
    {
        custom_defines.append("buffer_sm1(buf,i)", "buf[i]");
        custom_defines.append("buffer_sm4(buf,i)", "buf[i]");

        custom_defines.append("lfp2afp(v)", "v");
        custom_defines.append("afp2lfp(v)", "v");
        custom_defines.append("lfp2afpvec4(v)", "v");
        custom_defines.append("afp2lfpvec4(v)", "v");
    }
    else if (opt.use_fp16_storage && opt.use_fp16_arithmetic)
    {
        custom_defines.append("buffer_sm1(buf,i)", "float(buf[i])");
        custom_defines.append("buffer_sm4(buf,i)", "pack64(halfBitsToUint16(buf[i]))");

        custom_defines.append("lfp2afp(v)", "float16_t(v)");
        custom_defines.append("afp2lfp(v)", "float(v)");
        custom_defines.append("lfp2afpvec4(v)", "uint16BitsToHalf(unpack16(v))");
        custom_defines.append("afp2lfpvec4(v)", "pack64(halfBitsToUint16(v))");
    }
    else if (opt.use_fp16_packed && opt.use_fp16_arithmetic)
    {
        custom_defines.append("buffer_sm1(buf,i)", "unpackHalf2x16(buf[(i)/2])[(i)%2]");
        custom_defines.append("buffer_sm4(buf,i)", "buf[i]");

        custom_defines.append("lfp2afp(v)", "float16_t(v)");
        custom_defines.append("afp2lfp(v)", "float(v)");
        custom_defines.append("lfp2afpvec4(v)", "f16vec4(unpackFloat2x16(v.x),unpackFloat2x16(v.y))");
        custom_defines.append("afp2lfpvec4(v)", "uvec2(packFloat2x16(v.rg),packFloat2x16(v.ba))");
    }
    else if (opt.use_fp16_storage)
    {
        custom_defines.append("buffer_sm1(buf,i)", "float(buf[i])");
        custom_defines.append("buffer_sm4(buf,i)", "uvec2(packHalf2x16(vec4(buf[i]).rg),packHalf2x16(vec4(buf[i]).ba))");

        custom_defines.append("lfp2afp(v)", "v");
        custom_defines.append("afp2lfp(v)", "float(v)");
        custom_defines.append("lfp2afpvec4(v)", "vec4(unpackHalf2x16(v.x),unpackHalf2x16(v.y))");
        custom_defines.append("afp2lfpvec4(v)", "uvec2(packHalf2x16(v.rg),packHalf2x16(v.ba))");
    }
    else if (opt.use_fp16_packed)
    {
        custom_defines.append("buffer_sm1(buf,i)", "unpackHalf2x16(buf[(i)/2])[(i)%2]");
        custom_defines.append("buffer_sm4(buf,i)", "buf[i]");

        custom_defines.append("lfp2afp(v)", "v");
        custom_defines.append("afp2lfp(v)", "v");
        custom_defines.append("lfp2afpvec4(v)", "vec4(unpackHalf2x16(v.x),unpackHalf2x16(v.y))");
        custom_defines.append("afp2lfpvec4(v)", "uvec2(packHalf2x16(v.rg),packHalf2x16(v.ba))");
    }
    else
    {
        custom_defines.append("buffer_sm1(buf,i)", "buf[i]");
        custom_defines.append("buffer_sm4(buf,i)", "buf[i]");

        custom_defines.append("lfp2afp(v)", "v");
        custom_defines.append("afp2lfp(v)", "v");
        custom_defines.append("lfp2afpvec4(v)", "v");
        custom_defines.append("afp2lfpvec4(v)", "v");
    }

    if (opt.use_bf16_storage)
    {
        custom_defines.append("buffer_ld1(buf,i)", "float(buf[i])");
        custom_defines.append("buffer_st1(buf,i,v)", "{buf[i]=bfloat16_t(v);}");
        custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i].r=sbuf[si4.r];buf[i].g=sbuf[si4.g];buf[i].b=sbuf[si4.b];buf[i].a=sbuf[si4.a];}");
        custom_defines.append("buffer_ld2(buf,i)", "vec2(buf[i])");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=bf16vec2(v);}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "vec4(buf[i])");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=bf16vec4(v);}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{buf[i4.r]=sbuf[si].r;buf[i4.g]=sbuf[si].g;buf[i4.b]=sbuf[si].b;buf[i4.a]=sbuf[si].a;}");
    }
    else if (opt.use_bf16_packed)
    {
        if (support_fp16_storage)
        {
            custom_defines.append("buffer_ld1(buf,i)", "uintBitsToFloat(uint(buf[i])<<16)");
            custom_defines.append("buffer_st1(buf,i,v)", "{buf[i]=uint16_t(floatBitsToUint(v)>>16);}");
            custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");

            custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{buf[i]=uvec2(pack32(u16vec2(sbuf[si4.r],sbuf[si4.g])),pack32(u16vec2(sbuf[si4.b],sbuf[si4.a])));}");
            custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{buf[i4.r]=unpack16(sbuf[si].x).x;buf[i4.g]=unpack16(sbuf[si].x).y;buf[i4.b]=unpack16(sbuf[si].y).x;buf[i4.a]=unpack16(sbuf[si].y).y;}");
        }
        else
        {
            custom_defines.append("buffer_ld1(buf,i)", "unpackBFloat2x16(buf[(i)/2])[(i)%2]");
            custom_defines.append("buffer_st1(buf,i,v)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;float _vs=float(v);uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackBFloat2x16(_old_v);_v[_im2]=_vs;_new_v=packBFloat2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");
            custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;uint _si=uint(si);uint _sid2=_si/2;uint _sim2=_si%2;float v=unpackBFloat2x16(sbuf[_sid2])[_sim2];uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackBFloat2x16(_old_v);_v[_im2]=v;_new_v=packBFloat2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");

            custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{uvec4 _si4d2=uvec4(si4)/2;uvec4 _si4m2=uvec4(si4)%2; buf[i]=uvec2(packBFloat2x16(vec2(unpackBFloat2x16(sbuf[_si4d2.r])[_si4m2.r],unpackBFloat2x16(sbuf[_si4d2.g])[_si4m2.g])),packBFloat2x16(vec2(unpackBFloat2x16(sbuf[_si4d2.b])[_si4m2.b],unpackBFloat2x16(sbuf[_si4d2.a])[_si4m2.a])));}");
            custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{uvec2 _v=sbuf[si];vec2 _v0=unpackBFloat2x16(_v.x);vec2 _v1=unpackBFloat2x16(_v.y);buffer_st1(buf,i4.r,_v0.r);buffer_st1(buf,i4.g,_v0.g);buffer_st1(buf,i4.b,_v1.r);buffer_st1(buf,i4.a,_v1.g);}");
        }

        custom_defines.append("buffer_ld2(buf,i)", "unpackBFloat2x16(buf[i])");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=packBFloat2x16(v);}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "vec4(unpackBFloat2x16(buf[i].x),unpackBFloat2x16(buf[i].y))");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=uvec2(packBFloat2x16(v.rg),packBFloat2x16(v.ba));}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
    }
    else if (opt.use_fp16_storage && opt.use_fp16_arithmetic)
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
        custom_defines.append("buffer_ld1(buf,i)", "float16_t(unpackHalf2x16(buf[(i)/2])[(i)%2])");
        custom_defines.append("buffer_st1(buf,i,v)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;float _vs=float(v);uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackHalf2x16(_old_v);_v[_im2]=_vs;_new_v=packHalf2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");
        custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;uint _si=uint(si);uint _sid2=_si/2;uint _sim2=_si%2;float v=unpackHalf2x16(sbuf[_sid2])[_sim2];uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackHalf2x16(_old_v);_v[_im2]=v;_new_v=packHalf2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");

        custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{uvec4 _si4d2=uvec4(si4)/2;uvec4 _si4m2=uvec4(si4)%2; buf[i]=uvec2(packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.r])[_si4m2.r],unpackHalf2x16(sbuf[_si4d2.g])[_si4m2.g])),packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.b])[_si4m2.b],unpackHalf2x16(sbuf[_si4d2.a])[_si4m2.a])));}");

        custom_defines.append("buffer_ld2(buf,i)", "unpackFloat2x16(buf[i])");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=packFloat2x16(v)}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "f16vec4(unpackFloat2x16(buf[i].x),unpackFloat2x16(buf[i].y))");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=uvec2(packFloat2x16(v.rg),packFloat2x16(v.ba));}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");

        custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{uvec2 _v=sbuf[si];vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y);buffer_st1(buf,i4.r,_v0.r);buffer_st1(buf,i4.g,_v0.g);buffer_st1(buf,i4.b,_v1.r);buffer_st1(buf,i4.a,_v1.g);}");
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
        custom_defines.append("buffer_ld1(buf,i)", "unpackHalf2x16(buf[(i)/2])[(i)%2]");
        custom_defines.append("buffer_st1(buf,i,v)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;float _vs=float(v);uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackHalf2x16(_old_v);_v[_im2]=_vs;_new_v=packHalf2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");
        custom_defines.append("buffer_cp1(buf,i,sbuf,si)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;uint _si=uint(si);uint _sid2=_si/2;uint _sim2=_si%2;float v=unpackHalf2x16(sbuf[_sid2])[_sim2];uint _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);vec2 _v=unpackHalf2x16(_old_v);_v[_im2]=v;_new_v=packHalf2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");

        custom_defines.append("buffer_cp1to4(buf,i,sbuf,si4)", "{uvec4 _si4d2=uvec4(si4)/2;uvec4 _si4m2=uvec4(si4)%2; buf[i]=uvec2(packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.r])[_si4m2.r],unpackHalf2x16(sbuf[_si4d2.g])[_si4m2.g])),packHalf2x16(vec2(unpackHalf2x16(sbuf[_si4d2.b])[_si4m2.b],unpackHalf2x16(sbuf[_si4d2.a])[_si4m2.a])));}");

        custom_defines.append("buffer_ld2(buf,i)", "unpackHalf2x16(buf[i])");
        custom_defines.append("buffer_st2(buf,i,v)", "{buf[i]=packHalf2x16(v);}");
        custom_defines.append("buffer_cp2(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
        custom_defines.append("buffer_ld4(buf,i)", "vec4(unpackHalf2x16(buf[i].x),unpackHalf2x16(buf[i].y))");
        custom_defines.append("buffer_st4(buf,i,v)", "{buf[i]=uvec2(packHalf2x16(v.rg),packHalf2x16(v.ba));}");
        custom_defines.append("buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");

        custom_defines.append("buffer_cp4to1(buf,i4,sbuf,si)", "{uvec2 _v=sbuf[si];vec2 _v0=unpackHalf2x16(_v.x);vec2 _v1=unpackHalf2x16(_v.y);buffer_st1(buf,i4.r,_v0.r);buffer_st1(buf,i4.g,_v0.g);buffer_st1(buf,i4.b,_v1.r);buffer_st1(buf,i4.a,_v1.g);}");
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

    if (opt.use_int16_storage)
    {
        custom_defines.append("NCNN_int16_storage", 1);
        custom_defines.append("sint16", "int16_t");
        custom_defines.append("sint16vec4", "i16vec4");
        custom_defines.append("lint16", "int16_t");
        custom_defines.append("lint16vec4", "i16vec4");
        custom_defines.append("aint16", support_int16_arithmetic ? "int16_t" : "int");
        custom_defines.append("aint16vec4", support_int16_arithmetic ? "i16vec4" : "ivec4");
        custom_defines.append("lint162aint16(v)", support_int16_arithmetic ? "v" : "int(v)");
        custom_defines.append("lint162aint16vec4(v)", support_int16_arithmetic ? "v" : "ivec4(v)");
    }
    else if (opt.use_int16_packed)
    {
        custom_defines.append("NCNN_int16_packed", 1);
        custom_defines.append("sint16", "int");
        custom_defines.append("sint16vec4", "ivec2");
        custom_defines.append("lint16", "int");
        custom_defines.append("lint16vec4", "ivec2");
        custom_defines.append("aint16", support_int16_arithmetic ? "int16_t" : "int");
        custom_defines.append("aint16vec4", support_int16_arithmetic ? "i16vec4" : "ivec4");
        custom_defines.append("lint162aint16(v)", support_int16_arithmetic ? "int16_t(v)" : "int(v)");
        custom_defines.append("lint162aint16vec4(v)", support_int16_arithmetic ? "i16vec4(unpack16(v.r),unpack16(v.g))" : "ivec4(unpackInt2x16(v.r),unpackInt2x16(v.g))");
    }
    else
    {
        custom_defines.append("sint16", "int");
        custom_defines.append("sint16vec4", "ivec4");
        custom_defines.append("lint16", "int");
        custom_defines.append("lint16vec4", "ivec4");
        custom_defines.append("aint16", "int");
        custom_defines.append("aint16vec4", "ivec4");
        custom_defines.append("lint162aint16(v)", "v");
        custom_defines.append("lint162aint16vec4(v)", "v");
    }

    custom_defines.append("sint8vec4", "int");

    custom_defines.append("aint8", "int");
    custom_defines.append("aint8vec4", "ivec4");

    custom_defines.append("unpackInt4x8(v)", "ivec4((v<<24)>>24,(v<<16)>>24,(v<<8)>>24,v>>24)");
    custom_defines.append("packInt4x8(v)", "int((uint(v.r)&0xFFu)|((uint(v.g)&0xFFu)<<8)|((uint(v.b)&0xFFu)<<16)|((uint(v.a)&0xFFu)<<24))");
    custom_defines.append("unpackInt2x16(v)", "ivec2((int(v)<<16)>>16,int(v)>>16)");
    custom_defines.append("packInt2x16(v)", "int((uint(v.r)&0xFFFFu)|((uint(v.g)&0xFFFFu)<<16))");
    custom_defines.append("float2int8(v)", "int(clamp(float(v)+(float(v)>=0.f?0.5f:-0.5f),-127.f,127.f))");
    custom_defines.append("float2int8vec4(v)", "ivec4(clamp(vec4(v)+mix(vec4(-0.5f),vec4(0.5f),greaterThanEqual(vec4(v),vec4(0.f))),vec4(-127.f),vec4(127.f)))");

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
    custom_defines.append("i8buffer_sm4(buf,i)", "buf[i]");
    custom_defines.append("i8buffer_st4(buf,i,v)", "{buf[i]=packInt4x8(v);}");
    custom_defines.append("i8buffer_cp4(buf,i,sbuf,si)", "{buf[i]=sbuf[si];}");
    custom_defines.append("i8buffer_cp1to4(buf,i,sbuf,si)", "{ivec4 _v=ivec4(i8buffer_ld1(sbuf,si.r),i8buffer_ld1(sbuf,si.g),i8buffer_ld1(sbuf,si.b),i8buffer_ld1(sbuf,si.a));i8buffer_st4(buf,i,_v);}");
    custom_defines.append("i8buffer_cp4to1(buf,i4,sbuf,si)", "{ivec4 _v=i8buffer_ld4(sbuf,si);i8buffer_st1(buf,i4.r,_v.r);i8buffer_st1(buf,i4.g,_v.g);i8buffer_st1(buf,i4.b,_v.b);i8buffer_st1(buf,i4.a,_v.a);}");

    if (opt.use_int16_storage)
    {
        custom_defines.append("i16buffer_ld1(buf,i)", "int(buf[i])");
        custom_defines.append("i16buffer_st1(buf,i,v)", "{buf[i]=int16_t(v);}");
    }
    else if (opt.use_int16_packed)
    {
        custom_defines.append("i16buffer_ld1(buf,i)", "unpackInt2x16(buf[(i)/2])[(i)%2]");
        custom_defines.append("i16buffer_st1(buf,i,v)", "{uint _i=uint(i);uint _id2=_i/2;uint _im2=_i%2;int _vs=int(v);int _old_v, _new_v;do{_old_v=atomicCompSwap(buf[_id2],0,0);ivec2 _v=unpackInt2x16(_old_v);_v[_im2]=_vs;_new_v=packInt2x16(_v);} while(atomicCompSwap(buf[_id2],_old_v,_new_v)!=_old_v);}");
    }
    else
    {
        custom_defines.append("i16buffer_ld1(buf,i)", "int(buf[i])");
        custom_defines.append("i16buffer_st1(buf,i,v)", "{buf[i]=int(v);}");
    }
    custom_defines.append("i16buffer_ld2(buf,i)", "ivec2(i16buffer_ld1(buf,i),i16buffer_ld1(buf,(i)+1))");
    if (opt.use_int16_storage)
    {
        custom_defines.append("i16buffer_st2(buf,i,v)", "{ivec2 _v=ivec2(v);buf[i]=int16_t(_v.r);buf[(i)+1]=int16_t(_v.g);}");
        custom_defines.append("i16buffer_sm4(buf,i)", "buf[i]");
        custom_defines.append("i16buffer_ld4(buf,i)", support_int16_arithmetic ? "buf[i]" : "ivec4(buf[i])");
        custom_defines.append("i16buffer_st4(buf,i,v)", "{buf[i]=i16vec4(v);}");
    }
    else if (opt.use_int16_packed)
    {
        custom_defines.append("i16buffer_st2(buf,i,v)", "{uint _i=uint(i);ivec2 _v=ivec2(v);if((_i&1u)==0u){buf[_i/2]=packInt2x16(_v);}else{i16buffer_st1(buf,int(_i),_v.r);i16buffer_st1(buf,int(_i)+1,_v.g);}}");
        custom_defines.append("i16buffer_sm4(buf,i)", "buf[i]");
        custom_defines.append("i16buffer_ld4(buf,i)", support_int16_arithmetic ? "i16vec4(unpack16(buf[i].r),unpack16(buf[i].g))" : "ivec4(unpackInt2x16(buf[i].r),unpackInt2x16(buf[i].g))");
        custom_defines.append("i16buffer_st4(buf,i,v)", "{ivec4 _v=ivec4(v);buf[i]=ivec2(packInt2x16(ivec2(_v.r,_v.g)),packInt2x16(ivec2(_v.b,_v.a)));}");
    }
    else
    {
        custom_defines.append("i16buffer_st2(buf,i,v)", "{ivec2 _v=ivec2(v);buf[i]=int(_v.r);buf[(i)+1]=int(_v.g);}");
        custom_defines.append("i16buffer_sm4(buf,i)", "buf[i]");
        custom_defines.append("i16buffer_ld4(buf,i)", "ivec4(buf[i])");
        custom_defines.append("i16buffer_st4(buf,i,v)", "{buf[i]=ivec4(v);}");
    }

    custom_defines.append("psc(x)", "(x==0?p.x:x)");

    if (opt.use_bf16_storage)
    {
        custom_defines.append("NCNN_bf16_storage", 1);
    }
    else if (opt.use_bf16_packed)
    {
        custom_defines.append("NCNN_bf16_packed", 1);
    }
    else if (opt.use_fp16_storage)
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

    std::string define_macro_data;
    for (size_t i = 0; i < custom_defines.definitions.size(); i++)
    {
        const char* key = custom_defines.definitions[i].first;
        const DefinitionCollector::typed_value& def = custom_defines.definitions[i].second;

        char defstr[256];
        if (def.type == 0)
        {
            define_macro_data += std::string("#define ") + key + " " + def.s + "\n";
            continue;
        }
        if (def.type == 1)
            sprintf(defstr, "%u", def.u8);
        if (def.type == 2)
            sprintf(defstr, "%u", def.u32);
        if (def.type == 3)
            sprintf(defstr, "%d", def.i32);
        if (def.type == 4)
            sprintf(defstr, "%lluull", (unsigned long long)def.u64);
        if (def.type == 5)
            sprintf(defstr, "%e", def.f32);

        define_macro_data += std::string("#define ") + key + " " + defstr + "\n";
    }

    preamble = "#extension GL_EXT_shader_explicit_arithmetic_types_int64: require\n"
               "#extension GL_EXT_shader_explicit_arithmetic_types_int16: require\n";
    preamble += define_macro_data;
    return 0;
}

class WebGpuShaderIncluder : public glslang::TShader::Includer
{
public:
    virtual glslang::TShader::Includer::IncludeResult* includeLocal(const char* header_name, const char*, size_t)
    {
        if (strcmp(header_name, "vulkan_activation.comp") == 0)
            return new glslang::TShader::Includer::IncludeResult(header_name, vulkan_activation_comp_data, sizeof(vulkan_activation_comp_data), 0);

        return 0;
    }

    virtual void releaseInclude(glslang::TShader::Includer::IncludeResult* result)
    {
        delete result;
    }
};

static int normalize_webgpu_spirv(std::vector<uint32_t>& spirv)
{
    if (spirv.size() < 5 || spirv[0] != (uint32_t)spv::MagicNumber)
        return -1;

    const uint32_t bound = spirv[3];
    uint32_t local_size_ids[3] = {0, 0, 0};
    uint32_t workgroup_size_builtin_id = 0;
    std::vector<int32_t> spec_ids(bound, -1);
    std::vector<uint32_t> workgroup_size_members;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        if (opcode == spv::Op::OpExecutionModeId && word_count >= 6 && spirv[i + 2] == (uint32_t)spv::ExecutionMode::LocalSizeId)
        {
            local_size_ids[0] = spirv[i + 3];
            local_size_ids[1] = spirv[i + 4];
            local_size_ids[2] = spirv[i + 5];
        }
        else if (opcode == spv::Op::OpDecorate && word_count >= 4 && spirv[i + 1] < bound)
        {
            if (spirv[i + 2] == (uint32_t)spv::Decoration::SpecId)
                spec_ids[spirv[i + 1]] = spirv[i + 3];
            else if (spirv[i + 2] == (uint32_t)spv::Decoration::BuiltIn && spirv[i + 3] == (uint32_t)spv::BuiltIn::WorkgroupSize)
                workgroup_size_builtin_id = spirv[i + 1];
        }
        else if (opcode == spv::Op::OpSpecConstantComposite && word_count == 6 && spirv[i + 2] == workgroup_size_builtin_id)
        {
            workgroup_size_members.assign(spirv.begin() + i + 3, spirv.begin() + i + 6);
        }

        i += word_count;
    }

    bool has_workgroup_size_builtin = workgroup_size_members.size() == 3;
    for (int i = 0; i < 3 && has_workgroup_size_builtin; i++)
    {
        const uint32_t id = workgroup_size_members[i];
        if (id >= bound || spec_ids[id] != 233 + i)
            has_workgroup_size_builtin = false;
    }

    const bool has_local_size_id_mode = local_size_ids[0] != 0 && local_size_ids[1] != 0 && local_size_ids[2] != 0;

    std::vector<uint32_t> normalized;
    normalized.reserve(spirv.size());
    normalized.insert(normalized.end(), spirv.begin(), spirv.begin() + 5);

    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        const bool non_readable_decorate = opcode == spv::Op::OpDecorate && word_count >= 3 && spirv[i + 2] == (uint32_t)spv::Decoration::NonReadable;
        const bool non_readable_member_decorate = opcode == spv::Op::OpMemberDecorate && word_count >= 4 && spirv[i + 3] == (uint32_t)spv::Decoration::NonReadable;
        const bool literal_local_size = has_workgroup_size_builtin && opcode == spv::Op::OpExecutionMode && word_count == 6 && spirv[i + 2] == (uint32_t)spv::ExecutionMode::LocalSize;
        bool duplicate_local_size_spec_id = false;
        if (has_local_size_id_mode && opcode == spv::Op::OpDecorate && word_count >= 4 && spirv[i + 2] == (uint32_t)spv::Decoration::SpecId)
        {
            const uint32_t spec_id = spirv[i + 3];
            if (spec_id >= 233 && spec_id <= 235 && spirv[i + 1] != local_size_ids[spec_id - 233])
                duplicate_local_size_spec_id = true;
        }

        if (!non_readable_decorate && !non_readable_member_decorate && !literal_local_size && !duplicate_local_size_spec_id)
            normalized.insert(normalized.end(), spirv.begin() + i, spirv.begin() + i + word_count);

        i += word_count;
    }

    spirv.swap(normalized);
    return 0;
}

static int compile_glsl_to_spirv(const char* source, int source_size, const std::string& preamble, std::vector<uint32_t>& spirv)
{
    spirv.clear();

    if (!source || source_size <= 0 || g_webgpu_compiler.initialize() != 0)
        return -1;

    int version_end_pos = -1;
    for (int i = 0; i < source_size - 8; i++)
    {
        if (strncmp(source + i, "#version", 8) != 0)
            continue;
        if (i != 0 && source[i - 1] != '\n')
            continue;

        int nversion = 0;
        sscanf(source + i, "#version %*d\n%n", &nversion);
        if (nversion == 0)
            continue;

        version_end_pos = i + nversion;
        break;
    }

    if (version_end_pos == -1)
    {
        NCNN_LOGE("WebGPU shader source has no #version token");
        return -1;
    }

    const size_t define_pos = preamble.find("#define");
    const std::string custom_exts = preamble.substr(0, define_pos);
    const std::string define_macro_data = define_pos == std::string::npos ? std::string() : preamble.substr(define_pos);
    const char* source_2 = source + version_end_pos;
    const int source_size_1 = version_end_pos;
    const int source_size_2 = source_size - source_size_1;
    const bool has_local_size = strstr(source_2, "local_size_x") != 0;
    const std::string webgpu_local_size = has_local_size ? std::string() : "layout(local_size_x_id=233,local_size_y_id=234,local_size_z_id=235) in;\n";
    const char* source_strings[5] = {source, custom_exts.c_str(), define_macro_data.c_str(), webgpu_local_size.c_str(), source_2};
    const int source_sizes[5] = {source_size_1, (int)custom_exts.size(), (int)define_macro_data.size(), (int)webgpu_local_size.size(), source_size_2};

    glslang::TShader shader(EShLangCompute);
    shader.setStringsWithLengths(source_strings, source_sizes, 5);
    shader.setEntryPoint("main");
    shader.setSourceEntryPoint("main");
    shader.setEnvInput(glslang::EShSourceGlsl, EShLangCompute, glslang::EShClientVulkan, 1);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
    shader.setEnvTarget(glslang::EshTargetSpv, glslang::EShTargetSpv_1_3);

    WebGpuShaderIncluder includer;
    if (!shader.parse(GetDefaultResources(), 100, ENoProfile, false, false, EShMsgDefault, includer))
    {
        NCNN_LOGE("WebGPU glslang parse failed: %s", shader.getInfoLog());
        NCNN_LOGE("%s", shader.getInfoDebugLog());
        return -1;
    }

    glslang::GlslangToSpv(*shader.getIntermediate(), spirv);

    if (normalize_webgpu_spirv(spirv) != 0)
    {
        NCNN_LOGE("WebGPU glslang produced an invalid SPIR-V module");
        spirv.clear();
        return -1;
    }

    return 0;
}

struct WebGpuSpirvType
{
    WebGpuSpirvType()
        : opcode(spv::Op::OpNop), width(0), signedness(0), storage_class(0), pointee_type(0), element_count(0), length_id(0), array_stride(0)
    {
    }

    spv::Op opcode;
    uint32_t width;
    uint32_t signedness;
    uint32_t storage_class;
    uint32_t pointee_type;
    uint32_t element_count;
    uint32_t length_id;
    uint32_t array_stride;
    std::vector<uint32_t> member_types;
};

static WebGpuScalarType scalar_type_from_spirv(const WebGpuSpirvType& type)
{
    if (type.opcode == spv::Op::OpTypeBool)
        return NCNN_WEBGPU_SCALAR_BOOL;
    if (type.opcode == spv::Op::OpTypeInt && type.width == 32)
        return type.signedness ? NCNN_WEBGPU_SCALAR_I32 : NCNN_WEBGPU_SCALAR_U32;
    if (type.opcode == spv::Op::OpTypeFloat && type.width == 32)
        return NCNN_WEBGPU_SCALAR_F32;
    if (type.opcode == spv::Op::OpTypeFloat && type.width == 16)
        return NCNN_WEBGPU_SCALAR_F16;

    return (WebGpuScalarType)0;
}

static int resolve_webgpu_type_layout(uint32_t type_id, const std::vector<WebGpuSpirvType>& types, const std::vector<uint32_t>& constants, const std::vector<bool>& constant_known, const std::vector<std::vector<int32_t> >& member_offsets, uint64_t& size, uint64_t& alignment)
{
    if (type_id >= types.size())
        return -1;

    const WebGpuSpirvType& type = types[type_id];
    if (type.opcode == spv::Op::OpTypeBool
            || type.opcode == spv::Op::OpTypeInt
            || type.opcode == spv::Op::OpTypeFloat)
    {
        if (type.opcode != spv::Op::OpTypeBool && type.width != 32)
            return -1;

        size = 4;
        alignment = 4;
        return 0;
    }

    if (type.opcode == spv::Op::OpTypeVector)
    {
        uint64_t scalar_size;
        uint64_t scalar_alignment;
        if (resolve_webgpu_type_layout(type.pointee_type, types, constants, constant_known, member_offsets, scalar_size, scalar_alignment) != 0
                || type.element_count < 2 || type.element_count > 4)
            return -1;

        size = scalar_size * type.element_count;
        alignment = scalar_alignment * (type.element_count == 2 ? 2 : 4);
        return 0;
    }

    if (type.opcode == spv::Op::OpTypeArray)
    {
        if (type.length_id >= constants.size() || !constant_known[type.length_id])
            return -1;

        uint64_t element_size;
        uint64_t element_alignment;
        if (resolve_webgpu_type_layout(type.pointee_type, types, constants, constant_known, member_offsets, element_size, element_alignment) != 0)
            return -1;

        const uint64_t stride = type.array_stride ? type.array_stride : (element_size + element_alignment - 1) / element_alignment * element_alignment;
        if (stride < element_size || stride % element_alignment != 0)
            return -1;
        if (constants[type.length_id] != 0 && stride > UINT64_MAX / constants[type.length_id])
            return -1;
        size = stride * constants[type.length_id];
        alignment = element_alignment;
        return 0;
    }

    if (type.opcode == spv::Op::OpTypeStruct)
    {
        if (type_id >= member_offsets.size() || member_offsets[type_id].size() != type.member_types.size())
            return -1;

        size = 0;
        alignment = 1;
        for (size_t i = 0; i < type.member_types.size(); i++)
        {
            uint64_t member_size;
            uint64_t member_alignment;
            if (member_offsets[type_id][i] < 0
                    || resolve_webgpu_type_layout(type.member_types[i], types, constants, constant_known, member_offsets, member_size, member_alignment) != 0)
                return -1;

            const uint64_t member_offset = member_offsets[type_id][i];
            if (member_offset % member_alignment != 0 || member_offset + member_size < member_offset)
                return -1;

            size = std::max(size, member_offset + member_size);
            alignment = std::max(alignment, member_alignment);
        }

        size = (size + alignment - 1) / alignment * alignment;
        return 0;
    }

    return -1;
}

static int reflect_webgpu_shader(const std::vector<uint32_t>& spirv, WebGpuShaderInfo& info)
{
    info.immediate_size = 0;
    info.workgroup_storage_size = 0;
    info.required_feature_bits = 0;
    info.bindings.clear();
    info.overrides.clear();
    info.immediate_members.clear();

    if (spirv.size() < 5 || spirv[0] != (uint32_t)spv::MagicNumber)
    {
        NCNN_LOGE("WebGPU SPIR-V reflection found an invalid module header");
        return -1;
    }

    const uint32_t bound = spirv[3];
    std::vector<WebGpuSpirvType> types(bound);
    std::vector<uint32_t> variable_types(bound);
    std::vector<uint32_t> variable_storage_classes(bound);
    std::vector<int32_t> descriptor_sets(bound, -1);
    std::vector<int32_t> binding_numbers(bound, -1);
    std::vector<int32_t> spec_ids(bound, -1);
    std::vector<uint32_t> spec_types(bound);
    std::vector<uint32_t> constants(bound);
    std::vector<bool> constant_known(bound);
    std::vector<bool> non_writable(bound);
    std::vector<std::vector<int32_t> > member_offsets(bound);
    std::vector<std::vector<uint32_t> > composite_members(bound);

    uint32_t push_constant_struct = 0;
    uint32_t workgroup_size_builtin_id = 0;
    uint32_t local_size_ids[3] = {0, 0, 0};
    bool has_local_size_id_mode = false;
    int entry_point_count = 0;

    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t instruction = spirv[i];
        const uint32_t word_count = instruction >> 16;
        const spv::Op opcode = (spv::Op)(instruction & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
        {
            NCNN_LOGE("WebGPU SPIR-V reflection found a malformed instruction");
            return -1;
        }

        const uint32_t* p = &spirv[i];
        if (opcode == spv::Op::OpEntryPoint)
        {
            const char* name = (const char*)&p[3];
            if (word_count < 4 || p[1] != (uint32_t)spv::ExecutionModel::GLCompute || strcmp(name, "main") != 0)
            {
                NCNN_LOGE("WebGPU shader must have one compute entry point named main");
                return -1;
            }
            entry_point_count++;
        }
        else if (opcode == spv::Op::OpExecutionModeId && word_count >= 6 && p[2] == (uint32_t)spv::ExecutionMode::LocalSizeId)
        {
            local_size_ids[0] = p[3];
            local_size_ids[1] = p[4];
            local_size_ids[2] = p[5];
            has_local_size_id_mode = true;
        }
        else if (opcode == spv::Op::OpDecorate && word_count >= 3 && p[1] < bound)
        {
            if (p[2] == (uint32_t)spv::Decoration::DescriptorSet && word_count >= 4)
                descriptor_sets[p[1]] = p[3];
            else if (p[2] == (uint32_t)spv::Decoration::Binding && word_count >= 4)
                binding_numbers[p[1]] = p[3];
            else if (p[2] == (uint32_t)spv::Decoration::SpecId && word_count >= 4)
                spec_ids[p[1]] = p[3];
            else if (p[2] == (uint32_t)spv::Decoration::ArrayStride && word_count >= 4)
                types[p[1]].array_stride = p[3];
            else if (p[2] == (uint32_t)spv::Decoration::NonWritable)
                non_writable[p[1]] = true;
            else if (p[2] == (uint32_t)spv::Decoration::BuiltIn && word_count >= 4 && p[3] == (uint32_t)spv::BuiltIn::WorkgroupSize)
                workgroup_size_builtin_id = p[1];
        }
        else if (opcode == spv::Op::OpMemberDecorate && word_count >= 4 && p[1] < bound)
        {
            if (p[3] == (uint32_t)spv::Decoration::Offset && word_count >= 5)
            {
                if (member_offsets[p[1]].size() <= p[2])
                    member_offsets[p[1]].resize(p[2] + 1, -1);
                member_offsets[p[1]][p[2]] = p[4];
            }
            else if (p[3] == (uint32_t)spv::Decoration::NonWritable)
            {
                non_writable[p[1]] = true;
            }
        }
        else if ((opcode == spv::Op::OpTypeBool) && word_count >= 2 && p[1] < bound)
        {
            types[p[1]].opcode = opcode;
        }
        else if ((opcode == spv::Op::OpTypeInt || opcode == spv::Op::OpTypeFloat) && word_count >= 3 && p[1] < bound)
        {
            types[p[1]].opcode = opcode;
            types[p[1]].width = p[2];
            if (opcode == spv::Op::OpTypeInt && word_count >= 4)
                types[p[1]].signedness = p[3];
        }
        else if (opcode == spv::Op::OpTypeRuntimeArray && word_count >= 3 && p[1] < bound)
        {
            types[p[1]].opcode = opcode;
            types[p[1]].pointee_type = p[2];
        }
        else if (opcode == spv::Op::OpTypeVector && word_count >= 4 && p[1] < bound)
        {
            types[p[1]].opcode = opcode;
            types[p[1]].pointee_type = p[2];
            types[p[1]].element_count = p[3];
        }
        else if (opcode == spv::Op::OpTypeArray && word_count >= 4 && p[1] < bound)
        {
            types[p[1]].opcode = opcode;
            types[p[1]].pointee_type = p[2];
            types[p[1]].length_id = p[3];
        }
        else if (opcode == spv::Op::OpTypeStruct && word_count >= 2 && p[1] < bound)
        {
            types[p[1]].opcode = opcode;
            types[p[1]].member_types.assign(p + 2, p + word_count);
        }
        else if (opcode == spv::Op::OpTypePointer && word_count >= 4 && p[1] < bound)
        {
            types[p[1]].opcode = opcode;
            types[p[1]].storage_class = p[2];
            types[p[1]].pointee_type = p[3];
        }
        else if (opcode == spv::Op::OpVariable && word_count >= 4 && p[2] < bound)
        {
            variable_types[p[2]] = p[1];
            variable_storage_classes[p[2]] = p[3];
        }
        else if ((opcode == spv::Op::OpSpecConstantTrue || opcode == spv::Op::OpSpecConstantFalse || opcode == spv::Op::OpSpecConstant)
                 && word_count >= 3 && p[2] < bound)
        {
            spec_types[p[2]] = p[1];
        }
        else if (opcode == spv::Op::OpConstant && word_count >= 4 && p[2] < bound)
        {
            constants[p[2]] = p[3];
            constant_known[p[2]] = true;
        }
        else if (opcode == spv::Op::OpSpecConstantComposite && word_count >= 4 && p[2] < bound)
        {
            spec_types[p[2]] = p[1];
            composite_members[p[2]].assign(p + 3, p + word_count);
        }

        i += word_count;
    }

    if (entry_point_count != 1)
    {
        NCNN_LOGE("WebGPU shader has %d entry points, expected one", entry_point_count);
        return -1;
    }

    if (local_size_ids[0] == 0 && workgroup_size_builtin_id < bound && composite_members[workgroup_size_builtin_id].size() == 3)
    {
        local_size_ids[0] = composite_members[workgroup_size_builtin_id][0];
        local_size_ids[1] = composite_members[workgroup_size_builtin_id][1];
        local_size_ids[2] = composite_members[workgroup_size_builtin_id][2];
    }
    const bool has_workgroup_size_builtin = workgroup_size_builtin_id < bound && composite_members[workgroup_size_builtin_id].size() == 3;
    if (!has_local_size_id_mode && !has_workgroup_size_builtin)
    {
        NCNN_LOGE("WebGPU shader has no specialized workgroup size");
        return -1;
    }

    for (uint32_t id = 1; id < bound; id++)
    {
        if (variable_storage_classes[id] == (uint32_t)spv::StorageClass::Workgroup)
        {
            const uint32_t pointer_type = variable_types[id];
            if (pointer_type >= bound || types[pointer_type].opcode != spv::Op::OpTypePointer)
                return -1;

            uint64_t variable_size;
            uint64_t variable_alignment;
            if (resolve_webgpu_type_layout(types[pointer_type].pointee_type, types, constants, constant_known, member_offsets, variable_size, variable_alignment) != 0)
            {
                NCNN_LOGE("WebGPU shader workgroup variable %u has unsupported layout", id);
                return -1;
            }

            uint64_t workgroup_size = info.workgroup_storage_size;
            workgroup_size = (workgroup_size + variable_alignment - 1) / variable_alignment * variable_alignment;
            workgroup_size += variable_size;
            if (workgroup_size > UINT_MAX)
                return -1;
            info.workgroup_storage_size = workgroup_size;
        }

        if (variable_storage_classes[id] == (uint32_t)spv::StorageClass::PushConstant)
        {
            const uint32_t pointer_type = variable_types[id];
            if (pointer_type >= bound || types[pointer_type].opcode != spv::Op::OpTypePointer)
                return -1;
            if (push_constant_struct != 0)
            {
                NCNN_LOGE("WebGPU shader has multiple immediate blocks");
                return -1;
            }
            push_constant_struct = types[pointer_type].pointee_type;
        }

        if (binding_numbers[id] < 0)
            continue;

        if (descriptor_sets[id] != 0)
        {
            NCNN_LOGE("WebGPU shader binding %d uses descriptor set %d", binding_numbers[id], descriptor_sets[id]);
            return -1;
        }

        const uint32_t storage_class = variable_storage_classes[id];
        if (storage_class != (uint32_t)spv::StorageClass::StorageBuffer)
        {
            NCNN_LOGE("WebGPU shader binding %d is not a storage buffer", binding_numbers[id]);
            return -1;
        }

        WebGpuBindingInfo binding;
        binding.binding = binding_numbers[id];
        const uint32_t pointer_type = variable_types[id];
        const uint32_t pointee_type = pointer_type < bound ? types[pointer_type].pointee_type : 0;
        binding.access = (non_writable[id] || (pointer_type < bound && non_writable[pointer_type]) || (pointee_type < bound && non_writable[pointee_type])) ? NCNN_WEBGPU_BINDING_READ : NCNN_WEBGPU_BINDING_READ_WRITE;
        binding.min_binding_size = 0;
        uint64_t binding_size;
        uint64_t binding_alignment;
        if (resolve_webgpu_type_layout(pointee_type, types, constants, constant_known, member_offsets, binding_size, binding_alignment) == 0)
            binding.min_binding_size = binding_size;
        info.bindings.push_back(binding);
    }

    std::sort(info.bindings.begin(), info.bindings.end(), [](const WebGpuBindingInfo& a, const WebGpuBindingInfo& b) {
        return a.binding < b.binding;
    });

    for (size_t i = 1; i < info.bindings.size(); i++)
    {
        if (info.bindings[i - 1].binding == info.bindings[i].binding)
        {
            NCNN_LOGE("WebGPU shader has duplicate binding %u", info.bindings[i].binding);
            return -1;
        }
    }

    if (push_constant_struct)
    {
        if (push_constant_struct >= bound || types[push_constant_struct].opcode != spv::Op::OpTypeStruct)
            return -1;

        const WebGpuSpirvType& block_type = types[push_constant_struct];
        if (member_offsets[push_constant_struct].size() != block_type.member_types.size())
        {
            NCNN_LOGE("WebGPU immediate block is missing member offsets");
            return -1;
        }

        for (size_t i = 0; i < block_type.member_types.size(); i++)
        {
            const uint32_t member_type_id = block_type.member_types[i];
            if (member_type_id >= bound)
                return -1;

            const WebGpuScalarType scalar_type = scalar_type_from_spirv(types[member_type_id]);
            const int32_t byte_offset = member_offsets[push_constant_struct][i];
            if (scalar_type == 0 || byte_offset < 0 || byte_offset % 4 != 0 || scalar_type == NCNN_WEBGPU_SCALAR_F16)
            {
                NCNN_LOGE("WebGPU immediate member %zu is not an aligned 32-bit scalar", i);
                return -1;
            }

            WebGpuImmediateMember member;
            member.ncnn_constant_index = i;
            member.byte_offset = byte_offset;
            member.type = scalar_type;
            info.immediate_members.push_back(member);
            info.immediate_size = std::max(info.immediate_size, member.byte_offset + 4);
        }

        for (size_t i = 1; i < info.immediate_members.size(); i++)
        {
            if (info.immediate_members[i - 1].byte_offset + 4 > info.immediate_members[i].byte_offset)
            {
                NCNN_LOGE("WebGPU immediate members overlap");
                return -1;
            }
        }

        if (info.immediate_size > 64)
        {
            NCNN_LOGE("WebGPU immediate block size %u exceeds first-phase limit 64", info.immediate_size);
            return -1;
        }
    }

    for (uint32_t id = 1; id < bound; id++)
    {
        if (spec_ids[id] < 0)
            continue;

        const uint32_t type_id = spec_types[id];
        if (type_id >= bound)
            return -1;

        const WebGpuScalarType scalar_type = scalar_type_from_spirv(types[type_id]);
        if (scalar_type == 0 || scalar_type == NCNN_WEBGPU_SCALAR_F16)
        {
            NCNN_LOGE("WebGPU specialization id %d has unsupported type", spec_ids[id]);
            return -1;
        }

        WebGpuOverrideInfo override_info;
        override_info.spec_id = spec_ids[id];
        override_info.type = scalar_type;
        override_info.is_workgroup_size = id == local_size_ids[0] || id == local_size_ids[1] || id == local_size_ids[2];
        override_info.ncnn_specialization_index = override_info.is_workgroup_size ? -1 : spec_ids[id];
        info.overrides.push_back(override_info);
    }

    std::sort(info.overrides.begin(), info.overrides.end(), [](const WebGpuOverrideInfo& a, const WebGpuOverrideInfo& b) {
        return a.spec_id < b.spec_id;
    });

    for (size_t i = 1; i < info.overrides.size(); i++)
    {
        if (info.overrides[i - 1].spec_id == info.overrides[i].spec_id)
        {
            NCNN_LOGE("WebGPU shader has duplicate specialization id %u", info.overrides[i].spec_id);
            return -1;
        }
    }

    const uint32_t expected_local_size_spec_ids[3] = {233, 234, 235};
    for (int i = 0; i < 3; i++)
    {
        bool found = false;
        for (size_t j = 0; j < info.overrides.size(); j++)
        {
            if (info.overrides[j].is_workgroup_size && info.overrides[j].spec_id == expected_local_size_spec_ids[i])
                found = true;
        }
        if (!found)
        {
            NCNN_LOGE("WebGPU shader is missing local-size specialization id %u", expected_local_size_spec_ids[i]);
            return -1;
        }
    }

    return 0;
}

static int convert_spirv_to_wgsl(const std::vector<uint32_t>& spirv, std::string& wgsl, WebGpuShaderInfo& info)
{
    wgsl.clear();

    if (g_webgpu_compiler.initialize() != 0 || reflect_webgpu_shader(spirv, info) != 0)
        return -1;

    tint::wgsl::writer::Options options;
    options.allowed_features.features.insert(tint::wgsl::LanguageFeature::kImmediateAddressSpace);
    tint::Result<std::string> result = tint::SpirvToWgsl(spirv, options);
    if (result != tint::Success)
    {
        NCNN_LOGE("WebGPU Tint SPIR-V conversion failed: %s", result.Failure().reason.c_str());
        return -1;
    }

    wgsl = result.Get();
    return 0;
}

static int translate_webgpu_shader(const WebGpuShaderCompileOptions& options, WebGpuTranslatedShader& translated)
{
    translated.spirv.clear();
    translated.wgsl.clear();
    translated.shader_info = WebGpuShaderInfo();
    translated.compile_key.shader_type = -1;
    translated.compile_key.option_bits = 0;
    translated.compile_key.feature_bits = 0;

    if (validate_webgpu_shader_compile_options(options) != 0)
        return -1;

    std::string preamble;
    if (build_webgpu_shader_preamble(*options.opt, preamble) != 0)
        return -1;

    const layer_shader_registry_entry& shader_entry = layer_shader_registry[options.shader_type];
    if (compile_glsl_to_spirv(shader_entry.comp_data, shader_entry.comp_data_size, preamble, translated.spirv) != 0)
    {
        NCNN_LOGE("WebGPU shader translation failed type=%d stage=glslang", options.shader_type);
        return -1;
    }
    if (convert_spirv_to_wgsl(translated.spirv, translated.wgsl, translated.shader_info) != 0)
    {
        NCNN_LOGE("WebGPU shader translation failed type=%d stage=spirv-to-wgsl", options.shader_type);
        return -1;
    }

    translated.compile_key.shader_type = options.shader_type;
    translated.compile_key.option_bits = get_webgpu_shader_option_bits(*options.opt);
    const uint32_t profile = options.opt->use_fp16_packed ? WEBGPU_SHADER_PROFILE_FP16_PACKED : WEBGPU_SHADER_PROFILE_FP32;
    translated.compile_key.feature_bits = profile | (translated.shader_info.required_feature_bits << 8);
    return 0;
}

struct WebGpuCompilationResult
{
    WGPUCompilationInfoRequestStatus status;
    bool completed;
    bool has_error;
};

static void webgpu_compilation_info_callback(WGPUCompilationInfoRequestStatus status, const WGPUCompilationInfo* info, void* userdata1, void*)
{
    WebGpuCompilationResult* result = (WebGpuCompilationResult*)userdata1;
    result->status = status;
    result->completed = true;

    if (!info)
    {
        result->has_error = true;
        return;
    }

    for (size_t i = 0; i < info->messageCount; i++)
    {
        const WGPUCompilationMessage& message = info->messages[i];
        if (message.type == WGPUCompilationMessageType_Error)
            result->has_error = true;

        NCNN_LOGE("WebGPU shader compilation type=%d line=%llu column=%llu %.*s",
                  (int)message.type,
                  (unsigned long long)message.lineNum,
                  (unsigned long long)message.linePos,
                  (int)message.message.length,
                  message.message.data ? message.message.data : "");
    }
}

static int check_webgpu_shader_compilation(const VulkanDevice* vkdev, WGPUShaderModule shader_module)
{
    WebGpuCompilationResult result;
    result.status = WGPUCompilationInfoRequestStatus_CallbackCancelled;
    result.completed = false;
    result.has_error = false;

    WGPUCompilationInfoCallbackInfo callback_info = WGPU_COMPILATION_INFO_CALLBACK_INFO_INIT;
    callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
    callback_info.callback = webgpu_compilation_info_callback;
    callback_info.userdata1 = &result;

    WGPUFuture future = wgpuShaderModuleGetCompilationInfo(shader_module, callback_info);
    WGPUFutureWaitInfo wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
    if (vkdev->wait_webgpu_future(future, &wait_info, "shader-compilation-info") != 0
            || !result.completed
            || result.status != WGPUCompilationInfoRequestStatus_Success
            || result.has_error)
        return -1;

    return 0;
}

static WGPUShaderModule create_wgpu_shader_module(WGPUDevice device, const std::string& wgsl)
{
    WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
    source.code.data = wgsl.data();
    source.code.length = wgsl.size();

    WGPUShaderModuleDescriptor descriptor = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    descriptor.nextInChain = &source.chain;
    return wgpuDeviceCreateShaderModule(device, &descriptor);
}

int create_webgpu_shader_module(const VulkanDevice* vkdev, int shader_type_index, const Option& opt,
                                uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                                WGPUShaderModule* shader_module, WebGpuShaderInfo* shader_info)
{
    if (!vkdev || !vkdev->is_valid() || !shader_module || !shader_info)
        return -1;

    *shader_module = 0;
    *shader_info = WebGpuShaderInfo();

    WebGpuShaderCompileOptions options;
    options.shader_type = shader_type_index;
    options.opt = &opt;
    options.local_size_x = local_size_x;
    options.local_size_y = local_size_y;
    options.local_size_z = local_size_z;

    WebGpuTranslatedShader translated;
    if (translate_webgpu_shader(options, translated) != 0)
        return -1;

    WGPUShaderModule module = create_wgpu_shader_module(vkdev->wgpu_device(), translated.wgsl);
    if (!module || check_webgpu_shader_compilation(vkdev, module) != 0)
    {
        if (module)
            wgpuShaderModuleRelease(module);
        return -1;
    }

    *shader_module = module;
    *shader_info = translated.shader_info;
    return 0;
}

int pack_webgpu_immediates(const WebGpuShaderInfo& shader_info, const std::vector<vk_constant_type>& constants, std::vector<unsigned char>& packed_immediate)
{
    packed_immediate.clear();

    if (shader_info.immediate_size > 64)
        return -1;

    packed_immediate.resize(shader_info.immediate_size, 0);
    for (size_t i = 0; i < shader_info.immediate_members.size(); i++)
    {
        const WebGpuImmediateMember& member = shader_info.immediate_members[i];
        if (member.ncnn_constant_index >= constants.size() || member.byte_offset + 4 > packed_immediate.size())
        {
            packed_immediate.clear();
            return -1;
        }

        memcpy(&packed_immediate[member.byte_offset], &constants[member.ncnn_constant_index], 4);
    }

    return 0;
}

int compile_spirv_module(const char* comp_string, const Option& opt, std::vector<uint32_t>& spirv)
{
    if (!comp_string)
        return -1;

    const int length = strlen(comp_string) - 1;
    return compile_spirv_module(comp_string, length, opt, spirv);
}

int compile_spirv_module(const char* comp_data, int comp_data_size, const Option& opt, std::vector<uint32_t>& spirv)
{
    std::string preamble;
    if (build_webgpu_shader_preamble(opt, preamble) != 0)
        return -1;

    return compile_glsl_to_spirv(comp_data, comp_data_size, preamble, spirv);
}

int compile_spirv_module(int shader_type_index, const Option& opt, std::vector<uint32_t>& spirv)
{
    if (shader_type_index < 0 || shader_type_index >= layer_shader_registry_entry_count)
    {
        NCNN_LOGE("no such shader module %d", shader_type_index);
        return -1;
    }

    const layer_shader_registry_entry& entry = layer_shader_registry[shader_type_index];
    return compile_spirv_module(entry.comp_data, entry.comp_data_size, opt, spirv);
}

VulkanDevice* get_gpu_device(int device_index)
{
    if (g_webgpu.state == WEBGPU_INSTANCE_UNINITIALIZED)
        create_gpu_instance();

    if (g_webgpu.state != WEBGPU_INSTANCE_READY || device_index != 0)
        return 0;

    if (!g_webgpu.vkdev)
        g_webgpu.vkdev = new VulkanDevice(device_index);

    return g_webgpu.vkdev;
}

} // namespace ncnn

#endif // NCNN_WEBGPU
