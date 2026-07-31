// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#if !defined(NCNN_SIMPLEVK) && defined(__has_include)
#if __has_include("platform.h") && __has_include("simplevk.h")
#include "platform.h"
#endif
#endif

#if defined(NCNN_SIMPLEVK) && NCNN_SIMPLEVK
#include "simplevk.h"
#include "vulkan_header_fix.h"
#else
#include <vulkan/vulkan.h>
#endif
#include <webgpu/webgpu.h>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "spirv-tools/optimizer.hpp"
#include "spirv/unified1/spirv.hpp11"
#include "src/tint/api/common/binding_point.h"
#include "src/tint/api/tint.h"
#include "src/tint/lang/core/ir/instruction.h"
#include "src/tint/lang/core/ir/module.h"
#include "src/tint/lang/core/ir/var.h"
#include "src/tint/lang/core/ir/transform/change_immediate_to_uniform.h"
#include "src/tint/lang/core/ir/transform/dead_code_elimination.h"
#include "src/tint/lang/core/ir/transform/single_entry_point.h"
#include "src/tint/lang/core/type/bool.h"
#include "src/tint/lang/core/type/f32.h"
#include "src/tint/lang/core/type/i32.h"
#include "src/tint/lang/core/type/pointer.h"
#include "src/tint/lang/core/type/u32.h"
#include "src/tint/lang/spirv/reader/reader.h"
#include "src/tint/lang/wgsl/allowed_features.h"
#include "src/tint/lang/wgsl/inspector/inspector.h"
#include "src/tint/lang/wgsl/writer/writer.h"

#if defined(__EMSCRIPTEN__)
EM_JS(int, vkwebgpu_read_adapter_info,
      (uintptr_t adapter,
       char* vendor, uint32_t vendor_size,
       char* architecture, uint32_t architecture_size,
       char* device, uint32_t device_size,
       char* description, uint32_t description_size,
       uint32_t* subgroup_min_size, uint32_t* subgroup_max_size), {
    try {
        var js_adapter = WebGPU.getJsObject(adapter);
        var info = js_adapter && js_adapter.info;
        if (!info)
            return 0;

        stringToUTF8(info.vendor || "", vendor, vendor_size);
        stringToUTF8(info.architecture || "", architecture, architecture_size);
        stringToUTF8(info.device || "", device, device_size);
        stringToUTF8(info.description || "", description, description_size);

        var subgroup_min = Number(info.subgroupMinSize);
        var subgroup_max = Number(info.subgroupMaxSize);
        HEAPU32[subgroup_min_size >> 2] =
            Number.isFinite(subgroup_min) && subgroup_min >= 0 ? subgroup_min : 0;
        HEAPU32[subgroup_max_size >> 2] =
            Number.isFinite(subgroup_max) && subgroup_max >= 0 ? subgroup_max : 0;
        return info.isFallbackAdapter ? 2 : 1;
    } catch (error) {
        return 0;
    }
});
#endif

namespace vkwebgpu_detail {

// common helpers

static std::atomic<uint64_t> g_next_object_id(1);

static uint64_t next_object_id()
{
    return g_next_object_id.fetch_add(1, std::memory_order_relaxed);
}

static void log_error(const char* format, ...)
{
    fprintf(stderr, "[vkwebgpu] ");

    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);

    fprintf(stderr, "\n");
}

static void log_info(const char* format, ...)
{
    fprintf(stderr, "[vkwebgpu] ");

    va_list ap;
    va_start(ap, format);
    vfprintf(stderr, format, ap);
    va_end(ap);

    fprintf(stderr, "\n");
}

static uint64_t monotonic_time_ns()
{
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static uint64_t remaining_timeout_ns(uint64_t begin, uint64_t timeout)
{
    if (timeout == UINT64_MAX)
        return UINT64_MAX;

    const uint64_t now = monotonic_time_ns();
    const uint64_t elapsed = now >= begin ? now - begin : 0;
    return elapsed >= timeout ? 0 : timeout - elapsed;
}

static std::string string_from_webgpu(WGPUStringView value)
{
    if (!value.data)
        return std::string();

    return std::string(value.data, value.length);
}

static uint64_t align_up(uint64_t value, uint64_t alignment)
{
    if (alignment <= 1)
        return value;

    const uint64_t remainder = value % alignment;
    if (remainder == 0)
        return value;
    if (value > std::numeric_limits<uint64_t>::max() - (alignment - remainder))
        return 0;

    return value + alignment - remainder;
}

static bool checked_add(uint64_t a, uint64_t b, uint64_t& result)
{
    if (a > std::numeric_limits<uint64_t>::max() - b)
        return false;

    result = a + b;
    return true;
}

struct ByteRange
{
    uint64_t offset;
    uint64_t size;
};

static bool add_byte_range(std::vector<ByteRange>& ranges, uint64_t offset, uint64_t size)
{
    uint64_t end = 0;
    if (size == 0 || !checked_add(offset, size, end))
        return false;

    for (size_t i = 0; i < ranges.size();)
    {
        uint64_t range_end = 0;
        if (!checked_add(ranges[i].offset, ranges[i].size, range_end))
            return false;
        if (end < ranges[i].offset || range_end < offset)
        {
            i++;
            continue;
        }

        offset = std::min(offset, ranges[i].offset);
        end = std::max(end, range_end);
        ranges.erase(ranges.begin() + i);
        i = 0;
    }

    ByteRange range;
    range.offset = offset;
    range.size = end - offset;
    ranges.push_back(range);
    return true;
}

static void subtract_byte_range(std::vector<ByteRange>& ranges, uint64_t offset, uint64_t size)
{
    uint64_t end = 0;
    if (size == 0 || !checked_add(offset, size, end))
        return;

    std::vector<ByteRange> remaining;
    for (size_t i = 0; i < ranges.size(); i++)
    {
        uint64_t range_end = 0;
        if (!checked_add(ranges[i].offset, ranges[i].size, range_end))
            continue;
        if (end <= ranges[i].offset || range_end <= offset)
        {
            remaining.push_back(ranges[i]);
            continue;
        }

        if (ranges[i].offset < offset)
        {
            ByteRange left;
            left.offset = ranges[i].offset;
            left.size = offset - ranges[i].offset;
            remaining.push_back(left);
        }
        if (range_end > end)
        {
            ByteRange right;
            right.offset = end;
            right.size = range_end - end;
            remaining.push_back(right);
        }
    }
    ranges.swap(remaining);
}

template<typename Handle>
static uint64_t handle_to_id(Handle handle)
{
    if constexpr (std::is_pointer<Handle>::value)
        return (uint64_t)(uintptr_t)handle;
    else
        return (uint64_t)handle;
}

template<typename Handle>
static Handle id_to_handle(uint64_t id)
{
    if constexpr (std::is_pointer<Handle>::value)
        return (Handle)(uintptr_t)id;
    else
        return (Handle)id;
}

template<typename T>
class HandleTable
{
public:
    uint64_t insert(const std::shared_ptr<T>& object)
    {
        const uint64_t id = next_object_id();
        std::lock_guard<std::mutex> lock(mutex);
        objects[id] = object;
        return id;
    }

    std::shared_ptr<T> get(uint64_t id) const
    {
        if (id == 0)
            return std::shared_ptr<T>();

        std::lock_guard<std::mutex> lock(mutex);
        typename std::unordered_map<uint64_t, std::shared_ptr<T> >::const_iterator it = objects.find(id);
        if (it == objects.end())
            return std::shared_ptr<T>();

        return it->second;
    }

    void erase(uint64_t id)
    {
        if (id == 0)
            return;

        std::lock_guard<std::mutex> lock(mutex);
        objects.erase(id);
    }

    template<typename Predicate>
    std::vector<std::shared_ptr<T> > take_if(Predicate predicate)
    {
        std::vector<std::shared_ptr<T> > values;
        std::lock_guard<std::mutex> lock(mutex);
        typename std::unordered_map<uint64_t, std::shared_ptr<T> >::iterator it = objects.begin();
        while (it != objects.end())
        {
            if (!predicate(it->second))
            {
                ++it;
                continue;
            }

            values.push_back(it->second);
            it = objects.erase(it);
        }
        return values;
    }

private:
    mutable std::mutex mutex;
    std::unordered_map<uint64_t, std::shared_ptr<T> > objects;
};

enum BufferAccess
{
    BUFFER_ACCESS_READ,
    BUFFER_ACCESS_READ_WRITE
};

enum CommandBufferState
{
    COMMAND_BUFFER_INITIAL,
    COMMAND_BUFFER_RECORDING,
    COMMAND_BUFFER_EXECUTABLE,
    COMMAND_BUFFER_PENDING,
    COMMAND_BUFFER_INVALID
};

enum CommandType
{
    COMMAND_DISPATCH,
    COMMAND_COPY_BUFFER,
    COMMAND_BARRIER
};

struct Instance;
struct PhysicalDevice;
struct Device;
struct DeviceMemory;
struct Buffer;
struct DescriptorSetLayout;
struct DescriptorUpdateTemplate;
struct PipelineLayout;
struct ShaderModule;
struct ComputePipeline;
struct CommandPool;
struct CommandBuffer;
struct Fence;
struct PipelineCache;
struct Submission;

struct Instance : public std::enable_shared_from_this<Instance>
{
    Instance()
        : id(next_object_id()),
          instance(0),
          physical_devices_enumerated(false),
          immediate_supported(false)
    {
    }

    ~Instance();

    uint64_t id;
    WGPUInstance instance;
    std::vector<VkPhysicalDevice> physical_devices;
    bool physical_devices_enumerated;
    tint::wgsl::AllowedFeatures wgsl_allowed_features;
    bool immediate_supported;
};

struct PhysicalDevice
{
    PhysicalDevice()
        : id(next_object_id()),
          owner(0),
          adapter(0),
          vendor_id(0),
          device_id(0),
          backend_type(WGPUBackendType_Undefined),
          adapter_type(WGPUAdapterType_Unknown),
          subgroup_supported(false),
          subgroup_size_control_supported(false),
          subgroup_min_size(0),
          subgroup_max_size(0)
    {
        limits = WGPU_LIMITS_INIT;
    }

    ~PhysicalDevice()
    {
        if (adapter)
            wgpuAdapterRelease(adapter);
    }

    uint64_t id;
    Instance* owner;
    WGPUAdapter adapter;
    WGPULimits limits;
    uint32_t vendor_id;
    uint32_t device_id;
    WGPUBackendType backend_type;
    WGPUAdapterType adapter_type;
    bool subgroup_supported;
    bool subgroup_size_control_supported;
    uint32_t subgroup_min_size;
    uint32_t subgroup_max_size;
    std::string device_name;
    std::string driver_name;
    std::string vendor_name;
    std::string architecture_name;
};

struct Device
{
    Device()
        : id(next_object_id()),
          device(0),
          queue(0),
          queue_handle(0),
          lost(false),
          error(false),
          pipeline_translate_count(0),
          pipeline_translate_ns(0),
          webgpu_pipeline_count(0),
          webgpu_pipeline_ns(0),
          submit_count(0),
          submit_cpu_ns(0),
          bind_group_count(0),
          alias_snapshot_count(0),
          alias_snapshot_bytes(0),
          push_uniform_bytes(0),
          host_upload_bytes(0),
          host_readback_count(0),
          host_readback_bytes(0)
    {
        limits = WGPU_LIMITS_INIT;
    }

    ~Device();

    uint64_t id;
    std::shared_ptr<Instance> instance;
    std::shared_ptr<PhysicalDevice> physical_device;
    WGPUDevice device;
    WGPUQueue queue;
    WGPULimits limits;
    VkQueue queue_handle;
    std::atomic<bool> lost;
    std::atomic<bool> error;
    std::atomic<uint64_t> pipeline_translate_count;
    std::atomic<uint64_t> pipeline_translate_ns;
    std::atomic<uint64_t> webgpu_pipeline_count;
    std::atomic<uint64_t> webgpu_pipeline_ns;
    std::atomic<uint64_t> submit_count;
    std::atomic<uint64_t> submit_cpu_ns;
    std::atomic<uint64_t> bind_group_count;
    std::atomic<uint64_t> alias_snapshot_count;
    std::atomic<uint64_t> alias_snapshot_bytes;
    std::atomic<uint64_t> push_uniform_bytes;
    std::atomic<uint64_t> host_upload_bytes;
    std::atomic<uint64_t> host_readback_count;
    std::atomic<uint64_t> host_readback_bytes;
    std::unordered_set<std::string> enabled_extensions;
    std::mutex mutex;
    std::vector<std::shared_ptr<Submission> > submissions;
    std::vector<std::shared_ptr<Submission> > abandoned_submissions;
};

struct DeviceMemory
{
    DeviceMemory()
        : id(next_object_id()),
          owner(0),
          size(0),
          padded_size(0),
          memory_type_index(0),
          buffer(0),
          mapped(false),
          mapped_offset(0),
          mapped_size(0),
          live(true)
    {
    }

    ~DeviceMemory()
    {
        if (buffer)
            wgpuBufferRelease(buffer);
    }

    uint64_t id;
    Device* owner;
    VkDeviceSize size;
    VkDeviceSize padded_size;
    uint32_t memory_type_index;
    WGPUBuffer buffer;
    std::vector<unsigned char> host_shadow;
    bool mapped;
    VkDeviceSize mapped_offset;
    VkDeviceSize mapped_size;
    bool live;
    std::vector<ByteRange> dirty_ranges;
};

struct MemoryRange
{
    std::shared_ptr<DeviceMemory> memory;
    uint64_t offset;
    uint64_t size;
};

static bool append_memory_transfer_range(std::vector<MemoryRange>& ranges,
                                         const std::shared_ptr<DeviceMemory>& memory,
                                         uint64_t offset, uint64_t size)
{
    if (!memory || size == 0)
        return false;

    uint64_t end = 0;
    if (!checked_add(offset, size, end) || end > memory->size)
        return false;

    const uint64_t aligned_offset = offset - offset % 4;
    const uint64_t aligned_end = align_up(end, 4);
    if (aligned_end == 0 || aligned_end > memory->padded_size)
        return false;

    uint64_t merged_offset = aligned_offset;
    uint64_t merged_end = aligned_end;
    for (size_t i = 0; i < ranges.size();)
    {
        if (ranges[i].memory.get() != memory.get())
        {
            i++;
            continue;
        }

        uint64_t range_end = 0;
        if (!checked_add(ranges[i].offset, ranges[i].size, range_end))
            return false;
        if (merged_end < ranges[i].offset || range_end < merged_offset)
        {
            i++;
            continue;
        }

        merged_offset = std::min(merged_offset, ranges[i].offset);
        merged_end = std::max(merged_end, range_end);
        ranges.erase(ranges.begin() + i);
        i = 0;
    }

    MemoryRange range;
    range.memory = memory;
    range.offset = merged_offset;
    range.size = merged_end - merged_offset;
    ranges.push_back(range);
    return true;
}

static bool append_host_upload_ranges(std::vector<MemoryRange>& ranges,
                                      const std::shared_ptr<DeviceMemory>& memory,
                                      uint64_t offset, uint64_t size)
{
    if (!memory || memory->host_shadow.empty())
        return true;

    uint64_t end = 0;
    if (size == 0 || !checked_add(offset, size, end) || end > memory->size)
        return false;

    if (memory->mapped)
    {
        uint64_t mapped_end = 0;
        if (!checked_add(memory->mapped_offset, memory->mapped_size, mapped_end))
            return false;
        const uint64_t intersection_offset = std::max<uint64_t>(offset, memory->mapped_offset);
        const uint64_t intersection_end = std::min<uint64_t>(end, mapped_end);
        if (intersection_offset < intersection_end
                && !append_memory_transfer_range(ranges, memory, intersection_offset,
                                                 intersection_end - intersection_offset))
            return false;
    }

    for (size_t i = 0; i < memory->dirty_ranges.size(); i++)
    {
        uint64_t dirty_end = 0;
        if (!checked_add(memory->dirty_ranges[i].offset, memory->dirty_ranges[i].size, dirty_end))
            return false;
        const uint64_t intersection_offset = std::max<uint64_t>(offset, memory->dirty_ranges[i].offset);
        const uint64_t intersection_end = std::min<uint64_t>(end, dirty_end);
        if (intersection_offset < intersection_end
                && !append_memory_transfer_range(ranges, memory, intersection_offset,
                                                 intersection_end - intersection_offset))
            return false;
    }

    return true;
}

struct Buffer
{
    Buffer()
        : id(next_object_id()),
          owner(0),
          size(0),
          usage(0),
          memory_offset(0),
          live(true)
    {
    }

    uint64_t id;
    Device* owner;
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    std::shared_ptr<DeviceMemory> memory;
    VkDeviceSize memory_offset;
    bool live;
};

static uint64_t buffer_memory_alignment(const Device& device, const Buffer& buffer)
{
    uint64_t alignment = 4;
    if ((buffer.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0)
        alignment = std::max<uint64_t>(alignment, device.limits.minStorageBufferOffsetAlignment);
    if ((buffer.usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0)
        alignment = std::max<uint64_t>(alignment, device.limits.minUniformBufferOffsetAlignment);
    return alignment;
}

struct DescriptorBinding
{
    uint32_t binding;
    VkDescriptorType descriptor_type;
    uint32_t descriptor_count;
    VkShaderStageFlags stage_flags;
};

struct DescriptorSetLayout
{
    DescriptorSetLayout()
        : id(next_object_id()),
          owner(0),
          push_descriptor(false)
    {
    }

    uint64_t id;
    Device* owner;
    bool push_descriptor;
    std::vector<DescriptorBinding> bindings;
};

struct DescriptorTemplateEntry
{
    uint32_t binding;
    uint32_t array_element;
    uint32_t descriptor_count;
    VkDescriptorType descriptor_type;
    size_t offset;
    size_t stride;
};

struct DescriptorUpdateTemplate
{
    DescriptorUpdateTemplate()
        : id(next_object_id()),
          owner(0),
          template_type(VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET),
          bind_point(VK_PIPELINE_BIND_POINT_COMPUTE),
          set(0)
    {
    }

    uint64_t id;
    Device* owner;
    VkDescriptorUpdateTemplateType template_type;
    VkPipelineBindPoint bind_point;
    uint32_t set;
    std::shared_ptr<DescriptorSetLayout> set_layout;
    std::shared_ptr<PipelineLayout> pipeline_layout;
    std::vector<DescriptorTemplateEntry> entries;
};

struct PushConstantRange
{
    VkShaderStageFlags stage_flags;
    uint32_t offset;
    uint32_t size;
};

struct PipelineLayout
{
    PipelineLayout()
        : id(next_object_id()),
          owner(0),
          push_constant_size(0)
    {
    }

    uint64_t id;
    Device* owner;
    std::vector<std::shared_ptr<DescriptorSetLayout> > set_layouts;
    std::vector<PushConstantRange> push_constant_ranges;
    uint32_t push_constant_size;
};

struct ShaderModule
{
    ShaderModule()
        : id(next_object_id()),
          owner(0)
    {
    }

    uint64_t id;
    Device* owner;
    std::vector<uint32_t> spirv;
};

struct ActiveBinding
{
    uint32_t group;
    uint32_t binding;
    VkDescriptorType descriptor_type;
    BufferAccess access;
    BufferAccess layout_access;
    uint64_t min_binding_size;
    bool internal_uniform;
};

struct ComputePipeline
{
    ComputePipeline()
        : id(next_object_id()),
          owner(0),
          shader_module(0),
          bind_group_layout(0),
          immediate_bind_group_layout(0),
          pipeline_layout(0),
          pipeline(0),
          immediate_size(0),
          push_constant_uniform(false),
          push_constant_data_size(0),
          push_constant_uniform_size(0),
          workgroup_size_x(1),
          workgroup_size_y(1),
          workgroup_size_z(1),
          live(true)
    {
    }

    ~ComputePipeline()
    {
        if (pipeline)
            wgpuComputePipelineRelease(pipeline);
        if (pipeline_layout)
            wgpuPipelineLayoutRelease(pipeline_layout);
        if (immediate_bind_group_layout)
            wgpuBindGroupLayoutRelease(immediate_bind_group_layout);
        if (bind_group_layout)
            wgpuBindGroupLayoutRelease(bind_group_layout);
        if (shader_module)
            wgpuShaderModuleRelease(shader_module);
    }

    uint64_t id;
    Device* owner;
    std::shared_ptr<PipelineLayout> vk_pipeline_layout;
    WGPUShaderModule shader_module;
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroupLayout immediate_bind_group_layout;
    WGPUPipelineLayout pipeline_layout;
    WGPUComputePipeline pipeline;
    std::vector<ActiveBinding> active_bindings;
    uint32_t immediate_size;
    bool push_constant_uniform;
    uint32_t push_constant_data_size;
    uint32_t push_constant_uniform_size;
    uint32_t workgroup_size_x;
    uint32_t workgroup_size_y;
    uint32_t workgroup_size_z;
    bool live;
};

struct ResolvedBinding
{
    uint32_t binding;
    VkDescriptorType descriptor_type;
    BufferAccess access;
    BufferAccess layout_access;
    uint64_t min_binding_size;
    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<DeviceMemory> memory;
    uint64_t offset;
    uint64_t size;
};

struct CopyRegion
{
    uint64_t src_offset;
    uint64_t dst_offset;
    uint64_t size;
};

struct Command
{
    Command()
        : type(COMMAND_BARRIER),
          group_count_x(0),
          group_count_y(0),
          group_count_z(0)
    {
    }

    CommandType type;
    std::shared_ptr<ComputePipeline> pipeline;
    std::vector<ResolvedBinding> bindings;
    std::vector<unsigned char> push_constants;
    uint32_t group_count_x;
    uint32_t group_count_y;
    uint32_t group_count_z;
    std::shared_ptr<Buffer> src_buffer;
    std::shared_ptr<Buffer> dst_buffer;
    std::vector<CopyRegion> copy_regions;
    std::vector<std::shared_ptr<Buffer> > barrier_buffers;
};

struct CommandPool
{
    CommandPool()
        : id(next_object_id()),
          owner(0),
          flags(0),
          queue_family_index(0)
    {
    }

    uint64_t id;
    Device* owner;
    VkCommandPoolCreateFlags flags;
    uint32_t queue_family_index;
    std::mutex mutex;
    std::unordered_set<VkCommandBuffer> command_buffers;
};

struct DescriptorValue
{
    VkDescriptorType descriptor_type;
    VkDescriptorBufferInfo buffer_info;
};

struct CommandBuffer
{
    CommandBuffer()
        : id(next_object_id()),
          owner(0),
          state(COMMAND_BUFFER_INITIAL),
          pending_count(0),
          usage_flags(0),
          error(VK_SUCCESS)
    {
    }

    uint64_t id;
    Device* owner;
    std::shared_ptr<CommandPool> pool;
    CommandBufferState state;
    uint32_t pending_count;
    VkCommandBufferUsageFlags usage_flags;
    VkResult error;
    std::shared_ptr<ComputePipeline> current_pipeline;
    std::shared_ptr<PipelineLayout> descriptor_pipeline_layout;
    std::unordered_map<uint32_t, DescriptorValue> descriptors;
    std::shared_ptr<PipelineLayout> push_constant_pipeline_layout;
    std::vector<unsigned char> push_constants;
    std::vector<Command> commands;
};

struct PipelineCache
{
    PipelineCache()
        : id(next_object_id()),
          owner(0)
    {
    }

    uint64_t id;
    Device* owner;
};

struct MapResult
{
    MapResult()
        : status(WGPUMapAsyncStatus_Error),
          completed(false)
    {
    }

    WGPUMapAsyncStatus status;
    bool completed;
};

struct Readback
{
    Readback()
        : buffer(0),
          offset(0),
          size(0),
          map_started(false),
          copied(false)
    {
        future.id = 0;
    }

    WGPUBuffer buffer;
    std::shared_ptr<DeviceMemory> memory;
    uint64_t offset;
    uint64_t size;
    WGPUFuture future;
    std::shared_ptr<MapResult> map_result;
    bool map_started;
    bool copied;
};

struct Submission
{
    Submission()
        : id(next_object_id()),
          owner(0),
          completed(false),
          processed(false),
          status(WGPUQueueWorkDoneStatus_Error),
          result(VK_NOT_READY)
    {
        future.id = 0;
    }

    ~Submission()
    {
        release_resources();
    }

    void release_resources()
    {
        for (size_t i = 0; i < bind_groups.size(); i++)
            wgpuBindGroupRelease(bind_groups[i]);
        bind_groups.clear();
        for (size_t i = 0; i < temporary_buffers.size(); i++)
            wgpuBufferRelease(temporary_buffers[i]);
        temporary_buffers.clear();
        for (size_t i = 0; i < readbacks.size(); i++)
            wgpuBufferRelease(readbacks[i].buffer);
        readbacks.clear();
        memories.clear();
        buffers.clear();
        pipelines.clear();
        command_buffers.clear();
    }

    uint64_t id;
    Device* owner;
    WGPUFuture future;
    bool completed;
    bool processed;
    WGPUQueueWorkDoneStatus status;
    VkResult result;
    std::weak_ptr<Fence> fence;
    std::vector<WGPUBindGroup> bind_groups;
    std::vector<WGPUBuffer> temporary_buffers;
    std::vector<Readback> readbacks;
    std::vector<std::shared_ptr<DeviceMemory> > memories;
    std::vector<std::shared_ptr<Buffer> > buffers;
    std::vector<std::shared_ptr<ComputePipeline> > pipelines;
    std::vector<std::shared_ptr<CommandBuffer> > command_buffers;
};

struct Fence
{
    Fence()
        : id(next_object_id()),
          owner(0),
          signaled(false),
          failed(false)
    {
    }

    uint64_t id;
    Device* owner;
    bool signaled;
    bool failed;
    std::shared_ptr<Submission> submission;
};

template<typename T>
static bool contains_object(const std::vector<std::shared_ptr<T> >& objects, const T* object)
{
    for (size_t i = 0; i < objects.size(); i++)
    {
        if (objects[i].get() == object)
            return true;
    }
    return false;
}

static bool memory_in_flight(Device* device, const DeviceMemory* memory)
{
    std::lock_guard<std::mutex> lock(device->mutex);
    for (size_t i = 0; i < device->submissions.size(); i++)
    {
        if (!device->submissions[i]->processed
                && contains_object(device->submissions[i]->memories, memory))
            return true;
    }
    for (size_t i = 0; i < device->abandoned_submissions.size(); i++)
    {
        if (contains_object(device->abandoned_submissions[i]->memories, memory))
            return true;
    }
    return false;
}

static bool buffer_in_flight(Device* device, const Buffer* buffer)
{
    std::lock_guard<std::mutex> lock(device->mutex);
    for (size_t i = 0; i < device->submissions.size(); i++)
    {
        if (!device->submissions[i]->processed
                && contains_object(device->submissions[i]->buffers, buffer))
            return true;
    }
    for (size_t i = 0; i < device->abandoned_submissions.size(); i++)
    {
        if (contains_object(device->abandoned_submissions[i]->buffers, buffer))
            return true;
    }
    return false;
}

static bool pipeline_in_flight(Device* device, const ComputePipeline* pipeline)
{
    std::lock_guard<std::mutex> lock(device->mutex);
    for (size_t i = 0; i < device->submissions.size(); i++)
    {
        if (!device->submissions[i]->processed
                && contains_object(device->submissions[i]->pipelines, pipeline))
            return true;
    }
    for (size_t i = 0; i < device->abandoned_submissions.size(); i++)
    {
        if (contains_object(device->abandoned_submissions[i]->pipelines, pipeline))
            return true;
    }
    return false;
}

static HandleTable<DeviceMemory> g_memories;
static HandleTable<Buffer> g_buffers;
static HandleTable<DescriptorSetLayout> g_descriptor_set_layouts;
static HandleTable<DescriptorUpdateTemplate> g_descriptor_update_templates;
static HandleTable<PipelineLayout> g_pipeline_layouts;
static HandleTable<ShaderModule> g_shader_modules;
static HandleTable<ComputePipeline> g_compute_pipelines;
static HandleTable<CommandPool> g_command_pools;
static HandleTable<Fence> g_fences;
static HandleTable<PipelineCache> g_pipeline_caches;

static VkResult wait_device_submissions(Device* device);
static void release_device_objects(Device* device);

template<typename Handle, typename T>
static Handle make_handle(HandleTable<T>& table, const std::shared_ptr<T>& object)
{
    return id_to_handle<Handle>(table.insert(object));
}

template<typename T, typename Handle>
static std::shared_ptr<T> get_handle(const HandleTable<T>& table, Handle handle)
{
    return table.get(handle_to_id(handle));
}

template<typename T, typename Handle>
static void erase_handle(HandleTable<T>& table, Handle handle)
{
    table.erase(handle_to_id(handle));
}

} // namespace vkwebgpu_detail

struct VkInstance_T
{
    std::shared_ptr<vkwebgpu_detail::Instance> impl;
};

struct VkPhysicalDevice_T
{
    std::shared_ptr<vkwebgpu_detail::PhysicalDevice> impl;
};

struct VkDevice_T
{
    std::shared_ptr<vkwebgpu_detail::Device> impl;
};

struct VkQueue_T
{
    vkwebgpu_detail::Device* device;
};

struct VkCommandBuffer_T
{
    std::shared_ptr<vkwebgpu_detail::CommandBuffer> impl;
};

namespace vkwebgpu_detail {

Instance::~Instance()
{
    for (size_t i = 0; i < physical_devices.size(); i++)
        delete physical_devices[i];
    physical_devices.clear();
    if (instance)
        wgpuInstanceRelease(instance);
}

Device::~Device()
{
    if (queue_handle)
    {
        delete queue_handle;
        queue_handle = 0;
    }
    log_info("device %llu stats pipelines=%llu/%llu spirv-wgsl=%.3fms webgpu-pipeline=%.3fms submits=%llu submit-cpu=%.3fms bind-groups=%llu snapshots=%llu/%lluB push-uniform=%lluB host-upload=%lluB host-readback=%llu/%lluB",
             (unsigned long long)id,
             (unsigned long long)pipeline_translate_count.load(),
             (unsigned long long)webgpu_pipeline_count.load(),
             pipeline_translate_ns.load() / 1000000.0,
             webgpu_pipeline_ns.load() / 1000000.0,
             (unsigned long long)submit_count.load(),
             submit_cpu_ns.load() / 1000000.0,
             (unsigned long long)bind_group_count.load(),
             (unsigned long long)alias_snapshot_count.load(),
             (unsigned long long)alias_snapshot_bytes.load(),
             (unsigned long long)push_uniform_bytes.load(),
             (unsigned long long)host_upload_bytes.load(),
             (unsigned long long)host_readback_count.load(),
             (unsigned long long)host_readback_bytes.load());
    if (device)
        wgpuDeviceDestroy(device);
    submissions.clear();
    abandoned_submissions.clear();
    if (queue)
        wgpuQueueRelease(queue);
    if (device)
        wgpuDeviceRelease(device);
}

static Instance* unwrap(VkInstance instance)
{
    return instance && instance->impl ? instance->impl.get() : 0;
}

static PhysicalDevice* unwrap(VkPhysicalDevice physical_device)
{
    return physical_device && physical_device->impl ? physical_device->impl.get() : 0;
}

static Device* unwrap(VkDevice device)
{
    return device && device->impl ? device->impl.get() : 0;
}

static Device* unwrap(VkQueue queue)
{
    return queue ? queue->device : 0;
}

static bool device_failed(const Device* device)
{
    return device && (device->lost || device->error);
}

static CommandBuffer* unwrap(VkCommandBuffer command_buffer)
{
    return command_buffer && command_buffer->impl ? command_buffer->impl.get() : 0;
}

static thread_local uint32_t g_wait_depth = 0;

static int wait_any(Instance* instance, size_t future_count, WGPUFutureWaitInfo* wait_infos,
                    uint64_t timeout, const char* stage, WGPUWaitStatus& status)
{
    if (!instance || !instance->instance || future_count == 0 || !wait_infos)
        return -1;
    if (g_wait_depth != 0)
    {
        log_error("instance %llu nested WaitAny stage=%s depth=%u",
                  (unsigned long long)instance->id, stage ? stage : "unknown", g_wait_depth);
        return -1;
    }

    g_wait_depth++;
    status = wgpuInstanceWaitAny(instance->instance, future_count, wait_infos, timeout);
    g_wait_depth--;
    return 0;
}

static int wait_future(Instance* instance, WGPUFuture future, uint64_t timeout, const char* stage)
{
    if (!instance || !instance->instance || future.id == 0)
        return -1;

    WGPUFutureWaitInfo wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
    wait_info.future = future;
    WGPUWaitStatus status;
    if (wait_any(instance, 1, &wait_info, timeout, stage, status) != 0)
        return -1;
    if (status == WGPUWaitStatus_TimedOut)
        return 1;
    if (status != WGPUWaitStatus_Success || wait_info.completed != WGPU_TRUE)
        return -1;

    return 0;
}

struct AdapterResult
{
    AdapterResult()
        : status(WGPURequestAdapterStatus_Error),
          adapter(0)
    {
    }

    ~AdapterResult()
    {
        if (adapter)
            wgpuAdapterRelease(adapter);
    }

    WGPURequestAdapterStatus status;
    WGPUAdapter adapter;
    std::string message;
};

struct DeviceResult
{
    DeviceResult()
        : status(WGPURequestDeviceStatus_Error),
          device(0)
    {
    }

    ~DeviceResult()
    {
        if (device)
            wgpuDeviceRelease(device);
    }

    WGPURequestDeviceStatus status;
    WGPUDevice device;
    std::shared_ptr<Device> owner;
};

static void request_adapter_callback(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1, void*)
{
    std::shared_ptr<AdapterResult>* context = (std::shared_ptr<AdapterResult>*)userdata1;
    if (!context)
    {
        if (adapter)
            wgpuAdapterRelease(adapter);
        return;
    }
    std::shared_ptr<AdapterResult> result = *context;
    delete context;
    if (!result)
    {
        if (adapter)
            wgpuAdapterRelease(adapter);
        return;
    }

    result->status = status;
    result->adapter = adapter;
    result->message = string_from_webgpu(message);
}

static void request_device_callback(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1, void*)
{
    std::shared_ptr<DeviceResult>* context = (std::shared_ptr<DeviceResult>*)userdata1;
    if (!context)
    {
        if (device)
            wgpuDeviceRelease(device);
        return;
    }
    std::shared_ptr<DeviceResult> result = *context;
    delete context;
    if (!result)
    {
        if (device)
            wgpuDeviceRelease(device);
        return;
    }

    result->status = status;
    result->device = device;

    if (status != WGPURequestDeviceStatus_Success)
        log_error("request device failed: %.*s", (int)message.length, message.data ? message.data : "");
}

static void device_lost_callback(const WGPUDevice*, WGPUDeviceLostReason reason, WGPUStringView message, void* userdata1, void*)
{
    Device* device = (Device*)userdata1;
    if (!device || reason == WGPUDeviceLostReason_Destroyed)
        return;

    device->lost = true;
    log_error("device %llu lost reason=%d: %.*s", (unsigned long long)device->id, (int)reason, (int)message.length, message.data ? message.data : "");
}

static void uncaptured_error_callback(const WGPUDevice*, WGPUErrorType type, WGPUStringView message, void* userdata1, void*)
{
    Device* device = (Device*)userdata1;
    if (device)
        device->error = true;

    log_error("device %llu uncaptured error type=%d: %.*s",
              device ? (unsigned long long)device->id : 0,
              (int)type, (int)message.length, message.data ? message.data : "");
}

static bool same_physical_device(const PhysicalDevice& a, const PhysicalDevice& b)
{
    return a.vendor_id == b.vendor_id
           && a.device_id == b.device_id
           && a.backend_type == b.backend_type
           && a.adapter_type == b.adapter_type
           && a.device_name == b.device_name
           && a.driver_name == b.driver_name
           && a.vendor_name == b.vendor_name
           && a.architecture_name == b.architecture_name;
}

static std::shared_ptr<PhysicalDevice> request_physical_device(Instance* instance,
                                                               WGPUPowerPreference power_preference,
                                                               bool force_fallback,
                                                               std::string& error_message)
{
    std::shared_ptr<AdapterResult> result = std::make_shared<AdapterResult>();
    WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    options.powerPreference = power_preference;
    options.forceFallbackAdapter = force_fallback ? WGPU_TRUE : WGPU_FALSE;
    WGPURequestAdapterCallbackInfo callback_info = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
    callback_info.callback = request_adapter_callback;
    std::shared_ptr<AdapterResult>* callback_context = new std::shared_ptr<AdapterResult>(result);
    callback_info.userdata1 = callback_context;

    WGPUFuture future = wgpuInstanceRequestAdapter(instance->instance, &options, callback_info);
    if (future.id == 0)
    {
        delete callback_context;
        error_message = "request adapter returned an invalid future";
        return std::shared_ptr<PhysicalDevice>();
    }
    if (wait_future(instance, future, UINT64_MAX, "request-adapter") != 0
            || result->status != WGPURequestAdapterStatus_Success
            || !result->adapter)
    {
        error_message = result->message;
        return std::shared_ptr<PhysicalDevice>();
    }

    std::shared_ptr<PhysicalDevice> physical_device = std::make_shared<PhysicalDevice>();
    physical_device->owner = instance;
    physical_device->adapter = result->adapter;
    result->adapter = 0;

    if (wgpuAdapterGetLimits(physical_device->adapter, &physical_device->limits) != WGPUStatus_Success)
    {
        error_message = "get adapter limits failed";
        return std::shared_ptr<PhysicalDevice>();
    }

    bool adapter_info_available = false;
#if defined(__EMSCRIPTEN__)
    // Read GPUAdapter.info once and sanitize every optional field here.
    // Emdawnwebgpu's wgpuAdapterGetInfo() directly dereferences adapter.info
    // and therefore throws in browsers that only expose requestAdapterInfo().
    {
        char vendor[256] = {0};
        char architecture[256] = {0};
        char device_name[256] = {0};
        char description[256] = {0};
        uint32_t subgroup_min_size = 0;
        uint32_t subgroup_max_size = 0;
        const int info_result = vkwebgpu_read_adapter_info(
            (uintptr_t)physical_device->adapter,
            vendor, sizeof(vendor),
            architecture, sizeof(architecture),
            device_name, sizeof(device_name),
            description, sizeof(description),
            &subgroup_min_size, &subgroup_max_size);
        adapter_info_available = info_result != 0;
        if (adapter_info_available)
        {
            physical_device->backend_type = WGPUBackendType_WebGPU;
            physical_device->adapter_type =
                info_result == 2 ? WGPUAdapterType_CPU : WGPUAdapterType_Unknown;
            physical_device->device_name = device_name;
            physical_device->driver_name = description;
            physical_device->vendor_name = vendor;
            physical_device->architecture_name = architecture;
            physical_device->subgroup_min_size = subgroup_min_size;
            physical_device->subgroup_max_size = subgroup_max_size;
        }
    }
#else
    WGPUAdapterInfo adapter_info = WGPU_ADAPTER_INFO_INIT;
    if (wgpuAdapterGetInfo(physical_device->adapter, &adapter_info) == WGPUStatus_Success)
    {
        adapter_info_available = true;
        physical_device->vendor_id = adapter_info.vendorID;
        physical_device->device_id = adapter_info.deviceID;
        physical_device->backend_type = adapter_info.backendType;
        physical_device->adapter_type = adapter_info.adapterType;
        physical_device->device_name = string_from_webgpu(adapter_info.device);
        physical_device->driver_name = string_from_webgpu(adapter_info.description);
        physical_device->vendor_name = string_from_webgpu(adapter_info.vendor);
        physical_device->architecture_name = string_from_webgpu(adapter_info.architecture);
        physical_device->subgroup_min_size = adapter_info.subgroupMinSize;
        physical_device->subgroup_max_size = adapter_info.subgroupMaxSize;

        if (physical_device->device_name.empty())
            physical_device->device_name = physical_device->architecture_name;
        if (physical_device->driver_name.empty())
            physical_device->driver_name = physical_device->vendor_name;

        wgpuAdapterInfoFreeMembers(adapter_info);
    }
#endif

    if (physical_device->device_name.empty())
        physical_device->device_name = physical_device->architecture_name;
    if (physical_device->driver_name.empty())
        physical_device->driver_name = physical_device->vendor_name;

    physical_device->subgroup_supported =
        adapter_info_available
        && wgpuAdapterHasFeature(physical_device->adapter, WGPUFeatureName_Subgroups) == WGPU_TRUE
        && physical_device->subgroup_min_size != 0
        && physical_device->subgroup_max_size >= physical_device->subgroup_min_size;
    physical_device->subgroup_size_control_supported =
        physical_device->subgroup_supported
        && wgpuAdapterHasFeature(physical_device->adapter, WGPUFeatureName_SubgroupSizeControl) == WGPU_TRUE;
    if (!physical_device->subgroup_supported)
    {
        physical_device->subgroup_size_control_supported = false;
        physical_device->subgroup_min_size = 0;
        physical_device->subgroup_max_size = 0;
    }

    if (physical_device->device_name.empty())
        physical_device->device_name = "WebGPU";
    if (physical_device->driver_name.empty())
        physical_device->driver_name = "vkwebgpu";

    return physical_device;
}

static VkResult ensure_physical_devices(Instance* instance)
{
    if (!instance || !instance->instance)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (instance->physical_devices_enumerated)
        return instance->physical_devices.empty() ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS;

    const WGPUPowerPreference power_preferences[] = {
        WGPUPowerPreference_HighPerformance,
        WGPUPowerPreference_LowPower,
        WGPUPowerPreference_Undefined
    };
    const bool force_fallback[] = {false, false, true};
    std::string last_error;
    for (size_t i = 0; i < sizeof(power_preferences) / sizeof(power_preferences[0]); i++)
    {
        std::shared_ptr<PhysicalDevice> physical_device =
            request_physical_device(instance, power_preferences[i], force_fallback[i], last_error);
        if (!physical_device)
            continue;

        bool duplicate = false;
        for (size_t j = 0; j < instance->physical_devices.size(); j++)
        {
            if (same_physical_device(*instance->physical_devices[j]->impl, *physical_device))
            {
                duplicate = true;
                break;
            }
        }
        if (duplicate)
            continue;

        VkPhysicalDevice_T* handle = new VkPhysicalDevice_T;
        handle->impl = physical_device;
        instance->physical_devices.push_back(handle);
    }

    instance->physical_devices_enumerated = true;
    if (instance->physical_devices.empty())
    {
        log_error("request adapters failed: %s", last_error.empty() ? "no WebGPU adapter" : last_error.c_str());
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    return VK_SUCCESS;
}

// shader translation

static int normalize_spirv(std::vector<uint32_t>& spirv)
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

static int specialize_spirv(std::vector<uint32_t>& spirv, const VkSpecializationInfo* info)
{
    if (info && info->dataSize != 0 && !info->pData)
        return -1;
    if (info && info->mapEntryCount != 0 && !info->pMapEntries)
        return -1;

    std::unordered_map<uint32_t, std::vector<uint32_t> > values;
    if (info)
    {
        for (uint32_t i = 0; i < info->mapEntryCount; i++)
        {
            const VkSpecializationMapEntry& entry = info->pMapEntries[i];
            uint64_t end = 0;
            if (entry.size != 4 || !checked_add(entry.offset, entry.size, end) || end > info->dataSize)
            {
                log_error("specialization id %u has invalid offset=%u size=%llu data-size=%llu",
                          entry.constantID, entry.offset, (unsigned long long)entry.size,
                          (unsigned long long)info->dataSize);
                return -1;
            }

            if (values.find(entry.constantID) != values.end())
            {
                log_error("specialization id %u is specified more than once", entry.constantID);
                return -1;
            }

            std::vector<uint32_t>& value = values[entry.constantID];
            value.resize(1);
            memcpy(value.data(), (const unsigned char*)info->pData + entry.offset, entry.size);
        }
    }

    spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_1);
    optimizer.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
        log_error("SPIR-V optimizer word=%zu: %s", position.index, message ? message : "");
    });
    if (!values.empty())
        optimizer.RegisterPass(spvtools::CreateSetSpecConstantDefaultValuePass(values));
    optimizer.RegisterPass(spvtools::CreateFreezeSpecConstantValuePass());
    optimizer.RegisterPass(spvtools::CreateFoldSpecConstantOpAndCompositePass());
    optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
    optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass(false));
    optimizer.RegisterPass(spvtools::CreateRemoveUnusedInterfaceVariablesPass());

    std::vector<uint32_t> specialized;
    if (!optimizer.Run(spirv.data(), spirv.size(), &specialized))
    {
        log_error("SPIR-V specialization failed");
        return -1;
    }

    spirv.swap(specialized);
    return 0;
}

static int legalize_uniform_global_invocation_z(std::vector<uint32_t>& spirv)
{
    if (spirv.size() < 5 || spirv[0] != (uint32_t)spv::MagicNumber)
        return -1;

    const uint32_t bound = spirv[3];
    uint32_t global_invocation_id = 0;
    uint32_t global_invocation_pointer_type = 0;
    uint32_t local_size_z = 0;
    uint32_t local_size_z_id = 0;
    std::vector<uint32_t> constant_values(bound, 0);
    std::vector<unsigned char> constant_known(bound, 0);

    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        if (opcode == spv::Op::OpDecorate && word_count >= 4
                && spirv[i + 2] == (uint32_t)spv::Decoration::BuiltIn
                && spirv[i + 3] == (uint32_t)spv::BuiltIn::GlobalInvocationId)
        {
            global_invocation_id = spirv[i + 1];
        }
        else if (opcode == spv::Op::OpConstant && word_count == 4 && spirv[i + 2] < bound)
        {
            constant_values[spirv[i + 2]] = spirv[i + 3];
            constant_known[spirv[i + 2]] = 1;
        }
        else if (opcode == spv::Op::OpExecutionMode && word_count >= 6
                 && spirv[i + 2] == (uint32_t)spv::ExecutionMode::LocalSize)
        {
            local_size_z = spirv[i + 5];
        }
        else if (opcode == spv::Op::OpExecutionModeId && word_count >= 6
                 && spirv[i + 2] == (uint32_t)spv::ExecutionMode::LocalSizeId)
        {
            local_size_z_id = spirv[i + 5];
        }

        i += word_count;
    }

    if (local_size_z == 0 && local_size_z_id < bound && constant_known[local_size_z_id])
        local_size_z = constant_values[local_size_z_id];

    if (global_invocation_id == 0 || local_size_z != 1)
        return 0;

    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (opcode == spv::Op::OpVariable && word_count >= 4 && spirv[i + 2] == global_invocation_id)
        {
            global_invocation_pointer_type = spirv[i + 1];
            break;
        }
        i += word_count;
    }
    if (global_invocation_pointer_type == 0)
        return -1;

    bool rewrite_needed = false;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if ((opcode == spv::Op::OpAccessChain || opcode == spv::Op::OpInBoundsAccessChain)
                && word_count == 5 && spirv[i + 3] == global_invocation_id
                && spirv[i + 4] < bound && constant_known[spirv[i + 4]]
                && constant_values[spirv[i + 4]] == 2)
        {
            rewrite_needed = true;
            break;
        }
        i += word_count;
    }
    if (!rewrite_needed)
        return 0;

    const uint32_t workgroup_id = bound;
    std::vector<uint32_t> legalized;
    legalized.reserve(spirv.size() + 16);
    legalized.insert(legalized.end(), spirv.begin(), spirv.begin() + 5);
    legalized[3] = bound + 1;

    bool decorate_inserted = false;
    bool variable_inserted = false;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);

        if (!decorate_inserted && opcode >= spv::Op::OpTypeVoid && opcode <= spv::Op::OpTypeForwardPointer)
        {
            legalized.push_back((4u << 16) | (uint32_t)spv::Op::OpDecorate);
            legalized.push_back(workgroup_id);
            legalized.push_back((uint32_t)spv::Decoration::BuiltIn);
            legalized.push_back((uint32_t)spv::BuiltIn::WorkgroupId);
            decorate_inserted = true;
        }
        if (!variable_inserted && opcode == spv::Op::OpFunction)
        {
            legalized.push_back((4u << 16) | (uint32_t)spv::Op::OpVariable);
            legalized.push_back(global_invocation_pointer_type);
            legalized.push_back(workgroup_id);
            legalized.push_back((uint32_t)spv::StorageClass::Input);
            variable_inserted = true;
        }

        if (opcode == spv::Op::OpEntryPoint)
        {
            legalized.push_back(((word_count + 1) << 16) | (uint32_t)opcode);
            legalized.insert(legalized.end(), spirv.begin() + i + 1, spirv.begin() + i + word_count);
            legalized.push_back(workgroup_id);
        }
        else if ((opcode == spv::Op::OpAccessChain || opcode == spv::Op::OpInBoundsAccessChain)
                 && word_count == 5 && spirv[i + 3] == global_invocation_id
                 && spirv[i + 4] < bound && constant_known[spirv[i + 4]]
                 && constant_values[spirv[i + 4]] == 2)
        {
            legalized.insert(legalized.end(), spirv.begin() + i, spirv.begin() + i + word_count);
            legalized[legalized.size() - 2] = workgroup_id;
        }
        else
        {
            legalized.insert(legalized.end(), spirv.begin() + i, spirv.begin() + i + word_count);
        }

        i += word_count;
    }

    if (!decorate_inserted || !variable_inserted)
        return -1;

    spirv.swap(legalized);
    return 0;
}

static uint64_t binding_key(uint32_t group, uint32_t binding)
{
    return ((uint64_t)group << 32) | binding;
}

static bool storage_variable_is_written(tint::core::ir::Value* value)
{
    std::vector<tint::core::ir::Value*> pending;
    std::unordered_set<tint::core::ir::Value*> visited;
    pending.push_back(value);

    while (!pending.empty())
    {
        tint::core::ir::Value* current = pending.back();
        pending.pop_back();
        if (!current || !visited.insert(current).second)
            continue;

        for (const tint::core::ir::Usage& usage : current->UsagesUnsorted())
        {
            tint::core::ir::Instruction* instruction = usage.instruction;
            if (instruction->GetSideEffects().Contains(tint::core::ir::Instruction::Access::kStore))
                return true;

            for (tint::core::ir::InstructionResult* result : instruction->Results())
            {
                if (result->Type()->Is<tint::core::type::Pointer>())
                    pending.push_back(result);
            }
        }
    }

    return false;
}

static void collect_storage_access(tint::core::ir::Module& module,
                                   std::unordered_map<uint64_t, BufferAccess>& access)
{
    std::vector<tint::core::ir::Var*> inactive;
    for (tint::core::ir::Instruction* instruction : module.Instructions())
    {
        tint::core::ir::Var* var = instruction->As<tint::core::ir::Var>();
        if (!var)
            continue;

        std::optional<tint::BindingPoint> binding_point = var->BindingPoint();
        const tint::core::type::Pointer* pointer = var->Result()->Type()->As<tint::core::type::Pointer>();
        if (!binding_point.has_value() || !pointer
                || pointer->AddressSpace() != tint::core::AddressSpace::kStorage)
            continue;
        if (!var->Result()->IsUsed())
        {
            inactive.push_back(var);
            continue;
        }

        access[binding_key(binding_point->group, binding_point->binding)] =
            storage_variable_is_written(var->Result()) ? BUFFER_ACCESS_READ_WRITE : BUFFER_ACCESS_READ;
    }

    for (size_t i = 0; i < inactive.size(); i++)
        inactive[i]->Destroy();
}

struct TranslatedShader
{
    std::string wgsl;
    std::vector<ActiveBinding> bindings;
    uint32_t immediate_size;
    bool push_constant_uniform;
    uint32_t push_constant_data_size;
    uint32_t push_constant_uniform_size;
    uint32_t workgroup_size_x;
    uint32_t workgroup_size_y;
    uint32_t workgroup_size_z;
};

static std::once_flag g_tint_initialize_once;

static int translate_shader(const std::vector<uint32_t>& input_spirv, const char* entry_point,
                            const VkSpecializationInfo* specialization_info,
                            const PipelineLayout& pipeline_layout, const Device& device,
                            TranslatedShader& translated)
{
    translated = TranslatedShader();
    if (!entry_point || !entry_point[0])
        return -1;

    std::vector<uint32_t> spirv = input_spirv;
    if (normalize_spirv(spirv) != 0)
    {
        log_error("SPIR-V normalization failed");
        return -1;
    }
    if (specialize_spirv(spirv, specialization_info) != 0)
        return -1;
    if (legalize_uniform_global_invocation_z(spirv) != 0)
    {
        log_error("SPIR-V uniform control-flow legalization failed");
        return -1;
    }

    std::call_once(g_tint_initialize_once, []() { tint::Initialize(); });

    tint::Result<tint::core::ir::Module> ir_result = tint::spirv::reader::ReadIR(spirv);
    if (ir_result != tint::Success)
    {
        log_error("Tint SPIR-V ReadIR failed: %s", ir_result.Failure().reason.c_str());
        return -1;
    }

    tint::core::ir::Module& module = ir_result.Get();
    tint::Result<tint::SuccessType> result = tint::core::ir::transform::SingleEntryPoint(module, entry_point);
    if (result != tint::Success)
    {
        log_error("Tint single entry point failed: %s", result.Failure().reason.c_str());
        return -1;
    }

    result = tint::core::ir::transform::DeadCodeElimination(module);
    if (result != tint::Success)
    {
        log_error("Tint dead code elimination failed: %s", result.Failure().reason.c_str());
        return -1;
    }

    std::unordered_map<uint64_t, BufferAccess> storage_access;
    collect_storage_access(module, storage_access);

    const uint32_t max_immediate_size = device.limits.maxImmediateSize;
    const bool immediate_supported = device.instance->immediate_supported;
    const bool use_uniform = pipeline_layout.push_constant_size != 0
                             && (!immediate_supported || pipeline_layout.push_constant_size > max_immediate_size);
    if (use_uniform)
    {
        tint::core::ir::transform::ChangeImmediateToUniformConfig config;
        config.immediate_binding_point = tint::BindingPoint{1, 0};
        result = tint::core::ir::transform::ChangeImmediateToUniform(module, config);
        if (result != tint::Success)
        {
            log_error("Tint immediate to uniform failed: %s", result.Failure().reason.c_str());
            return -1;
        }
    }

    tint::wgsl::writer::Options writer_options;
    writer_options.allowed_features = device.instance->wgsl_allowed_features;
    tint::Result<tint::Program> program_result = tint::wgsl::writer::ProgramFromIR(module, writer_options);
    if (program_result != tint::Success)
    {
        log_error("Tint IR to program failed: %s", program_result.Failure().reason.c_str());
        return -1;
    }

    tint::Program program = program_result.Move();
    tint::inspector::Inspector inspector(program);
    tint::inspector::EntryPoint reflected_entry_point = inspector.GetEntryPoint(entry_point);
    if (inspector.has_error() || reflected_entry_point.stage != tint::inspector::PipelineStage::kCompute
            || !reflected_entry_point.workgroup_size.has_value())
    {
        log_error("Tint compute entry point reflection failed: %s", inspector.error().c_str());
        return -1;
    }

    translated.immediate_size = reflected_entry_point.immediate_data_size;
    if (translated.immediate_size > pipeline_layout.push_constant_size)
    {
        log_error("shader immediate size=%u exceeds Vulkan push constant layout size=%u",
                  translated.immediate_size, pipeline_layout.push_constant_size);
        return -1;
    }
    translated.workgroup_size_x = reflected_entry_point.workgroup_size->x;
    translated.workgroup_size_y = reflected_entry_point.workgroup_size->y;
    translated.workgroup_size_z = reflected_entry_point.workgroup_size->z;

    std::vector<tint::inspector::ResourceBinding> resources = inspector.GetResourceBindings(entry_point);
    if (inspector.has_error())
    {
        log_error("Tint resource reflection failed: %s", inspector.error().c_str());
        return -1;
    }

    for (size_t i = 0; i < resources.size(); i++)
    {
        const tint::inspector::ResourceBinding& resource = resources[i];
        ActiveBinding binding;
        binding.group = resource.bind_group;
        binding.binding = resource.binding;
        binding.min_binding_size = resource.size;
        binding.internal_uniform = resource.bind_group == 1 && resource.binding == 0 && use_uniform;
        std::unordered_map<uint64_t, BufferAccess>::const_iterator access_it =
            storage_access.find(binding_key(resource.bind_group, resource.binding));

        if (resource.resource_type == tint::inspector::ResourceBinding::ResourceType::kReadOnlyStorageBuffer)
        {
            if (access_it == storage_access.end())
                continue;
            binding.descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.access = BUFFER_ACCESS_READ;
            binding.layout_access = BUFFER_ACCESS_READ;
        }
        else if (resource.resource_type == tint::inspector::ResourceBinding::ResourceType::kStorageBuffer)
        {
            if (access_it == storage_access.end())
                continue;
            binding.descriptor_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binding.access = access_it->second;
            binding.layout_access = BUFFER_ACCESS_READ_WRITE;
        }
        else if (resource.resource_type == tint::inspector::ResourceBinding::ResourceType::kUniformBuffer)
        {
            binding.descriptor_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.access = BUFFER_ACCESS_READ;
            binding.layout_access = BUFFER_ACCESS_READ;
            if (resource.size == 0
                    || resource.size > device.limits.maxUniformBufferBindingSize
                    || resource.size > UINT32_MAX)
            {
                log_error("invalid uniform buffer size=%llu group=%u binding=%u",
                          (unsigned long long)resource.size, binding.group, binding.binding);
                return -1;
            }
            if (binding.internal_uniform && translated.push_constant_uniform)
            {
                log_error("multiple internal push uniform bindings");
                return -1;
            }
            if (binding.internal_uniform)
            {
                translated.push_constant_uniform = true;
                translated.push_constant_data_size =
                    (uint32_t)std::min<uint64_t>(pipeline_layout.push_constant_size, resource.size);
                translated.push_constant_uniform_size = (uint32_t)resource.size;
            }
        }
        else
        {
            log_error("unsupported shader resource group=%u binding=%u type=%d",
                      resource.bind_group, resource.binding, (int)resource.resource_type);
            return -1;
        }

        if (!binding.internal_uniform && binding.group != 0)
        {
            log_error("only Vulkan descriptor set 0 is supported, shader uses group=%u binding=%u", binding.group, binding.binding);
            return -1;
        }

        translated.bindings.push_back(binding);
    }

    tint::Result<tint::wgsl::writer::Output> wgsl_result = tint::wgsl::writer::Generate(program, writer_options);
    if (wgsl_result != tint::Success)
    {
        log_error("Tint WGSL generation failed: %s", wgsl_result.Failure().reason.c_str());
        return -1;
    }

    translated.wgsl = wgsl_result->wgsl;
    return 0;
}

struct ErrorScopeResult
{
    ErrorScopeResult()
        : status(WGPUPopErrorScopeStatus_Error),
          type(WGPUErrorType_Unknown),
          completed(false)
    {
    }

    WGPUPopErrorScopeStatus status;
    WGPUErrorType type;
    bool completed;
    std::string message;
};

static void error_scope_callback(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void* userdata1, void*)
{
    std::shared_ptr<ErrorScopeResult>* context = (std::shared_ptr<ErrorScopeResult>*)userdata1;
    if (!context)
        return;
    std::shared_ptr<ErrorScopeResult> result = *context;
    delete context;
    if (!result)
        return;

    result->status = status;
    result->type = type;
    result->completed = true;
    result->message = string_from_webgpu(message);
}

static int pop_error_scope(Device* device, const char* stage)
{
    std::shared_ptr<ErrorScopeResult> result = std::make_shared<ErrorScopeResult>();
    WGPUPopErrorScopeCallbackInfo callback_info = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
    callback_info.callback = error_scope_callback;
    std::shared_ptr<ErrorScopeResult>* callback_context = new std::shared_ptr<ErrorScopeResult>(result);
    callback_info.userdata1 = callback_context;

    WGPUFuture future = wgpuDevicePopErrorScope(device->device, callback_info);
    if (future.id == 0)
    {
        delete callback_context;
        device->error = true;
        log_error("device %llu %s returned an invalid future",
                  (unsigned long long)device->id, stage);
        return -1;
    }
    if (wait_future(device->instance.get(), future, UINT64_MAX, stage) != 0
            || !result->completed
            || result->status != WGPUPopErrorScopeStatus_Success)
    {
        device->error = true;
        log_error("device %llu %s failed status=%d type=%d: %s",
                  (unsigned long long)device->id, stage,
                  (int)result->status, (int)result->type, result->message.c_str());
        return -1;
    }
    if (result->type != WGPUErrorType_NoError)
    {
        // The scope captured this operation's error and the WebGPU device is
        // still usable. Propagate failure to the Vulkan caller without
        // converting a validation error into permanent device loss.
        log_error("device %llu %s failed type=%d: %s",
                  (unsigned long long)device->id, stage,
                  (int)result->type, result->message.c_str());
        return -1;
    }

    return 0;
}

static WGPUShaderModule create_shader_module(Device* device, const std::string& wgsl)
{
    WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
    source.code.data = wgsl.data();
    source.code.length = wgsl.size();

    WGPUShaderModuleDescriptor descriptor = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    descriptor.nextInChain = &source.chain;
    return wgpuDeviceCreateShaderModule(device->device, &descriptor);
}

// instance, physical device and device

static VkResult enumerate_properties(uint32_t available_count, const VkExtensionProperties* available,
                                     uint32_t* property_count, VkExtensionProperties* properties)
{
    if (!property_count)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (!properties)
    {
        *property_count = available_count;
        return VK_SUCCESS;
    }

    const uint32_t requested_count = *property_count;
    const uint32_t copy_count = std::min(requested_count, available_count);
    if (copy_count != 0)
        memcpy(properties, available, copy_count * sizeof(VkExtensionProperties));
    *property_count = copy_count;
    return copy_count < available_count ? VK_INCOMPLETE : VK_SUCCESS;
}

static const VkExtensionProperties g_instance_extensions[] = {
    {"VK_KHR_get_physical_device_properties2", 2}
};

static bool instance_extension_supported(const char* name)
{
    if (!name)
        return false;

    for (size_t i = 0; i < sizeof(g_instance_extensions) / sizeof(g_instance_extensions[0]); i++)
    {
        if (strcmp(name, g_instance_extensions[i].extensionName) == 0)
            return true;
    }

    return false;
}

static VkResult impl_enumerate_instance_extension_properties(const char* layer_name, uint32_t* property_count, VkExtensionProperties* properties)
{
    if (layer_name)
        return VK_ERROR_LAYER_NOT_PRESENT;

    return enumerate_properties(sizeof(g_instance_extensions) / sizeof(g_instance_extensions[0]),
                                g_instance_extensions, property_count, properties);
}

static VkResult impl_enumerate_instance_layer_properties(uint32_t* property_count, VkLayerProperties*)
{
    if (!property_count)
        return VK_ERROR_INITIALIZATION_FAILED;

    *property_count = 0;
    return VK_SUCCESS;
}

static VkResult impl_enumerate_instance_version(uint32_t* api_version)
{
    if (!api_version)
        return VK_ERROR_INITIALIZATION_FAILED;

    *api_version = VK_MAKE_VERSION(1, 1, 0);
    return VK_SUCCESS;
}

static VkResult impl_create_instance(const VkInstanceCreateInfo* create_info, const VkAllocationCallbacks* allocator, VkInstance* instance)
{
    if (!create_info || create_info->sType != VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO || !instance)
        return VK_ERROR_INITIALIZATION_FAILED;
    *instance = VK_NULL_HANDLE;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || create_info->flags != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->enabledLayerCount != 0)
        return VK_ERROR_LAYER_NOT_PRESENT;
    if (create_info->enabledExtensionCount != 0 && !create_info->ppEnabledExtensionNames)
        return VK_ERROR_INITIALIZATION_FAILED;
    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++)
    {
        if (!instance_extension_supported(create_info->ppEnabledExtensionNames[i]))
            return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    if (create_info->pApplicationInfo)
    {
        const VkApplicationInfo& application_info = *create_info->pApplicationInfo;
        if (application_info.sType != VK_STRUCTURE_TYPE_APPLICATION_INFO || application_info.pNext)
            return VK_ERROR_INITIALIZATION_FAILED;
        const uint32_t api_major = VK_VERSION_MAJOR(application_info.apiVersion);
        const uint32_t api_minor = VK_VERSION_MINOR(application_info.apiVersion);
        if (api_major > 1 || (api_major == 1 && api_minor > 1))
            return VK_ERROR_INCOMPATIBLE_DRIVER;
    }

    WGPUInstanceFeatureName instance_feature = WGPUInstanceFeatureName_TimedWaitAny;
    WGPUInstanceLimits instance_limits = WGPU_INSTANCE_LIMITS_INIT;
    instance_limits.timedWaitAnyMaxCount = (size_t)std::numeric_limits<int32_t>::max();
    WGPUInstanceDescriptor descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    descriptor.requiredFeatureCount = 1;
    descriptor.requiredFeatures = &instance_feature;
    descriptor.requiredLimits = &instance_limits;

    std::shared_ptr<Instance> impl = std::make_shared<Instance>();
    impl->instance = wgpuCreateInstance(&descriptor);
    if (!impl->instance)
        return VK_ERROR_INITIALIZATION_FAILED;

    const auto allow_wgsl_feature = [&](WGPUWGSLLanguageFeatureName webgpu_feature,
                                        tint::wgsl::LanguageFeature tint_feature) {
        if (wgpuInstanceHasWGSLLanguageFeature(impl->instance, webgpu_feature) == WGPU_TRUE)
            impl->wgsl_allowed_features.features.insert(tint_feature);
    };
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_ReadonlyAndReadwriteStorageTextures,
                       tint::wgsl::LanguageFeature::kReadonlyAndReadwriteStorageTextures);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_Packed4x8IntegerDotProduct,
                       tint::wgsl::LanguageFeature::kPacked4X8IntegerDotProduct);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_UnrestrictedPointerParameters,
                       tint::wgsl::LanguageFeature::kUnrestrictedPointerParameters);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_PointerCompositeAccess,
                       tint::wgsl::LanguageFeature::kPointerCompositeAccess);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_UniformBufferStandardLayout,
                       tint::wgsl::LanguageFeature::kUniformBufferStandardLayout);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_SubgroupId,
                       tint::wgsl::LanguageFeature::kSubgroupId);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_TextureAndSamplerLet,
                       tint::wgsl::LanguageFeature::kTextureAndSamplerLet);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_SubgroupUniformity,
                       tint::wgsl::LanguageFeature::kSubgroupUniformity);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_TextureFormatsTier1,
                       tint::wgsl::LanguageFeature::kTextureFormatsTier1);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_LinearIndexing,
                       tint::wgsl::LanguageFeature::kLinearIndexing);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_ImmediateAddressSpace,
                       tint::wgsl::LanguageFeature::kImmediateAddressSpace);

    impl->immediate_supported = impl->wgsl_allowed_features.features.count(
                                    tint::wgsl::LanguageFeature::kImmediateAddressSpace)
                                != 0;

    VkInstance_T* handle = new VkInstance_T;
    handle->impl = impl;
    *instance = handle;
    return VK_SUCCESS;
}

static void impl_destroy_instance(VkInstance instance, const VkAllocationCallbacks* allocator)
{
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    delete instance;
}

static VkResult impl_enumerate_physical_devices(VkInstance instance, uint32_t* physical_device_count, VkPhysicalDevice* physical_devices)
{
    Instance* impl = unwrap(instance);
    if (!impl || !physical_device_count)
        return VK_ERROR_INITIALIZATION_FAILED;

    VkResult result = ensure_physical_devices(impl);
    if (result != VK_SUCCESS)
        return result;

    const uint32_t available_count = (uint32_t)impl->physical_devices.size();
    if (!physical_devices)
    {
        *physical_device_count = available_count;
        return VK_SUCCESS;
    }

    const uint32_t write_count = std::min(*physical_device_count, available_count);
    for (uint32_t i = 0; i < write_count; i++)
        physical_devices[i] = impl->physical_devices[i];

    *physical_device_count = write_count;
    return write_count < available_count ? VK_INCOMPLETE : VK_SUCCESS;
}

static void impl_get_physical_device_features(VkPhysicalDevice physical_device, VkPhysicalDeviceFeatures* features)
{
    if (!unwrap(physical_device) || !features)
        return;

    memset(features, 0, sizeof(VkPhysicalDeviceFeatures));
}

static VkPhysicalDeviceType physical_device_type(WGPUAdapterType adapter_type)
{
    if (adapter_type == WGPUAdapterType_DiscreteGPU)
        return VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    if (adapter_type == WGPUAdapterType_IntegratedGPU)
        return VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    if (adapter_type == WGPUAdapterType_CPU)
        return VK_PHYSICAL_DEVICE_TYPE_CPU;

    return VK_PHYSICAL_DEVICE_TYPE_OTHER;
}

static void impl_get_physical_device_properties(VkPhysicalDevice physical_device, VkPhysicalDeviceProperties* properties)
{
    PhysicalDevice* impl = unwrap(physical_device);
    if (!impl || !properties)
        return;

    memset(properties, 0, sizeof(VkPhysicalDeviceProperties));
    properties->apiVersion = VK_MAKE_VERSION(1, 1, 0);
    properties->driverVersion = VK_MAKE_VERSION(0, 1, 0);
    properties->vendorID = impl->vendor_id;
    properties->deviceID = impl->device_id;
    properties->deviceType = physical_device_type(impl->adapter_type);
    strncpy(properties->deviceName, impl->device_name.c_str(), VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);

    VkPhysicalDeviceLimits& limits = properties->limits;
    limits.maxImageDimension1D = 0;
    limits.maxImageDimension2D = 0;
    limits.maxImageDimension3D = 0;
    limits.maxImageDimensionCube = 0;
    limits.maxImageArrayLayers = 0;
    limits.maxTexelBufferElements = 0;
    limits.maxUniformBufferRange = (uint32_t)std::min<uint64_t>(impl->limits.maxUniformBufferBindingSize, UINT32_MAX);
    limits.maxStorageBufferRange = (uint32_t)std::min<uint64_t>(impl->limits.maxStorageBufferBindingSize, UINT32_MAX);
    limits.maxPushConstantsSize =
        (uint32_t)std::max<uint64_t>(128, std::min<uint64_t>(impl->limits.maxUniformBufferBindingSize, 4096));
    limits.maxMemoryAllocationCount = 4096;
    limits.maxSamplerAllocationCount = 0;
    limits.bufferImageGranularity = 1;
    limits.sparseAddressSpaceSize = 0;
    limits.maxBoundDescriptorSets = 1;
    limits.maxPerStageDescriptorStorageBuffers = impl->limits.maxStorageBuffersPerShaderStage;
    limits.maxDescriptorSetStorageBuffers = impl->limits.maxStorageBuffersPerShaderStage;
    limits.maxPerStageDescriptorUniformBuffers = impl->limits.maxUniformBuffersPerShaderStage;
    limits.maxDescriptorSetUniformBuffers = impl->limits.maxUniformBuffersPerShaderStage;
    limits.maxPerStageResources = (uint32_t)std::min<uint64_t>(
        impl->limits.maxBindingsPerBindGroup,
        (uint64_t)limits.maxPerStageDescriptorStorageBuffers
        + limits.maxPerStageDescriptorUniformBuffers);
    limits.maxComputeSharedMemorySize = impl->limits.maxComputeWorkgroupStorageSize;
    limits.maxComputeWorkGroupCount[0] = impl->limits.maxComputeWorkgroupsPerDimension;
    limits.maxComputeWorkGroupCount[1] = impl->limits.maxComputeWorkgroupsPerDimension;
    limits.maxComputeWorkGroupCount[2] = impl->limits.maxComputeWorkgroupsPerDimension;
    limits.maxComputeWorkGroupInvocations = impl->limits.maxComputeInvocationsPerWorkgroup;
    limits.maxComputeWorkGroupSize[0] = impl->limits.maxComputeWorkgroupSizeX;
    limits.maxComputeWorkGroupSize[1] = impl->limits.maxComputeWorkgroupSizeY;
    limits.maxComputeWorkGroupSize[2] = impl->limits.maxComputeWorkgroupSizeZ;
    limits.minMemoryMapAlignment = 8;
    limits.minTexelBufferOffsetAlignment = 1;
    limits.minUniformBufferOffsetAlignment = impl->limits.minUniformBufferOffsetAlignment;
    limits.minStorageBufferOffsetAlignment = impl->limits.minStorageBufferOffsetAlignment;
    limits.nonCoherentAtomSize = 4;
}

static void impl_get_physical_device_queue_family_properties(VkPhysicalDevice physical_device, uint32_t* property_count,
                                                              VkQueueFamilyProperties* properties)
{
    if (!unwrap(physical_device) || !property_count)
        return;

    if (!properties)
    {
        *property_count = 1;
        return;
    }

    if (*property_count == 0)
        return;

    memset(&properties[0], 0, sizeof(VkQueueFamilyProperties));
    properties[0].queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    properties[0].queueCount = 1;
    properties[0].timestampValidBits = 0;
    properties[0].minImageTransferGranularity.width = 1;
    properties[0].minImageTransferGranularity.height = 1;
    properties[0].minImageTransferGranularity.depth = 1;
    *property_count = 1;
}

static void impl_get_physical_device_memory_properties(VkPhysicalDevice physical_device, VkPhysicalDeviceMemoryProperties* properties)
{
    PhysicalDevice* impl = unwrap(physical_device);
    if (!impl || !properties)
        return;

    memset(properties, 0, sizeof(VkPhysicalDeviceMemoryProperties));
    properties->memoryTypeCount = 2;
    properties->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    properties->memoryTypes[0].heapIndex = 0;
    properties->memoryTypes[1].propertyFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    properties->memoryTypes[1].heapIndex = 0;
    properties->memoryHeapCount = 1;
    properties->memoryHeaps[0].size = impl->limits.maxBufferSize;
    properties->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
}

static void impl_get_physical_device_format_properties(VkPhysicalDevice physical_device, VkFormat, VkFormatProperties* properties)
{
    if (!unwrap(physical_device) || !properties)
        return;

    memset(properties, 0, sizeof(VkFormatProperties));
}

static VkResult impl_get_physical_device_image_format_properties(VkPhysicalDevice, VkFormat, VkImageType, VkImageTiling,
                                                                  VkImageUsageFlags, VkImageCreateFlags, VkImageFormatProperties*)
{
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
}

static void impl_get_physical_device_features2(VkPhysicalDevice physical_device, VkPhysicalDeviceFeatures2KHR* features)
{
    if (!features || features->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR)
        return;

    impl_get_physical_device_features(physical_device, &features->features);

    struct StructureHeader
    {
        VkStructureType sType;
        void* pNext;
    };

    StructureHeader* next = (StructureHeader*)features->pNext;
    while (next)
    {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT)
        {
            VkPhysicalDeviceSubgroupSizeControlFeaturesEXT* subgroup =
                (VkPhysicalDeviceSubgroupSizeControlFeaturesEXT*)next;
            // Dawn requires workgroup_size.x, not only the total invocation count,
            // to be a multiple of @subgroup_size. ncnn only guarantees the total.
            subgroup->subgroupSizeControl = VK_FALSE;
            subgroup->computeFullSubgroups = VK_FALSE;
        }

        next = (StructureHeader*)next->pNext;
    }
}

static void impl_get_physical_device_properties2(VkPhysicalDevice physical_device, VkPhysicalDeviceProperties2KHR* properties)
{
    if (!properties || properties->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR)
        return;

    PhysicalDevice* impl = unwrap(physical_device);
    impl_get_physical_device_properties(physical_device, &properties->properties);

    struct StructureHeader
    {
        VkStructureType sType;
        void* pNext;
    };

    StructureHeader* next = (StructureHeader*)properties->pNext;
    while (next)
    {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES)
        {
            VkPhysicalDeviceSubgroupProperties* subgroup = (VkPhysicalDeviceSubgroupProperties*)next;
            subgroup->subgroupSize =
                impl && impl->subgroup_supported ? impl->subgroup_max_size : 4;
            subgroup->supportedStages = 0;
            subgroup->supportedOperations = 0;
            subgroup->quadOperationsInAllStages = VK_FALSE;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT)
        {
            VkPhysicalDeviceSubgroupSizeControlPropertiesEXT* subgroup =
                (VkPhysicalDeviceSubgroupSizeControlPropertiesEXT*)next;
            const uint32_t min_subgroup_size = impl && impl->subgroup_size_control_supported
                                               ? impl->subgroup_min_size : 4;
            const uint32_t max_subgroup_size = impl && impl->subgroup_size_control_supported
                                               ? impl->subgroup_max_size : 4;
            subgroup->minSubgroupSize = min_subgroup_size;
            subgroup->maxSubgroupSize = max_subgroup_size;
            subgroup->maxComputeWorkgroupSubgroups =
                impl ? std::max(impl->limits.maxComputeInvocationsPerWorkgroup / min_subgroup_size, 1u) : 1;
            subgroup->requiredSubgroupSizeStages = 0;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES)
        {
            VkPhysicalDeviceMaintenance3Properties* maintenance3 =
                (VkPhysicalDeviceMaintenance3Properties*)next;
            maintenance3->maxPerSetDescriptors = impl ? impl->limits.maxBindingsPerBindGroup : 0;
            maintenance3->maxMemoryAllocationSize = impl ? impl->limits.maxBufferSize : 0;
        }

        next = (StructureHeader*)next->pNext;
    }
}

static void impl_get_physical_device_format_properties2(VkPhysicalDevice physical_device, VkFormat format,
                                                         VkFormatProperties2KHR* properties)
{
    if (!properties || properties->sType != VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2_KHR)
        return;

    impl_get_physical_device_format_properties(physical_device, format, &properties->formatProperties);
}

static VkResult impl_get_physical_device_image_format_properties2(
    VkPhysicalDevice physical_device, const VkPhysicalDeviceImageFormatInfo2KHR* format_info,
    VkImageFormatProperties2KHR* properties)
{
    if (!format_info || format_info->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2_KHR
            || !properties || properties->sType != VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2_KHR)
        return VK_ERROR_INITIALIZATION_FAILED;

    return impl_get_physical_device_image_format_properties(physical_device, format_info->format, format_info->type,
                                                             format_info->tiling, format_info->usage, format_info->flags,
                                                             &properties->imageFormatProperties);
}

static void impl_get_physical_device_queue_family_properties2(
    VkPhysicalDevice physical_device, uint32_t* property_count, VkQueueFamilyProperties2KHR* properties)
{
    if (!properties)
    {
        impl_get_physical_device_queue_family_properties(physical_device, property_count, 0);
        return;
    }
    if (!property_count || *property_count == 0
            || properties[0].sType != VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2_KHR)
        return;

    VkQueueFamilyProperties queue_properties;
    impl_get_physical_device_queue_family_properties(physical_device, property_count, &queue_properties);
    if (*property_count != 0)
        properties[0].queueFamilyProperties = queue_properties;
}

static void impl_get_physical_device_memory_properties2(
    VkPhysicalDevice physical_device, VkPhysicalDeviceMemoryProperties2KHR* properties)
{
    if (!properties || properties->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR)
        return;

    impl_get_physical_device_memory_properties(physical_device, &properties->memoryProperties);
}

// VK_KHR_bind_memory2 and VK_KHR_get_memory_requirements2 also require image
// entry points. M1 exposes their Vulkan 1.1 buffer commands, but does not
// advertise the incomplete KHR extensions.
static const VkExtensionProperties g_device_extensions[] = {
    {"VK_KHR_descriptor_update_template", 1},
    {"VK_KHR_maintenance1", 2},
    {"VK_KHR_maintenance3", 1},
    {"VK_KHR_push_descriptor", 2},
    {"VK_KHR_storage_buffer_storage_class", 1}
};

static const VkExtensionProperties g_subgroup_size_control_extension = {
    "VK_EXT_subgroup_size_control", 2
};

static VkResult impl_enumerate_device_extension_properties(VkPhysicalDevice physical_device, const char* layer_name,
                                                            uint32_t* property_count, VkExtensionProperties* properties)
{
    if (!unwrap(physical_device))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (layer_name)
        return VK_ERROR_LAYER_NOT_PRESENT;

    PhysicalDevice* impl = unwrap(physical_device);
    VkExtensionProperties extensions[sizeof(g_device_extensions) / sizeof(g_device_extensions[0]) + 1];
    memcpy(extensions, g_device_extensions, sizeof(g_device_extensions));
    uint32_t extension_count = sizeof(g_device_extensions) / sizeof(g_device_extensions[0]);
    if (impl->subgroup_size_control_supported)
        extensions[extension_count++] = g_subgroup_size_control_extension;

    return enumerate_properties(extension_count, extensions, property_count, properties);
}

static VkResult impl_enumerate_device_layer_properties(VkPhysicalDevice physical_device, uint32_t* property_count, VkLayerProperties*)
{
    if (!unwrap(physical_device) || !property_count)
        return VK_ERROR_INITIALIZATION_FAILED;

    *property_count = 0;
    return VK_SUCCESS;
}

static bool device_extension_supported(const PhysicalDevice* physical_device, const char* name)
{
    if (!name)
        return false;

    for (size_t i = 0; i < sizeof(g_device_extensions) / sizeof(g_device_extensions[0]); i++)
    {
        if (strcmp(name, g_device_extensions[i].extensionName) == 0)
            return true;
    }

    if (physical_device && physical_device->subgroup_size_control_supported
            && strcmp(name, g_subgroup_size_control_extension.extensionName) == 0)
        return true;

    return false;
}

static VkResult impl_create_device(VkPhysicalDevice physical_device, const VkDeviceCreateInfo* create_info,
                                   const VkAllocationCallbacks* allocator, VkDevice* device)
{
    PhysicalDevice* physical_impl = unwrap(physical_device);
    if (!physical_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO || !device)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->flags != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->enabledLayerCount != 0)
        return VK_ERROR_LAYER_NOT_PRESENT;
    if ((create_info->enabledExtensionCount != 0 && !create_info->ppEnabledExtensionNames)
            || (create_info->enabledLayerCount != 0 && !create_info->ppEnabledLayerNames))
        return VK_ERROR_INITIALIZATION_FAILED;

    if (create_info->pEnabledFeatures)
    {
        const VkBool32* features = (const VkBool32*)create_info->pEnabledFeatures;
        for (size_t i = 0; i < sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32); i++)
        {
            if (features[i] != VK_FALSE)
                return VK_ERROR_FEATURE_NOT_PRESENT;
        }
    }

    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++)
    {
        if (!device_extension_supported(physical_impl, create_info->ppEnabledExtensionNames[i]))
            return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    struct StructureHeader
    {
        VkStructureType sType;
        const void* pNext;
    };
    const StructureHeader* next = (const StructureHeader*)create_info->pNext;
    while (next)
    {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT)
        {
            const VkPhysicalDeviceSubgroupSizeControlFeaturesEXT* subgroup =
                (const VkPhysicalDeviceSubgroupSizeControlFeaturesEXT*)next;
            if (subgroup->subgroupSizeControl == VK_TRUE || subgroup->computeFullSubgroups == VK_TRUE)
                return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        else
        {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }

        next = (const StructureHeader*)next->pNext;
    }

    if (create_info->queueCreateInfoCount != 1 || !create_info->pQueueCreateInfos
            || create_info->pQueueCreateInfos[0].queueFamilyIndex != 0
            || create_info->pQueueCreateInfos[0].queueCount != 1
            || create_info->pQueueCreateInfos[0].sType != VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
            || create_info->pQueueCreateInfos[0].pNext
            || create_info->pQueueCreateInfos[0].flags != 0
            || !create_info->pQueueCreateInfos[0].pQueuePriorities
            || create_info->pQueueCreateInfos[0].pQueuePriorities[0] < 0.f
            || create_info->pQueueCreateInfos[0].pQueuePriorities[0] > 1.f)
        return VK_ERROR_INITIALIZATION_FAILED;

    std::shared_ptr<Device> impl = std::make_shared<Device>();
    impl->instance = physical_impl->owner->shared_from_this();
    impl->physical_device = physical_device->impl;
    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++)
        impl->enabled_extensions.insert(create_info->ppEnabledExtensionNames[i]);

    WGPULimits required_limits = WGPU_LIMITS_INIT;
    required_limits.maxBufferSize = physical_impl->limits.maxBufferSize;
    required_limits.maxStorageBufferBindingSize = physical_impl->limits.maxStorageBufferBindingSize;
    required_limits.maxStorageBuffersPerShaderStage = physical_impl->limits.maxStorageBuffersPerShaderStage;
    required_limits.minStorageBufferOffsetAlignment = physical_impl->limits.minStorageBufferOffsetAlignment;
    required_limits.maxUniformBufferBindingSize = physical_impl->limits.maxUniformBufferBindingSize;
    required_limits.maxUniformBuffersPerShaderStage = physical_impl->limits.maxUniformBuffersPerShaderStage;
    required_limits.minUniformBufferOffsetAlignment = physical_impl->limits.minUniformBufferOffsetAlignment;
    required_limits.maxBindGroups = 2;
    required_limits.maxBindingsPerBindGroup = physical_impl->limits.maxBindingsPerBindGroup;
    required_limits.maxComputeWorkgroupStorageSize = physical_impl->limits.maxComputeWorkgroupStorageSize;
    required_limits.maxComputeInvocationsPerWorkgroup = physical_impl->limits.maxComputeInvocationsPerWorkgroup;
    required_limits.maxComputeWorkgroupSizeX = physical_impl->limits.maxComputeWorkgroupSizeX;
    required_limits.maxComputeWorkgroupSizeY = physical_impl->limits.maxComputeWorkgroupSizeY;
    required_limits.maxComputeWorkgroupSizeZ = physical_impl->limits.maxComputeWorkgroupSizeZ;
    required_limits.maxComputeWorkgroupsPerDimension = physical_impl->limits.maxComputeWorkgroupsPerDimension;
    if (physical_impl->owner->immediate_supported)
        required_limits.maxImmediateSize = std::min(64u, physical_impl->limits.maxImmediateSize);

    WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    descriptor.requiredLimits = &required_limits;
    descriptor.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    descriptor.deviceLostCallbackInfo.callback = device_lost_callback;
    descriptor.deviceLostCallbackInfo.userdata1 = impl.get();
    descriptor.uncapturedErrorCallbackInfo.callback = uncaptured_error_callback;
    descriptor.uncapturedErrorCallbackInfo.userdata1 = impl.get();

    std::shared_ptr<DeviceResult> result = std::make_shared<DeviceResult>();
    result->owner = impl;
    WGPURequestDeviceCallbackInfo callback_info = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
    callback_info.callback = request_device_callback;
    std::shared_ptr<DeviceResult>* callback_context = new std::shared_ptr<DeviceResult>(result);
    callback_info.userdata1 = callback_context;

    WGPUFuture future = wgpuAdapterRequestDevice(physical_impl->adapter, &descriptor, callback_info);
    if (future.id == 0)
    {
        delete callback_context;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (wait_future(physical_impl->owner, future, UINT64_MAX, "request-device") != 0
            || result->status != WGPURequestDeviceStatus_Success
            || !result->device)
        return VK_ERROR_INITIALIZATION_FAILED;

    impl->device = result->device;
    result->device = 0;
    result->owner.reset();
    impl->queue = wgpuDeviceGetQueue(impl->device);
    if (!impl->queue || wgpuDeviceGetLimits(impl->device, &impl->limits) != WGPUStatus_Success)
        return VK_ERROR_INITIALIZATION_FAILED;

    VkDevice_T* handle = new VkDevice_T;
    handle->impl = impl;
    VkQueue_T* queue_handle = new VkQueue_T;
    queue_handle->device = impl.get();
    impl->queue_handle = queue_handle;
    *device = handle;
    return VK_SUCCESS;
}

static void impl_destroy_device(VkDevice device, const VkAllocationCallbacks* allocator)
{
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    Device* impl = unwrap(device);
    if (impl)
    {
        const VkResult result = wait_device_submissions(impl);
        if (result != VK_SUCCESS)
            log_error("device %llu destroy wait failed result=%d", (unsigned long long)impl->id, (int)result);
        release_device_objects(impl);
    }

    delete device;
}

static void impl_get_device_queue(VkDevice device, uint32_t queue_family_index, uint32_t queue_index, VkQueue* queue)
{
    Device* impl = unwrap(device);
    if (!impl || !queue)
        return;

    *queue = queue_family_index == 0 && queue_index == 0 ? impl->queue_handle : VK_NULL_HANDLE;
}

// memory and buffer

static VkResult impl_allocate_memory(VkDevice device, const VkMemoryAllocateInfo* allocate_info,
                                     const VkAllocationCallbacks* allocator, VkDeviceMemory* memory)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !allocate_info || allocate_info->sType != VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO || !memory)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (allocate_info->pNext)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (allocate_info->allocationSize == 0 || allocate_info->memoryTypeIndex >= 2)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (allocate_info->allocationSize > device_impl->limits.maxBufferSize)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    const uint64_t padded_size = align_up(allocate_info->allocationSize, 4);
    if (padded_size == 0 || padded_size > device_impl->limits.maxBufferSize)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.size = padded_size;
    descriptor.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Uniform
                       | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;

    std::shared_ptr<DeviceMemory> impl = std::make_shared<DeviceMemory>();
    impl->owner = device_impl;
    impl->size = allocate_info->allocationSize;
    impl->padded_size = padded_size;
    impl->memory_type_index = allocate_info->memoryTypeIndex;
    impl->buffer = wgpuDeviceCreateBuffer(device_impl->device, &descriptor);
    if (!impl->buffer)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    if (allocate_info->memoryTypeIndex == 1)
        impl->host_shadow.resize(padded_size, 0);

    *memory = make_handle<VkDeviceMemory>(g_memories, impl);
    return VK_SUCCESS;
}

static void impl_free_memory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<DeviceMemory> impl = get_handle(g_memories, memory);
    if (!device_impl || !impl || impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    if (memory_in_flight(device_impl, impl.get()))
    {
        log_error("device %llu memory %llu freed while in flight",
                  (unsigned long long)device_impl->id, (unsigned long long)impl->id);
        return;
    }

    impl->live = false;
    impl->mapped = false;
    impl->mapped_offset = 0;
    impl->mapped_size = 0;
    impl->dirty_ranges.clear();
    erase_handle(g_memories, memory);
}

static VkResult impl_map_memory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size,
                                VkMemoryMapFlags flags, void** data)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<DeviceMemory> impl = get_handle(g_memories, memory);
    if (!device_impl || !impl || impl->owner != device_impl || !data || flags != 0 || impl->host_shadow.empty())
        return VK_ERROR_MEMORY_MAP_FAILED;
    if (impl->mapped)
        return VK_ERROR_MEMORY_MAP_FAILED;

    const uint64_t map_size = size == VK_WHOLE_SIZE ? impl->size - std::min<uint64_t>(offset, impl->size) : size;
    uint64_t end = 0;
    if (map_size == 0 || offset >= impl->size || offset % 8 != 0
            || !checked_add(offset, map_size, end) || end > impl->size)
        return VK_ERROR_MEMORY_MAP_FAILED;

    impl->mapped = true;
    impl->mapped_offset = offset;
    impl->mapped_size = map_size;
    *data = impl->host_shadow.data() + offset;
    return VK_SUCCESS;
}

static void impl_unmap_memory(VkDevice device, VkDeviceMemory memory)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<DeviceMemory> impl = get_handle(g_memories, memory);
    if (!device_impl || !impl || impl->owner != device_impl || !impl->mapped)
        return;

    if (!add_byte_range(impl->dirty_ranges, impl->mapped_offset, impl->mapped_size))
    {
        device_impl->error = true;
        log_error("device %llu memory %llu unmap range overflow",
                  (unsigned long long)device_impl->id, (unsigned long long)impl->id);
    }
    impl->mapped = false;
    impl->mapped_offset = 0;
    impl->mapped_size = 0;
}

static VkResult impl_flush_mapped_memory_ranges(VkDevice device, uint32_t range_count, const VkMappedMemoryRange* ranges)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || (range_count != 0 && !ranges))
        return VK_ERROR_INITIALIZATION_FAILED;

    for (uint32_t i = 0; i < range_count; i++)
    {
        std::shared_ptr<DeviceMemory> impl = get_handle(g_memories, ranges[i].memory);
        if (ranges[i].sType != VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE || ranges[i].pNext
                || !impl || impl->owner != device_impl || impl->host_shadow.empty() || !impl->mapped)
            return VK_ERROR_MEMORY_MAP_FAILED;

        const uint64_t range_size =
            ranges[i].size == VK_WHOLE_SIZE ? impl->size - std::min<uint64_t>(ranges[i].offset, impl->size) : ranges[i].size;
        uint64_t end = 0;
        uint64_t mapped_end = 0;
        if (range_size == 0 || ranges[i].offset >= impl->size
                || !checked_add(ranges[i].offset, range_size, end) || end > impl->size
                || !checked_add(impl->mapped_offset, impl->mapped_size, mapped_end)
                || ranges[i].offset < impl->mapped_offset || end > mapped_end)
            return VK_ERROR_MEMORY_MAP_FAILED;

        if (!add_byte_range(impl->dirty_ranges, ranges[i].offset, range_size))
            return VK_ERROR_MEMORY_MAP_FAILED;
    }

    return VK_SUCCESS;
}

static VkResult impl_invalidate_mapped_memory_ranges(VkDevice device, uint32_t range_count, const VkMappedMemoryRange* ranges)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || (range_count != 0 && !ranges))
        return VK_ERROR_INITIALIZATION_FAILED;

    for (uint32_t i = 0; i < range_count; i++)
    {
        std::shared_ptr<DeviceMemory> impl = get_handle(g_memories, ranges[i].memory);
        if (ranges[i].sType != VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE || ranges[i].pNext
                || !impl || impl->owner != device_impl || impl->host_shadow.empty() || !impl->mapped)
            return VK_ERROR_MEMORY_MAP_FAILED;

        const uint64_t range_size =
            ranges[i].size == VK_WHOLE_SIZE ? impl->size - std::min<uint64_t>(ranges[i].offset, impl->size) : ranges[i].size;
        uint64_t end = 0;
        uint64_t mapped_end = 0;
        if (range_size == 0 || ranges[i].offset >= impl->size
                || !checked_add(ranges[i].offset, range_size, end) || end > impl->size
                || !checked_add(impl->mapped_offset, impl->mapped_size, mapped_end)
                || ranges[i].offset < impl->mapped_offset || end > mapped_end)
            return VK_ERROR_MEMORY_MAP_FAILED;
    }

    const VkResult wait_result = wait_device_submissions(device_impl);
    if (wait_result != VK_SUCCESS)
        return wait_result;

    return VK_SUCCESS;
}

static void impl_get_device_memory_commitment(VkDevice device, VkDeviceMemory memory, VkDeviceSize* committed_memory)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<DeviceMemory> impl = get_handle(g_memories, memory);
    if (!device_impl || !impl || impl->owner != device_impl || !committed_memory)
        return;

    *committed_memory = impl->size;
}

static VkResult impl_create_buffer(VkDevice device, const VkBufferCreateInfo* create_info,
                                   const VkAllocationCallbacks* allocator, VkBuffer* buffer)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO || !buffer)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || create_info->flags != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->size == 0 || create_info->size > device_impl->limits.maxBufferSize)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (create_info->sharingMode != VK_SHARING_MODE_EXCLUSIVE)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    const VkBufferUsageFlags supported_usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
        | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (create_info->usage == 0 || (create_info->usage & ~supported_usage) != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<Buffer> impl = std::make_shared<Buffer>();
    impl->owner = device_impl;
    impl->size = create_info->size;
    impl->usage = create_info->usage;
    *buffer = make_handle<VkBuffer>(g_buffers, impl);
    return VK_SUCCESS;
}

static void impl_destroy_buffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<Buffer> impl = get_handle(g_buffers, buffer);
    if (!device_impl || !impl || impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    if (buffer_in_flight(device_impl, impl.get()))
    {
        log_error("device %llu buffer %llu destroyed while in flight",
                  (unsigned long long)device_impl->id, (unsigned long long)impl->id);
        return;
    }

    impl->live = false;
    erase_handle(g_buffers, buffer);
}

static void impl_get_buffer_memory_requirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* requirements)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<Buffer> impl = get_handle(g_buffers, buffer);
    if (!device_impl || !impl || impl->owner != device_impl || !requirements)
        return;

    requirements->size = impl->size;
    requirements->alignment = buffer_memory_alignment(*device_impl, *impl);
    requirements->memoryTypeBits = 3;
}

static void impl_get_buffer_memory_requirements2(VkDevice device,
                                                  const VkBufferMemoryRequirementsInfo2* info,
                                                  VkMemoryRequirements2* requirements)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !info || info->sType != VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2
            || info->pNext || !requirements
            || requirements->sType != VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2)
        return;

    std::shared_ptr<Buffer> buffer_impl = get_handle(g_buffers, info->buffer);
    if (!buffer_impl || buffer_impl->owner != device_impl || !buffer_impl->live)
        return;

    impl_get_buffer_memory_requirements(device, info->buffer, &requirements->memoryRequirements);

    struct StructureHeader
    {
        VkStructureType sType;
        void* pNext;
    };

    StructureHeader* next = (StructureHeader*)requirements->pNext;
    while (next)
    {
        if (next->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS)
        {
            VkMemoryDedicatedRequirements* dedicated = (VkMemoryDedicatedRequirements*)next;
            dedicated->prefersDedicatedAllocation = VK_FALSE;
            dedicated->requiresDedicatedAllocation = VK_FALSE;
        }
        next = (StructureHeader*)next->pNext;
    }
}

static VkResult validate_buffer_memory_binding(Device* device_impl,
                                                const std::shared_ptr<Buffer>& buffer_impl,
                                                const std::shared_ptr<DeviceMemory>& memory_impl,
                                                VkDeviceSize memory_offset)
{
    if (!device_impl || !buffer_impl || !memory_impl
            || buffer_impl->owner != device_impl || memory_impl->owner != device_impl)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!buffer_impl->live || !memory_impl->live || buffer_impl->memory)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (memory_offset % buffer_memory_alignment(*device_impl, *buffer_impl) != 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    uint64_t end = 0;
    if (!checked_add(memory_offset, buffer_impl->size, end) || end > memory_impl->size)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    return VK_SUCCESS;
}

static VkResult impl_bind_buffer_memory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memory_offset)
{
    Device* device_impl = unwrap(device);
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;

    std::shared_ptr<Buffer> buffer_impl = get_handle(g_buffers, buffer);
    std::shared_ptr<DeviceMemory> memory_impl = get_handle(g_memories, memory);
    const VkResult result = validate_buffer_memory_binding(device_impl, buffer_impl, memory_impl, memory_offset);
    if (result != VK_SUCCESS)
        return result;

    buffer_impl->memory = memory_impl;
    buffer_impl->memory_offset = memory_offset;
    return VK_SUCCESS;
}

static VkResult impl_bind_buffer_memory2(VkDevice device, uint32_t bind_info_count,
                                         const VkBindBufferMemoryInfo* bind_infos)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || (bind_info_count != 0 && !bind_infos))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;

    struct PendingBinding
    {
        std::shared_ptr<Buffer> buffer;
        std::shared_ptr<DeviceMemory> memory;
        VkDeviceSize memory_offset;
    };

    std::vector<PendingBinding> pending;
    pending.reserve(bind_info_count);
    std::unordered_set<Buffer*> seen_buffers;
    for (uint32_t i = 0; i < bind_info_count; i++)
    {
        const VkBindBufferMemoryInfo& info = bind_infos[i];
        if (info.sType != VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO || info.pNext)
            return VK_ERROR_FEATURE_NOT_PRESENT;

        PendingBinding binding;
        binding.buffer = get_handle(g_buffers, info.buffer);
        binding.memory = get_handle(g_memories, info.memory);
        binding.memory_offset = info.memoryOffset;
        if (!binding.buffer || !seen_buffers.insert(binding.buffer.get()).second)
            return VK_ERROR_INITIALIZATION_FAILED;

        const VkResult result = validate_buffer_memory_binding(
            device_impl, binding.buffer, binding.memory, binding.memory_offset);
        if (result != VK_SUCCESS)
            return result;
        pending.push_back(binding);
    }

    for (size_t i = 0; i < pending.size(); i++)
    {
        pending[i].buffer->memory = pending[i].memory;
        pending[i].buffer->memory_offset = pending[i].memory_offset;
    }

    return VK_SUCCESS;
}

// descriptor layouts, shader modules and pipelines

static VkResult validate_descriptor_set_layout_create_info(
    Device* device_impl, const VkDescriptorSetLayoutCreateInfo* create_info,
    std::vector<DescriptorBinding>* bindings, bool* push_descriptor)
{
    if (!device_impl || !create_info
            || create_info->sType != VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (create_info->pNext)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    const VkDescriptorSetLayoutCreateFlags supported_flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    if ((create_info->flags & ~supported_flags) != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->bindingCount != 0 && !create_info->pBindings)
        return VK_ERROR_INITIALIZATION_FAILED;

    const bool is_push_descriptor =
        (create_info->flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR) != 0;
    if (is_push_descriptor
            && device_impl->enabled_extensions.find("VK_KHR_push_descriptor") == device_impl->enabled_extensions.end())
        return VK_ERROR_EXTENSION_NOT_PRESENT;

    std::vector<DescriptorBinding> validated_bindings;
    validated_bindings.reserve(create_info->bindingCount);

    std::unordered_set<uint32_t> seen_bindings;
    uint32_t storage_buffer_count = 0;
    uint32_t uniform_buffer_count = 0;
    for (uint32_t i = 0; i < create_info->bindingCount; i++)
    {
        const VkDescriptorSetLayoutBinding& source = create_info->pBindings[i];
        if ((source.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
             && source.descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                || source.descriptorCount != 1
                || source.stageFlags != VK_SHADER_STAGE_COMPUTE_BIT
                || source.pImmutableSamplers
                || source.binding >= device_impl->limits.maxBindingsPerBindGroup
                || !seen_bindings.insert(source.binding).second)
            return VK_ERROR_FEATURE_NOT_PRESENT;

        if (source.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            storage_buffer_count++;
        else
            uniform_buffer_count++;

        DescriptorBinding binding;
        binding.binding = source.binding;
        binding.descriptor_type = source.descriptorType;
        binding.descriptor_count = source.descriptorCount;
        binding.stage_flags = source.stageFlags;
        validated_bindings.push_back(binding);
    }

    if (storage_buffer_count > device_impl->limits.maxStorageBuffersPerShaderStage
            || uniform_buffer_count > device_impl->limits.maxUniformBuffersPerShaderStage
            || create_info->bindingCount > device_impl->limits.maxBindingsPerBindGroup)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    if (bindings)
        bindings->swap(validated_bindings);
    if (push_descriptor)
        *push_descriptor = is_push_descriptor;
    return VK_SUCCESS;
}

static VkResult impl_create_descriptor_set_layout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* create_info,
                                                   const VkAllocationCallbacks* allocator, VkDescriptorSetLayout* set_layout)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !set_layout)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<DescriptorSetLayout> impl = std::make_shared<DescriptorSetLayout>();
    impl->owner = device_impl;
    const VkResult result = validate_descriptor_set_layout_create_info(
        device_impl, create_info, &impl->bindings, &impl->push_descriptor);
    if (result != VK_SUCCESS)
        return result;

    *set_layout = make_handle<VkDescriptorSetLayout>(g_descriptor_set_layouts, impl);
    return VK_SUCCESS;
}

static void impl_get_descriptor_set_layout_support(
    VkDevice device, const VkDescriptorSetLayoutCreateInfo* create_info,
    VkDescriptorSetLayoutSupport* support)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !support || support->sType != VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT)
        return;

    support->supported = VK_FALSE;
    if (device_failed(device_impl))
        return;

    support->supported = validate_descriptor_set_layout_create_info(
                             device_impl, create_info, 0, 0)
                             == VK_SUCCESS
                         ? VK_TRUE : VK_FALSE;
}

static void impl_destroy_descriptor_set_layout(VkDevice device, VkDescriptorSetLayout set_layout,
                                               const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<DescriptorSetLayout> impl = get_handle(g_descriptor_set_layouts, set_layout);
    if (!device_impl || !impl || impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    erase_handle(g_descriptor_set_layouts, set_layout);
}

static VkResult impl_create_pipeline_layout(VkDevice device, const VkPipelineLayoutCreateInfo* create_info,
                                            const VkAllocationCallbacks* allocator, VkPipelineLayout* pipeline_layout)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO || !pipeline_layout)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || create_info->flags != 0 || create_info->setLayoutCount > 1
            || (create_info->setLayoutCount != 0 && !create_info->pSetLayouts)
            || (create_info->pushConstantRangeCount != 0 && !create_info->pPushConstantRanges))
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<PipelineLayout> impl = std::make_shared<PipelineLayout>();
    impl->owner = device_impl;
    for (uint32_t i = 0; i < create_info->setLayoutCount; i++)
    {
        std::shared_ptr<DescriptorSetLayout> set_layout =
            get_handle(g_descriptor_set_layouts, create_info->pSetLayouts[i]);
        if (!set_layout || set_layout->owner != device_impl)
            return VK_ERROR_INITIALIZATION_FAILED;
        impl->set_layouts.push_back(set_layout);
    }

    for (uint32_t i = 0; i < create_info->pushConstantRangeCount; i++)
    {
        const VkPushConstantRange& source = create_info->pPushConstantRanges[i];
        uint64_t end = 0;
        if (source.size == 0 || source.offset % 4 != 0 || source.size % 4 != 0
                || source.stageFlags != VK_SHADER_STAGE_COMPUTE_BIT
                || !checked_add(source.offset, source.size, end)
                || end > std::min<uint64_t>(device_impl->limits.maxUniformBufferBindingSize, 4096))
            return VK_ERROR_FEATURE_NOT_PRESENT;

        for (size_t j = 0; j < impl->push_constant_ranges.size(); j++)
        {
            const PushConstantRange& other = impl->push_constant_ranges[j];
            const uint64_t other_end = (uint64_t)other.offset + other.size;
            if ((uint64_t)source.offset < other_end && (uint64_t)other.offset < end)
                return VK_ERROR_FEATURE_NOT_PRESENT;
        }

        PushConstantRange range;
        range.stage_flags = source.stageFlags;
        range.offset = source.offset;
        range.size = source.size;
        impl->push_constant_ranges.push_back(range);
        impl->push_constant_size = std::max<uint32_t>(impl->push_constant_size, (uint32_t)end);
    }

    *pipeline_layout = make_handle<VkPipelineLayout>(g_pipeline_layouts, impl);
    return VK_SUCCESS;
}

static void impl_destroy_pipeline_layout(VkDevice device, VkPipelineLayout pipeline_layout,
                                         const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<PipelineLayout> impl = get_handle(g_pipeline_layouts, pipeline_layout);
    if (!device_impl || !impl || impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    erase_handle(g_pipeline_layouts, pipeline_layout);
}

static VkResult impl_create_shader_module(VkDevice device, const VkShaderModuleCreateInfo* create_info,
                                          const VkAllocationCallbacks* allocator, VkShaderModule* shader_module)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO || !shader_module)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || create_info->flags != 0
            || !create_info->pCode || create_info->codeSize < 20 || create_info->codeSize % 4 != 0)
        return VK_ERROR_INVALID_SHADER_NV;

    std::shared_ptr<ShaderModule> impl = std::make_shared<ShaderModule>();
    impl->owner = device_impl;
    impl->spirv.assign(create_info->pCode, create_info->pCode + create_info->codeSize / 4);
    if (impl->spirv[0] != (uint32_t)spv::MagicNumber)
        return VK_ERROR_INVALID_SHADER_NV;

    *shader_module = make_handle<VkShaderModule>(g_shader_modules, impl);
    return VK_SUCCESS;
}

static void impl_destroy_shader_module(VkDevice device, VkShaderModule shader_module, const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<ShaderModule> impl = get_handle(g_shader_modules, shader_module);
    if (!device_impl || !impl || impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    erase_handle(g_shader_modules, shader_module);
}

static const DescriptorBinding* find_layout_binding(const DescriptorSetLayout& layout, uint32_t binding)
{
    for (size_t i = 0; i < layout.bindings.size(); i++)
    {
        if (layout.bindings[i].binding == binding)
            return &layout.bindings[i];
    }

    return 0;
}

static VkResult create_compute_pipeline(Device* device_impl, const VkComputePipelineCreateInfo& create_info,
                                        std::shared_ptr<ComputePipeline>& pipeline)
{
    if (create_info.sType != VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
            || create_info.pNext
            || create_info.flags != 0
            || create_info.stage.sType != VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
            || create_info.stage.pNext
            || create_info.stage.stage != VK_SHADER_STAGE_COMPUTE_BIT
            || create_info.stage.flags != 0
            || create_info.basePipelineHandle != VK_NULL_HANDLE)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<ShaderModule> shader = get_handle(g_shader_modules, create_info.stage.module);
    std::shared_ptr<PipelineLayout> layout = get_handle(g_pipeline_layouts, create_info.layout);
    if (!shader || !layout || shader->owner != device_impl || layout->owner != device_impl)
        return VK_ERROR_INITIALIZATION_FAILED;

    TranslatedShader translated;
    const uint64_t translate_begin = monotonic_time_ns();
    const int translate_result =
        translate_shader(shader->spirv, create_info.stage.pName, create_info.stage.pSpecializationInfo,
                         *layout, *device_impl, translated);
    device_impl->pipeline_translate_count.fetch_add(1, std::memory_order_relaxed);
    device_impl->pipeline_translate_ns.fetch_add(monotonic_time_ns() - translate_begin, std::memory_order_relaxed);
    if (translate_result != 0)
        return VK_ERROR_INVALID_SHADER_NV;

    wgpuDevicePushErrorScope(device_impl->device, WGPUErrorFilter_Validation);

    pipeline = std::make_shared<ComputePipeline>();
    pipeline->owner = device_impl;
    pipeline->vk_pipeline_layout = layout;
    pipeline->active_bindings = translated.bindings;
    pipeline->immediate_size = translated.immediate_size;
    pipeline->push_constant_uniform = translated.push_constant_uniform;
    pipeline->push_constant_data_size = translated.push_constant_data_size;
    pipeline->push_constant_uniform_size = translated.push_constant_uniform_size;
    pipeline->workgroup_size_x = translated.workgroup_size_x;
    pipeline->workgroup_size_y = translated.workgroup_size_y;
    pipeline->workgroup_size_z = translated.workgroup_size_z;
    pipeline->shader_module = create_shader_module(device_impl, translated.wgsl);
    if (!pipeline->shader_module)
    {
        pop_error_scope(device_impl, "shader module");
        return VK_ERROR_INVALID_SHADER_NV;
    }

    std::vector<WGPUBindGroupLayoutEntry> group_entries;
    for (size_t i = 0; i < translated.bindings.size(); i++)
    {
        const ActiveBinding& binding = translated.bindings[i];
        if (binding.internal_uniform)
            continue;
        if (layout->set_layouts.empty())
        {
            log_error("pipeline active binding=%u has no Vulkan descriptor set layout", binding.binding);
            pop_error_scope(device_impl, "pipeline descriptor layout");
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        const DescriptorBinding* layout_binding = find_layout_binding(*layout->set_layouts[0], binding.binding);
        if (!layout_binding || layout_binding->descriptor_type != binding.descriptor_type)
        {
            log_error("pipeline active binding=%u is missing from Vulkan descriptor set layout", binding.binding);
            pop_error_scope(device_impl, "pipeline descriptor binding");
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entry.binding = binding.binding;
        entry.visibility = WGPUShaderStage_Compute;
        entry.buffer = WGPU_BUFFER_BINDING_LAYOUT_INIT;
        entry.buffer.type = binding.descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                            ? WGPUBufferBindingType_Uniform
                            : binding.layout_access == BUFFER_ACCESS_READ
                              ? WGPUBufferBindingType_ReadOnlyStorage : WGPUBufferBindingType_Storage;
        entry.buffer.minBindingSize = binding.min_binding_size;
        group_entries.push_back(entry);
    }

    WGPUBindGroupLayoutDescriptor group_descriptor = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    group_descriptor.entryCount = group_entries.size();
    group_descriptor.entries = group_entries.empty() ? 0 : group_entries.data();
    pipeline->bind_group_layout = wgpuDeviceCreateBindGroupLayout(device_impl->device, &group_descriptor);
    if (!pipeline->bind_group_layout)
    {
        pop_error_scope(device_impl, "bind group layout");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    WGPUBindGroupLayout layouts[2] = {pipeline->bind_group_layout, 0};
    uint32_t layout_count = 1;
    if (translated.push_constant_uniform)
    {
        WGPUBindGroupLayoutEntry uniform_entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        uniform_entry.binding = 0;
        uniform_entry.visibility = WGPUShaderStage_Compute;
        uniform_entry.buffer = WGPU_BUFFER_BINDING_LAYOUT_INIT;
        uniform_entry.buffer.type = WGPUBufferBindingType_Uniform;
        uniform_entry.buffer.hasDynamicOffset = WGPU_FALSE;
        uniform_entry.buffer.minBindingSize = translated.push_constant_uniform_size;

        WGPUBindGroupLayoutDescriptor uniform_descriptor = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        uniform_descriptor.entryCount = 1;
        uniform_descriptor.entries = &uniform_entry;
        pipeline->immediate_bind_group_layout = wgpuDeviceCreateBindGroupLayout(device_impl->device, &uniform_descriptor);
        if (!pipeline->immediate_bind_group_layout)
        {
            pop_error_scope(device_impl, "push uniform bind group layout");
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        layouts[1] = pipeline->immediate_bind_group_layout;
        layout_count = 2;
    }

    WGPUPipelineLayoutDescriptor pipeline_layout_descriptor = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipeline_layout_descriptor.bindGroupLayoutCount = layout_count;
    pipeline_layout_descriptor.bindGroupLayouts = layouts;
    pipeline_layout_descriptor.immediateSize = translated.immediate_size;
    pipeline->pipeline_layout = wgpuDeviceCreatePipelineLayout(device_impl->device, &pipeline_layout_descriptor);
    if (!pipeline->pipeline_layout)
    {
        pop_error_scope(device_impl, "pipeline layout");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    WGPUComputePipelineDescriptor pipeline_descriptor = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pipeline_descriptor.layout = pipeline->pipeline_layout;
    pipeline_descriptor.compute.module = pipeline->shader_module;
    pipeline_descriptor.compute.entryPoint.data = create_info.stage.pName;
    pipeline_descriptor.compute.entryPoint.length = strlen(create_info.stage.pName);
    const uint64_t pipeline_begin = monotonic_time_ns();
    pipeline->pipeline = wgpuDeviceCreateComputePipeline(device_impl->device, &pipeline_descriptor);
    device_impl->webgpu_pipeline_count.fetch_add(1, std::memory_order_relaxed);
    device_impl->webgpu_pipeline_ns.fetch_add(monotonic_time_ns() - pipeline_begin, std::memory_order_relaxed);
    if (pop_error_scope(device_impl, "compute pipeline") != 0 || !pipeline->pipeline)
        return VK_ERROR_INITIALIZATION_FAILED;

    // Dispatch only needs the compute pipeline and the bind group layouts.
    // Releasing construction-only objects here avoids retaining one shader
    // module and one pipeline layout for every ncnn layer pipeline.
    wgpuPipelineLayoutRelease(pipeline->pipeline_layout);
    pipeline->pipeline_layout = 0;
    wgpuShaderModuleRelease(pipeline->shader_module);
    pipeline->shader_module = 0;

    return VK_SUCCESS;
}

static VkResult impl_create_compute_pipelines(VkDevice device, VkPipelineCache pipeline_cache, uint32_t create_info_count,
                                              const VkComputePipelineCreateInfo* create_infos,
                                              const VkAllocationCallbacks* allocator, VkPipeline* pipelines)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || create_info_count == 0 || !create_infos || !pipelines)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (pipeline_cache != VK_NULL_HANDLE)
    {
        std::shared_ptr<PipelineCache> cache = get_handle(g_pipeline_caches, pipeline_cache);
        if (!cache || cache->owner != device_impl)
            return VK_ERROR_INITIALIZATION_FAILED;
    }

    for (uint32_t i = 0; i < create_info_count; i++)
        pipelines[i] = VK_NULL_HANDLE;

    for (uint32_t i = 0; i < create_info_count; i++)
    {
        std::shared_ptr<ComputePipeline> pipeline;
        VkResult result = create_compute_pipeline(device_impl, create_infos[i], pipeline);
        if (result != VK_SUCCESS)
        {
            std::shared_ptr<ShaderModule> shader = get_handle(g_shader_modules, create_infos[i].stage.module);
            std::shared_ptr<PipelineLayout> layout = get_handle(g_pipeline_layouts, create_infos[i].layout);
            log_error("device %llu vkCreateComputePipelines index=%u shader=%llu layout=%llu entry=%s failed result=%d",
                      (unsigned long long)device_impl->id, i,
                      shader ? (unsigned long long)shader->id : 0,
                      layout ? (unsigned long long)layout->id : 0,
                      create_infos[i].stage.pName ? create_infos[i].stage.pName : "<null>",
                      (int)result);
            return result;
        }

        pipelines[i] = make_handle<VkPipeline>(g_compute_pipelines, pipeline);
    }

    return VK_SUCCESS;
}

static void impl_destroy_pipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<ComputePipeline> impl = get_handle(g_compute_pipelines, pipeline);
    if (!device_impl || !impl || impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    if (pipeline_in_flight(device_impl, impl.get()))
    {
        log_error("device %llu pipeline %llu destroyed while in flight",
                  (unsigned long long)device_impl->id, (unsigned long long)impl->id);
        return;
    }

    impl->live = false;
    erase_handle(g_compute_pipelines, pipeline);
}

static VkResult impl_create_descriptor_update_template(VkDevice device,
                                                        const VkDescriptorUpdateTemplateCreateInfo* create_info,
                                                        const VkAllocationCallbacks* allocator,
                                                        VkDescriptorUpdateTemplate* descriptor_update_template)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || !descriptor_update_template
            || create_info->sType != VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || create_info->flags != 0
            || (create_info->descriptorUpdateEntryCount != 0 && !create_info->pDescriptorUpdateEntries))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (create_info->templateType != VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR
            || create_info->pipelineBindPoint != VK_PIPELINE_BIND_POINT_COMPUTE
            || create_info->set != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<PipelineLayout> pipeline_layout = get_handle(g_pipeline_layouts, create_info->pipelineLayout);
    std::shared_ptr<DescriptorSetLayout> set_layout = get_handle(g_descriptor_set_layouts, create_info->descriptorSetLayout);
    if (!pipeline_layout || !set_layout
            || pipeline_layout->owner != device_impl || set_layout->owner != device_impl
            || !set_layout->push_descriptor || pipeline_layout->set_layouts.empty()
            || pipeline_layout->set_layouts[0].get() != set_layout.get())
        return VK_ERROR_INITIALIZATION_FAILED;

    std::shared_ptr<DescriptorUpdateTemplate> impl = std::make_shared<DescriptorUpdateTemplate>();
    impl->owner = device_impl;
    impl->template_type = create_info->templateType;
    impl->bind_point = create_info->pipelineBindPoint;
    impl->set = create_info->set;
    impl->set_layout = set_layout;
    impl->pipeline_layout = pipeline_layout;

    std::unordered_set<uint32_t> updated_bindings;
    for (uint32_t i = 0; i < create_info->descriptorUpdateEntryCount; i++)
    {
        const VkDescriptorUpdateTemplateEntry& source = create_info->pDescriptorUpdateEntries[i];
        const DescriptorBinding* layout_binding = find_layout_binding(*set_layout, source.dstBinding);
        if ((source.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
             && source.descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                || source.descriptorCount != 1
                || source.dstArrayElement != 0 || !layout_binding
                || layout_binding->descriptor_type != source.descriptorType
                || !updated_bindings.insert(source.dstBinding).second)
            return VK_ERROR_FEATURE_NOT_PRESENT;

        DescriptorTemplateEntry entry;
        entry.binding = source.dstBinding;
        entry.array_element = source.dstArrayElement;
        entry.descriptor_count = source.descriptorCount;
        entry.descriptor_type = source.descriptorType;
        entry.offset = source.offset;
        entry.stride = source.stride;
        impl->entries.push_back(entry);
    }

    *descriptor_update_template =
        make_handle<VkDescriptorUpdateTemplate>(g_descriptor_update_templates, impl);
    return VK_SUCCESS;
}

static void impl_destroy_descriptor_update_template(VkDevice device,
                                                     VkDescriptorUpdateTemplate descriptor_update_template,
                                                     const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<DescriptorUpdateTemplate> impl =
        get_handle(g_descriptor_update_templates, descriptor_update_template);
    if (!device_impl || !impl || impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    erase_handle(g_descriptor_update_templates, descriptor_update_template);
}

// command pools, command buffers and command recording

static void reset_command_buffer(CommandBuffer& command_buffer, bool release_resources = false)
{
    command_buffer.state = COMMAND_BUFFER_INITIAL;
    command_buffer.pending_count = 0;
    command_buffer.usage_flags = 0;
    command_buffer.error = VK_SUCCESS;
    command_buffer.current_pipeline.reset();
    command_buffer.descriptor_pipeline_layout.reset();
    command_buffer.descriptors.clear();
    command_buffer.push_constant_pipeline_layout.reset();
    command_buffer.push_constants.clear();
    command_buffer.commands.clear();

    if (release_resources)
    {
        std::unordered_map<uint32_t, DescriptorValue>().swap(command_buffer.descriptors);
        std::vector<unsigned char>().swap(command_buffer.push_constants);
        std::vector<Command>().swap(command_buffer.commands);
    }
}

static void release_device_objects(Device* device)
{
    if (!device)
        return;

    size_t object_count = 0;

    std::vector<std::shared_ptr<Fence> > fences =
        g_fences.take_if([&](const std::shared_ptr<Fence>& value) { return value->owner == device; });
    object_count += fences.size();

    std::vector<std::shared_ptr<CommandPool> > command_pools =
        g_command_pools.take_if([&](const std::shared_ptr<CommandPool>& value) { return value->owner == device; });
    object_count += command_pools.size();
    for (size_t i = 0; i < command_pools.size(); i++)
    {
        std::vector<VkCommandBuffer> command_buffers;
        {
            std::lock_guard<std::mutex> lock(command_pools[i]->mutex);
            command_buffers.assign(command_pools[i]->command_buffers.begin(), command_pools[i]->command_buffers.end());
            command_pools[i]->command_buffers.clear();
        }
        object_count += command_buffers.size();
        for (size_t j = 0; j < command_buffers.size(); j++)
            delete command_buffers[j];
    }

    std::vector<std::shared_ptr<PipelineCache> > pipeline_caches =
        g_pipeline_caches.take_if([&](const std::shared_ptr<PipelineCache>& value) { return value->owner == device; });
    object_count += pipeline_caches.size();
    std::vector<std::shared_ptr<ComputePipeline> > compute_pipelines =
        g_compute_pipelines.take_if([&](const std::shared_ptr<ComputePipeline>& value) { return value->owner == device; });
    object_count += compute_pipelines.size();
    std::vector<std::shared_ptr<DescriptorUpdateTemplate> > descriptor_update_templates =
        g_descriptor_update_templates.take_if(
            [&](const std::shared_ptr<DescriptorUpdateTemplate>& value) { return value->owner == device; });
    object_count += descriptor_update_templates.size();
    std::vector<std::shared_ptr<ShaderModule> > shader_modules =
        g_shader_modules.take_if([&](const std::shared_ptr<ShaderModule>& value) { return value->owner == device; });
    object_count += shader_modules.size();
    std::vector<std::shared_ptr<PipelineLayout> > pipeline_layouts =
        g_pipeline_layouts.take_if([&](const std::shared_ptr<PipelineLayout>& value) { return value->owner == device; });
    object_count += pipeline_layouts.size();
    std::vector<std::shared_ptr<DescriptorSetLayout> > descriptor_set_layouts =
        g_descriptor_set_layouts.take_if(
            [&](const std::shared_ptr<DescriptorSetLayout>& value) { return value->owner == device; });
    object_count += descriptor_set_layouts.size();
    std::vector<std::shared_ptr<Buffer> > buffers =
        g_buffers.take_if([&](const std::shared_ptr<Buffer>& value) { return value->owner == device; });
    object_count += buffers.size();
    std::vector<std::shared_ptr<DeviceMemory> > memories =
        g_memories.take_if([&](const std::shared_ptr<DeviceMemory>& value) { return value->owner == device; });
    object_count += memories.size();

    if (object_count != 0)
        log_error("device %llu destroyed with %llu live child objects",
                  (unsigned long long)device->id, (unsigned long long)object_count);
}

static VkResult impl_create_command_pool(VkDevice device, const VkCommandPoolCreateInfo* create_info,
                                         const VkAllocationCallbacks* allocator, VkCommandPool* command_pool)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO || !command_pool)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->queueFamilyIndex != 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    const VkCommandPoolCreateFlags supported_flags =
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if ((create_info->flags & ~supported_flags) != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<CommandPool> impl = std::make_shared<CommandPool>();
    impl->owner = device_impl;
    impl->flags = create_info->flags;
    impl->queue_family_index = create_info->queueFamilyIndex;
    *command_pool = make_handle<VkCommandPool>(g_command_pools, impl);
    return VK_SUCCESS;
}

static void impl_destroy_command_pool(VkDevice device, VkCommandPool command_pool, const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<CommandPool> impl = get_handle(g_command_pools, command_pool);
    if (!device_impl || !impl || impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    std::vector<VkCommandBuffer> command_buffers;
    {
        std::lock_guard<std::mutex> lock(impl->mutex);
        for (std::unordered_set<VkCommandBuffer>::const_iterator it = impl->command_buffers.begin();
                it != impl->command_buffers.end(); ++it)
        {
            CommandBuffer* command_buffer = unwrap(*it);
            if (command_buffer && command_buffer->pending_count != 0)
            {
                log_error("command pool %llu destroyed with pending command buffer %llu",
                          (unsigned long long)impl->id, (unsigned long long)command_buffer->id);
                return;
            }
        }
        command_buffers.assign(impl->command_buffers.begin(), impl->command_buffers.end());
        impl->command_buffers.clear();
    }
    for (size_t i = 0; i < command_buffers.size(); i++)
        delete command_buffers[i];

    erase_handle(g_command_pools, command_pool);
}

static VkResult impl_allocate_command_buffers(VkDevice device, const VkCommandBufferAllocateInfo* allocate_info,
                                              VkCommandBuffer* command_buffers)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !allocate_info || allocate_info->sType != VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
            || allocate_info->commandBufferCount == 0 || !command_buffers)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocate_info->pNext || allocate_info->level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<CommandPool> pool = get_handle(g_command_pools, allocate_info->commandPool);
    if (!pool || pool->owner != device_impl)
        return VK_ERROR_INITIALIZATION_FAILED;

    for (uint32_t i = 0; i < allocate_info->commandBufferCount; i++)
        command_buffers[i] = VK_NULL_HANDLE;

    for (uint32_t i = 0; i < allocate_info->commandBufferCount; i++)
    {
        std::shared_ptr<CommandBuffer> impl = std::make_shared<CommandBuffer>();
        impl->owner = device_impl;
        impl->pool = pool;
        VkCommandBuffer_T* handle = new VkCommandBuffer_T;
        handle->impl = impl;
        command_buffers[i] = handle;

        std::lock_guard<std::mutex> lock(pool->mutex);
        pool->command_buffers.insert(handle);
    }

    return VK_SUCCESS;
}

static void impl_free_command_buffers(VkDevice device, VkCommandPool command_pool, uint32_t command_buffer_count,
                                      const VkCommandBuffer* command_buffers)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<CommandPool> pool = get_handle(g_command_pools, command_pool);
    if (!device_impl || !pool || pool->owner != device_impl || (command_buffer_count != 0 && !command_buffers))
        return;

    for (uint32_t i = 0; i < command_buffer_count; i++)
    {
        CommandBuffer* command_buffer = unwrap(command_buffers[i]);
        if (!command_buffer || command_buffer->owner != device_impl || command_buffer->pool.get() != pool.get()
                || command_buffer->pending_count != 0)
            continue;

        {
            std::lock_guard<std::mutex> lock(pool->mutex);
            pool->command_buffers.erase(command_buffers[i]);
        }
        delete command_buffers[i];
    }
}

static VkResult impl_reset_command_pool(VkDevice device, VkCommandPool command_pool, VkCommandPoolResetFlags flags)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<CommandPool> pool = get_handle(g_command_pools, command_pool);
    const VkCommandPoolResetFlags supported_flags = VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT;
    if (!device_impl || !pool || pool->owner != device_impl || (flags & ~supported_flags) != 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    std::lock_guard<std::mutex> lock(pool->mutex);
    for (std::unordered_set<VkCommandBuffer>::const_iterator it = pool->command_buffers.begin();
            it != pool->command_buffers.end(); ++it)
    {
        CommandBuffer* command_buffer = unwrap(*it);
        if (!command_buffer || command_buffer->pending_count != 0)
            return VK_ERROR_INITIALIZATION_FAILED;
    }
    for (std::unordered_set<VkCommandBuffer>::const_iterator it = pool->command_buffers.begin();
            it != pool->command_buffers.end(); ++it)
        reset_command_buffer(*unwrap(*it), (flags & VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT) != 0);

    return VK_SUCCESS;
}

static void impl_trim_command_pool(VkDevice device, VkCommandPool command_pool, VkCommandPoolTrimFlags flags)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<CommandPool> pool = get_handle(g_command_pools, command_pool);
    if (!device_impl || !pool || pool->owner != device_impl || flags != 0)
    {
        log_error("trim command pool received invalid arguments");
        return;
    }

    std::lock_guard<std::mutex> lock(pool->mutex);
    for (std::unordered_set<VkCommandBuffer>::const_iterator it = pool->command_buffers.begin();
            it != pool->command_buffers.end(); ++it)
    {
        CommandBuffer* command_buffer = unwrap(*it);
        if (!command_buffer)
            continue;
        command_buffer->push_constants.shrink_to_fit();
        command_buffer->commands.shrink_to_fit();
        command_buffer->descriptors.rehash(command_buffer->descriptors.size());
    }
}

static VkResult impl_reset_command_buffer(VkCommandBuffer command_buffer, VkCommandBufferResetFlags flags)
{
    CommandBuffer* impl = unwrap(command_buffer);
    const VkCommandBufferResetFlags supported_flags = VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT;
    if (!impl || (flags & ~supported_flags) != 0
            || impl->pending_count != 0 || !impl->pool
            || (impl->pool->flags & VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT) == 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    reset_command_buffer(*impl, (flags & VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT) != 0);
    return VK_SUCCESS;
}

static VkResult impl_begin_command_buffer(VkCommandBuffer command_buffer, const VkCommandBufferBeginInfo* begin_info)
{
    CommandBuffer* impl = unwrap(command_buffer);
    if (!impl || !begin_info || begin_info->sType != VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
            || begin_info->pNext || begin_info->pInheritanceInfo || impl->pending_count != 0
            || (impl->state != COMMAND_BUFFER_INITIAL && impl->state != COMMAND_BUFFER_EXECUTABLE))
        return VK_ERROR_INITIALIZATION_FAILED;

    const VkCommandBufferUsageFlags supported_flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((begin_info->flags & ~supported_flags) != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    reset_command_buffer(*impl);
    impl->state = COMMAND_BUFFER_RECORDING;
    impl->usage_flags = begin_info->flags;
    return VK_SUCCESS;
}

static VkResult impl_end_command_buffer(VkCommandBuffer command_buffer)
{
    CommandBuffer* impl = unwrap(command_buffer);
    if (!impl || impl->state != COMMAND_BUFFER_RECORDING)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (impl->error != VK_SUCCESS)
    {
        impl->state = COMMAND_BUFFER_INVALID;
        return impl->error;
    }

    impl->state = COMMAND_BUFFER_EXECUTABLE;
    return VK_SUCCESS;
}

static void set_command_error(CommandBuffer* command_buffer, VkResult error, const char* operation)
{
    if (!command_buffer || command_buffer->error != VK_SUCCESS)
        return;

    command_buffer->error = error;
    log_error("command buffer %llu %s failed", command_buffer ? (unsigned long long)command_buffer->id : 0, operation);
}

static void impl_cmd_bind_pipeline(VkCommandBuffer command_buffer, VkPipelineBindPoint pipeline_bind_point, VkPipeline pipeline)
{
    CommandBuffer* impl = unwrap(command_buffer);
    std::shared_ptr<ComputePipeline> pipeline_impl = get_handle(g_compute_pipelines, pipeline);
    if (!impl || impl->state != COMMAND_BUFFER_RECORDING || pipeline_bind_point != VK_PIPELINE_BIND_POINT_COMPUTE
            || !pipeline_impl || pipeline_impl->owner != impl->owner || !pipeline_impl->live)
    {
        set_command_error(impl, VK_ERROR_INITIALIZATION_FAILED, "bind pipeline");
        return;
    }

    impl->current_pipeline = pipeline_impl;
}

static int update_descriptor(CommandBuffer* command_buffer, const DescriptorSetLayout& set_layout,
                             uint32_t binding, VkDescriptorType descriptor_type,
                             const VkDescriptorBufferInfo* buffer_info)
{
    if (!command_buffer
            || (descriptor_type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                && descriptor_type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
            || !buffer_info)
        return -1;

    const DescriptorBinding* layout_binding = find_layout_binding(set_layout, binding);
    std::shared_ptr<Buffer> buffer = get_handle(g_buffers, buffer_info->buffer);
    if (!layout_binding || layout_binding->descriptor_type != descriptor_type
            || layout_binding->descriptor_count != 1
            || !buffer || buffer->owner != command_buffer->owner || !buffer->live
            || !buffer->memory || !buffer->memory->live
            || buffer_info->range == 0 || buffer_info->offset >= buffer->size)
        return -1;

    const bool uniform = descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    const VkBufferUsageFlags required_usage = uniform
                                               ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                               : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if ((buffer->usage & required_usage) == 0)
        return -1;

    const uint64_t range = buffer_info->range == VK_WHOLE_SIZE
                           ? buffer->size - buffer_info->offset : buffer_info->range;
    const uint64_t max_binding_size = uniform
                                      ? command_buffer->owner->limits.maxUniformBufferBindingSize
                                      : command_buffer->owner->limits.maxStorageBufferBindingSize;
    const uint64_t offset_alignment = uniform
                                      ? command_buffer->owner->limits.minUniformBufferOffsetAlignment
                                      : command_buffer->owner->limits.minStorageBufferOffsetAlignment;
    uint64_t end = 0;
    if (!checked_add(buffer_info->offset, range, end) || end > buffer->size
            || range > max_binding_size
            || buffer_info->offset % std::max<uint64_t>(4, offset_alignment) != 0)
        return -1;

    DescriptorValue value;
    value.descriptor_type = descriptor_type;
    value.buffer_info = *buffer_info;
    command_buffer->descriptors[binding] = value;
    return 0;
}

static void impl_cmd_push_descriptor_set_with_template(VkCommandBuffer command_buffer,
                                                        VkDescriptorUpdateTemplate descriptor_update_template,
                                                        VkPipelineLayout pipeline_layout, uint32_t set,
                                                        const void* data)
{
    CommandBuffer* impl = unwrap(command_buffer);
    std::shared_ptr<DescriptorUpdateTemplate> template_impl =
        get_handle(g_descriptor_update_templates, descriptor_update_template);
    std::shared_ptr<PipelineLayout> layout_impl = get_handle(g_pipeline_layouts, pipeline_layout);
    if (!impl || impl->state != COMMAND_BUFFER_RECORDING || !template_impl || !layout_impl
            || (!data && !template_impl->entries.empty())
            || template_impl->owner != impl->owner || layout_impl->owner != impl->owner
            || template_impl->pipeline_layout.get() != layout_impl.get()
            || template_impl->set != set)
    {
        set_command_error(impl, VK_ERROR_INITIALIZATION_FAILED, "push descriptor template");
        return;
    }

    if (!template_impl->entries.empty())
    {
        if (impl->descriptor_pipeline_layout.get() != layout_impl.get())
            impl->descriptors.clear();
        impl->descriptor_pipeline_layout = layout_impl;
    }

    const unsigned char* bytes = (const unsigned char*)data;
    for (size_t i = 0; i < template_impl->entries.size(); i++)
    {
        const DescriptorTemplateEntry& entry = template_impl->entries[i];
        VkDescriptorBufferInfo buffer_info;
        memcpy(&buffer_info, bytes + entry.offset, sizeof(buffer_info));
        if (update_descriptor(impl, *template_impl->set_layout,
                              entry.binding, entry.descriptor_type, &buffer_info) != 0)
        {
            set_command_error(impl, VK_ERROR_INITIALIZATION_FAILED, "push descriptor template value");
            return;
        }
    }
}

static void impl_cmd_push_descriptor_set(VkCommandBuffer command_buffer, VkPipelineBindPoint pipeline_bind_point,
                                         VkPipelineLayout pipeline_layout, uint32_t set, uint32_t descriptor_write_count,
                                         const VkWriteDescriptorSet* descriptor_writes)
{
    CommandBuffer* impl = unwrap(command_buffer);
    std::shared_ptr<PipelineLayout> layout_impl = get_handle(g_pipeline_layouts, pipeline_layout);
    if (!impl || impl->state != COMMAND_BUFFER_RECORDING || pipeline_bind_point != VK_PIPELINE_BIND_POINT_COMPUTE
            || !layout_impl || layout_impl->owner != impl->owner || set != 0
            || layout_impl->set_layouts.empty() || !layout_impl->set_layouts[0]->push_descriptor
            || (descriptor_write_count != 0 && !descriptor_writes))
    {
        set_command_error(impl, VK_ERROR_INITIALIZATION_FAILED, "push descriptor set");
        return;
    }

    if (descriptor_write_count != 0)
    {
        if (impl->descriptor_pipeline_layout.get() != layout_impl.get())
            impl->descriptors.clear();
        impl->descriptor_pipeline_layout = layout_impl;
    }

    for (uint32_t i = 0; i < descriptor_write_count; i++)
    {
        const VkWriteDescriptorSet& write = descriptor_writes[i];
        if (write.sType != VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET || write.pNext
                || write.descriptorCount != 1 || write.dstArrayElement != 0
                || update_descriptor(impl, *layout_impl->set_layouts[0],
                                     write.dstBinding, write.descriptorType, write.pBufferInfo) != 0)
        {
            set_command_error(impl, VK_ERROR_FEATURE_NOT_PRESENT, "push descriptor set value");
            return;
        }
    }
}

static bool pipeline_layout_supports_push_constant_range(const PipelineLayout& layout,
                                                         VkShaderStageFlags stage_flags,
                                                         uint32_t offset, uint32_t size)
{
    if (stage_flags != VK_SHADER_STAGE_COMPUTE_BIT || size == 0)
        return false;

    uint64_t end = 0;
    if (!checked_add(offset, size, end))
        return false;

    uint64_t covered = offset;
    while (covered < end)
    {
        uint64_t next = covered;
        for (size_t i = 0; i < layout.push_constant_ranges.size(); i++)
        {
            const PushConstantRange& range = layout.push_constant_ranges[i];
            const uint64_t range_end = (uint64_t)range.offset + range.size;
            if ((range.stage_flags & stage_flags) == stage_flags
                    && range.offset <= covered && range_end > next)
                next = range_end;
        }
        if (next == covered)
            return false;
        covered = std::min<uint64_t>(next, end);
    }

    return true;
}

static void impl_cmd_push_constants(VkCommandBuffer command_buffer, VkPipelineLayout pipeline_layout,
                                    VkShaderStageFlags stage_flags, uint32_t offset, uint32_t size, const void* values)
{
    CommandBuffer* impl = unwrap(command_buffer);
    std::shared_ptr<PipelineLayout> layout = get_handle(g_pipeline_layouts, pipeline_layout);
    uint64_t end = 0;
    if (!impl || impl->state != COMMAND_BUFFER_RECORDING || !layout || layout->owner != impl->owner
            || !values
            || offset % 4 != 0 || size % 4 != 0 || !checked_add(offset, size, end)
            || end > layout->push_constant_size
            || !pipeline_layout_supports_push_constant_range(*layout, stage_flags, offset, size))
    {
        set_command_error(impl, VK_ERROR_INITIALIZATION_FAILED, "push constants");
        return;
    }

    if (impl->push_constant_pipeline_layout.get() != layout.get())
        impl->push_constants.clear();
    impl->push_constant_pipeline_layout = layout;

    if (impl->push_constants.size() < layout->push_constant_size)
        impl->push_constants.resize(layout->push_constant_size, 0);
    memcpy(impl->push_constants.data() + offset, values, size);
}

static int resolve_dispatch_binding(CommandBuffer& command_buffer, const ActiveBinding& active,
                                    ResolvedBinding& resolved)
{
    std::unordered_map<uint32_t, DescriptorValue>::const_iterator it =
        command_buffer.descriptors.find(active.binding);
    if (it == command_buffer.descriptors.end())
        return -1;
    if (it->second.descriptor_type != active.descriptor_type)
        return -1;

    const VkDescriptorBufferInfo& info = it->second.buffer_info;
    std::shared_ptr<Buffer> buffer = get_handle(g_buffers, info.buffer);
    if (!buffer || buffer->owner != command_buffer.owner || !buffer->live
            || !buffer->memory || !buffer->memory->live)
        return -1;
    if (info.offset > buffer->size)
        return -1;

    const uint64_t size = info.range == VK_WHOLE_SIZE ? buffer->size - info.offset : info.range;
    uint64_t buffer_end = 0;
    uint64_t memory_offset = 0;
    uint64_t memory_end = 0;
    if (size < active.min_binding_size
            || !checked_add(info.offset, size, buffer_end) || buffer_end > buffer->size
            || !checked_add(buffer->memory_offset, info.offset, memory_offset)
            || !checked_add(memory_offset, size, memory_end) || memory_end > buffer->memory->size)
        return -1;

    resolved.binding = active.binding;
    resolved.descriptor_type = active.descriptor_type;
    resolved.access = active.access;
    resolved.layout_access = active.layout_access;
    resolved.min_binding_size = active.min_binding_size;
    resolved.buffer = buffer;
    resolved.memory = buffer->memory;
    resolved.offset = memory_offset;
    resolved.size = size;
    return 0;
}

static void impl_cmd_dispatch(VkCommandBuffer command_buffer, uint32_t group_count_x,
                              uint32_t group_count_y, uint32_t group_count_z)
{
    CommandBuffer* impl = unwrap(command_buffer);
    if (!impl || impl->state != COMMAND_BUFFER_RECORDING || !impl->current_pipeline)
    {
        set_command_error(impl, VK_ERROR_INITIALIZATION_FAILED, "dispatch pipeline");
        return;
    }

    const PipelineLayout& pipeline_layout = *impl->current_pipeline->vk_pipeline_layout;
    bool has_external_binding = false;
    for (size_t i = 0; i < impl->current_pipeline->active_bindings.size(); i++)
    {
        if (!impl->current_pipeline->active_bindings[i].internal_uniform)
        {
            has_external_binding = true;
            break;
        }
    }
    if (has_external_binding
            && impl->descriptor_pipeline_layout.get() != &pipeline_layout)
    {
        set_command_error(impl, VK_ERROR_INITIALIZATION_FAILED, "dispatch descriptor layout");
        return;
    }
    if ((impl->current_pipeline->immediate_size != 0 || impl->current_pipeline->push_constant_uniform)
            && impl->push_constant_pipeline_layout.get() != &pipeline_layout)
    {
        set_command_error(impl, VK_ERROR_INITIALIZATION_FAILED, "dispatch push constant layout");
        return;
    }

    const WGPULimits& limits = impl->owner->limits;
    if (group_count_x > limits.maxComputeWorkgroupsPerDimension
            || group_count_y > limits.maxComputeWorkgroupsPerDimension
            || group_count_z > limits.maxComputeWorkgroupsPerDimension)
    {
        impl->error = VK_ERROR_INITIALIZATION_FAILED;
        log_error("command buffer %llu pipeline %llu dispatch size failed groups=%u,%u,%u limit=%u local=%u,%u,%u",
                  (unsigned long long)impl->id,
                  (unsigned long long)impl->current_pipeline->id,
                  group_count_x, group_count_y, group_count_z,
                  limits.maxComputeWorkgroupsPerDimension,
                  impl->current_pipeline->workgroup_size_x,
                  impl->current_pipeline->workgroup_size_y,
                  impl->current_pipeline->workgroup_size_z);
        return;
    }

    Command command;
    command.type = COMMAND_DISPATCH;
    command.pipeline = impl->current_pipeline;
    command.group_count_x = group_count_x;
    command.group_count_y = group_count_y;
    command.group_count_z = group_count_z;

    for (size_t i = 0; i < impl->current_pipeline->active_bindings.size(); i++)
    {
        const ActiveBinding& active = impl->current_pipeline->active_bindings[i];
        if (active.internal_uniform)
            continue;

        ResolvedBinding binding;
        if (resolve_dispatch_binding(*impl, active, binding) != 0)
        {
            set_command_error(impl, VK_ERROR_INITIALIZATION_FAILED, "dispatch descriptor");
            return;
        }
        command.bindings.push_back(binding);
    }

    const uint32_t push_size = impl->current_pipeline->vk_pipeline_layout->push_constant_size;
    command.push_constants.resize(push_size, 0);
    if (!impl->push_constants.empty())
        memcpy(command.push_constants.data(), impl->push_constants.data(), std::min<size_t>(push_size, impl->push_constants.size()));

    impl->commands.push_back(command);
}

static void impl_cmd_copy_buffer(VkCommandBuffer command_buffer, VkBuffer src_buffer, VkBuffer dst_buffer,
                                 uint32_t region_count, const VkBufferCopy* regions)
{
    CommandBuffer* impl = unwrap(command_buffer);
    std::shared_ptr<Buffer> src = get_handle(g_buffers, src_buffer);
    std::shared_ptr<Buffer> dst = get_handle(g_buffers, dst_buffer);
    if (!impl || impl->state != COMMAND_BUFFER_RECORDING || !src || !dst
            || src->owner != impl->owner || dst->owner != impl->owner
            || !src->live || !dst->live
            || !src->memory || !dst->memory || !src->memory->live || !dst->memory->live
            || (src->usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0
            || (dst->usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) == 0
            || region_count == 0 || !regions)
    {
        set_command_error(impl, VK_ERROR_INITIALIZATION_FAILED, "copy buffer");
        return;
    }

    Command command;
    command.type = COMMAND_COPY_BUFFER;
    command.src_buffer = src;
    command.dst_buffer = dst;
    for (uint32_t i = 0; i < region_count; i++)
    {
        uint64_t src_end = 0;
        uint64_t dst_end = 0;
        if (regions[i].size == 0
                || !checked_add(regions[i].srcOffset, regions[i].size, src_end) || src_end > src->size
                || !checked_add(regions[i].dstOffset, regions[i].size, dst_end) || dst_end > dst->size
                || regions[i].srcOffset % 4 != 0 || regions[i].dstOffset % 4 != 0 || regions[i].size % 4 != 0)
        {
            set_command_error(impl, VK_ERROR_INITIALIZATION_FAILED, "copy buffer range");
            return;
        }

        CopyRegion region;
        region.src_offset = regions[i].srcOffset;
        region.dst_offset = regions[i].dstOffset;
        region.size = regions[i].size;
        command.copy_regions.push_back(region);
    }

    impl->commands.push_back(command);
}

static void impl_cmd_pipeline_barrier(VkCommandBuffer command_buffer,
                                      VkPipelineStageFlags src_stage_mask,
                                      VkPipelineStageFlags dst_stage_mask,
                                      VkDependencyFlags dependency_flags,
                                      uint32_t memory_barrier_count,
                                      const VkMemoryBarrier* memory_barriers, uint32_t buffer_memory_barrier_count,
                                      const VkBufferMemoryBarrier* buffer_memory_barriers,
                                      uint32_t image_memory_barrier_count, const VkImageMemoryBarrier*)
{
    CommandBuffer* impl = unwrap(command_buffer);
    const VkPipelineStageFlags supported_stages =
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
        | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        | VK_PIPELINE_STAGE_TRANSFER_BIT
        | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
        | VK_PIPELINE_STAGE_HOST_BIT
        | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    const VkAccessFlags supported_access =
        VK_ACCESS_UNIFORM_READ_BIT
        | VK_ACCESS_SHADER_READ_BIT
        | VK_ACCESS_SHADER_WRITE_BIT
        | VK_ACCESS_TRANSFER_READ_BIT
        | VK_ACCESS_TRANSFER_WRITE_BIT
        | VK_ACCESS_HOST_READ_BIT
        | VK_ACCESS_HOST_WRITE_BIT
        | VK_ACCESS_MEMORY_READ_BIT
        | VK_ACCESS_MEMORY_WRITE_BIT;
    if (!impl || impl->state != COMMAND_BUFFER_RECORDING
            || src_stage_mask == 0 || dst_stage_mask == 0
            || (src_stage_mask & ~supported_stages) != 0
            || (dst_stage_mask & ~supported_stages) != 0
            || dependency_flags != 0
            || (memory_barrier_count != 0 && !memory_barriers)
            || (buffer_memory_barrier_count != 0 && !buffer_memory_barriers)
            || image_memory_barrier_count != 0)
    {
        set_command_error(impl, VK_ERROR_FEATURE_NOT_PRESENT, "pipeline barrier");
        return;
    }

    Command command;
    command.type = COMMAND_BARRIER;
    for (uint32_t i = 0; i < memory_barrier_count; i++)
    {
        if (memory_barriers[i].sType != VK_STRUCTURE_TYPE_MEMORY_BARRIER
                || memory_barriers[i].pNext
                || (memory_barriers[i].srcAccessMask & ~supported_access) != 0
                || (memory_barriers[i].dstAccessMask & ~supported_access) != 0)
        {
            set_command_error(impl, VK_ERROR_FEATURE_NOT_PRESENT, "memory barrier");
            return;
        }
    }
    for (uint32_t i = 0; i < buffer_memory_barrier_count; i++)
    {
        const VkBufferMemoryBarrier& barrier = buffer_memory_barriers[i];
        std::shared_ptr<Buffer> buffer = get_handle(g_buffers, barrier.buffer);
        if (barrier.sType != VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER
                || barrier.pNext
                || !buffer || buffer->owner != impl->owner || !buffer->live
                || !buffer->memory || !buffer->memory->live
                || (barrier.srcAccessMask & ~supported_access) != 0
                || (barrier.dstAccessMask & ~supported_access) != 0
                || barrier.srcQueueFamilyIndex != barrier.dstQueueFamilyIndex
                || (barrier.srcQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED
                    && barrier.srcQueueFamilyIndex != 0))
        {
            set_command_error(impl, VK_ERROR_FEATURE_NOT_PRESENT, "buffer barrier");
            return;
        }

        const uint64_t size =
            barrier.size == VK_WHOLE_SIZE
            ? buffer->size - std::min<uint64_t>(barrier.offset, buffer->size)
            : barrier.size;
        uint64_t end = 0;
        if (size == 0 || !checked_add(barrier.offset, size, end) || end > buffer->size)
        {
            set_command_error(impl, VK_ERROR_INITIALIZATION_FAILED, "buffer barrier range");
            return;
        }
        command.barrier_buffers.push_back(buffer);
    }
    impl->commands.push_back(command);
}

// command replay, queue and synchronization

static bool ranges_overlap(uint64_t a_offset, uint64_t a_size, uint64_t b_offset, uint64_t b_size)
{
    uint64_t a_end = 0;
    uint64_t b_end = 0;
    return checked_add(a_offset, a_size, a_end) && checked_add(b_offset, b_size, b_end)
           && a_offset < b_end && b_offset < a_end;
}

template<typename T>
static void append_unique(std::vector<std::shared_ptr<T> >& values, const std::shared_ptr<T>& value)
{
    if (!value)
        return;

    for (size_t i = 0; i < values.size(); i++)
    {
        if (values[i].get() == value.get())
            return;
    }
    values.push_back(value);
}

struct ReplayState
{
    ReplayState()
        : encoder(0),
          compute_pass(0),
          push_uniform_buffer(0),
          push_uniform_offset(0),
          push_uniform_size(0),
          push_uniform_alignment(4)
    {
    }

    WGPUCommandEncoder encoder;
    WGPUComputePassEncoder compute_pass;
    std::unordered_map<uint64_t, BufferAccess> compute_pass_buffer_access;
    WGPUBuffer push_uniform_buffer;
    uint64_t push_uniform_offset;
    uint64_t push_uniform_size;
    uint64_t push_uniform_alignment;
};

static void end_compute_pass(ReplayState& replay)
{
    if (!replay.compute_pass)
        return;

    wgpuComputePassEncoderEnd(replay.compute_pass);
    wgpuComputePassEncoderRelease(replay.compute_pass);
    replay.compute_pass = 0;
    replay.compute_pass_buffer_access.clear();
}

static bool compute_pass_compatible(const ReplayState& replay,
                                    const std::vector<WGPUBuffer>& buffers,
                                    const std::vector<ResolvedBinding>& bindings)
{
    for (size_t i = 0; i < buffers.size(); i++)
    {
        const uint64_t buffer_id = handle_to_id(buffers[i]);
        std::unordered_map<uint64_t, BufferAccess>::const_iterator it =
            replay.compute_pass_buffer_access.find(buffer_id);
        if (it != replay.compute_pass_buffer_access.end() && it->second != bindings[i].layout_access)
            return false;
    }

    return true;
}

static int begin_compute_pass(ReplayState& replay)
{
    if (replay.compute_pass)
        return 0;

    WGPUComputePassDescriptor descriptor = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    replay.compute_pass = wgpuCommandEncoderBeginComputePass(replay.encoder, &descriptor);
    return replay.compute_pass ? 0 : -1;
}

static int encode_dispatch(Device* device, ReplayState& replay, const Command& command,
                           const std::shared_ptr<Submission>& submission)
{
    std::vector<ResolvedBinding> bindings = command.bindings;
    enum SnapshotMode
    {
        SNAPSHOT_NONE,
        SNAPSHOT_READ
    };
    std::vector<SnapshotMode> snapshots(bindings.size(), SNAPSHOT_NONE);
    for (size_t i = 0; i < bindings.size(); i++)
    {
        for (size_t j = i + 1; j < bindings.size(); j++)
        {
            if (bindings[i].memory.get() != bindings[j].memory.get())
                continue;

            const bool layout_write_i = bindings[i].layout_access == BUFFER_ACCESS_READ_WRITE;
            const bool layout_write_j = bindings[j].layout_access == BUFFER_ACCESS_READ_WRITE;
            if (!layout_write_i && !layout_write_j)
                continue;

            const bool overlap = ranges_overlap(bindings[i].offset, bindings[i].size,
                                                bindings[j].offset, bindings[j].size);
            const bool actual_write_i = bindings[i].access == BUFFER_ACCESS_READ_WRITE;
            const bool actual_write_j = bindings[j].access == BUFFER_ACCESS_READ_WRITE;
            if (actual_write_i && actual_write_j)
            {
                if (!overlap)
                    continue;

                log_error("dispatch writable alias binding=%u and binding=%u memory=%llu",
                          bindings[i].binding, bindings[j].binding,
                          (unsigned long long)bindings[i].memory->id);
                return -1;
            }

            if (layout_write_i != layout_write_j)
            {
                snapshots[layout_write_i ? j : i] = SNAPSHOT_READ;
                continue;
            }

            if (!overlap)
                continue;

            snapshots[actual_write_i ? j : i] = SNAPSHOT_READ;
        }
    }

    bool snapshot_required = false;
    for (size_t i = 0; i < snapshots.size(); i++)
    {
        if (snapshots[i] == SNAPSHOT_READ)
        {
            snapshot_required = true;
            break;
        }
    }
    if (snapshot_required)
        end_compute_pass(replay);

    std::vector<WGPUBuffer> bind_buffers(bindings.size(), 0);
    std::vector<uint64_t> bind_offsets(bindings.size(), 0);
    for (size_t i = 0; i < bindings.size(); i++)
    {
        if (snapshots[i] == SNAPSHOT_NONE)
        {
            bind_buffers[i] = bindings[i].memory->buffer;
            bind_offsets[i] = bindings[i].offset;
            continue;
        }

        const uint64_t copy_size = align_up(bindings[i].size, 4);
        if (copy_size == 0 || copy_size > bindings[i].memory->padded_size
                || bindings[i].offset > bindings[i].memory->padded_size - copy_size)
            return -1;

        WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
        descriptor.size = copy_size;
        descriptor.usage = WGPUBufferUsage_CopyDst
                           | (bindings[i].descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                              ? WGPUBufferUsage_Uniform : WGPUBufferUsage_Storage);
        WGPUBuffer temporary = wgpuDeviceCreateBuffer(device->device, &descriptor);
        if (!temporary)
            return -1;

        wgpuCommandEncoderCopyBufferToBuffer(replay.encoder, bindings[i].memory->buffer, bindings[i].offset,
                                             temporary, 0, copy_size);
        submission->temporary_buffers.push_back(temporary);
        device->alias_snapshot_count.fetch_add(1, std::memory_order_relaxed);
        device->alias_snapshot_bytes.fetch_add(copy_size, std::memory_order_relaxed);
        bind_buffers[i] = temporary;
    }

    std::vector<WGPUBindGroupEntry> entries(bindings.size());
    for (size_t i = 0; i < bindings.size(); i++)
    {
        entries[i] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[i].binding = bindings[i].binding;
        entries[i].buffer = bind_buffers[i];
        entries[i].offset = bind_offsets[i];
        entries[i].size = bindings[i].size;
    }

    WGPUBindGroupDescriptor bind_group_descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bind_group_descriptor.layout = command.pipeline->bind_group_layout;
    bind_group_descriptor.entryCount = entries.size();
    bind_group_descriptor.entries = entries.empty() ? 0 : entries.data();
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device->device, &bind_group_descriptor);
    if (!bind_group)
        return -1;
    submission->bind_groups.push_back(bind_group);
    device->bind_group_count.fetch_add(1, std::memory_order_relaxed);

    WGPUBindGroup push_bind_group = 0;
    if (command.pipeline->push_constant_uniform)
    {
        const uint64_t size = align_up(command.pipeline->push_constant_uniform_size, 4);
        const uint64_t offset = align_up(replay.push_uniform_offset, replay.push_uniform_alignment);
        uint64_t end = 0;
        if (!replay.push_uniform_buffer
                || size == 0
                || command.push_constants.size() < command.pipeline->push_constant_data_size
                || !checked_add(offset, size, end)
                || end > replay.push_uniform_size)
            return -1;

        WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
        entry.binding = 0;
        entry.buffer = replay.push_uniform_buffer;
        entry.offset = offset;
        entry.size = command.pipeline->push_constant_uniform_size;
        WGPUBindGroupDescriptor descriptor_group = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        descriptor_group.layout = command.pipeline->immediate_bind_group_layout;
        descriptor_group.entryCount = 1;
        descriptor_group.entries = &entry;
        push_bind_group = wgpuDeviceCreateBindGroup(device->device, &descriptor_group);
        if (!push_bind_group)
            return -1;
        submission->bind_groups.push_back(push_bind_group);
        device->bind_group_count.fetch_add(1, std::memory_order_relaxed);
        replay.push_uniform_offset = end;
    }

    if (replay.compute_pass && !compute_pass_compatible(replay, bind_buffers, bindings))
        end_compute_pass(replay);
    if (begin_compute_pass(replay) != 0)
        return -1;

    wgpuComputePassEncoderSetPipeline(replay.compute_pass, command.pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(replay.compute_pass, 0, bind_group, 0, 0);
    if (push_bind_group)
        wgpuComputePassEncoderSetBindGroup(replay.compute_pass, 1, push_bind_group, 0, 0);
    if (command.pipeline->immediate_size != 0)
    {
        if (command.push_constants.size() < command.pipeline->immediate_size)
            return -1;
        wgpuComputePassEncoderSetImmediates(replay.compute_pass, 0, command.push_constants.data(), command.pipeline->immediate_size);
    }
    wgpuComputePassEncoderDispatchWorkgroups(replay.compute_pass, command.group_count_x, command.group_count_y, command.group_count_z);

    for (size_t i = 0; i < bind_buffers.size(); i++)
        replay.compute_pass_buffer_access[handle_to_id(bind_buffers[i])] = bindings[i].layout_access;

    append_unique(submission->pipelines, command.pipeline);
    for (size_t i = 0; i < command.bindings.size(); i++)
    {
        append_unique(submission->buffers, command.bindings[i].buffer);
        append_unique(submission->memories, command.bindings[i].memory);
    }
    return 0;
}

static int encode_command_buffer(Device* device, ReplayState& replay,
                                 const std::shared_ptr<CommandBuffer>& command_buffer,
                                 const std::shared_ptr<Submission>& submission,
                                 std::vector<MemoryRange>& readback_ranges)
{
    for (size_t i = 0; i < command_buffer->commands.size(); i++)
    {
        const Command& command = command_buffer->commands[i];
        if (command.type == COMMAND_DISPATCH)
        {
            if (encode_dispatch(device, replay, command, submission) != 0)
                return -1;

            for (size_t j = 0; j < command.bindings.size(); j++)
            {
                if (command.bindings[j].access == BUFFER_ACCESS_READ_WRITE
                        && !command.bindings[j].memory->host_shadow.empty()
                        && !append_memory_transfer_range(readback_ranges,
                                                         command.bindings[j].memory,
                                                         command.bindings[j].offset,
                                                         command.bindings[j].size))
                    return -1;
            }
        }
        else if (command.type == COMMAND_COPY_BUFFER)
        {
            end_compute_pass(replay);
            for (size_t j = 0; j < command.copy_regions.size(); j++)
            {
                const CopyRegion& region = command.copy_regions[j];
                const uint64_t src_offset = command.src_buffer->memory_offset + region.src_offset;
                const uint64_t dst_offset = command.dst_buffer->memory_offset + region.dst_offset;
                if (command.src_buffer->memory.get() == command.dst_buffer->memory.get())
                {
                    if (src_offset == dst_offset)
                        continue;

                    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
                    descriptor.size = align_up(region.size, 4);
                    descriptor.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
                    WGPUBuffer temporary = wgpuDeviceCreateBuffer(device->device, &descriptor);
                    if (!temporary)
                        return -1;
                    submission->temporary_buffers.push_back(temporary);

                    wgpuCommandEncoderCopyBufferToBuffer(
                        replay.encoder, command.src_buffer->memory->buffer, src_offset, temporary, 0, region.size);
                    wgpuCommandEncoderCopyBufferToBuffer(
                        replay.encoder, temporary, 0, command.dst_buffer->memory->buffer, dst_offset, region.size);
                }
                else
                {
                    wgpuCommandEncoderCopyBufferToBuffer(
                        replay.encoder,
                        command.src_buffer->memory->buffer, src_offset,
                        command.dst_buffer->memory->buffer, dst_offset,
                        region.size);
                }
            }

            append_unique(submission->buffers, command.src_buffer);
            append_unique(submission->buffers, command.dst_buffer);
            append_unique(submission->memories, command.src_buffer->memory);
            append_unique(submission->memories, command.dst_buffer->memory);
            if (!command.dst_buffer->memory->host_shadow.empty())
            {
                for (size_t j = 0; j < command.copy_regions.size(); j++)
                {
                    uint64_t dst_offset = 0;
                    if (!checked_add(command.dst_buffer->memory_offset,
                                     command.copy_regions[j].dst_offset, dst_offset)
                            || !append_memory_transfer_range(readback_ranges,
                                                             command.dst_buffer->memory,
                                                             dst_offset,
                                                             command.copy_regions[j].size))
                        return -1;
                }
            }
        }
        else if (command.type == COMMAND_BARRIER)
        {
            end_compute_pass(replay);
            for (size_t j = 0; j < command.barrier_buffers.size(); j++)
            {
                append_unique(submission->buffers, command.barrier_buffers[j]);
                append_unique(submission->memories, command.barrier_buffers[j]->memory);
            }
        }
    }

    append_unique(submission->command_buffers, command_buffer);
    return 0;
}

static void queue_work_done_callback(WGPUQueueWorkDoneStatus status, WGPUStringView message, void* userdata1, void*)
{
    std::shared_ptr<Submission>* context = (std::shared_ptr<Submission>*)userdata1;
    if (!context)
        return;
    std::shared_ptr<Submission> submission = *context;
    delete context;
    if (!submission)
        return;

    submission->status = status;
    submission->completed = true;
    if (status != WGPUQueueWorkDoneStatus_Success)
        log_error("submission %llu failed status=%d: %.*s", (unsigned long long)submission->id,
                  (int)status, (int)message.length, message.data ? message.data : "");
}

static void map_callback(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void*)
{
    std::shared_ptr<MapResult>* context = (std::shared_ptr<MapResult>*)userdata1;
    if (!context)
        return;
    std::shared_ptr<MapResult> result = *context;
    delete context;
    if (!result)
        return;

    result->status = status;
    result->completed = true;
    if (status != WGPUMapAsyncStatus_Success)
        log_error("readback map failed status=%d: %.*s", (int)status,
                  (int)message.length, message.data ? message.data : "");
}

static VkResult finish_submission(const std::shared_ptr<Submission>& submission, VkResult result)
{
    if (!submission)
        return result;
    if (submission->processed)
        return submission->result;

    for (size_t i = 0; i < submission->command_buffers.size(); i++)
    {
        CommandBuffer& command_buffer = *submission->command_buffers[i];
        if (command_buffer.pending_count != 0)
            command_buffer.pending_count--;
        if (command_buffer.pending_count == 0)
        {
            command_buffer.state =
                (command_buffer.usage_flags & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
                ? COMMAND_BUFFER_INVALID : COMMAND_BUFFER_EXECUTABLE;
        }
    }

    submission->result = result;
    submission->processed = true;

    std::shared_ptr<Fence> fence = submission->fence.lock();
    if (fence)
    {
        fence->signaled = result == VK_SUCCESS;
        fence->failed = result != VK_SUCCESS;
    }

    submission->release_resources();
    if (submission->owner)
    {
        std::lock_guard<std::mutex> lock(submission->owner->mutex);
        std::vector<std::shared_ptr<Submission> >& submissions = submission->owner->submissions;
        submissions.erase(
            std::remove_if(
                submissions.begin(), submissions.end(),
                [&](const std::shared_ptr<Submission>& value) {
                    return value.get() == submission.get();
                }),
            submissions.end());
    }

    return result;
}

static VkResult abandon_submission(const std::shared_ptr<Submission>& submission, VkResult result)
{
    if (!submission)
        return result;
    if (submission->processed)
        return submission->result;

    for (size_t i = 0; i < submission->command_buffers.size(); i++)
    {
        CommandBuffer& command_buffer = *submission->command_buffers[i];
        if (command_buffer.pending_count != 0)
            command_buffer.pending_count--;
        command_buffer.state = COMMAND_BUFFER_INVALID;
        command_buffer.error = result;
    }

    submission->result = result;
    submission->processed = true;

    std::shared_ptr<Fence> fence = submission->fence.lock();
    if (fence)
    {
        fence->signaled = false;
        fence->failed = true;
    }

    if (submission->owner)
    {
        std::lock_guard<std::mutex> lock(submission->owner->mutex);
        std::vector<std::shared_ptr<Submission> >& submissions = submission->owner->submissions;
        submissions.erase(
            std::remove_if(
                submissions.begin(), submissions.end(),
                [&](const std::shared_ptr<Submission>& value) {
                    return value.get() == submission.get();
                }),
            submissions.end());

        // A failed WaitAny does not prove that WebGPU has stopped using the
        // recorded resources. Keep them alive until the device is destroyed.
        submission->owner->abandoned_submissions.push_back(submission);
    }

    return result;
}

static VkResult process_submission(const std::shared_ptr<Submission>& submission, uint64_t timeout)
{
    if (!submission)
        return VK_ERROR_DEVICE_LOST;
    if (submission->processed)
        return submission->result;

    const uint64_t wait_begin = monotonic_time_ns();
    if (!submission->completed)
    {
        const int wait_result = wait_future(
            submission->owner->instance.get(), submission->future, timeout, "queue-completion");
        if (wait_result == 1)
            return VK_TIMEOUT;
        if (wait_result != 0)
        {
            submission->owner->lost = true;
            return abandon_submission(submission, VK_ERROR_DEVICE_LOST);
        }
    }
    if (submission->status != WGPUQueueWorkDoneStatus_Success)
    {
        submission->owner->lost = true;
        return finish_submission(submission, VK_ERROR_DEVICE_LOST);
    }

    for (size_t i = 0; i < submission->readbacks.size(); i++)
    {
        Readback& readback = submission->readbacks[i];
        if (readback.copied)
            continue;

        if (!readback.map_started)
        {
            readback.map_result = std::make_shared<MapResult>();
            std::shared_ptr<MapResult>* callback_context =
                new std::shared_ptr<MapResult>(readback.map_result);
            WGPUBufferMapCallbackInfo callback_info = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
            callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
            callback_info.callback = map_callback;
            callback_info.userdata1 = callback_context;
            readback.future =
                wgpuBufferMapAsync(readback.buffer, WGPUMapMode_Read, 0, readback.size, callback_info);
            if (readback.future.id == 0)
            {
                delete callback_context;
                submission->owner->error = true;
                return finish_submission(submission, VK_ERROR_MEMORY_MAP_FAILED);
            }
            readback.map_started = true;
        }

        if (!readback.map_result->completed)
        {
            const uint64_t remaining = remaining_timeout_ns(wait_begin, timeout);
            const int wait_result =
                wait_future(submission->owner->instance.get(), readback.future, remaining, "readback-map");
            if (wait_result == 1)
                return VK_TIMEOUT;
            if (wait_result != 0)
            {
                submission->owner->error = true;
                return finish_submission(submission, VK_ERROR_MEMORY_MAP_FAILED);
            }
        }
        if (!readback.map_result->completed
                || readback.map_result->status != WGPUMapAsyncStatus_Success)
        {
            submission->owner->error = true;
            return finish_submission(submission, VK_ERROR_MEMORY_MAP_FAILED);
        }

        const void* mapped = wgpuBufferGetConstMappedRange(readback.buffer, 0, readback.size);
        if (!mapped)
        {
            wgpuBufferUnmap(readback.buffer);
            submission->owner->error = true;
            return finish_submission(submission, VK_ERROR_MEMORY_MAP_FAILED);
        }
        memcpy(readback.memory->host_shadow.data() + readback.offset, mapped, readback.size);
        subtract_byte_range(readback.memory->dirty_ranges, readback.offset, readback.size);
        wgpuBufferUnmap(readback.buffer);
        readback.copied = true;
    }

    return finish_submission(submission, VK_SUCCESS);
}

static VkResult reap_device_submissions(Device* device)
{
    if (!device)
        return VK_ERROR_DEVICE_LOST;

    std::vector<std::shared_ptr<Submission> > submissions;
    {
        std::lock_guard<std::mutex> lock(device->mutex);
        submissions = device->submissions;
    }

    for (size_t i = 0; i < submissions.size(); i++)
    {
        const VkResult result = process_submission(submissions[i], 0);
        if (result != VK_SUCCESS && result != VK_TIMEOUT)
            return result;
    }

    return VK_SUCCESS;
}

static bool command_resources_live(const Command& command)
{
    if (command.type == COMMAND_DISPATCH)
    {
        if (!command.pipeline || !command.pipeline->live)
            return false;
        for (size_t i = 0; i < command.bindings.size(); i++)
        {
            if (!command.bindings[i].buffer || !command.bindings[i].buffer->live
                    || !command.bindings[i].memory || !command.bindings[i].memory->live)
                return false;
        }
        return true;
    }
    if (command.type == COMMAND_COPY_BUFFER)
    {
        return command.src_buffer && command.src_buffer->live
               && command.src_buffer->memory && command.src_buffer->memory->live
               && command.dst_buffer && command.dst_buffer->live
               && command.dst_buffer->memory && command.dst_buffer->memory->live;
    }
    if (command.type == COMMAND_BARRIER)
    {
        for (size_t i = 0; i < command.barrier_buffers.size(); i++)
        {
            if (!command.barrier_buffers[i] || !command.barrier_buffers[i]->live
                    || !command.barrier_buffers[i]->memory || !command.barrier_buffers[i]->memory->live)
                return false;
        }
        return true;
    }

    return false;
}

static VkResult impl_queue_submit(VkQueue queue, uint32_t submit_count, const VkSubmitInfo* submits, VkFence fence)
{
    Device* device = unwrap(queue);
    if (!device || (submit_count != 0 && !submits))
        return VK_ERROR_DEVICE_LOST;
    if (device->lost || device->error)
        return VK_ERROR_DEVICE_LOST;

    const VkResult reap_result = reap_device_submissions(device);
    if (reap_result != VK_SUCCESS)
        return reap_result;
    if (device->lost || device->error)
        return VK_ERROR_DEVICE_LOST;

    const uint64_t submit_begin = monotonic_time_ns();

    std::shared_ptr<Fence> fence_impl;
    if (fence != VK_NULL_HANDLE)
    {
        fence_impl = get_handle(g_fences, fence);
        if (!fence_impl || fence_impl->owner != device || fence_impl->signaled || fence_impl->submission)
            return VK_ERROR_INITIALIZATION_FAILED;
    }

    std::shared_ptr<Submission> submission = std::make_shared<Submission>();
    submission->owner = device;
    std::vector<MemoryRange> readback_ranges;
    std::vector<MemoryRange> upload_ranges;
    std::vector<std::shared_ptr<CommandBuffer> > command_buffers;
    std::unordered_set<CommandBuffer*> unique_command_buffers;
    std::vector<unsigned char> push_uniform_data;
    const uint64_t push_uniform_alignment =
        std::max<uint64_t>(4, device->limits.minUniformBufferOffsetAlignment);
    for (uint32_t i = 0; i < submit_count; i++)
    {
        if (submits[i].sType != VK_STRUCTURE_TYPE_SUBMIT_INFO || submits[i].pNext
                || submits[i].waitSemaphoreCount != 0 || submits[i].signalSemaphoreCount != 0
                || (submits[i].commandBufferCount != 0 && !submits[i].pCommandBuffers))
            return VK_ERROR_FEATURE_NOT_PRESENT;

        for (uint32_t j = 0; j < submits[i].commandBufferCount; j++)
        {
            VkCommandBuffer handle = submits[i].pCommandBuffers[j];
            CommandBuffer* command_buffer = unwrap(handle);
            if (!command_buffer || command_buffer->owner != device
                    || command_buffer->state != COMMAND_BUFFER_EXECUTABLE || command_buffer->error != VK_SUCCESS)
                return VK_ERROR_INITIALIZATION_FAILED;

            const bool first_use = unique_command_buffers.insert(command_buffer).second;
            if (!first_use || command_buffer->pending_count != 0)
                return VK_ERROR_INITIALIZATION_FAILED;
            for (size_t k = 0; k < command_buffer->commands.size(); k++)
            {
                if (!command_resources_live(command_buffer->commands[k]))
                {
                    log_error("device %llu command buffer %llu submit uses destroyed resource at command %llu",
                              (unsigned long long)device->id,
                              (unsigned long long)command_buffer->id,
                              (unsigned long long)k);
                    return VK_ERROR_INITIALIZATION_FAILED;
                }
            }
            command_buffers.push_back(handle->impl);
        }
    }

    for (size_t i = 0; i < command_buffers.size(); i++)
    {
        const std::vector<Command>& commands = command_buffers[i]->commands;
        for (size_t j = 0; j < commands.size(); j++)
        {
            if (commands[j].type == COMMAND_DISPATCH)
            {
                for (size_t k = 0; k < commands[j].bindings.size(); k++)
                {
                    append_unique(submission->memories, commands[j].bindings[k].memory);
                    if (!append_host_upload_ranges(upload_ranges,
                                                   commands[j].bindings[k].memory,
                                                   commands[j].bindings[k].offset,
                                                   commands[j].bindings[k].size))
                        return VK_ERROR_INITIALIZATION_FAILED;
                }

                if (commands[j].pipeline->push_constant_uniform)
                {
                    const uint64_t offset = align_up(push_uniform_data.size(), push_uniform_alignment);
                    const uint64_t size = align_up(commands[j].pipeline->push_constant_uniform_size, 4);
                    uint64_t end = 0;
                    if (offset == 0 && !push_uniform_data.empty())
                        return VK_ERROR_OUT_OF_HOST_MEMORY;
                    if (size == 0
                            || commands[j].push_constants.size() < commands[j].pipeline->push_constant_data_size
                            || !checked_add(offset, size, end)
                            || end > device->limits.maxBufferSize
                            || end > std::numeric_limits<size_t>::max())
                        return VK_ERROR_OUT_OF_HOST_MEMORY;

                    push_uniform_data.resize((size_t)end, 0);
                    memcpy(push_uniform_data.data() + offset,
                           commands[j].push_constants.data(),
                           commands[j].pipeline->push_constant_data_size);
                }
            }
            else if (commands[j].type == COMMAND_COPY_BUFFER)
            {
                append_unique(submission->memories, commands[j].src_buffer->memory);
                append_unique(submission->memories, commands[j].dst_buffer->memory);
                for (size_t k = 0; k < commands[j].copy_regions.size(); k++)
                {
                    uint64_t src_offset = 0;
                    if (!checked_add(commands[j].src_buffer->memory_offset,
                                     commands[j].copy_regions[k].src_offset, src_offset)
                            || !append_host_upload_ranges(upload_ranges,
                                                          commands[j].src_buffer->memory,
                                                          src_offset,
                                                          commands[j].copy_regions[k].size))
                        return VK_ERROR_INITIALIZATION_FAILED;
                }
            }
        }
    }

    for (size_t i = 0; i < upload_ranges.size(); i++)
    {
        MemoryRange& range = upload_ranges[i];
        wgpuQueueWriteBuffer(device->queue, range.memory->buffer, range.offset,
                             range.memory->host_shadow.data() + range.offset, range.size);
        subtract_byte_range(range.memory->dirty_ranges, range.offset, range.size);
        device->host_upload_bytes.fetch_add(range.size, std::memory_order_relaxed);
    }

    WGPUBuffer push_uniform_buffer = 0;
    if (!push_uniform_data.empty())
    {
        WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
        descriptor.size = align_up(push_uniform_data.size(), 4);
        descriptor.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        push_uniform_buffer = wgpuDeviceCreateBuffer(device->device, &descriptor);
        if (!push_uniform_buffer)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;

        submission->temporary_buffers.push_back(push_uniform_buffer);
        wgpuQueueWriteBuffer(device->queue, push_uniform_buffer, 0,
                             push_uniform_data.data(), push_uniform_data.size());
        device->push_uniform_bytes.fetch_add(push_uniform_data.size(), std::memory_order_relaxed);
    }

    wgpuDevicePushErrorScope(device->device, WGPUErrorFilter_Validation);
    WGPUCommandEncoderDescriptor encoder_descriptor = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device->device, &encoder_descriptor);
    if (!encoder)
    {
        pop_error_scope(device, "command encoder");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    ReplayState replay;
    replay.encoder = encoder;
    replay.push_uniform_buffer = push_uniform_buffer;
    replay.push_uniform_size = push_uniform_data.size();
    replay.push_uniform_alignment = push_uniform_alignment;
    for (size_t i = 0; i < command_buffers.size(); i++)
    {
        if (encode_command_buffer(device, replay, command_buffers[i], submission, readback_ranges) != 0)
        {
            end_compute_pass(replay);
            wgpuCommandEncoderRelease(encoder);
            pop_error_scope(device, "command replay");
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }
    end_compute_pass(replay);

    for (size_t i = 0; i < readback_ranges.size(); i++)
    {
        MemoryRange& range = readback_ranges[i];
        WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
        descriptor.size = range.size;
        descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        WGPUBuffer readback_buffer = wgpuDeviceCreateBuffer(device->device, &descriptor);
        if (!readback_buffer)
        {
            wgpuCommandEncoderRelease(encoder);
            pop_error_scope(device, "readback allocation");
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        wgpuCommandEncoderCopyBufferToBuffer(encoder, range.memory->buffer, range.offset,
                                             readback_buffer, 0, range.size);
        Readback readback;
        readback.buffer = readback_buffer;
        readback.memory = range.memory;
        readback.offset = range.offset;
        readback.size = range.size;
        submission->readbacks.push_back(readback);
        device->host_readback_count.fetch_add(1, std::memory_order_relaxed);
        device->host_readback_bytes.fetch_add(range.size, std::memory_order_relaxed);
    }

    WGPUCommandBufferDescriptor command_buffer_descriptor = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer webgpu_command_buffer = wgpuCommandEncoderFinish(encoder, &command_buffer_descriptor);
    wgpuCommandEncoderRelease(encoder);
    if (!webgpu_command_buffer || pop_error_scope(device, "command encoding") != 0)
    {
        if (webgpu_command_buffer)
            wgpuCommandBufferRelease(webgpu_command_buffer);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    wgpuQueueSubmit(device->queue, 1, &webgpu_command_buffer);
    wgpuCommandBufferRelease(webgpu_command_buffer);

    WGPUQueueWorkDoneCallbackInfo callback_info = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
    callback_info.callback = queue_work_done_callback;
    std::shared_ptr<Submission>* callback_context = new std::shared_ptr<Submission>(submission);
    callback_info.userdata1 = callback_context;
    submission->future = wgpuQueueOnSubmittedWorkDone(device->queue, callback_info);
    if (submission->future.id == 0)
    {
        delete callback_context;
        device->lost = true;
        for (size_t i = 0; i < submission->command_buffers.size(); i++)
        {
            submission->command_buffers[i]->state = COMMAND_BUFFER_INVALID;
            submission->command_buffers[i]->error = VK_ERROR_DEVICE_LOST;
        }
        if (fence_impl)
        {
            fence_impl->signaled = false;
            fence_impl->failed = true;
        }
        {
            // Queue work has already been submitted, but without a completion
            // future there is no safe point at which its resources can be
            // released. Keep them alive until wgpuDeviceDestroy().
            std::lock_guard<std::mutex> lock(device->mutex);
            device->abandoned_submissions.push_back(submission);
        }
        log_error("device %llu submission %llu returned an invalid completion future",
                  (unsigned long long)device->id, (unsigned long long)submission->id);
        return VK_ERROR_DEVICE_LOST;
    }

    for (size_t i = 0; i < submission->command_buffers.size(); i++)
    {
        CommandBuffer& command_buffer = *submission->command_buffers[i];
        command_buffer.pending_count++;
        command_buffer.state = COMMAND_BUFFER_PENDING;
    }

    if (fence_impl)
    {
        submission->fence = fence_impl;
        fence_impl->submission = submission;
        fence_impl->signaled = false;
        fence_impl->failed = false;
    }
    {
        std::lock_guard<std::mutex> lock(device->mutex);
        device->submissions.push_back(submission);
    }

    device->submit_count.fetch_add(1, std::memory_order_relaxed);
    device->submit_cpu_ns.fetch_add(monotonic_time_ns() - submit_begin, std::memory_order_relaxed);
    return VK_SUCCESS;
}

static VkResult wait_device_submissions(Device* device)
{
    if (!device)
        return VK_ERROR_DEVICE_LOST;

    std::vector<std::shared_ptr<Submission> > submissions;
    {
        std::lock_guard<std::mutex> lock(device->mutex);
        submissions = device->submissions;
    }

    VkResult first_error = device->lost || device->error ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
    for (size_t i = 0; i < submissions.size(); i++)
    {
        const VkResult result = process_submission(submissions[i], UINT64_MAX);
        if (result != VK_SUCCESS && first_error == VK_SUCCESS)
            first_error = result;
    }
    return first_error;
}

static VkResult impl_queue_wait_idle(VkQueue queue)
{
    return wait_device_submissions(unwrap(queue));
}

static VkResult impl_device_wait_idle(VkDevice device)
{
    return wait_device_submissions(unwrap(device));
}

static VkResult impl_create_fence(VkDevice device, const VkFenceCreateInfo* create_info,
                                  const VkAllocationCallbacks* allocator, VkFence* fence)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_FENCE_CREATE_INFO || !fence)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || (create_info->flags & ~VK_FENCE_CREATE_SIGNALED_BIT) != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<Fence> impl = std::make_shared<Fence>();
    impl->owner = device_impl;
    impl->signaled = (create_info->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0;
    *fence = make_handle<VkFence>(g_fences, impl);
    return VK_SUCCESS;
}

static void impl_destroy_fence(VkDevice device, VkFence fence, const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<Fence> impl = get_handle(g_fences, fence);
    if (!device_impl || !impl || impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    if (impl->submission && !impl->submission->processed)
    {
        log_error("device %llu fence %llu destroyed while pending",
                  (unsigned long long)device_impl->id, (unsigned long long)impl->id);
        return;
    }

    erase_handle(g_fences, fence);
}

static VkResult impl_get_fence_status(VkDevice device, VkFence fence)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<Fence> impl = get_handle(g_fences, fence);
    if (!device_impl || !impl || impl->owner != device_impl)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_impl->lost || device_impl->error)
    {
        impl->failed = true;
        return VK_ERROR_DEVICE_LOST;
    }
    if (impl->failed)
        return VK_ERROR_DEVICE_LOST;
    if (impl->signaled)
        return VK_SUCCESS;
    if (impl->submission)
    {
        const VkResult result = process_submission(impl->submission, 0);
        if (result == VK_SUCCESS)
        {
            impl->signaled = true;
            return VK_SUCCESS;
        }
        if (result != VK_TIMEOUT)
        {
            impl->failed = true;
            return result;
        }
    }

    return VK_NOT_READY;
}

static VkResult impl_reset_fences(VkDevice device, uint32_t fence_count, const VkFence* fences)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || fence_count == 0 || !fences)
        return VK_ERROR_INITIALIZATION_FAILED;

    for (uint32_t i = 0; i < fence_count; i++)
    {
        std::shared_ptr<Fence> impl = get_handle(g_fences, fences[i]);
        if (!impl || impl->owner != device_impl || (impl->submission && !impl->submission->processed))
            return VK_ERROR_INITIALIZATION_FAILED;
    }
    for (uint32_t i = 0; i < fence_count; i++)
    {
        std::shared_ptr<Fence> impl = get_handle(g_fences, fences[i]);
        impl->signaled = false;
        impl->failed = false;
        impl->submission.reset();
    }
    return VK_SUCCESS;
}

static VkResult impl_wait_for_fences(VkDevice device, uint32_t fence_count, const VkFence* fences,
                                     VkBool32 wait_all, uint64_t timeout)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || fence_count == 0 || !fences)
        return VK_ERROR_INITIALIZATION_FAILED;

    std::vector<std::shared_ptr<Fence> > fence_impls;
    for (uint32_t i = 0; i < fence_count; i++)
    {
        std::shared_ptr<Fence> impl = get_handle(g_fences, fences[i]);
        if (!impl || impl->owner != device_impl)
            return VK_ERROR_INITIALIZATION_FAILED;
        fence_impls.push_back(impl);
    }

    if (device_impl->lost || device_impl->error)
    {
        for (size_t i = 0; i < fence_impls.size(); i++)
            fence_impls[i]->failed = true;
        return VK_ERROR_DEVICE_LOST;
    }

    const uint64_t wait_begin = monotonic_time_ns();
    if (!wait_all)
    {
        for (;;)
        {
            std::vector<WGPUFutureWaitInfo> wait_infos;
            for (size_t i = 0; i < fence_impls.size(); i++)
            {
                std::shared_ptr<Fence>& fence = fence_impls[i];
                if (fence->signaled)
                    return VK_SUCCESS;
                if (fence->failed)
                    return VK_ERROR_DEVICE_LOST;
                if (!fence->submission)
                    continue;

                const VkResult result = process_submission(fence->submission, 0);
                if (result == VK_SUCCESS)
                {
                    fence->signaled = true;
                    return VK_SUCCESS;
                }
                if (result != VK_TIMEOUT)
                {
                    fence->failed = true;
                    return result;
                }

                WGPUFuture future = {};
                if (!fence->submission->completed)
                {
                    future = fence->submission->future;
                }
                else
                {
                    for (size_t j = 0; j < fence->submission->readbacks.size(); j++)
                    {
                        const Readback& readback = fence->submission->readbacks[j];
                        if (!readback.copied && readback.map_started
                                && readback.map_result && !readback.map_result->completed)
                        {
                            future = readback.future;
                            break;
                        }
                    }
                }

                if (future.id != 0)
                {
                    bool duplicate = false;
                    for (size_t j = 0; j < wait_infos.size(); j++)
                    {
                        if (wait_infos[j].future.id == future.id)
                        {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate)
                    {
                        WGPUFutureWaitInfo wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
                        wait_info.future = future;
                        wait_infos.push_back(wait_info);
                    }
                }
            }

            const uint64_t remaining = remaining_timeout_ns(wait_begin, timeout);
            if (remaining == 0 || wait_infos.empty())
                return VK_TIMEOUT;

            WGPUWaitStatus status;
            if (wait_any(device_impl->instance.get(), wait_infos.size(), wait_infos.data(),
                         remaining, "fence-wait-any", status) != 0)
            {
                device_impl->lost = true;
                for (size_t i = 0; i < fence_impls.size(); i++)
                    fence_impls[i]->failed = true;
                return VK_ERROR_DEVICE_LOST;
            }
            if (status == WGPUWaitStatus_TimedOut)
                return VK_TIMEOUT;
            if (status != WGPUWaitStatus_Success)
            {
                device_impl->lost = true;
                for (size_t i = 0; i < fence_impls.size(); i++)
                    fence_impls[i]->failed = true;
                return VK_ERROR_DEVICE_LOST;
            }

            bool completed = false;
            for (size_t i = 0; i < wait_infos.size(); i++)
                completed = completed || wait_infos[i].completed == WGPU_TRUE;
            if (!completed)
            {
                device_impl->lost = true;
                for (size_t i = 0; i < fence_impls.size(); i++)
                    fence_impls[i]->failed = true;
                return VK_ERROR_DEVICE_LOST;
            }
        }
    }

    for (size_t i = 0; i < fence_impls.size(); i++)
    {
        if (fence_impls[i]->signaled)
            continue;
        if (fence_impls[i]->failed)
            return VK_ERROR_DEVICE_LOST;
        if (!fence_impls[i]->submission)
            return VK_TIMEOUT;

        const uint64_t remaining = remaining_timeout_ns(wait_begin, timeout);
        const VkResult result = process_submission(fence_impls[i]->submission, remaining);
        if (result != VK_SUCCESS)
        {
            if (result != VK_TIMEOUT)
                fence_impls[i]->failed = true;
            return result;
        }
        fence_impls[i]->signaled = true;
    }

    return VK_SUCCESS;
}

// pipeline cache is an in-memory no-op object in the first phase

static VkResult impl_create_pipeline_cache(VkDevice device, const VkPipelineCacheCreateInfo* create_info,
                                           const VkAllocationCallbacks* allocator, VkPipelineCache* pipeline_cache)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO || !pipeline_cache)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || create_info->flags != 0
            || (create_info->initialDataSize != 0 && !create_info->pInitialData))
        return VK_ERROR_INITIALIZATION_FAILED;

    std::shared_ptr<PipelineCache> impl = std::make_shared<PipelineCache>();
    impl->owner = device_impl;
    *pipeline_cache = make_handle<VkPipelineCache>(g_pipeline_caches, impl);
    return VK_SUCCESS;
}

static void impl_destroy_pipeline_cache(VkDevice device, VkPipelineCache pipeline_cache,
                                        const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<PipelineCache> impl = get_handle(g_pipeline_caches, pipeline_cache);
    if (!device_impl || !impl || impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    erase_handle(g_pipeline_caches, pipeline_cache);
}

static VkResult impl_get_pipeline_cache_data(VkDevice device, VkPipelineCache pipeline_cache, size_t* data_size, void*)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<PipelineCache> impl = get_handle(g_pipeline_caches, pipeline_cache);
    if (!device_impl || !impl || impl->owner != device_impl || !data_size)
        return VK_ERROR_INITIALIZATION_FAILED;

    *data_size = 0;
    return VK_SUCCESS;
}

static VkResult impl_merge_pipeline_caches(VkDevice device, VkPipelineCache dst_cache,
                                           uint32_t src_cache_count, const VkPipelineCache* src_caches)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<PipelineCache> dst = get_handle(g_pipeline_caches, dst_cache);
    if (!device_impl || !dst || dst->owner != device_impl || (src_cache_count != 0 && !src_caches))
        return VK_ERROR_INITIALIZATION_FAILED;

    for (uint32_t i = 0; i < src_cache_count; i++)
    {
        std::shared_ptr<PipelineCache> src = get_handle(g_pipeline_caches, src_caches[i]);
        if (!src || src->owner != device_impl)
            return VK_ERROR_INITIALIZATION_FAILED;
    }
    return VK_SUCCESS;
}

static PFN_vkVoidFunction find_device_proc(const char* name)
{
    if (!name)
        return 0;

#define VKWEBGPU_DEVICE_PROC(api_name_, impl_name_) \
    if (strcmp(name, "vk" #api_name_) == 0) return (PFN_vkVoidFunction)impl_##impl_name_

    VKWEBGPU_DEVICE_PROC(DestroyDevice, destroy_device);
    VKWEBGPU_DEVICE_PROC(GetDeviceQueue, get_device_queue);
    VKWEBGPU_DEVICE_PROC(AllocateMemory, allocate_memory);
    VKWEBGPU_DEVICE_PROC(FreeMemory, free_memory);
    VKWEBGPU_DEVICE_PROC(MapMemory, map_memory);
    VKWEBGPU_DEVICE_PROC(UnmapMemory, unmap_memory);
    VKWEBGPU_DEVICE_PROC(FlushMappedMemoryRanges, flush_mapped_memory_ranges);
    VKWEBGPU_DEVICE_PROC(InvalidateMappedMemoryRanges, invalidate_mapped_memory_ranges);
    VKWEBGPU_DEVICE_PROC(GetDeviceMemoryCommitment, get_device_memory_commitment);
    VKWEBGPU_DEVICE_PROC(CreateBuffer, create_buffer);
    VKWEBGPU_DEVICE_PROC(DestroyBuffer, destroy_buffer);
    VKWEBGPU_DEVICE_PROC(GetBufferMemoryRequirements, get_buffer_memory_requirements);
    VKWEBGPU_DEVICE_PROC(GetBufferMemoryRequirements2, get_buffer_memory_requirements2);
    VKWEBGPU_DEVICE_PROC(BindBufferMemory, bind_buffer_memory);
    VKWEBGPU_DEVICE_PROC(BindBufferMemory2, bind_buffer_memory2);
    VKWEBGPU_DEVICE_PROC(CreateDescriptorSetLayout, create_descriptor_set_layout);
    VKWEBGPU_DEVICE_PROC(DestroyDescriptorSetLayout, destroy_descriptor_set_layout);
    VKWEBGPU_DEVICE_PROC(GetDescriptorSetLayoutSupportKHR, get_descriptor_set_layout_support);
    VKWEBGPU_DEVICE_PROC(GetDescriptorSetLayoutSupport, get_descriptor_set_layout_support);
    VKWEBGPU_DEVICE_PROC(CreatePipelineLayout, create_pipeline_layout);
    VKWEBGPU_DEVICE_PROC(DestroyPipelineLayout, destroy_pipeline_layout);
    VKWEBGPU_DEVICE_PROC(CreateShaderModule, create_shader_module);
    VKWEBGPU_DEVICE_PROC(DestroyShaderModule, destroy_shader_module);
    VKWEBGPU_DEVICE_PROC(CreateComputePipelines, create_compute_pipelines);
    VKWEBGPU_DEVICE_PROC(DestroyPipeline, destroy_pipeline);
    VKWEBGPU_DEVICE_PROC(CreateDescriptorUpdateTemplateKHR, create_descriptor_update_template);
    VKWEBGPU_DEVICE_PROC(CreateDescriptorUpdateTemplate, create_descriptor_update_template);
    VKWEBGPU_DEVICE_PROC(DestroyDescriptorUpdateTemplateKHR, destroy_descriptor_update_template);
    VKWEBGPU_DEVICE_PROC(DestroyDescriptorUpdateTemplate, destroy_descriptor_update_template);
    VKWEBGPU_DEVICE_PROC(CreateCommandPool, create_command_pool);
    VKWEBGPU_DEVICE_PROC(DestroyCommandPool, destroy_command_pool);
    VKWEBGPU_DEVICE_PROC(AllocateCommandBuffers, allocate_command_buffers);
    VKWEBGPU_DEVICE_PROC(FreeCommandBuffers, free_command_buffers);
    VKWEBGPU_DEVICE_PROC(ResetCommandPool, reset_command_pool);
    VKWEBGPU_DEVICE_PROC(TrimCommandPoolKHR, trim_command_pool);
    VKWEBGPU_DEVICE_PROC(TrimCommandPool, trim_command_pool);
    VKWEBGPU_DEVICE_PROC(ResetCommandBuffer, reset_command_buffer);
    VKWEBGPU_DEVICE_PROC(BeginCommandBuffer, begin_command_buffer);
    VKWEBGPU_DEVICE_PROC(EndCommandBuffer, end_command_buffer);
    VKWEBGPU_DEVICE_PROC(CmdBindPipeline, cmd_bind_pipeline);
    VKWEBGPU_DEVICE_PROC(CmdPushDescriptorSetWithTemplateKHR, cmd_push_descriptor_set_with_template);
    VKWEBGPU_DEVICE_PROC(CmdPushDescriptorSetKHR, cmd_push_descriptor_set);
    VKWEBGPU_DEVICE_PROC(CmdPushConstants, cmd_push_constants);
    VKWEBGPU_DEVICE_PROC(CmdDispatch, cmd_dispatch);
    VKWEBGPU_DEVICE_PROC(CmdCopyBuffer, cmd_copy_buffer);
    VKWEBGPU_DEVICE_PROC(CmdPipelineBarrier, cmd_pipeline_barrier);
    VKWEBGPU_DEVICE_PROC(QueueSubmit, queue_submit);
    VKWEBGPU_DEVICE_PROC(QueueWaitIdle, queue_wait_idle);
    VKWEBGPU_DEVICE_PROC(DeviceWaitIdle, device_wait_idle);
    VKWEBGPU_DEVICE_PROC(CreateFence, create_fence);
    VKWEBGPU_DEVICE_PROC(DestroyFence, destroy_fence);
    VKWEBGPU_DEVICE_PROC(GetFenceStatus, get_fence_status);
    VKWEBGPU_DEVICE_PROC(ResetFences, reset_fences);
    VKWEBGPU_DEVICE_PROC(WaitForFences, wait_for_fences);
    VKWEBGPU_DEVICE_PROC(CreatePipelineCache, create_pipeline_cache);
    VKWEBGPU_DEVICE_PROC(DestroyPipelineCache, destroy_pipeline_cache);
    VKWEBGPU_DEVICE_PROC(GetPipelineCacheData, get_pipeline_cache_data);
    VKWEBGPU_DEVICE_PROC(MergePipelineCaches, merge_pipeline_caches);

#undef VKWEBGPU_DEVICE_PROC
    return 0;
}

static PFN_vkVoidFunction impl_get_device_proc_addr(VkDevice device, const char* name)
{
    Device* impl = unwrap(device);
    if (!impl || !name)
        return 0;

    if ((strcmp(name, "vkCmdPushDescriptorSetKHR") == 0
            || strcmp(name, "vkCmdPushDescriptorSetWithTemplateKHR") == 0)
            && impl->enabled_extensions.find("VK_KHR_push_descriptor") == impl->enabled_extensions.end())
        return 0;
    if ((strcmp(name, "vkCreateDescriptorUpdateTemplateKHR") == 0
            || strcmp(name, "vkDestroyDescriptorUpdateTemplateKHR") == 0)
            && impl->enabled_extensions.find("VK_KHR_descriptor_update_template") == impl->enabled_extensions.end())
        return 0;
    if (strcmp(name, "vkTrimCommandPoolKHR") == 0
            && impl->enabled_extensions.find("VK_KHR_maintenance1") == impl->enabled_extensions.end())
        return 0;
    if (strcmp(name, "vkGetDescriptorSetLayoutSupportKHR") == 0
            && impl->enabled_extensions.find("VK_KHR_maintenance3") == impl->enabled_extensions.end())
        return 0;

    return find_device_proc(name);
}

static PFN_vkVoidFunction find_instance_proc(const char* name)
{
    if (!name)
        return 0;

#define VKWEBGPU_INSTANCE_PROC(api_name_, impl_name_) \
    if (strcmp(name, "vk" #api_name_) == 0) return (PFN_vkVoidFunction)impl_##impl_name_

    VKWEBGPU_INSTANCE_PROC(EnumerateInstanceVersion, enumerate_instance_version);
    VKWEBGPU_INSTANCE_PROC(DestroyInstance, destroy_instance);
    VKWEBGPU_INSTANCE_PROC(EnumeratePhysicalDevices, enumerate_physical_devices);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceFeatures, get_physical_device_features);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceProperties, get_physical_device_properties);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceQueueFamilyProperties, get_physical_device_queue_family_properties);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceMemoryProperties, get_physical_device_memory_properties);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceFormatProperties, get_physical_device_format_properties);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceImageFormatProperties, get_physical_device_image_format_properties);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceFeatures2KHR, get_physical_device_features2);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceFeatures2, get_physical_device_features2);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceProperties2KHR, get_physical_device_properties2);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceProperties2, get_physical_device_properties2);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceFormatProperties2KHR, get_physical_device_format_properties2);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceFormatProperties2, get_physical_device_format_properties2);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceImageFormatProperties2KHR, get_physical_device_image_format_properties2);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceImageFormatProperties2, get_physical_device_image_format_properties2);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceQueueFamilyProperties2KHR, get_physical_device_queue_family_properties2);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceQueueFamilyProperties2, get_physical_device_queue_family_properties2);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceMemoryProperties2KHR, get_physical_device_memory_properties2);
    VKWEBGPU_INSTANCE_PROC(GetPhysicalDeviceMemoryProperties2, get_physical_device_memory_properties2);
    VKWEBGPU_INSTANCE_PROC(EnumerateDeviceExtensionProperties, enumerate_device_extension_properties);
    VKWEBGPU_INSTANCE_PROC(EnumerateDeviceLayerProperties, enumerate_device_layer_properties);
    VKWEBGPU_INSTANCE_PROC(CreateDevice, create_device);
    VKWEBGPU_INSTANCE_PROC(GetDeviceProcAddr, get_device_proc_addr);

#undef VKWEBGPU_INSTANCE_PROC

    PFN_vkVoidFunction device_proc = find_device_proc(name);
    if (device_proc)
        return device_proc;

    return 0;
}

} // namespace vkwebgpu_detail

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char* layer_name, uint32_t* property_count, VkExtensionProperties* properties)
{
    return vkwebgpu_detail::impl_enumerate_instance_extension_properties(layer_name, property_count, properties);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
    const VkInstanceCreateInfo* create_info, const VkAllocationCallbacks* allocator, VkInstance* instance)
{
    return vkwebgpu_detail::impl_create_instance(create_info, allocator, instance);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
    uint32_t* property_count, VkLayerProperties* properties)
{
    return vkwebgpu_detail::impl_enumerate_instance_layer_properties(property_count, properties);
}

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* name)
{
    if (!name)
        return 0;
    if (strcmp(name, "vkGetInstanceProcAddr") == 0)
        return (PFN_vkVoidFunction)vkGetInstanceProcAddr;
    if (strcmp(name, "vkEnumerateInstanceExtensionProperties") == 0)
        return (PFN_vkVoidFunction)vkEnumerateInstanceExtensionProperties;
    if (strcmp(name, "vkCreateInstance") == 0)
        return (PFN_vkVoidFunction)vkCreateInstance;
    if (strcmp(name, "vkEnumerateInstanceLayerProperties") == 0)
        return (PFN_vkVoidFunction)vkEnumerateInstanceLayerProperties;
    if (strcmp(name, "vkEnumerateInstanceVersion") == 0)
        return (PFN_vkVoidFunction)vkwebgpu_detail::impl_enumerate_instance_version;

    if (!vkwebgpu_detail::unwrap(instance))
        return 0;

    return vkwebgpu_detail::find_instance_proc(name);
}
