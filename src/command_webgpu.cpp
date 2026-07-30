// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "command.h"

#if NCNN_WEBGPU

#include <algorithm>
#include <string.h>

#include "allocator.h"
#include "gpu.h"
#include "option.h"
#include "pipeline.h"

namespace ncnn {

uint64_t begin_webgpu_sync_operation(const char* operation);
int finish_webgpu_sync_operation(uint64_t operation_id, int result);

int acquire_webgpu_allocation_in_flight(VkAllocator* allocator, VkBufferMemory* memory);
void release_webgpu_allocation_in_flight(VkAllocator* allocator, VkBufferMemory* memory);

struct WebGpuReadback
{
    WGPUBuffer buffer;
    Mat* destination;
    size_t size;
};

struct WebGpuAllocationReference
{
    VkAllocator* allocator;
    VkBufferMemory* memory;
};

enum WebGpuCommandState
{
    WEBGPU_COMMAND_RECORDING,
    WEBGPU_COMMAND_WAITING_QUEUE,
    WEBGPU_COMMAND_WAITING_MAP,
    WEBGPU_COMMAND_COMPLETED,
    WEBGPU_COMMAND_FAILED
};

class VkComputePrivate
{
public:
    VkComputePrivate()
        : command_encoder(0), compute_pass(0), validation_scope(false), queue_write_pending(false), state(WEBGPU_COMMAND_RECORDING), record_error(0), pending_dispatch_total(0)
    {
    }

    WGPUCommandEncoder command_encoder;
    WGPUComputePassEncoder compute_pass;
    bool validation_scope;
    bool queue_write_pending;
    WebGpuCommandState state;
    int record_error;
    uint64_t pending_dispatch_total;
    std::vector<WGPUBindGroup> bind_groups;
    std::vector<WGPUBuffer> binding_copy_buffers;
    std::vector<WebGpuAllocationReference> allocation_references;
    std::vector<WebGpuReadback> readbacks;
};

VkCompute::VkCompute(const VulkanDevice* _vkdev)
    : vkdev(_vkdev), d(new VkComputePrivate)
{
}

VkCompute::~VkCompute()
{
    reset();
    delete d;
}

static void set_webgpu_record_error(VkComputePrivate* d)
{
    if (d->record_error == 0)
        d->record_error = -1;
}

static int acquire_webgpu_allocation(std::vector<WebGpuAllocationReference>& references, const VkMat& mat)
{
    if (!mat.data || !mat.allocator)
        return -1;

    for (size_t i = 0; i < references.size(); i++)
    {
        if (references[i].memory == mat.data)
            return 0;
    }

    if (acquire_webgpu_allocation_in_flight(mat.allocator, mat.data) != 0)
        return -1;

    WebGpuAllocationReference reference;
    reference.allocator = mat.allocator;
    reference.memory = mat.data;
    references.push_back(reference);
    return 0;
}

static void release_webgpu_allocations(std::vector<WebGpuAllocationReference>& references)
{
    for (size_t i = 0; i < references.size(); i++)
        release_webgpu_allocation_in_flight(references[i].allocator, references[i].memory);
    references.clear();
}

static int ensure_webgpu_command_encoder(const VulkanDevice* vkdev, VkComputePrivate* d)
{
    if (d->command_encoder)
        return 0;

    wgpuDevicePushErrorScope(vkdev->wgpu_device(), WGPUErrorFilter_Validation);
    d->validation_scope = true;
    WGPUCommandEncoderDescriptor descriptor = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    d->command_encoder = wgpuDeviceCreateCommandEncoder(vkdev->wgpu_device(), &descriptor);
    return d->command_encoder ? 0 : -1;
}

static void end_webgpu_compute_pass(VkComputePrivate* d)
{
    if (!d->compute_pass)
        return;

    wgpuComputePassEncoderEnd(d->compute_pass);
    wgpuComputePassEncoderRelease(d->compute_pass);
    d->compute_pass = 0;
}

static int queue_write_webgpu_buffer(WGPUQueue queue, WGPUBuffer buffer, size_t buffer_offset, const void* data, size_t size)
{
    if (!buffer || !data)
        return -1;

    const size_t write_chunk_size = 16 * 1024 * 1024;
    const unsigned char* data_ptr = (const unsigned char*)data;
    size_t offset = 0;
    while (offset + 4 <= size)
    {
        size_t write_size = std::min(write_chunk_size, size - offset);
        write_size &= ~(size_t)3;
        if (write_size == 0)
            break;
        wgpuQueueWriteBuffer(queue, buffer, buffer_offset + offset, data_ptr + offset, write_size);
        offset += write_size;
    }

    if (offset < size)
    {
        unsigned char tail[4] = {0, 0, 0, 0};
        memcpy(tail, data_ptr + offset, size - offset);
        wgpuQueueWriteBuffer(queue, buffer, buffer_offset + offset, tail, 4);
    }

    return 0;
}

void VkCompute::record_upload(const Mat& src, VkMat& dst, const Option& opt)
{
    if (d->record_error != 0)
        return;
    if (!opt.blob_vkallocator)
    {
        set_webgpu_record_error(d);
        return;
    }

    Mat src_upload = src;
    Mat src_fp16;
    if (src.elemsize == src.elempack * 4u && opt.use_fp16_packed)
    {
        cast_float32_to_float16(src, src_fp16, opt);
        if (src_fp16.empty())
        {
            set_webgpu_record_error(d);
            return;
        }
        src_upload = src_fp16;
    }

    dst.create_like(src_upload, opt.blob_vkallocator);
    if (dst.empty())
    {
        set_webgpu_record_error(d);
        return;
    }

    const size_t size = src_upload.total() * src_upload.elemsize;
    if (size > dst.buffer_capacity()
            || queue_write_webgpu_buffer(vkdev->wgpu_queue(), dst.buffer(), dst.buffer_offset(), src_upload.data, size) != 0)
    {
        set_webgpu_record_error(d);
        return;
    }
    d->queue_write_pending = true;
    if (acquire_webgpu_allocation(d->allocation_references, dst) != 0)
        set_webgpu_record_error(d);
}

void VkCompute::record_download(const VkMat& src, Mat& dst, const Option& opt)
{
    if (d->record_error != 0)
        return;
    if (src.empty() || !opt.blob_allocator || ensure_webgpu_command_encoder(vkdev, d) != 0)
    {
        set_webgpu_record_error(d);
        return;
    }

    if (src.data->host_shadow_dirty && src.allocator->flush(src.data) != 0)
    {
        set_webgpu_record_error(d);
        return;
    }

    dst.create_like(src, opt.blob_allocator);
    if (dst.empty())
    {
        set_webgpu_record_error(d);
        return;
    }

    end_webgpu_compute_pass(d);
    const size_t size = src.total() * src.elemsize;
    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    descriptor.size = std::max(alignSize(size, 4), (size_t)4);
    WGPUBuffer readback_buffer = wgpuDeviceCreateBuffer(vkdev->wgpu_device(), &descriptor);
    if (!readback_buffer)
    {
        set_webgpu_record_error(d);
        return;
    }

    wgpuCommandEncoderCopyBufferToBuffer(d->command_encoder, src.buffer(), src.buffer_offset(), readback_buffer, 0, alignSize(size, 4));
    WebGpuReadback readback;
    readback.buffer = readback_buffer;
    readback.destination = &dst;
    readback.size = size;
    d->readbacks.push_back(readback);
    if (acquire_webgpu_allocation(d->allocation_references, src) != 0)
        set_webgpu_record_error(d);
}

void VkCompute::record_clone(const Mat& src, VkMat& dst, const Option& opt)
{
    record_upload(src, dst, opt);
}

void VkCompute::record_clone(const Mat&, VkImageMat&, const Option&)
{
    set_webgpu_record_error(d);
}

void VkCompute::record_clone(const VkMat& src, Mat& dst, const Option& opt)
{
    record_download(src, dst, opt);
}

void VkCompute::record_clone(const VkImageMat&, Mat&, const Option&)
{
    set_webgpu_record_error(d);
}

void VkCompute::record_clone(const VkMat& src, VkMat& dst, const Option& opt)
{
    if (d->record_error != 0)
        return;
    if (src.empty() || !opt.blob_vkallocator || ensure_webgpu_command_encoder(vkdev, d) != 0)
    {
        set_webgpu_record_error(d);
        return;
    }

    if (src.data->host_shadow_dirty && src.allocator->flush(src.data) != 0)
    {
        set_webgpu_record_error(d);
        return;
    }

    dst.create_like(src, opt.blob_vkallocator);
    if (dst.empty())
    {
        set_webgpu_record_error(d);
        return;
    }

    end_webgpu_compute_pass(d);
    const size_t size = src.total() * src.elemsize;
    const size_t aligned_size = alignSize(size, 4);
    if (src.buffer() == dst.buffer())
    {
        WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
        descriptor.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
        descriptor.size = aligned_size;
        WGPUBuffer temporary_buffer = wgpuDeviceCreateBuffer(vkdev->wgpu_device(), &descriptor);
        if (!temporary_buffer)
        {
            set_webgpu_record_error(d);
            return;
        }
        d->binding_copy_buffers.push_back(temporary_buffer);
        wgpuCommandEncoderCopyBufferToBuffer(d->command_encoder, src.buffer(), src.buffer_offset(), temporary_buffer, 0, aligned_size);
        wgpuCommandEncoderCopyBufferToBuffer(d->command_encoder, temporary_buffer, 0, dst.buffer(), dst.buffer_offset(), aligned_size);
    }
    else
    {
        wgpuCommandEncoderCopyBufferToBuffer(d->command_encoder, src.buffer(), src.buffer_offset(), dst.buffer(), dst.buffer_offset(), aligned_size);
    }
    if (acquire_webgpu_allocation(d->allocation_references, src) != 0
            || acquire_webgpu_allocation(d->allocation_references, dst) != 0)
        set_webgpu_record_error(d);
}

void VkCompute::record_clone(const VkImageMat&, VkImageMat&, const Option&)
{
    set_webgpu_record_error(d);
}

void VkCompute::record_clone(const VkMat&, VkImageMat&, const Option&)
{
    set_webgpu_record_error(d);
}

void VkCompute::record_clone(const VkImageMat&, VkMat&, const Option&)
{
    set_webgpu_record_error(d);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher)
{
    if (d->record_error != 0)
        return;
    if (dispatcher.w == 0 || dispatcher.h == 0 || dispatcher.c == 0)
        return;
    if (!pipeline || !pipeline->webgpu_bundle() || !pipeline->webgpu_bundle()->pipeline)
    {
        set_webgpu_record_error(d);
        return;
    }

    const WebGpuPipelineBundle* bundle = pipeline->webgpu_bundle();
    const WebGpuShaderInfo& shader_info = bundle->shader_info;
    std::vector<WGPUBindGroupEntry> bind_group_entries(shader_info.bindings.size());
    std::vector<VkMat> resolved_bindings(shader_info.bindings.size());
    for (size_t i = 0; i < shader_info.bindings.size(); i++)
    {
        const WebGpuBindingInfo& binding_info = shader_info.bindings[i];
        const VkMat binding = binding_info.binding < bindings.size() && !bindings[binding_info.binding].empty()
                              ? bindings[binding_info.binding]
                              : vkdev->get_dummy_buffer(binding_info.binding);
        if (binding.empty())
        {
            set_webgpu_record_error(d);
            return;
        }
        if (binding.data->host_shadow_dirty && binding.allocator->flush(binding.data) != 0)
        {
            set_webgpu_record_error(d);
            return;
        }
        if (binding.buffer_offset() % vkdev->info.buffer_offset_alignment() != 0
                || binding.buffer_capacity() < binding_info.min_binding_size)
        {
            set_webgpu_record_error(d);
            return;
        }

        WGPUBindGroupEntry& entry = bind_group_entries[i];
        entry = WGPU_BIND_GROUP_ENTRY_INIT;
        entry.binding = binding_info.binding;
        entry.buffer = binding.buffer();
        entry.offset = binding.buffer_offset();
        entry.size = binding.buffer_capacity();
        resolved_bindings[i] = binding;
    }

    for (size_t i = 0; i < bind_group_entries.size(); i++)
    {
        const WGPUBindGroupEntry& a = bind_group_entries[i];
        for (size_t j = i + 1; j < bind_group_entries.size(); j++)
        {
            const WGPUBindGroupEntry& b = bind_group_entries[j];
            if (a.buffer != b.buffer)
                continue;

            const uint64_t a_end = a.offset + a.size;
            const uint64_t b_end = b.offset + b.size;
            if (a_end < a.offset || b_end < b.offset)
            {
                set_webgpu_record_error(d);
                return;
            }

            const bool overlaps = a.offset < b_end && b.offset < a_end;
            const bool has_write = shader_info.bindings[i].access == NCNN_WEBGPU_BINDING_READ_WRITE
                                   || shader_info.bindings[j].access == NCNN_WEBGPU_BINDING_READ_WRITE;
            if (overlaps && has_write)
            {
                NCNN_LOGE("WebGPU bindings %u and %u overlap writable storage ranges",
                          shader_info.bindings[i].binding, shader_info.bindings[j].binding);
                set_webgpu_record_error(d);
                return;
            }
        }
    }

    for (size_t i = 0; i < bind_group_entries.size(); i++)
    {
        if (shader_info.bindings[i].access != NCNN_WEBGPU_BINDING_READ)
            continue;

        bool shares_writable_buffer = false;
        for (size_t j = 0; j < bind_group_entries.size(); j++)
        {
            if (shader_info.bindings[j].access == NCNN_WEBGPU_BINDING_READ_WRITE
                    && bind_group_entries[i].buffer == bind_group_entries[j].buffer)
            {
                shares_writable_buffer = true;
                break;
            }
        }
        if (!shares_writable_buffer)
            continue;

        if (ensure_webgpu_command_encoder(vkdev, d) != 0)
        {
            set_webgpu_record_error(d);
            return;
        }
        end_webgpu_compute_pass(d);

        WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
        descriptor.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        descriptor.size = bind_group_entries[i].size;
        WGPUBuffer binding_copy_buffer = wgpuDeviceCreateBuffer(vkdev->wgpu_device(), &descriptor);
        if (!binding_copy_buffer)
        {
            set_webgpu_record_error(d);
            return;
        }
        d->binding_copy_buffers.push_back(binding_copy_buffer);
        wgpuCommandEncoderCopyBufferToBuffer(d->command_encoder,
                                             bind_group_entries[i].buffer, bind_group_entries[i].offset,
                                             binding_copy_buffer, 0, bind_group_entries[i].size);
        bind_group_entries[i].buffer = binding_copy_buffer;
        bind_group_entries[i].offset = 0;
    }

    std::vector<unsigned char> packed_immediate;
    if (pack_webgpu_immediates(shader_info, constants, packed_immediate) != 0
            || ensure_webgpu_command_encoder(vkdev, d) != 0)
    {
        set_webgpu_record_error(d);
        return;
    }

    WGPUBindGroupDescriptor bind_group_descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bind_group_descriptor.layout = bundle->bind_group_layout;
    bind_group_descriptor.entryCount = bind_group_entries.size();
    bind_group_descriptor.entries = bind_group_entries.data();
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(vkdev->wgpu_device(), &bind_group_descriptor);
    if (!bind_group)
    {
        set_webgpu_record_error(d);
        return;
    }

    if (!d->compute_pass)
    {
        WGPUComputePassDescriptor pass_descriptor = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        d->compute_pass = wgpuCommandEncoderBeginComputePass(d->command_encoder, &pass_descriptor);
    }
    if (!d->compute_pass)
    {
        wgpuBindGroupRelease(bind_group);
        set_webgpu_record_error(d);
        return;
    }

    const uint32_t group_count_x = (dispatcher.w + pipeline->local_size_x() - 1) / pipeline->local_size_x();
    const uint32_t group_count_y = (dispatcher.h + pipeline->local_size_y() - 1) / pipeline->local_size_y();
    const uint32_t group_count_z = (dispatcher.c + pipeline->local_size_z() - 1) / pipeline->local_size_z();
    if (group_count_x > vkdev->info.max_workgroup_count_x()
            || group_count_y > vkdev->info.max_workgroup_count_y()
            || group_count_z > vkdev->info.max_workgroup_count_z())
    {
        wgpuBindGroupRelease(bind_group);
        set_webgpu_record_error(d);
        return;
    }

    wgpuComputePassEncoderSetPipeline(d->compute_pass, bundle->pipeline);
    wgpuComputePassEncoderSetBindGroup(d->compute_pass, 0, bind_group, 0, 0);
    if (!packed_immediate.empty())
        wgpuComputePassEncoderSetImmediates(d->compute_pass, 0, packed_immediate.data(), packed_immediate.size());
    wgpuComputePassEncoderDispatchWorkgroups(d->compute_pass, group_count_x, group_count_y, group_count_z);
    end_webgpu_compute_pass(d);

    d->bind_groups.push_back(bind_group);
    for (size_t i = 0; i < resolved_bindings.size(); i++)
    {
        if (acquire_webgpu_allocation(d->allocation_references, resolved_bindings[i]) != 0)
        {
            set_webgpu_record_error(d);
            return;
        }
    }
    d->pending_dispatch_total += (uint64_t)group_count_x * group_count_y * group_count_z;
}

void VkCompute::record_pipeline(const Pipeline*, const std::vector<VkImageMat>&, const std::vector<vk_constant_type>&, const VkImageMat&)
{
    set_webgpu_record_error(d);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const VkMat& dispatcher)
{
    if (!image_bindings.empty())
    {
        set_webgpu_record_error(d);
        return;
    }
    record_pipeline(pipeline, buffer_bindings, constants, dispatcher);
}

void VkCompute::record_pipeline(const Pipeline*, const std::vector<VkMat>&, const std::vector<VkImageMat>&, const std::vector<vk_constant_type>&, const VkImageMat&)
{
    set_webgpu_record_error(d);
}

void VkCompute::record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& buffer_bindings, const std::vector<VkImageMat>& image_bindings, const std::vector<vk_constant_type>& constants, const Mat& dispatcher)
{
    if (!image_bindings.empty())
    {
        set_webgpu_record_error(d);
        return;
    }

    VkMat dispatcher_gpu;
    dispatcher_gpu.dims = dispatcher.dims;
    dispatcher_gpu.w = dispatcher.w;
    dispatcher_gpu.h = dispatcher.h;
    dispatcher_gpu.c = dispatcher.c;
    record_pipeline(pipeline, buffer_bindings, constants, dispatcher_gpu);
}

#if NCNN_BENCHMARK
void VkCompute::record_write_timestamp(uint32_t)
{
}
#endif // NCNN_BENCHMARK

struct WebGpuQueueResult
{
    WGPUQueueWorkDoneStatus status;
    bool completed;
};

static void webgpu_queue_callback(WGPUQueueWorkDoneStatus status, WGPUStringView message, void* userdata1, void*)
{
    WebGpuQueueResult* result = (WebGpuQueueResult*)userdata1;
    result->status = status;
    result->completed = true;
    if (status != WGPUQueueWorkDoneStatus_Success)
        NCNN_LOGE("WebGPU queue failed status=%d %.*s", (int)status, (int)message.length, message.data ? message.data : "");
}

static int wait_webgpu_queue(const VulkanDevice* vkdev, const char* operation)
{
    WebGpuQueueResult result;
    result.status = WGPUQueueWorkDoneStatus_Error;
    result.completed = false;
    WGPUQueueWorkDoneCallbackInfo callback_info = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
    callback_info.callback = webgpu_queue_callback;
    callback_info.userdata1 = &result;
    WGPUFuture future = wgpuQueueOnSubmittedWorkDone(vkdev->wgpu_queue(), callback_info);
    WGPUFutureWaitInfo wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
    if (vkdev->wait_webgpu_future(future, &wait_info, operation) != 0
            || !result.completed
            || result.status != WGPUQueueWorkDoneStatus_Success)
        return -1;

    return 0;
}

struct WebGpuMapResult
{
    WGPUMapAsyncStatus status;
    WGPUMapAsyncStatus actual_status;
    bool completed;
};

static void webgpu_map_callback(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void*)
{
    WebGpuMapResult* result = (WebGpuMapResult*)userdata1;
    result->actual_status = status;
    result->status = status;
    result->completed = true;
    if (status != WGPUMapAsyncStatus_Success)
        NCNN_LOGE("WebGPU map failed status=%d %.*s", (int)status, (int)message.length, message.data ? message.data : "");
}

struct WebGpuCommandErrorScopeResult
{
    WGPUPopErrorScopeStatus status;
    WGPUErrorType type;
    bool completed;
};

static void webgpu_command_error_scope_callback(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void* userdata1, void*)
{
    WebGpuCommandErrorScopeResult* result = (WebGpuCommandErrorScopeResult*)userdata1;
    result->status = status;
    result->type = type;
    result->completed = true;
    if (status != WGPUPopErrorScopeStatus_Success || type != WGPUErrorType_NoError)
        NCNN_LOGE("WebGPU command validation failed status=%d type=%d %.*s", (int)status, (int)type, (int)message.length, message.data ? message.data : "");
}

static int pop_webgpu_command_error_scope(const VulkanDevice* vkdev, VkComputePrivate* d)
{
    if (!d->validation_scope)
        return 0;
    d->validation_scope = false;

    WebGpuCommandErrorScopeResult result;
    result.status = WGPUPopErrorScopeStatus_Error;
    result.type = WGPUErrorType_Unknown;
    result.completed = false;
    WGPUPopErrorScopeCallbackInfo callback_info = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
    callback_info.callback = webgpu_command_error_scope_callback;
    callback_info.userdata1 = &result;
    WGPUFuture future = wgpuDevicePopErrorScope(vkdev->wgpu_device(), callback_info);
    WGPUFutureWaitInfo wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
    if (vkdev->wait_webgpu_future(future, &wait_info, "command-error-scope") != 0
            || !result.completed
            || result.status != WGPUPopErrorScopeStatus_Success
            || result.type != WGPUErrorType_NoError)
        return -1;

    return 0;
}

static void release_webgpu_command_resources(VkComputePrivate* d)
{
    end_webgpu_compute_pass(d);
    if (d->command_encoder)
    {
        wgpuCommandEncoderRelease(d->command_encoder);
    }
    d->command_encoder = 0;

    for (size_t i = 0; i < d->bind_groups.size(); i++)
    {
        wgpuBindGroupRelease(d->bind_groups[i]);
    }
    d->bind_groups.clear();
    for (size_t i = 0; i < d->binding_copy_buffers.size(); i++)
    {
        wgpuBufferRelease(d->binding_copy_buffers[i]);
    }
    d->binding_copy_buffers.clear();
    for (size_t i = 0; i < d->readbacks.size(); i++)
    {
        wgpuBufferRelease(d->readbacks[i].buffer);
    }
    d->readbacks.clear();
    release_webgpu_allocations(d->allocation_references);
    d->queue_write_pending = false;
    d->pending_dispatch_total = 0;
}

int VkCompute::submit_and_wait()
{
    const uint64_t operation_id = begin_webgpu_sync_operation("compute-submit");
    if (operation_id == 0)
    {
        release_webgpu_command_resources(d);
        d->record_error = 0;
        d->state = WEBGPU_COMMAND_RECORDING;
        return -1;
    }

    if (d->record_error != 0 || get_webgpu_last_error() != 0)
    {
        d->state = WEBGPU_COMMAND_FAILED;
        if (d->queue_write_pending)
            wait_webgpu_queue(vkdev, "queue-work-done-after-record-error");
        pop_webgpu_command_error_scope(vkdev, d);
        release_webgpu_command_resources(d);
        d->record_error = 0;
        d->state = WEBGPU_COMMAND_RECORDING;
        return finish_webgpu_sync_operation(operation_id, -1);
    }

    WGPUCommandBuffer command_buffer = 0;
    if (d->command_encoder)
    {
        end_webgpu_compute_pass(d);
        WGPUCommandBufferDescriptor descriptor = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
        command_buffer = wgpuCommandEncoderFinish(d->command_encoder, &descriptor);
        wgpuCommandEncoderRelease(d->command_encoder);
        d->command_encoder = 0;
        if (!command_buffer)
        {
            pop_webgpu_command_error_scope(vkdev, d);
            release_webgpu_command_resources(d);
            return finish_webgpu_sync_operation(operation_id, -1);
        }
        wgpuQueueSubmit(vkdev->wgpu_queue(), 1, &command_buffer);
    }

    d->state = WEBGPU_COMMAND_WAITING_QUEUE;
    int ret = wait_webgpu_queue(vkdev, "queue-work-done");
    d->queue_write_pending = false;

    if (command_buffer)
    {
        wgpuCommandBufferRelease(command_buffer);
    }

    if (ret == 0)
        d->state = WEBGPU_COMMAND_WAITING_MAP;
    for (size_t i = 0; ret == 0 && i < d->readbacks.size(); i++)
    {
        WebGpuReadback& readback = d->readbacks[i];
        WebGpuMapResult map_result;
        map_result.status = WGPUMapAsyncStatus_Error;
        map_result.actual_status = WGPUMapAsyncStatus_Error;
        map_result.completed = false;
        WGPUBufferMapCallbackInfo map_callback_info = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        map_callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
        map_callback_info.callback = webgpu_map_callback;
        map_callback_info.userdata1 = &map_result;
        WGPUFuture map_future = wgpuBufferMapAsync(readback.buffer, WGPUMapMode_Read, 0, alignSize(readback.size, 4), map_callback_info);
        WGPUFutureWaitInfo map_wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
        if (vkdev->wait_webgpu_future(map_future, &map_wait_info, "buffer-map") != 0
                || !map_result.completed
                || map_result.status != WGPUMapAsyncStatus_Success)
        {
            if (map_result.actual_status == WGPUMapAsyncStatus_Success)
                wgpuBufferUnmap(readback.buffer);
            ret = -1;
            break;
        }

        const void* mapped_data = wgpuBufferGetConstMappedRange(readback.buffer, 0, alignSize(readback.size, 4));
        if (!mapped_data)
        {
            ret = -1;
            break;
        }
        memcpy(readback.destination->data, mapped_data, readback.size);
        wgpuBufferUnmap(readback.buffer);
    }

    if (pop_webgpu_command_error_scope(vkdev, d) != 0)
        ret = -1;
    if (get_webgpu_last_error() != 0)
        ret = -1;
    d->state = ret == 0 ? WEBGPU_COMMAND_COMPLETED : WEBGPU_COMMAND_FAILED;
    release_webgpu_command_resources(d);
    d->record_error = 0;
    d->state = WEBGPU_COMMAND_RECORDING;
    return finish_webgpu_sync_operation(operation_id, ret);
}

int VkCompute::reset()
{
    int ret = 0;
    if (d->queue_write_pending && wait_webgpu_queue(vkdev, "queue-work-done-reset") != 0)
        ret = -1;
    if (d->validation_scope)
    {
        if (pop_webgpu_command_error_scope(vkdev, d) != 0)
            ret = -1;
    }
    release_webgpu_command_resources(d);
    d->record_error = 0;
    d->state = WEBGPU_COMMAND_RECORDING;
    return ret;
}

uint64_t VkCompute::pending_dispatch_total() const
{
    return d->pending_dispatch_total;
}

#if NCNN_BENCHMARK
int VkCompute::create_query_pool(uint32_t)
{
    return -1;
}

int VkCompute::get_query_pool_results(uint32_t, uint32_t, std::vector<uint64_t>&)
{
    return -1;
}
#endif // NCNN_BENCHMARK

void VkCompute::barrier_readwrite(const VkMat&)
{
    end_webgpu_compute_pass(d);
}

void VkCompute::barrier_readwrite(const VkImageMat&)
{
}

void VkCompute::barrier_readonly(const VkImageMat&)
{
}

class VkTransferPrivate
{
public:
    VkTransferPrivate()
        : record_error(0), queue_write_pending(false), pending_upload_total(0)
    {
    }

    int record_error;
    bool queue_write_pending;
    uint64_t pending_upload_total;
    std::vector<WebGpuAllocationReference> allocation_references;
};

VkTransfer::VkTransfer(const VulkanDevice* _vkdev)
    : vkdev(_vkdev), d(new VkTransferPrivate)
{
}

VkTransfer::~VkTransfer()
{
    delete d;
}

void VkTransfer::record_upload(const Mat& src, VkMat& dst, const Option& opt, bool flatten)
{
    if (d->record_error != 0)
        return;
    if (!opt.blob_vkallocator)
    {
        d->record_error = -1;
        return;
    }

    if (src.elembits() == 32 && opt.use_fp16_packed)
    {
        Mat src_fp16;
        cast_float32_to_float16(src, src_fp16, opt);
        if (src_fp16.empty())
        {
            d->record_error = -1;
            return;
        }
        record_upload(src_fp16, dst, opt, flatten);
        return;
    }

    Mat src_flattened = flatten ? src.reshape(src.w * src.h * src.c) : src;
    dst.create_like(src_flattened, opt.blob_vkallocator);
    if (dst.empty())
    {
        d->record_error = -1;
        return;
    }

    const size_t size = src_flattened.total() * src_flattened.elemsize;
    if (size > dst.buffer_capacity()
            || queue_write_webgpu_buffer(vkdev->wgpu_queue(), dst.buffer(), dst.buffer_offset(), src_flattened.data, size) != 0)
    {
        d->record_error = -1;
        return;
    }
    d->queue_write_pending = true;

    if (acquire_webgpu_allocation(d->allocation_references, dst) != 0)
    {
        d->record_error = -1;
        return;
    }
    d->pending_upload_total += size;
}

int VkTransfer::submit_and_wait()
{
    const uint64_t operation_id = begin_webgpu_sync_operation("transfer-submit");
    if (operation_id == 0)
    {
        reset();
        return -1;
    }

    if (d->record_error != 0 || get_webgpu_last_error() != 0)
    {
        if (d->queue_write_pending)
        {
            wait_webgpu_queue(vkdev, "transfer-queue-work-done-after-record-error");
            d->queue_write_pending = false;
        }
        reset();
        return finish_webgpu_sync_operation(operation_id, -1);
    }

    const int ret = wait_webgpu_queue(vkdev, "transfer-queue-work-done");
    d->queue_write_pending = false;

    if (ret != 0 || get_webgpu_last_error() != 0)
    {
        reset();
        return finish_webgpu_sync_operation(operation_id, -1);
    }

    release_webgpu_allocations(d->allocation_references);
    d->pending_upload_total = 0;
    return finish_webgpu_sync_operation(operation_id, 0);
}

int VkTransfer::reset()
{
    int ret = 0;
    if (d->queue_write_pending && wait_webgpu_queue(vkdev, "transfer-queue-work-done-reset") != 0)
        ret = -1;
    d->queue_write_pending = false;
    release_webgpu_allocations(d->allocation_references);
    d->record_error = 0;
    d->pending_upload_total = 0;
    return ret;
}

uint64_t VkTransfer::pending_upload_total() const
{
    return d->pending_upload_total;
}

} // namespace ncnn

#endif // NCNN_WEBGPU
