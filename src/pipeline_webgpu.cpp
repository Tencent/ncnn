// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "pipeline.h"

#if NCNN_WEBGPU

#include <stdio.h>

#include <string>


#include "option.h"
#include "pipelinecache.h"

namespace ncnn {

uint64_t begin_webgpu_sync_operation(const char* operation);
int finish_webgpu_sync_operation(uint64_t operation_id, int result);

int create_webgpu_shader_module(const VulkanDevice* vkdev, int shader_type_index, const Option& opt,
                                uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                                WGPUShaderModule* shader_module, WebGpuShaderInfo* shader_info);


class PipelinePrivate
{
public:
    WebGpuPipelineBundle bundle;
    uint32_t local_size_x;
    uint32_t local_size_y;
    uint32_t local_size_z;
    uint32_t subgroup_size;
};

struct WebGpuErrorScopeResult
{
    WGPUPopErrorScopeStatus status;
    WGPUErrorType type;
    bool completed;
    std::string message;
};

static void webgpu_error_scope_callback(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void* userdata1, void*)
{
    WebGpuErrorScopeResult* result = (WebGpuErrorScopeResult*)userdata1;
    result->status = status;
    result->type = type;
    result->completed = true;
    if (message.data)
        result->message.assign(message.data, message.length);
}

static int pop_webgpu_validation_error_scope(const VulkanDevice* vkdev, const char* operation)
{
    WebGpuErrorScopeResult result;
    result.status = WGPUPopErrorScopeStatus_Error;
    result.type = WGPUErrorType_Unknown;
    result.completed = false;

    WGPUPopErrorScopeCallbackInfo callback_info = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
    callback_info.callback = webgpu_error_scope_callback;
    callback_info.userdata1 = &result;

    WGPUFuture future = wgpuDevicePopErrorScope(vkdev->wgpu_device(), callback_info);
    WGPUFutureWaitInfo wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
    if (vkdev->wait_webgpu_future(future, &wait_info, operation) != 0
        || !result.completed
        || result.status != WGPUPopErrorScopeStatus_Success
        || result.type != WGPUErrorType_NoError)
    {
        NCNN_LOGE("WebGPU validation failed operation=%s status=%d type=%d %s",
                  operation, (int)result.status, (int)result.type, result.message.c_str());
        return -1;
    }

    return 0;
}

static int get_webgpu_override_value(const WebGpuOverrideInfo& override_info, const std::vector<vk_specialization_type>& specializations,
                                     uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z, double& value)
{
    if (override_info.is_workgroup_size)
    {
        if (override_info.spec_id == 233)
            value = local_size_x;
        else if (override_info.spec_id == 234)
            value = local_size_y;
        else if (override_info.spec_id == 235)
            value = local_size_z;
        else
            return -1;
        return 0;
    }

    if (override_info.ncnn_specialization_index < 0
        || (size_t)override_info.ncnn_specialization_index >= specializations.size())
        return -1;

    const vk_specialization_type& specialization = specializations[override_info.ncnn_specialization_index];
    if (override_info.type == NCNN_WEBGPU_SCALAR_BOOL)
    {
        if (specialization.u32 > 1)
            return -1;
        value = specialization.u32;
    }
    else if (override_info.type == NCNN_WEBGPU_SCALAR_I32)
    {
        value = specialization.i;
    }
    else if (override_info.type == NCNN_WEBGPU_SCALAR_U32)
    {
        value = specialization.u32;
    }
    else if (override_info.type == NCNN_WEBGPU_SCALAR_F32 || override_info.type == NCNN_WEBGPU_SCALAR_F16)
    {
        value = specialization.f;
    }
    else
    {
        return -1;
    }

    return 0;
}

static void release_webgpu_pipeline_bundle(WebGpuPipelineBundle& bundle)
{
    if (bundle.pipeline)
    {
        wgpuComputePipelineRelease(bundle.pipeline);
    }
    if (bundle.pipeline_layout)
    {
        wgpuPipelineLayoutRelease(bundle.pipeline_layout);
    }
    if (bundle.bind_group_layout)
    {
        wgpuBindGroupLayoutRelease(bundle.bind_group_layout);
    }
    if (bundle.shader_module)
    {
        wgpuShaderModuleRelease(bundle.shader_module);
    }

    bundle = WebGpuPipelineBundle();
}

int create_webgpu_pipeline_bundle(const VulkanDevice* vkdev, int shader_type_index, const Option& opt,
                                  const std::vector<vk_specialization_type>& specializations,
                                  uint32_t local_size_x, uint32_t local_size_y, uint32_t local_size_z,
                                  WebGpuPipelineBundle* bundle)
{
    if (!bundle || !vkdev || !vkdev->is_valid())
        return -1;

    *bundle = WebGpuPipelineBundle();

    WGPUDevice device = vkdev->wgpu_device();
    wgpuDevicePushErrorScope(device, WGPUErrorFilter_Validation);

    if (create_webgpu_shader_module(vkdev, shader_type_index, opt,
                                    local_size_x, local_size_y, local_size_z,
                                    &bundle->shader_module, &bundle->shader_info)
        != 0)
    {
        pop_webgpu_validation_error_scope(vkdev, "shader-module-error-scope");
        release_webgpu_pipeline_bundle(*bundle);
        return -1;
    }

    std::vector<WGPUBindGroupLayoutEntry> bind_group_layout_entries(bundle->shader_info.bindings.size());
    for (size_t i = 0; i < bundle->shader_info.bindings.size(); i++)
    {
        const WebGpuBindingInfo& binding = bundle->shader_info.bindings[i];
        WGPUBindGroupLayoutEntry& entry = bind_group_layout_entries[i];
        entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entry.binding = binding.binding;
        entry.visibility = WGPUShaderStage_Compute;
        entry.buffer = WGPU_BUFFER_BINDING_LAYOUT_INIT;
        entry.buffer.type = binding.access == NCNN_WEBGPU_BINDING_READ ? WGPUBufferBindingType_ReadOnlyStorage : WGPUBufferBindingType_Storage;
        entry.buffer.minBindingSize = binding.min_binding_size;
    }

    WGPUBindGroupLayoutDescriptor bind_group_layout_descriptor = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bind_group_layout_descriptor.entryCount = bind_group_layout_entries.size();
    bind_group_layout_descriptor.entries = bind_group_layout_entries.data();
    bundle->bind_group_layout = wgpuDeviceCreateBindGroupLayout(device, &bind_group_layout_descriptor);
    if (!bundle->bind_group_layout)
    {
        pop_webgpu_validation_error_scope(vkdev, "bind-group-layout-error-scope");
        release_webgpu_pipeline_bundle(*bundle);
        return -1;
    }

    WGPUPipelineLayoutDescriptor pipeline_layout_descriptor = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipeline_layout_descriptor.bindGroupLayoutCount = 1;
    pipeline_layout_descriptor.bindGroupLayouts = &bundle->bind_group_layout;
    pipeline_layout_descriptor.immediateSize = bundle->shader_info.immediate_size;
    bundle->pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_descriptor);
    if (!bundle->pipeline_layout)
    {
        pop_webgpu_validation_error_scope(vkdev, "pipeline-layout-error-scope");
        release_webgpu_pipeline_bundle(*bundle);
        return -1;
    }

    std::vector<std::string> constant_keys(bundle->shader_info.overrides.size());
    std::vector<WGPUConstantEntry> constants(bundle->shader_info.overrides.size());
    for (size_t i = 0; i < bundle->shader_info.overrides.size(); i++)
    {
        char key[16];
        sprintf(key, "%u", bundle->shader_info.overrides[i].spec_id);
        constant_keys[i] = key;

        constants[i] = WGPU_CONSTANT_ENTRY_INIT;
        constants[i].key.data = constant_keys[i].data();
        constants[i].key.length = constant_keys[i].size();
        if (get_webgpu_override_value(bundle->shader_info.overrides[i], specializations,
                                      local_size_x, local_size_y, local_size_z, constants[i].value)
            != 0)
        {
            NCNN_LOGE("WebGPU shader %d specialization id %u has no valid value", shader_type_index, bundle->shader_info.overrides[i].spec_id);
            pop_webgpu_validation_error_scope(vkdev, "pipeline-specialization-error-scope");
            release_webgpu_pipeline_bundle(*bundle);
            return -1;
        }
    }

    WGPUComputePipelineDescriptor pipeline_descriptor = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pipeline_descriptor.layout = bundle->pipeline_layout;
    pipeline_descriptor.compute.module = bundle->shader_module;
    const char entry_point[] = "main";
    pipeline_descriptor.compute.entryPoint.data = entry_point;
    pipeline_descriptor.compute.entryPoint.length = sizeof(entry_point) - 1;
    pipeline_descriptor.compute.constantCount = constants.size();
    pipeline_descriptor.compute.constants = constants.data();
    bundle->pipeline = wgpuDeviceCreateComputePipeline(device, &pipeline_descriptor);

    if (pop_webgpu_validation_error_scope(vkdev, "compute-pipeline-error-scope") != 0 || !bundle->pipeline)
    {
        release_webgpu_pipeline_bundle(*bundle);
        return -1;
    }

    return 0;
}

Pipeline::Pipeline(const VulkanDevice* _vkdev)
    : vkdev(_vkdev), d(new PipelinePrivate)
{
    d->bundle = WebGpuPipelineBundle();
    d->local_size_x = 1;
    d->local_size_y = 1;
    d->local_size_z = 1;
    d->subgroup_size = 0;
}

Pipeline::~Pipeline()
{
    release_webgpu_pipeline_bundle(d->bundle);

    delete d;
}

Pipeline::Pipeline(const Pipeline&)
    : vkdev(0), d(0)
{
}

Pipeline& Pipeline::operator=(const Pipeline&)
{
    return *this;
}

void Pipeline::set_optimal_local_size_xyz(int w, int h, int c)
{
    set_optimal_local_size_xyz(Mat(w, h, c, (void*)0));
}

void Pipeline::set_optimal_local_size_xyz(const Mat& local_size_xyz)
{
    int w = local_size_xyz.w;
    int h = local_size_xyz.h;
    int c = local_size_xyz.c;

    if (w == 0 && h == 0 && c == 0)
    {
        w = 4;
        h = 4;
        c = 4;
    }

    w = std::min(w, (int)vkdev->info.max_workgroup_size_x());
    h = std::min(h, (int)vkdev->info.max_workgroup_size_y());
    c = std::min(c, (int)vkdev->info.max_workgroup_size_z());

    while (w * h * c > (int)vkdev->info.max_workgroup_invocations())
    {
        if (w >= h && w >= c && w > 1)
            w /= 2;
        else if (h >= c && h > 1)
            h /= 2;
        else if (c > 1)
            c /= 2;
        else
            break;
    }

    set_local_size_xyz(w, h, c);
}

void Pipeline::set_local_size_xyz(int w, int h, int c)
{
    d->local_size_x = std::max(w, 1);
    d->local_size_y = std::max(h, 1);
    d->local_size_z = std::max(c, 1);
}

void Pipeline::set_subgroup_size(uint32_t subgroup_size)
{
    d->subgroup_size = subgroup_size;
}

int Pipeline::create(const uint32_t*, size_t, const std::vector<vk_specialization_type>&)
{
    return -1;
}

int Pipeline::create(int shader_type_index, const Option& opt, const std::vector<vk_specialization_type>& specializations)
{
    if (d->bundle.shader_module || d->bundle.bind_group_layout || d->bundle.pipeline_layout || d->bundle.pipeline)
        return -1;

    const uint64_t operation_id = begin_webgpu_sync_operation("pipeline-create");
    if (operation_id == 0)
        return -1;

    const int ret = vkdev->get_pipeline_cache()->get_pipeline(shader_type_index, opt, specializations,
                                                              d->local_size_x, d->local_size_y, d->local_size_z,
                                                              &d->bundle);
    return finish_webgpu_sync_operation(operation_id, ret);
}

const WebGpuShaderInfo& Pipeline::shader_info() const
{
    return d->bundle.shader_info;
}

const WebGpuPipelineBundle* Pipeline::webgpu_bundle() const
{
    return &d->bundle;
}

uint32_t Pipeline::local_size_x() const
{
    return d->local_size_x;
}

uint32_t Pipeline::local_size_y() const
{
    return d->local_size_y;
}

uint32_t Pipeline::local_size_z() const
{
    return d->local_size_z;
}


} // namespace ncnn

#endif // NCNN_WEBGPU
