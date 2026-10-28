// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_weight_storage.h"
#include "test_weight_storage_failure.h"

static int test_weight_storage_cache_failure(VulkanFailure::Failure failure)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkWeightStagingAllocator upload_staging(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &upload_staging;
    ncnn::VkTransfer upload(vkdev);
    ncnn::Mat input = RandomMat(256);
    ncnn::VkMat weight;
    upload.record_upload(input, weight, opt);
    if (upload.submit_and_wait() != 0 || weights.finalize_weights() != 0)
    {
        fprintf(stderr, "test_weight_storage_cache_failure prepare failed\n");
        return -1;
    }
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    ncnn::VkCompute cmd(vkdev, opt);
    ncnn::Mat output;
    {
        // Cache allocation now happens while encoding the first cached record.
        VulkanFailure fail(failure);
        cmd.record_clone(weight, output, opt);
        if (cmd.submit_and_wait() == 0 || !fail.triggered)
        {
            fprintf(stderr, "test_weight_storage_cache_failure error missing failure=%d\n", failure);
            return -1;
        }
    }
    if (cmd.reset() != 0)
    {
        fprintf(stderr, "test_weight_storage_cache_failure reset failed failure=%d\n", failure);
        return -1;
    }
    cmd.record_clone(weight, output, opt);
    if (cmd.submit_and_wait() != 0 || CompareMat(input, output, 0.f) != 0)
    {
        fprintf(stderr, "test_weight_storage_cache_failure recovery failed failure=%d\n", failure);
        return -1;
    }
    return 0;
}

static int test_weight_storage_transfer_failure(VulkanFailure::Failure failure)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    if (failure == VulkanFailure::SECOND_SUBMIT && vkdev->info.unified_compute_transfer_queue())
        return 0;
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkWeightStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    ncnn::VkTransfer cmd(vkdev);
    ncnn::VkMat weight;
    cmd.record_upload(RandomMat(256), weight, opt);
    {
        VulkanFailure fail(failure);
        if (cmd.submit_and_wait() == 0 || !fail.triggered || cmd.reset() == 0)
        {
            fprintf(stderr, "test_weight_storage_transfer_failure error missing failure=%d\n", failure);
            return -1;
        }
    }
    if (cmd.submit_and_wait() == 0)
    {
        fprintf(stderr, "test_weight_storage_transfer_failure lost error failure=%d\n", failure);
        return -1;
    }
    return 0;
}

static int test_weight_storage_midflight_failure(int fail_at)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    // Two sources fit per unified-cache segment; retain enough segments
    // to inject early, middle, and late submission failures.
    const int count = 24;
    ncnn::Mat input[count];
    ncnn::VkMat weight[count];
    ncnn::VkTransfer upload(vkdev);
    for (int i = 0; i < count; i++)
    {
        input[i] = RandomMat(1024);
        upload.record_upload(input[i], weight[i], opt);
    }
    if (upload.submit_and_wait() != 0 || weights.finalize_weights() != 0)
        return -1;
    opt.blob_vkallocator = &blobs;
    opt.weight_device_cache_size = 8192;
    opt.weight_host_cache_size = 256;
    ncnn::VkCompute cmd(vkdev, opt);
    ncnn::Mat output[count];
    for (int i = 0; i < count; i++)
        cmd.record_clone(weight[i], output[i], opt);
    {
        VulkanFailure fail(VulkanFailure::SUBMIT, fail_at);
        if (cmd.submit_and_wait() == 0 || !fail.triggered)
        {
            fprintf(stderr, "test_weight_storage_midflight_failure missing failure at submit %d\n", fail_at);
            return -1;
        }
    }
    if (cmd.reset() != 0)
        return -1;
    for (int i = 0; i < count; i++)
        cmd.record_clone(weight[i], output[i], opt);
    if (cmd.submit_and_wait() != 0)
        return -1;
    for (int i = 0; i < count; i++)
        if (CompareMat(input[i], output[i], 0.f) != 0)
            return -1;
    return 0;
}

static int test_weight_storage_resident_batches()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    UnmappedWeightAllocator weights(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    weights.set_weight_memory_limit(65536, 65536, 8192, 4096);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    opt.weight_host_memory_limit = 1024;
    ncnn::Mat input[24];
    ncnn::VkMat weight[24];
    ncnn::VkTransfer upload(vkdev);
    {
        VulkanFailure count(VulkanFailure::SUBMIT, 1000000);
        for (int i = 0; i < 24; i++)
        {
            input[i] = RandomMat(i % 3 == 0 ? 4 : i % 3 == 1 ? 256
                                                             : 1024);
            upload.record_upload(input[i], weight[i], opt);
        }
        if (count.submission_count() != 0 || weight[0].empty() || weight[0].data->storage_type != 0)
            return -1;
        if (upload.submit_and_wait() || count.submission_count() < 2 || count.submission_count() > 100)
        {
            fprintf(stderr, "bounded resident upload did not batch weights: %d submits\n", count.submission_count());
            return -1;
        }
    }
    weight[2].release();
    if (weights.finalize_weights() || upload.reset())
        return -1;
    opt.blob_vkallocator = &blobs;
    ncnn::VkCompute cmd(vkdev);
    ncnn::Mat output[24];
    for (int i = 0; i < 24; i++)
        if (i != 2)
            cmd.record_clone(weight[i], output[i], opt);
    if (cmd.submit_and_wait())
        return -1;
    for (int i = 0; i < 24; i++)
        if (i != 2 && CompareMat(input[i], output[i], 0.f))
            return -1;
    return 0;
}

static int test_weight_storage_resident_failure()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    UnmappedWeightAllocator weights(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    weights.set_weight_memory_limit(16384, 16384, 4096, 4096);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.weight_host_memory_limit = 1024;
    ncnn::VkMat weight;
    ncnn::VkTransfer upload(vkdev);
    upload.record_upload(RandomMat(2048), weight, opt);
    if (upload.error() || weight.empty() || weight.data->storage_type != 0)
        return -1;
    VulkanFailure fail(VulkanFailure::SECOND_SUBMIT);
    if (upload.submit_and_wait() == 0 || !fail.triggered || upload.error() == 0 || upload.reset() == 0)
    {
        fprintf(stderr, "resident upload midflight failure was lost\n");
        return -1;
    }
    return 0;
}

static int test_weight_storage_cached_batch()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    opt.weight_device_cache_size = 16384;
    opt.weight_host_cache_size = 4096;
    ncnn::Mat input[48];
    ncnn::VkMat weight[48];
    ncnn::VkTransfer upload(vkdev);
    for (int i = 0; i < 48; i++)
    {
        input[i] = RandomMat(64);
        upload.record_upload(input[i], weight[i], opt);
    }
    if (upload.submit_and_wait() != 0 || weights.finalize_weights() != 0)
        return -1;
    opt.blob_vkallocator = &blobs;
    ncnn::VkCompute cmd(vkdev, opt);
    ncnn::Mat output[96];
    for (int round = 0; round < 2; round++)
    {
        for (int i = 0; i < 96; i++)
            cmd.record_clone(weight[i < 48 ? i : 95 - i], output[i], opt);
        {
            VulkanFailure count(VulkanFailure::SUBMIT, 1000000);
            if (cmd.submit_and_wait() != 0)
                return -1;
            // reset discards content hits; cold unified fills still share one compute submission.
            const bool unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
            if (unified && count.submission_count() != 1)
            {
                fprintf(stderr, "cached weights split into %d submissions round=%d unified=%d\n", count.submission_count(), round, unified);
                return -1;
            }
        }
        for (int i = 0; i < 96; i++)
            if (CompareMat(input[i < 48 ? i : 95 - i], output[i], 0.f) != 0)
                return -1;
        if (cmd.reset() != 0)
            return -1;
    }
    return 0;
}

// A missing first binding must not evict a later binding of the same dispatch.
// Subsequent dispatches reuse occupied ranges while earlier readers are pending.
static int test_weight_storage_dispatch_pinning()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    opt.weight_device_cache_size = 3072;
    opt.weight_host_cache_size = 2048;
    ncnn::Mat input[6];
    ncnn::VkMat weight[6];
    ncnn::VkTransfer upload(vkdev);
    for (int i = 0; i < 6; i++)
    {
        input[i] = RandomMat(256);
        upload.record_upload(input[i], weight[i], opt);
    }
    if (upload.submit_and_wait() || weights.finalize_weights())
        return -1;
    opt.blob_vkallocator = &blobs;
    const char shader[] = "#version 450\n"
                          "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
                          "layout(binding = 0) readonly buffer A { float a[]; };\n"
                          "layout(binding = 1) readonly buffer B { float b[]; };\n"
                          "layout(binding = 2) readonly buffer C { float c[]; };\n"
                          "layout(binding = 3) writeonly buffer D { float d[]; };\n"
                          "void main() { uint i = gl_GlobalInvocationID.x; if (i < 256) d[i] = a[i] + b[i] + c[i]; }\n";
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(shader, opt, spirv))
        return -1;
    ncnn::Pipeline pipeline(vkdev);
    pipeline.set_local_size_xyz(1, 1, 1);
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>()))
        return -1;
    ncnn::VkCompute cmd(vkdev, opt);
    const int indices[8][3] = {{3, 0, 1}, {4, 1, 3}, {5, 3, 4}, {0, 4, 5}, {1, 5, 0}, {2, 0, 1}, {3, 1, 2}, {0, 2, 3}};
    for (int round = 0; round < 2; round++)
    {
        ncnn::Mat warm[3];
        for (int i = 0; i < 3; i++)
            cmd.record_clone(weight[i], warm[i], opt);
        if (cmd.submit_and_wait() || cmd.reset())
            return -1;
        for (int i = 0; i < 3; i++)
            if (CompareMat(input[i], warm[i], 0.f))
                return -1;
        ncnn::Mat output[8];
        ncnn::VkMat results[8];
        for (int dispatch = 0; dispatch < 8; dispatch++)
        {
            ncnn::VkMat& result = results[dispatch];
            result.create(256, (size_t)4u, &blobs);
            if (result.empty())
                return -1;
            std::vector<ncnn::VkMat> bindings(4);
            for (int i = 0; i < 3; i++)
                bindings[i] = weight[indices[dispatch][i]];
            bindings[3] = result;
            cmd.record_pipeline(&pipeline, bindings, std::vector<ncnn::vk_constant_type>(), result);
            cmd.record_clone(result, output[dispatch], opt);
        }
        if (cmd.submit_and_wait())
        {
            fprintf(stderr, "dispatch pinning submit failed round=%d\n", round);
            return -1;
        }
        for (int dispatch = 0; dispatch < 8; dispatch++)
        {
            ncnn::Mat expected(256);
            for (int i = 0; i < 256; i++)
                ((float*)expected)[i] = ((const float*)input[indices[dispatch][0]])[i] + ((const float*)input[indices[dispatch][1]])[i] + ((const float*)input[indices[dispatch][2]])[i];
            if (CompareMat(expected, output[dispatch], 0.0001f))
            {
                fprintf(stderr, "dispatch pinning corrupted pending reader round=%d dispatch=%d\n", round, dispatch);
                return -1;
            }
        }
        if (cmd.reset())
            return -1;
    }
    return 0;
}

// Trace actual Vulkan encoding, not merely CPU-side recording metadata.
class WeightRecordingProbe;
static WeightRecordingProbe* g_weight_recording_probe = 0;
class WeightRecordingProbe
{
public:
    explicit WeightRecordingProbe(const ncnn::VulkanDevice* vkdev)
        : copies(0), dispatches(0), submissions(0), destroyed_recorded_buffer(false), allocations(0), device_bytes(0), host_bytes(0), peak_device_bytes(0), peak_host_bytes(0), copy(ncnn::vkCmdCopyBuffer), dispatch(ncnn::vkCmdDispatch), submit(ncnn::vkQueueSubmit), destroy(ncnn::vkDestroyBuffer), allocate(ncnn::vkAllocateMemory), free(ncnn::vkFreeMemory), properties(vkdev->info.physicalDeviceMemoryProperties())
    {
        g_weight_recording_probe = this;
        ncnn::vkCmdCopyBuffer = record_copy;
        ncnn::vkCmdDispatch = record_dispatch;
        ncnn::vkQueueSubmit = queue_submit;
        ncnn::vkDestroyBuffer = destroy_buffer;
        ncnn::vkAllocateMemory = allocate_memory;
        ncnn::vkFreeMemory = free_memory;
    }
    ~WeightRecordingProbe()
    {
        ncnn::vkCmdCopyBuffer = copy;
        ncnn::vkCmdDispatch = dispatch;
        ncnn::vkQueueSubmit = submit;
        ncnn::vkDestroyBuffer = destroy;
        ncnn::vkAllocateMemory = allocate;
        ncnn::vkFreeMemory = free;
        g_weight_recording_probe = 0;
    }
    int copies;
    int dispatches;
    int submissions;
    bool destroyed_recorded_buffer;
    int allocations;
    VkDeviceSize device_bytes;
    VkDeviceSize host_bytes;
    VkDeviceSize peak_device_bytes;
    VkDeviceSize peak_host_bytes;

private:
    struct Memory
    {
        VkDeviceMemory handle;
        VkDeviceSize size;
        bool device_local;
    };
    WeightRecordingProbe(const WeightRecordingProbe&);
    WeightRecordingProbe& operator=(const WeightRecordingProbe&);
    static VKAPI_ATTR void VKAPI_CALL record_copy(VkCommandBuffer command, VkBuffer src, VkBuffer dst, uint32_t count, const VkBufferCopy* regions)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        p->copies++;
        p->buffers.push_back(src);
        p->buffers.push_back(dst);
        p->copy(command, src, dst, count, regions);
    }
    static VKAPI_ATTR void VKAPI_CALL record_dispatch(VkCommandBuffer command, uint32_t x, uint32_t y, uint32_t z)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        p->dispatches++;
        p->dispatch(command, x, y, z);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, uint32_t count, const VkSubmitInfo* info, VkFence fence)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        p->submissions += count;
        return p->submit(queue, count, info, fence);
    }
    static VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* callbacks)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        for (size_t i = 0; i < p->buffers.size(); i++)
            if (p->buffers[i] == buffer)
                p->destroyed_recorded_buffer = true;
        p->destroy(device, buffer, callbacks);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(VkDevice device, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        const VkResult ret = p->allocate(device, info, callbacks, memory);
        if (ret == VK_SUCCESS)
        {
            Memory entry;
            entry.handle = *memory;
            entry.size = info->allocationSize;
            const uint32_t heap = p->properties.memoryTypes[info->memoryTypeIndex].heapIndex;
            entry.device_local = (p->properties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            p->memory.push_back(entry);
            p->allocations++;
            VkDeviceSize& bytes = entry.device_local ? p->device_bytes : p->host_bytes;
            VkDeviceSize& peak = entry.device_local ? p->peak_device_bytes : p->peak_host_bytes;
            bytes += entry.size;
            if (bytes > peak)
                peak = bytes;
        }
        return ret;
    }
    static VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        for (size_t i = 0; i < p->memory.size(); i++)
        {
            Memory& entry = p->memory[i];
            if (entry.handle == memory)
            {
                VkDeviceSize& bytes = entry.device_local ? p->device_bytes : p->host_bytes;
                bytes -= entry.size;
                entry.handle = 0;
                entry.size = 0;
                break;
            }
        }
        p->free(device, memory, callbacks);
    }
    PFN_vkCmdCopyBuffer copy;
    PFN_vkCmdDispatch dispatch;
    PFN_vkQueueSubmit submit;
    PFN_vkDestroyBuffer destroy;
    PFN_vkAllocateMemory allocate;
    PFN_vkFreeMemory free;
    VkPhysicalDeviceMemoryProperties properties;
    std::vector<Memory> memory;
    std::vector<VkBuffer> buffers;
};

static int test_weight_storage_record_time_encoding()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    const int elements = 256 * 1024;
    const size_t mib = 1024 * 1024;
    opt.weight_device_cache_size = 4 * mib;
    opt.weight_host_cache_size = mib / 4;
    opt.weight_device_memory_limit = 8 * mib;
    opt.weight_host_memory_limit = 8 * mib;
    ncnn::Mat input[4];
    ncnn::VkMat weight[4];
    ncnn::VkTransfer upload(vkdev);
    for (int i = 0; i < 4; i++)
    {
        input[i].create(elements);
        input[i].fill((float)i + 0.25f);
        upload.record_upload(input[i], weight[i], opt);
    }
    if (upload.submit_and_wait() || weights.finalize_weights())
        return -1;
    opt.blob_vkallocator = &blobs;
    const char shader[] = "#version 450\n"
                          "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
                          "layout(binding = 0) readonly buffer A { float a[]; };\n"
                          "layout(binding = 1) readonly buffer B { float b[]; };\n"
                          "layout(binding = 2) readonly buffer C { float c[]; };\n"
                          "layout(binding = 3) writeonly buffer D { float d[]; };\n"
                          "void main() { uint i = gl_GlobalInvocationID.x; if (i < 262144) d[i] = a[i] + b[i] + c[i]; }\n";
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(shader, opt, spirv))
        return -1;
    ncnn::Pipeline pipeline(vkdev);
    pipeline.set_local_size_xyz(64, 1, 1);
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>()))
        return -1;
    ncnn::Mat expected(elements);
    expected.fill(6.75f);
    ncnn::VkMat result(elements, (size_t)4u, &blobs);
    if (result.empty())
        return -1;
    std::vector<ncnn::VkMat> bindings(4);
    for (int i = 0; i < 3; i++)
        bindings[i] = weight[i + 1];
    bindings[3] = result;
    // Keep caller-owned GPU targets alive until after command destruction,
    // including a failed submit whose work still needs to be drained.
    ncnn::VkMat ordinary_devices[2];
    ncnn::VkMat first_devices[2];
    ncnn::VkCompute cmd(vkdev, opt);
    for (int round = 0; round < 2; round++)
    {
        ncnn::Mat ordinary(64);
        ordinary.fill(2.f);
        ncnn::Mat ordinary_expected = ordinary.clone();
        ncnn::Mat ordinary_output;
        ncnn::Mat first_output;
        ncnn::Mat output;
        ncnn::VkMat& ordinary_device = ordinary_devices[round];
        ncnn::VkMat& first_device = first_devices[round];
        {
            WeightRecordingProbe probe(vkdev);
            bool cached_copy_encoded = false;
            if (round == 1)
            {
                const int before = probe.copies;
                cmd.record_clone(weight[0], first_device, opt);
                cached_copy_encoded = probe.copies > before;
            }
            cmd.record_clone(ordinary, ordinary_device, opt);
            ordinary.fill(-1.f);
            cmd.record_clone(ordinary_device, ordinary_output, opt);
            if (round == 0)
            {
                const int before = probe.copies;
                cmd.record_clone(weight[0], first_device, opt);
                cached_copy_encoded = probe.copies > before;
            }
            if (!cached_copy_encoded || probe.submissions != 0)
            {
                fprintf(stderr, "cached clone was not encoded at record time, or submitted while recording\n");
                return -1;
            }
            cmd.record_clone(first_device, first_output, opt);
            const int before_dispatch = probe.dispatches;
            cmd.record_pipeline(&pipeline, bindings, std::vector<ncnn::vk_constant_type>(), result);
            if (probe.dispatches != before_dispatch + 1 || probe.submissions != 0 || probe.destroyed_recorded_buffer)
            {
                fprintf(stderr, "cached dispatch encoding/lifetime failed before submission round=%d\n", round);
                return -1;
            }
            cmd.record_clone(result, output, opt);
            const int copies = probe.copies;
            const int dispatches = probe.dispatches;
            if (cmd.submit_and_wait() || probe.copies != copies || probe.dispatches != dispatches || probe.submissions == 0)
            {
                fprintf(stderr, "cached submit re-encoded commands or failed round=%d\n", round);
                return -1;
            }
        }
        if (CompareMat(ordinary_expected, ordinary_output, 0.f) || CompareMat(input[0], first_output, 0.f) || CompareMat(expected, output, 0.f) || cmd.reset())
            return -1;
    }
    // Insufficient fixed capacity must reject before submission, without
    // discarding backing already referenced by an earlier encoded command.
    ncnn::Option small_opt = opt;
    small_opt.weight_device_cache_size = 2 * mib;
    ncnn::VkMat first_device;
    ncnn::VkCompute small(vkdev, small_opt);
    {
        WeightRecordingProbe probe(vkdev);
        small.record_clone(weight[0], first_device, small_opt);
        if (probe.copies == 0)
            return -1;
        small.record_pipeline(&pipeline, bindings, std::vector<ncnn::vk_constant_type>(), result);
        if (small.submit_and_wait() == 0 || probe.submissions != 0 || probe.destroyed_recorded_buffer)
        {
            fprintf(stderr, "insufficient cache submitted work or destroyed recorded backing\n");
            return -1;
        }
    }
    if (small.reset())
        return -1;
    ncnn::Mat recovered;
    small.record_clone(weight[0], recovered, small_opt);
    if (small.submit_and_wait() || CompareMat(input[0], recovered, 0.f))
        return -1;
    return 0;
}

// Count actual backing allocations (including Vulkan padding), not capacities.
// Caller outputs and ordinary transfer staging are prepared outside the probe.
static int test_weight_storage_automatic_allocation_budget()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    ncnn::Mat input = RandomMat(1024);
    ncnn::VkMat weight;
    ncnn::VkTransfer upload(vkdev);
    upload.record_upload(input, weight, opt);
    if (upload.submit_and_wait() || weights.finalize_weights())
        return -1;
    opt.blob_vkallocator = &blobs;
    // Derive a modest feasible bound from actual buffer requirements rather
    // than assuming every driver can allocate cache backing within 64 KiB.
    size_t capacity = std::max((size_t)65536, vkdev->info.buffer_offset_alignment());
    capacity = std::max(capacity, vkdev->info.non_coherent_atom_size());
    if (vkdev->info.support_VK_KHR_robustness2() || vkdev->info.support_VK_EXT_robustness2())
        capacity = std::max(capacity, (size_t)vkdev->info.queryRobustness2Properties().robustStorageBufferAccessSizeAlignment);
    size_t limit = capacity;
    const uint32_t families[2] = {vkdev->info.compute_queue_family_index(), vkdev->info.transfer_queue_family_index()};
    for (int sharing = 0; sharing < (families[0] == families[1] ? 1 : 2); sharing++)
    {
        VkBufferCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size = capacity;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = sharing ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
        info.queueFamilyIndexCount = sharing ? 2 : 0;
        info.pQueueFamilyIndices = sharing ? families : 0;
        VkBuffer buffer = 0;
        if (ncnn::vkCreateBuffer(vkdev->vkdevice(), &info, 0, &buffer) != VK_SUCCESS)
            return -1;
        VkMemoryRequirements requirements;
        ncnn::vkGetBufferMemoryRequirements(vkdev->vkdevice(), buffer, &requirements);
        ncnn::vkDestroyBuffer(vkdev->vkdevice(), buffer, 0);
        if (requirements.size > (size_t)-1)
            return -1;
        limit = std::max(limit, (size_t)requirements.size);
    }
    opt.weight_device_memory_limit = limit;
    opt.weight_host_memory_limit = limit;
    // Zero explicit capacities request bounded automatic defaults.
    opt.weight_device_cache_size = 0;
    opt.weight_host_cache_size = 0;
    ncnn::VkMat native(1024, (size_t)4u, &blobs);
    ncnn::VkMat target(1024, (size_t)4u, &blobs);
    if (native.empty() || target.empty())
        return -1;
    {
        ncnn::VkCompute prepare(vkdev);
        prepare.record_clone(input, native, opt);
        if (prepare.submit_and_wait())
            return -1;
    }
    {
        WeightRecordingProbe probe(vkdev);
        ncnn::VkCompute ordinary(vkdev, opt);
        ordinary.record_clone(native, target, opt);
        if (ordinary.submit_and_wait() || probe.allocations != 0)
        {
            fprintf(stderr, "ordinary preallocated copy allocated a weight cache\n");
            return -1;
        }
    }
    {
        WeightRecordingProbe probe(vkdev);
        ncnn::VkCompute cached(vkdev, opt);
        cached.record_clone(weight, target, opt);
        if (cached.submit_and_wait() || probe.allocations == 0 || probe.peak_device_bytes > limit || probe.peak_host_bytes > limit)
        {
            fprintf(stderr, "automatic cache exceeded actual backing budgets device=%llu host=%llu limit=%zu\n", (unsigned long long)probe.peak_device_bytes, (unsigned long long)probe.peak_host_bytes, limit);
            return -1;
        }
        if ((vkdev->info.type() == 1 || vkdev->info.type() == 3) && probe.peak_host_bytes != 0)
        {
            fprintf(stderr, "unified weight cache unexpectedly allocated host backing\n");
            return -1;
        }
    }
    ncnn::VkCompute download(vkdev);
    ncnn::Mat output;
    download.record_clone(target, output, opt);
    return download.submit_and_wait() || CompareMat(input, output, 0.f);
}

class CountingWeightStagingAllocator : public ncnn::VkStagingAllocator
{
public:
    explicit CountingWeightStagingAllocator(const ncnn::VulkanDevice* vkdev)
        : ncnn::VkStagingAllocator(vkdev), frees(0)
    {
    }
    virtual void fastFree(ncnn::VkBufferMemory* memory)
    {
        frees++;
        ncnn::VkStagingAllocator::fastFree(memory);
    }
    int frees;
};

static int test_weight_storage_transfer_wait_failure()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    UnmappedWeightAllocator weights(vkdev);
    CountingWeightStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.staging_vkallocator = &staging;
    opt.blob_vkallocator = &weights;
    ncnn::Mat input = RandomMat(256);
    ncnn::VkMat target;
    {
        VulkanFailure fail(VulkanFailure::FENCE_WAIT, 3);
        {
            ncnn::VkTransfer upload(vkdev);
            upload.record_upload(input, target, opt);
            if (upload.submit_and_wait() == 0 || !fail.triggered || staging.frees != 0 || upload.reset() == 0)
            {
                fprintf(stderr, "failed fence wait prematurely released transfer staging\n");
                return -1;
            }
        }
        // The destructor retries two more transient errors before the real wait.
    }
    return staging.frees == 1 ? 0 : -1;
}

static int test_weight_storage_cache_wait_failure(bool cached_weight)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    CountingWeightStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    ncnn::VkTransfer upload(vkdev);
    ncnn::VkMat weight;
    upload.record_upload(RandomMat(256), weight, opt);
    if (upload.submit_and_wait() != 0 || weights.finalize_weights() != 0)
        return -1;
    opt.blob_vkallocator = &blobs;
    ncnn::Mat output;
    ncnn::VkCompute* cmd = new ncnn::VkCompute(vkdev, opt);
    ncnn::VkMat ordinary;
    ncnn::Mat ordinary_input = RandomMat(256);
    if (cached_weight)
        cmd->record_clone(weight, output, opt);
    else
    {
        cmd->record_clone(ordinary_input, ordinary, opt);
        cmd->record_clone(ordinary, output, opt);
    }
    const int frees_before = staging.frees;
    VulkanFailure fail(VulkanFailure::FENCE_STATUS, 3);
    const int result = cmd->submit_and_wait();
    const bool retained = result != 0 && fail.triggered && cmd->reset() != 0 && staging.frees == frees_before;
    // Keep injection active during destruction: repeated query errors must not
    // release source/destination/staging resources before a real completed wait.
    delete cmd;
    if (!retained || fail.status_errors != 3 || staging.frees <= frees_before)
    {
        fprintf(stderr, "compute fence query failure lost pending execution or staging lifetime cached=%d\n", cached_weight);
        return -1;
    }
    return 0;
}

static int test_weight_storage_execution_resources(VulkanFailure::Failure failure)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    const int count = 192;
    ncnn::Mat input[count];
    ncnn::Mat output[count];
    ncnn::VkMat weight[count];
    ncnn::VkTransfer upload(vkdev);
    for (int i = 0; i < count; i++)
    {
        input[i] = RandomMat(64);
        upload.record_upload(input[i], weight[i], opt);
    }
    if (upload.submit_and_wait() != 0 || weights.finalize_weights() != 0)
        return -1;
    // Exercise resident-to-cache transfers even on unified-memory test devices.
    std::vector<int> storage_types(count);
    for (int i = 0; i < count; i++)
    {
        storage_types[i] = weight[i].data->storage_type;
        weight[i].data->storage_type = 1;
        weight[i].data->source_size = input[i].total() * input[i].elemsize;
    }
    opt.blob_vkallocator = &blobs;
    opt.weight_device_cache_size = 16384;
    int ret = 0;
    {
        ncnn::VkCompute cmd(vkdev, opt);
        {
            // Command/fence/semaphore allocation is part of record-time encoding.
            VulkanFailure check(failure, 5);
            for (int i = 0; i < count; i++)
                cmd.record_clone(weight[i], output[i], opt);
            ret = cmd.submit_and_wait();
            if (failure == VulkanFailure::TRACK_COMMANDS)
            {
                if (ret != 0 || check.reused_command || check.resets != 0 || check.submission_count() <= count + 4)
                {
                    fprintf(stderr, "execution resources reused a submitted command or insufficient task coverage count=%d reused=%d resets=%d\n", check.submission_count(), check.reused_command, check.resets);
                    ret = -1;
                }
            }
            else
            {
                ret = ret != 0 && check.triggered ? 0 : -1;
            }
        }
        if (ret == 0 && failure == VulkanFailure::TRACK_COMMANDS)
            for (int i = 0; i < count; i++)
                if (CompareMat(input[i], output[i], 0.f) != 0)
                    ret = -1;
        if (cmd.reset() != 0)
            ret = -1;
    }
    for (int i = 0; i < count; i++)
    {
        weight[i].data->storage_type = storage_types[i];
        if (storage_types[i] == 0)
            weight[i].data->source_size = 0;
    }
    return ret;
}

int main()
{
    SRAND(7767517);
    if (ncnn::get_gpu_count() == 0)
        return 0;

    WeightStorageTestDirectory directory;
    if (directory.create() != 0)
    {
        fprintf(stderr, "could not create a writable test-local weight directory\n");
        return 1;
    }

    const int ret = 0
                    || test_weight_storage_transfer_wait_failure()
                    || test_weight_storage_cache_wait_failure(true)
                    || test_weight_storage_cache_wait_failure(false)
                    || test_weight_storage_execution_resources(VulkanFailure::TRACK_COMMANDS)
                    || test_weight_storage_execution_resources(VulkanFailure::ALLOCATE_COMMAND)
                    || test_weight_storage_execution_resources(VulkanFailure::CREATE_FENCE)
                    || test_weight_storage_execution_resources(VulkanFailure::CREATE_SEMAPHORE)
                    || test_weight_storage_cache_failure(VulkanFailure::ALLOCATE_MEMORY)
                    || test_weight_storage_cache_failure(VulkanFailure::BIND_MEMORY)
                    || test_weight_storage_cache_failure(VulkanFailure::MAP_MEMORY)
                    || test_weight_storage_cache_failure(VulkanFailure::SUBMIT)
                    || test_weight_storage_transfer_failure(VulkanFailure::SUBMIT)
                    || test_weight_storage_transfer_failure(VulkanFailure::SECOND_SUBMIT)
                    || test_weight_storage_cached_batch()
                    || test_weight_storage_dispatch_pinning()
                    || test_weight_storage_record_time_encoding()
                    || test_weight_storage_automatic_allocation_budget()
                    || test_weight_storage_resident_batches()
                    || test_weight_storage_resident_failure()
                    || test_weight_storage_midflight_failure(2)
                    || test_weight_storage_midflight_failure(5)
                    || test_weight_storage_midflight_failure(9);
    if (directory.remove() != 0)
    {
        fprintf(stderr, "weight storage temporary files leaked after tests\n");
        return 1;
    }
    return ret;
}
