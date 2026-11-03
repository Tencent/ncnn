// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_weight_storage.h"
#include "test_weight_storage_failure.h"

static int test_weight_storage_allocator_failure(VulkanFailure::Failure failure)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator weights(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    if (weights.set_weight_memory_limit(16384, 16384))
        return -1;
    {
        VulkanFailure fail(failure);
        ncnn::VkMat rejected(256, (size_t)4u, &weights);
        if (!rejected.empty() || !fail.triggered)
        {
            fprintf(stderr, "resident allocation failure was not returned to the caller\n");
            return -1;
        }
    }
    uint64_t device_bytes, host_bytes;
    weights.get_weight_memory_usage(device_bytes, host_bytes);
    if (device_bytes || host_bytes)
        return -1;
    // One failed allocation does not disable later independent allocations.
    ncnn::VkMat weight(256, (size_t)4u, &weights);
    if (weight.empty() || weight.data->storage_type != 0 || weights.finalize_weights())
        return -1;
    return 0;
}

static int test_weight_storage_staging_allocator_failure(VulkanFailure::Failure failure)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightStagingAllocator staging(vkdev);
    {
        VulkanFailure fail(failure);
        ncnn::VkMat rejected(256, (size_t)4u, &staging);
        if (!rejected.empty() || !fail.triggered)
        {
            fprintf(stderr, "weight staging allocation failure was not returned failure=%d\n", failure);
            return -1;
        }
    }
    ncnn::VkMat buffer(256, (size_t)4u, &staging);
    if (buffer.empty() || !buffer.mapped_ptr())
        return -1;
    memset(buffer.mapped_ptr(), 0, buffer.buffer_capacity());
    return staging.flush(buffer.data);
}

static int test_weight_storage_workspace_failure(VulkanFailure::Failure failure)
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
        fprintf(stderr, "test_weight_storage_workspace_failure prepare failed\n");
        return -1;
    }
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    ncnn::Mat output;
    {
        // Failures during streaming recording stop the current void operation.
        VulkanFailure fail(failure);
        cmd.record_clone(weight, output, opt);
        const bool failed = output.empty();
        if (!failed || !fail.triggered)
        {
            fprintf(stderr, "test_weight_storage_workspace_failure error missing failure=%d\n", failure);
            return -1;
        }
    }
    if (cmd.reset() != 0)
    {
        fprintf(stderr, "test_weight_storage_workspace_failure reset failed failure=%d\n", failure);
        return -1;
    }
    cmd.record_clone(weight, output, opt);
    if (cmd.submit_and_wait() != 0 || CompareMat(input, output, 0.f) != 0)
    {
        fprintf(stderr, "test_weight_storage_workspace_failure recovery failed failure=%d\n", failure);
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
        const int ret = cmd.submit_and_wait();
        // A failed second submit can leave the upload in flight. Drain it in
        // the test before destroying the command and its staging allocations.
        ncnn::vkDeviceWaitIdle(vkdev->vkdevice());
        if (ret == 0 || !fail.triggered)
        {
            fprintf(stderr, "test_weight_storage_transfer_failure error missing failure=%d\n", failure);
            return -1;
        }
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
    // Retain enough compute batches to inject early, middle and late failures.
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

    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    ncnn::Mat output[count];
    {
        VulkanFailure fail(VulkanFailure::SUBMIT, fail_at);
        for (int i = 0; i < count && !fail.triggered; i++)
            cmd.record_clone(weight[i], output[i], opt);
        if (!fail.triggered)
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
    if (weights.set_weight_memory_limit(65536, 65536))
        return -1;
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    ncnn::Mat input[24];
    ncnn::VkMat weight[24];
    ncnn::VkTransfer upload(vkdev);
    {
        VulkanFailure count(VulkanFailure::SUBMIT, 1000000);
        for (int i = 0; i < 24; i++)
        {
            input[i] = RandomMat(i % 3 == 0 ? 4 : i % 3 == 1 ? 256 : 1024);
            upload.record_upload(input[i], weight[i], opt);
        }
        if (count.submission_count() != 0 || upload.pending_upload_total() == 0 || weight[0].empty() || weight[0].data->storage_type != 0)
            return -1;
        if (upload.submit_and_wait() || count.submission_count() != (vkdev->info.unified_compute_transfer_queue() ? 1 : 2))
        {
            fprintf(stderr, "resident weights did not share one upload batch: %d submits\n", count.submission_count());
            return -1;
        }
    }
    weight[2].release();
    if (weights.finalize_weights() || upload.reset())
        return -1;
    opt.blob_vkallocator = &blobs;
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
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
    ncnn::VkWeightStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    if (weights.set_weight_memory_limit(16384, 16384))
        return -1;
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    ncnn::VkMat weight;
    ncnn::VkTransfer upload(vkdev);
    VulkanFailure fail(VulkanFailure::SUBMIT);
    upload.record_upload(RandomMat(2048), weight, opt);
    if (weight.empty() || weight.data->storage_type != 0 || fail.triggered)
        return -1;
    if (upload.submit_and_wait() == 0 || !fail.triggered || fail.submission_count() != 1)
    {
        fprintf(stderr, "resident upload submission failure was lost\n");
        return -1;
    }
    return 0;
}

static int test_weight_storage_repeated_weights()
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

    ncnn::Mat input[48];
    ncnn::VkMat weight[48];
    ncnn::VkTransfer upload(vkdev);
    // Mix sizes so the workspace repeatedly splits and merges returned ranges.
    const int sizes[] = {64, 256, 1024, 4096, 128, 2048};
    for (int i = 0; i < 48; i++)
    {
        input[i] = RandomMat(sizes[i % 6]);
        upload.record_upload(input[i], weight[i], opt);
    }
    if (upload.submit_and_wait() != 0 || weights.finalize_weights() != 0)
        return -1;
    opt.blob_vkallocator = &blobs;
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    ncnn::Mat output[96];
    for (int round = 0; round < 2; round++)
    {
        {
            VulkanFailure count(VulkanFailure::SUBMIT, 1000000);
            for (int i = 0; i < 96; i++)
                cmd.record_clone(weight[i < 48 ? i : 95 - i], output[i], opt);
            if (cmd.submit_and_wait() != 0)
                return -1;
            // Repeated sources are uploaded again in ordered compute batches.
            const bool unified = vkdev->info.type() == 1 || vkdev->info.type() == 3;
            if (unified && count.submission_count() < 2)
            {
                fprintf(stderr, "repeated weights did not record multiple compute batches round=%d\n", round);
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

// Later resolves must not evict a binding already resolved for this dispatch.
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
                          "layout(push_constant) uniform parameter { float scale; float bias; } p;\n"
                          "void main() { uint i = gl_GlobalInvocationID.x; if (i < 256) d[i] = (a[i] + b[i] + c[i]) * p.scale + p.bias; }\n";
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(shader, opt, spirv))
        return -1;
    ncnn::Pipeline pipeline(vkdev);
    pipeline.set_local_size_xyz(1, 1, 1);
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>()))
        return -1;
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    const int indices[8][3] = {{3, 0, 1}, {3, 4, 1}, {5, 3, 4}, {0, 4, 5}, {1, 5, 0}, {2, 0, 1}, {3, 1, 2}, {0, 2, 3}};
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
        // Keep an earlier copy pending while resolving all dispatch bindings.
        cmd.record_clone(weight[0], warm[0], opt);
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
            std::vector<ncnn::vk_constant_type> constants(2);
            constants[0].f = 0.5f * (dispatch + 1);
            constants[1].f = (float)round;
            cmd.record_pipeline(&pipeline, bindings, constants, result);
            // Later edits to caller-owned parameters must not affect replay.
            constants[0].f = -100.f;
            constants[1].f = -100.f;
            cmd.record_clone(result, output[dispatch], opt);
        }
        if (cmd.submit_and_wait())
        {
            fprintf(stderr, "dispatch pinning submit failed round=%d\n", round);
            return -1;
        }
        if (CompareMat(input[0], warm[0], 0.f))
            return -1;
        for (int dispatch = 0; dispatch < 8; dispatch++)
        {
            ncnn::Mat expected(256);
            for (int i = 0; i < 256; i++)
                ((float*)expected)[i] = (((const float*)input[indices[dispatch][0]])[i] + ((const float*)input[indices[dispatch][1]])[i] + ((const float*)input[indices[dispatch][2]])[i]) * (0.5f * (dispatch + 1)) + (float)round;
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
    explicit WeightRecordingProbe(const ncnn::VulkanDevice* vkdev, bool force_staging = false)
        : copies(0), dispatches(0), submissions(0), semaphores(0), destroyed_semaphores(0), fences(0), destroyed_fences(0), command_buffers(0), freed_command_buffers(0), destroyed_recorded_buffer(false), allocations(0), device_bytes(0), host_bytes(0), peak_device_bytes(0), peak_host_bytes(0), hide_next_map(force_staging && (vkdev->info.type() == 1 || vkdev->info.type() == 3)), map(ncnn::vkMapMemory), copy(ncnn::vkCmdCopyBuffer), dispatch(ncnn::vkCmdDispatch), submit(ncnn::vkQueueSubmit), create_semaphore(ncnn::vkCreateSemaphore), destroy_semaphore(ncnn::vkDestroySemaphore), create_fence(ncnn::vkCreateFence), destroy_fence(ncnn::vkDestroyFence), allocate_command(ncnn::vkAllocateCommandBuffers), free_command(ncnn::vkFreeCommandBuffers), destroy(ncnn::vkDestroyBuffer), allocate(ncnn::vkAllocateMemory), free(ncnn::vkFreeMemory), properties(vkdev->info.physicalDeviceMemoryProperties())
    {
        g_weight_recording_probe = this;
        ncnn::vkMapMemory = map_memory;
        ncnn::vkCmdCopyBuffer = record_copy;
        ncnn::vkCmdDispatch = record_dispatch;
        ncnn::vkQueueSubmit = queue_submit;
        ncnn::vkCreateSemaphore = make_semaphore;
        ncnn::vkDestroySemaphore = free_semaphore;
        ncnn::vkCreateFence = make_fence;
        ncnn::vkDestroyFence = free_fence;
        ncnn::vkAllocateCommandBuffers = allocate_commands;
        ncnn::vkFreeCommandBuffers = free_commands;
        ncnn::vkDestroyBuffer = destroy_buffer;
        ncnn::vkAllocateMemory = allocate_memory;
        ncnn::vkFreeMemory = free_memory;
    }
    ~WeightRecordingProbe()
    {
        ncnn::vkMapMemory = map;
        ncnn::vkCmdCopyBuffer = copy;
        ncnn::vkCmdDispatch = dispatch;
        ncnn::vkQueueSubmit = submit;
        ncnn::vkCreateSemaphore = create_semaphore;
        ncnn::vkDestroySemaphore = destroy_semaphore;
        ncnn::vkCreateFence = create_fence;
        ncnn::vkDestroyFence = destroy_fence;
        ncnn::vkAllocateCommandBuffers = allocate_command;
        ncnn::vkFreeCommandBuffers = free_command;
        ncnn::vkDestroyBuffer = destroy;
        ncnn::vkAllocateMemory = allocate;
        ncnn::vkFreeMemory = free;
        g_weight_recording_probe = 0;
    }
    int copies;
    int dispatches;
    int submissions;
    int semaphores;
    int destroyed_semaphores;
    int fences;
    int destroyed_fences;
    int command_buffers;
    int freed_command_buffers;
    bool destroyed_recorded_buffer;
    int allocations;
    VkDeviceSize device_bytes;
    VkDeviceSize host_bytes;
    VkDeviceSize peak_device_bytes;
    VkDeviceSize peak_host_bytes;

    struct Copy
    {
        VkCommandBuffer command;
        int submission;
        bool source_destroyed;
        VkBuffer source;
        VkBuffer target;
        VkBufferCopy region;
    };
    std::vector<Copy> copy_regions;
    std::vector<VkCommandBuffer> dispatch_commands;
    std::vector<int> dispatch_submissions;

    struct Submission
    {
        VkCommandBuffer command;
        std::vector<VkSemaphore> waits;
        std::vector<VkSemaphore> signals;
        std::vector<int> wait_producers;
        std::vector<int> dependencies;
    };
    std::vector<Submission> submit_records;

    bool depends_on(int consumer, int producer) const
    {
        std::vector<bool> visited(submit_records.size(), false);
        std::vector<int> pending(1, consumer);
        visited[consumer] = true;
        for (size_t i = 0; i < pending.size(); i++)
        {
            const Submission& current = submit_records[pending[i]];
            for (size_t j = 0; j < current.dependencies.size(); j++)
            {
                const int p = current.dependencies[j];
                if (p == producer)
                    return true;
                if (p >= 0 && !visited[p])
                {
                    visited[p] = true;
                    pending.push_back(p);
                }
            }
        }
        return false;
    }

private:
    struct Memory
    {
        VkDeviceMemory handle;
        VkDeviceSize size;
        bool device_local;
    };
    WeightRecordingProbe(const WeightRecordingProbe&);
    WeightRecordingProbe& operator=(const WeightRecordingProbe&);
    static VKAPI_ATTR VkResult VKAPI_CALL map_memory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** data)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        const VkResult ret = p->map(device, memory, offset, size, flags, data);
        if (ret == VK_SUCCESS && p->hide_next_map)
        {
            *data = 0;
            p->hide_next_map = false;
        }
        return ret;
    }
    static VKAPI_ATTR void VKAPI_CALL record_copy(VkCommandBuffer command, VkBuffer src, VkBuffer dst, uint32_t count, const VkBufferCopy* regions)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        p->copies++;
        p->buffers.push_back(src);
        p->buffers.push_back(dst);
        for (uint32_t i = 0; i < count; i++)
        {
            Copy entry;
            entry.command = command;
            entry.submission = -1;
            entry.source_destroyed = false;
            entry.source = src;
            entry.target = dst;
            entry.region = regions[i];
            p->copy_regions.push_back(entry);
        }
        p->copy(command, src, dst, count, regions);
    }
    static VKAPI_ATTR void VKAPI_CALL record_dispatch(VkCommandBuffer command, uint32_t x, uint32_t y, uint32_t z)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        p->dispatches++;
        p->dispatch_commands.push_back(command);
        p->dispatch_submissions.push_back(-1);
        p->dispatch(command, x, y, z);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, uint32_t count, const VkSubmitInfo* info, VkFence fence)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        p->submissions += count;
        for (uint32_t i = 0; i < count; i++)
            for (uint32_t j = 0; j < std::max(1u, info[i].commandBufferCount); j++)
            {
                Submission entry;
                entry.command = info[i].commandBufferCount ? info[i].pCommandBuffers[j] : 0;
                for (uint32_t w = 0; w < info[i].waitSemaphoreCount; w++)
                {
                    entry.waits.push_back(info[i].pWaitSemaphores[w]);
                    int producer = -1;
                    for (int k = (int)p->submit_records.size() - 1; k >= 0 && producer < 0; k--)
                        for (size_t s = 0; s < p->submit_records[k].signals.size(); s++)
                            if (p->submit_records[k].signals[s] == info[i].pWaitSemaphores[w])
                                producer = k;
                    entry.wait_producers.push_back(producer);
                    if (producer >= 0)
                        entry.dependencies.push_back(producer);
                }
                // A notification batch signals completion of the preceding copy in this call.
                if (i > 0)
                    entry.dependencies.push_back((int)p->submit_records.size() - 1);
                for (uint32_t s = 0; s < info[i].signalSemaphoreCount; s++)
                    entry.signals.push_back(info[i].pSignalSemaphores[s]);
                const int submission = (int)p->submit_records.size();
                for (size_t c = 0; c < p->copy_regions.size(); c++)
                    if (p->copy_regions[c].command == entry.command && p->copy_regions[c].submission < 0)
                        p->copy_regions[c].submission = submission;
                for (size_t d = 0; d < p->dispatch_commands.size(); d++)
                    if (p->dispatch_commands[d] == entry.command && p->dispatch_submissions[d] < 0)
                        p->dispatch_submissions[d] = submission;
                p->submit_records.push_back(entry);
            }
        return p->submit(queue, count, info, fence);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL make_semaphore(VkDevice device, const VkSemaphoreCreateInfo* info, const VkAllocationCallbacks* callbacks, VkSemaphore* semaphore)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        const VkResult ret = p->create_semaphore(device, info, callbacks, semaphore);
        if (ret == VK_SUCCESS)
            p->semaphores++;
        return ret;
    }
    static VKAPI_ATTR void VKAPI_CALL free_semaphore(VkDevice device, VkSemaphore semaphore, const VkAllocationCallbacks* callbacks)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        if (semaphore)
            p->destroyed_semaphores++;
        p->destroy_semaphore(device, semaphore, callbacks);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL make_fence(VkDevice device, const VkFenceCreateInfo* info, const VkAllocationCallbacks* callbacks, VkFence* fence)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        const VkResult ret = p->create_fence(device, info, callbacks, fence);
        if (ret == VK_SUCCESS)
            p->fences++;
        return ret;
    }
    static VKAPI_ATTR void VKAPI_CALL free_fence(VkDevice device, VkFence fence, const VkAllocationCallbacks* callbacks)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        if (fence)
            p->destroyed_fences++;
        p->destroy_fence(device, fence, callbacks);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL allocate_commands(VkDevice device, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* commands)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        const VkResult ret = p->allocate_command(device, info, commands);
        if (ret == VK_SUCCESS)
            p->command_buffers += info->commandBufferCount;
        return ret;
    }
    static VKAPI_ATTR void VKAPI_CALL free_commands(VkDevice device, VkCommandPool pool, uint32_t count, const VkCommandBuffer* commands)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        p->freed_command_buffers += count;
        p->free_command(device, pool, count, commands);
    }
    static VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* callbacks)
    {
        WeightRecordingProbe* p = g_weight_recording_probe;
        for (size_t i = 0; i < p->buffers.size(); i++)
            if (p->buffers[i] == buffer)
                p->destroyed_recorded_buffer = true;
        for (size_t i = 0; i < p->copy_regions.size(); i++)
            if (p->copy_regions[i].source == buffer)
                p->copy_regions[i].source_destroyed = true;
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
    bool hide_next_map;
    PFN_vkMapMemory map;
    PFN_vkCmdCopyBuffer copy;
    PFN_vkCmdDispatch dispatch;
    PFN_vkQueueSubmit submit;
    PFN_vkCreateSemaphore create_semaphore;
    PFN_vkDestroySemaphore destroy_semaphore;
    PFN_vkCreateFence create_fence;
    PFN_vkDestroyFence destroy_fence;
    PFN_vkAllocateCommandBuffers allocate_command;
    PFN_vkFreeCommandBuffers free_command;
    PFN_vkDestroyBuffer destroy;
    PFN_vkAllocateMemory allocate;
    PFN_vkFreeMemory free;
    VkPhysicalDeviceMemoryProperties properties;
    std::vector<Memory> memory;
    std::vector<VkBuffer> buffers;
};

static int test_weight_storage_record_time_preparation()
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
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
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
            bool weight_copy_recorded = false;
            if (round == 1)
            {
                const int before = probe.copies;
                cmd.record_clone(weight[0], first_device, opt);
                weight_copy_recorded = probe.copies > before;
            }
            cmd.record_clone(ordinary, ordinary_device, opt);
            ordinary.fill(-1.f);
            cmd.record_clone(ordinary_device, ordinary_output, opt);
            if (round == 0)
            {
                const int before = probe.copies;
                cmd.record_clone(weight[0], first_device, opt);
                weight_copy_recorded = probe.copies > before;
            }
            if ((vkdev->info.support_VK_KHR_push_descriptor() && !weight_copy_recorded) || probe.submissions == 0)
            {
                fprintf(stderr, "weight clone did not stream its preceding batch\n");
                return -1;
            }
            cmd.record_clone(first_device, first_output, opt);
            const int before_dispatch = probe.dispatches;
            cmd.record_pipeline(&pipeline, bindings, std::vector<ncnn::vk_constant_type>(), result);
            const int expected_dispatches = before_dispatch + (vkdev->info.support_VK_KHR_push_descriptor() ? 1 : 0);
            if (probe.dispatches != expected_dispatches || probe.submissions == 0 || probe.destroyed_recorded_buffer)
            {
                fprintf(stderr, "weight dispatch recording or backing lifetime was incorrect round=%d\n", round);
                return -1;
            }
            cmd.record_clone(result, output, opt);
            const int copies = probe.copies;
            const int dispatches = probe.dispatches;
            if (cmd.submit_and_wait() || probe.submissions == 0
                    || (vkdev->info.support_VK_KHR_push_descriptor() ? (probe.copies != copies || probe.dispatches != dispatches) : (probe.copies < copies || probe.dispatches != dispatches + 1)))
            {
                fprintf(stderr, "weight submit failed to execute recorded commands round=%d\n", round);
                return -1;
            }
        }
        if (CompareMat(ordinary_expected, ordinary_output, 0.f) || CompareMat(input[0], first_output, 0.f) || CompareMat(expected, output, 0.f) || cmd.reset())
            return -1;
    }
    if (!vkdev->info.support_VK_KHR_push_descriptor())
    {
        const VulkanFailure::Failure failures[] = {VulkanFailure::CREATE_DESCRIPTOR_POOL, VulkanFailure::ALLOCATE_DESCRIPTOR_SET};
        for (int i = 0; i < 2; i++)
        {
            {
                VulkanFailure failure(failures[i]);
                cmd.record_pipeline(&pipeline, bindings, std::vector<ncnn::vk_constant_type>(), result);
                if (!failure.triggered || failure.submission_count() == 0)
                {
                    fprintf(stderr, "descriptor allocation failure did not reach the next streamed batch\n");
                    return -1;
                }
            }
            // A failed descriptor operation does not poison later valid recording.
            ncnn::Mat output;
            cmd.record_pipeline(&pipeline, bindings, std::vector<ncnn::vk_constant_type>(), result);
            cmd.record_clone(result, output, opt);
            if (cmd.submit_and_wait() || CompareMat(expected, output, 0.f) || cmd.reset())
                return -1;
        }
    }
    // Recording and submitting later groups must not allocate any command buffers.
    ncnn::VkMat first_device(elements, (size_t)4u, &blobs);
    ncnn::VkCompute rejected(vkdev);
    rejected.set_use_weight_upload_online(true);
    {
        WeightRecordingProbe probe(vkdev);
        rejected.record_clone(weight[0], first_device, opt);
        const int dispatches = probe.dispatches;
        VulkanFailure failure(VulkanFailure::ALLOCATE_COMMAND);
        rejected.record_pipeline(&pipeline, bindings, std::vector<ncnn::vk_constant_type>(), result);
        if (failure.triggered || probe.submissions == 0 || probe.dispatches != dispatches + (vkdev->info.support_VK_KHR_push_descriptor() ? 1 : 0) || probe.destroyed_recorded_buffer)
        {
            fprintf(stderr, "weight recording allocated a command buffer or failed to stream work\n");
            return -1;
        }
    }
    if (rejected.reset())
        return -1;
    ncnn::Mat recovered;
    rejected.record_pipeline(&pipeline, bindings, std::vector<ncnn::vk_constant_type>(), result);
    rejected.record_clone(result, recovered, opt);
    if (rejected.submit_and_wait() || CompareMat(expected, recovered, 0.f))
        return -1;

    return 0;
}

static int test_weight_storage_workspace_allocation()
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
    ncnn::VkMat native;
    ncnn::VkMat target(1024, (size_t)4u, &blobs);
    {
        ncnn::VkCompute prepare(vkdev);
        prepare.record_clone(input, native, opt);
        if (prepare.submit_and_wait())
            return -1;
    }
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    {
        WeightRecordingProbe probe(vkdev);
        cmd.record_clone(native, target, opt);
        if (cmd.submit_and_wait() || probe.allocations != 0 || probe.semaphores != 0 || probe.destroyed_semaphores != 0 || cmd.reset())
        {
            fprintf(stderr, "online recording allocated temporary weight memory or recreated semaphores\n");
            return -1;
        }
    }
    for (int round = 0; round < 2; round++)
    {
        WeightRecordingProbe probe(vkdev);
        cmd.record_clone(weight, target, opt);
        if ((round == 0 ? probe.allocations == 0 : probe.allocations != 0) || probe.submissions == 0 || probe.destroyed_recorded_buffer)
        {
            fprintf(stderr, "private weight pool was not allocated lazily or reused after reset round=%d\n", round);
            return -1;
        }
        ncnn::Mat output;
        cmd.record_clone(target, output, opt);
        if (cmd.submit_and_wait() || CompareMat(input, output, 0.f) || cmd.reset())
            return -1;
    }
    {
        WeightRecordingProbe probe(vkdev);
        {
            ncnn::VkCompute scoped(vkdev);
            scoped.set_use_weight_upload_online(true);
            scoped.record_clone(weight, target, opt);
            if (scoped.submit_and_wait() || probe.allocations == 0)
                return -1;
        }
        if (probe.device_bytes || probe.host_bytes)
        {
            fprintf(stderr, "command destruction retained its private weight pool\n");
            return -1;
        }
    }
    return 0;
}

// A returned temporary weight range must never enter an activation allocator.
// The ordinary upload after the weight read previously overwrote the same range
// when the temporary pool was supplied by the caller.
static int test_weight_storage_private_pool()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev, 4096);
    ncnn::VkStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    ncnn::Mat input = RandomMat(64);
    ncnn::Mat replacement = RandomMat(64);
    ncnn::VkMat weight;
    ncnn::VkTransfer upload(vkdev);
    upload.record_upload(input, weight, opt);
    if (upload.submit_and_wait() || weights.finalize_weights())
        return -1;
    opt.blob_vkallocator = &blobs;
    opt.workspace_vkallocator = &blobs;
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    for (int round = 0; round < 2; round++)
    {
        ncnn::VkMat first;
        ncnn::VkMat ordinary;
        ncnn::Mat first_output;
        ncnn::Mat ordinary_output;
        WeightRecordingProbe probe(vkdev);
        cmd.record_clone(weight, first, opt);
        cmd.record_clone(replacement, ordinary, opt);
        cmd.record_clone(first, first_output, opt);
        cmd.record_clone(ordinary, ordinary_output, opt);
        if (probe.submissions == 0 || probe.destroyed_recorded_buffer || cmd.submit_and_wait()
                || CompareMat(input, first_output, 0.f) || CompareMat(replacement, ordinary_output, 0.f))
            return -1;
        bool found_weight_read = false;
        for (size_t i = 0; i < probe.copy_regions.size(); i++)
        {
            const WeightRecordingProbe::Copy& copy = probe.copy_regions[i];
            if (copy.target != first.buffer() || copy.region.dstOffset != first.buffer_offset())
                continue;
            found_weight_read = true;
            if (copy.source == first.buffer() || copy.source == ordinary.buffer())
            {
                fprintf(stderr, "temporary weights shared activation backing\n");
                return -1;
            }
        }
        if (!found_weight_read || cmd.reset())
            return -1;
    }
    return 0;
}

static int test_weight_storage_sync_init_failure(VulkanFailure::Failure failure_type, int fail_at)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    const ncnn::Mat input = RandomMat(64);

    const bool command_failure = failure_type == VulkanFailure::ALLOCATE_COMMAND;
    const int count = command_failure || failure_type == VulkanFailure::CREATE_FENCE ? 2 : 4;
    const int rounds = failure_type == VulkanFailure::CREATE_SEMAPHORE ? 2 : 1;
    for (int recover = 0; recover < rounds; recover++)
    {
        WeightRecordingProbe probe(vkdev);
        const int& created = command_failure ? probe.command_buffers : failure_type == VulkanFailure::CREATE_FENCE ? probe.fences : probe.semaphores;
        const int& destroyed = command_failure ? probe.freed_command_buffers : failure_type == VulkanFailure::CREATE_FENCE ? probe.destroyed_fences : probe.destroyed_semaphores;
        {
            VulkanFailure failure(failure_type, fail_at);
            ncnn::VkCompute cmd(vkdev);
            if (!failure.triggered || created != fail_at - 1 || destroyed != 0 || cmd.submit_and_wait() == 0 || probe.submissions != 0)
            {
                fprintf(stderr, "sync initialization failure did not retain resources failure=%d at=%d\n", failure_type, fail_at);
                return -1;
            }
            if (recover)
            {
                // Reset can still create missing semaphores.
                if (cmd.reset() || created != count || destroyed != 0)
                {
                    fprintf(stderr, "sync initialization failure prevented reset failure=%d at=%d\n", failure_type, fail_at);
                    return -1;
                }
                cmd.set_use_weight_upload_online(true);
                ncnn::VkMat device;
                ncnn::Mat output;
                cmd.record_clone(input, device, opt);
                cmd.record_clone(device, output, opt);
                if (cmd.submit_and_wait() || CompareMat(input, output, 0.f))
                    return -1;
            }
        }
        if (created != (recover ? count : fail_at - 1) || probe.destroyed_semaphores != probe.semaphores || probe.destroyed_fences != probe.fences || probe.freed_command_buffers != probe.command_buffers)
        {
            fprintf(stderr, "sync resources leaked after initialization failure=%d at=%d recover=%d\n", failure_type, fail_at, recover);
            return -1;
        }
    }
    return 0;
}

// Keep adjacent complete weight groups disjoint, then reuse the older group.
// GPU-backed deferred sources exercise the upload graph even on unified memory.
static int test_weight_storage_upload_overlap(int submit_fail_at = 0, VulkanFailure::Failure failure_type = VulkanFailure::SUBMIT)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator weights(vkdev, true);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    ncnn::Mat input[3];
    ncnn::VkMat weight[3];
    ncnn::VkTransfer upload(vkdev);
    for (int i = 0; i < 3; i++)
    {
        input[i] = RandomMat(64);
        upload.record_upload(input[i], weight[i], opt);
    }
    if (upload.submit_and_wait() || weights.finalize_weights())
        return -1;
    int storage_types[3];
    for (int i = 0; i < 3; i++)
    {
        storage_types[i] = weight[i].data->storage_type;
        weight[i].data->storage_type = 1;
    }
    opt.blob_vkallocator = &blobs;
    int ret = 0;
    WeightRecordingProbe probe(vkdev);
    {
        ncnn::VkCompute cmd(vkdev);
        cmd.set_use_weight_upload_online(true);
        for (int round = 0; round < 2 && ret == 0; round++)
        {
            ncnn::VkMat results[3];
            ncnn::Mat output[3];
            probe.copy_regions.clear();
            probe.submit_records.clear();
            probe.submissions = 0;
            if (submit_fail_at && round == 0)
            {
                {
                    VulkanFailure failure(failure_type, submit_fail_at);
                    for (int i = 0; i < 3 && !failure.triggered; i++)
                        cmd.record_clone(weight[i], results[i], opt);
                    if (!failure.triggered)
                    {
                        fprintf(stderr, "streaming failure missing at %d type=%d\n", submit_fail_at, failure_type);
                        ret = -1;
                    }
                }
                // Reset waits for submitted work before discarding the open batch.
                if (cmd.reset() || probe.semaphores != 4 + probe.destroyed_semaphores)
                {
                    fprintf(stderr, "streaming failure did not reset synchronization at %d\n", submit_fail_at);
                    ret = -1;
                }
                continue;
            }
            for (int i = 0; i < 3; i++)
                cmd.record_clone(weight[i], results[i], opt);
            for (int i = 0; i < 3; i++)
                cmd.record_clone(results[i], output[i], opt);
            if (probe.submissions == 0 || cmd.submit_and_wait())
            {
                ret = -1;
                break;
            }
            int signal_count = 0;
            for (size_t i = 0; i < probe.submit_records.size(); i++)
            {
                const WeightRecordingProbe::Submission& submission = probe.submit_records[i];
                for (size_t s = 0; s < submission.signals.size(); s++)
                {
                    int wait_count = 0;
                    for (size_t j = i + 1; j < probe.submit_records.size(); j++)
                        for (size_t w = 0; w < probe.submit_records[j].waits.size(); w++)
                            if (probe.submit_records[j].waits[w] == submission.signals[s] && probe.submit_records[j].wait_producers[w] == (int)i)
                                wait_count++;
                    if (wait_count != 1)
                        ret = -1;
                    signal_count++;
                }
            }
            // The final compute also consumes the last upload-chain signal.
            if (ret != 0 || signal_count != 12 || probe.semaphores != 4 + probe.destroyed_semaphores)
            {
                fprintf(stderr, "weight semaphore consumers or lifetime incorrect round=%d created=%d signaled=%d destroyed=%d\n", round, probe.semaphores, signal_count, probe.destroyed_semaphores);
                ret = -1;
                break;
            }
            if (probe.fences != 2 || probe.destroyed_fences != 0 || probe.command_buffers != 2 || probe.freed_command_buffers != 0)
            {
                fprintf(stderr, "compute and upload did not retain their command buffers and fences round=%d fences=%d destroyed=%d\n", round, probe.fences, probe.destroyed_fences);
                ret = -1;
                break;
            }
            int uploads[3] = {-1, -1, -1};
            int computes[3] = {-1, -1, -1};
            VkBuffer targets[3] = {0, 0, 0};
            VkDeviceSize offsets[3] = {0, 0, 0};
            for (size_t c = 0; c < probe.copy_regions.size(); c++)
            {
                const WeightRecordingProbe::Copy& copy = probe.copy_regions[c];
                for (int i = 0; i < 3; i++)
                {
                    if (copy.source == weight[i].buffer() && copy.region.srcOffset == weight[i].buffer_offset())
                    {
                        uploads[i] = copy.submission;
                        targets[i] = copy.target;
                        offsets[i] = copy.region.dstOffset;
                    }
                    if (copy.target == results[i].buffer() && copy.region.dstOffset == results[i].buffer_offset())
                        computes[i] = copy.submission;
                }
            }
            for (int i = 0; i < 3; i++)
                if (uploads[i] < 0 || computes[i] < 0 || CompareMat(input[i], output[i], 0.f)
                        || !probe.depends_on(computes[i], uploads[i]))
                    ret = -1;
            if (ret != 0)
            {
                fprintf(stderr, "weight upload graph or output missing round=%d\n", round);
                break;
            }
            const VkDeviceSize size = weight[0].buffer_capacity();
            if (targets[0] != targets[1] || targets[0] != targets[2]
                    || (offsets[0] < offsets[1] + size && offsets[1] < offsets[0] + size)
                    || offsets[2] != offsets[0])
            {
                fprintf(stderr, "adjacent weight groups overlapped or older range was not reused round=%d\n", round);
                ret = -1;
            }
            if (probe.depends_on(uploads[1], computes[0])
                    || !probe.depends_on(uploads[2], computes[0])
                    || probe.depends_on(uploads[2], computes[1]))
            {
                fprintf(stderr, "weight upload waited for the adjacent consumer or missed the older reader round=%d\n", round);
                ret = -1;
            }
            if (cmd.reset())
                ret = -1;
        }
    }
    for (int i = 0; i < 3; i++)
    {
        weight[i].data->storage_type = storage_types[i];
    }
    if (probe.semaphores != probe.destroyed_semaphores || probe.fences != probe.destroyed_fences || probe.command_buffers != probe.freed_command_buffers)
    {
        fprintf(stderr, "weight synchronization resources leaked after command destruction\n");
        ret = -1;
    }
    return ret;
}

static int test_weight_storage_group_signal(bool staged)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator weights(vkdev, true);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    ncnn::Mat input[9];
    ncnn::VkMat weight[9];
    ncnn::VkTransfer upload(vkdev);
    for (int i = 0; i < 9; i++)
    {
        input[i] = RandomMat(64);
        upload.record_upload(input[i], weight[i], opt);
    }
    if (upload.submit_and_wait() || weights.finalize_weights())
        return -1;

    const char shader[] = "#version 450\n"
                          "layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n"
                          "layout(binding = 0) readonly buffer A { float a[]; };\n"
                          "layout(binding = 1) readonly buffer B { float b[]; };\n"
                          "layout(binding = 2) readonly buffer C { float c[]; };\n"
                          "layout(binding = 3) writeonly buffer D { float d[]; };\n"
                          "void main() { uint i = gl_GlobalInvocationID.x; if (i < 64) d[i] = a[i] + b[i] + c[i]; }\n";
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(shader, opt, spirv))
        return -1;
    ncnn::Pipeline pipeline(vkdev);
    pipeline.set_local_size_xyz(64, 1, 1);
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>()))
        return -1;

    opt.blob_vkallocator = &blobs;
    ncnn::VkMat results[3];
    for (int i = 0; i < 3; i++)
    {
        results[i].create(64, (size_t)4u, &blobs);
        if (results[i].empty())
            return -1;
    }
    int storage_types[9];
    for (int i = 0; i < 9; i++)
    {
        storage_types[i] = weight[i].data->storage_type;
        weight[i].data->storage_type = staged ? 2 : 1;
    }
    int ret = 0;
    WeightRecordingProbe probe(vkdev, staged);
    {
        ncnn::VkCompute cmd(vkdev);
        cmd.set_use_weight_upload_online(true);
        for (int round = 0; round < 2 && ret == 0; round++)
        {
            ncnn::Mat output[3];
            probe.dispatch_commands.clear();
            probe.dispatch_submissions.clear();
            probe.copy_regions.clear();
            probe.submit_records.clear();
            for (int i = 0; i < 3; i++)
            {
                std::vector<ncnn::VkMat> bindings(4);
                for (int j = 0; j < 3; j++)
                    bindings[j] = weight[i * 3 + j];
                bindings[3] = results[i];
                cmd.record_pipeline(&pipeline, bindings, std::vector<ncnn::vk_constant_type>(), results[i]);
                if (probe.copy_regions.size() != (size_t)(i + 1) * 3)
                {
                    ret = -1;
                    break;
                }
                for (int j = 0; j < 3; j++)
                {
                    const WeightRecordingProbe::Copy& copy = probe.copy_regions[i * 3 + j];
                    if (copy.submission < 0 || copy.submission != probe.copy_regions[i * 3].submission
                            || copy.region.size != weight[i * 3 + j].data->capacity || copy.source_destroyed)
                        ret = -1;
                    if (staged)
                        for (int k = 0; k < j; k++)
                            if (copy.source == probe.copy_regions[i * 3 + k].source)
                                ret = -1;
                }
                if (staged)
                    for (int j = 0; j < i * 3; j++)
                        if (!probe.copy_regions[j].source_destroyed)
                            ret = -1;
            }
            if (ret != 0)
                break;
            for (int i = 0; i < 3; i++)
                cmd.record_clone(results[i], output[i], opt);
            if (cmd.submit_and_wait() || probe.dispatch_commands.size() != 3 || probe.semaphores != 4 || probe.destroyed_semaphores != 0 || probe.fences != 2 || probe.destroyed_fences != 0 || probe.command_buffers != 2 || probe.freed_command_buffers != 0)
            {
                ret = -1;
                break;
            }
            int notifications = 0;
            for (size_t i = 0; i < probe.submit_records.size(); i++)
                if (!probe.submit_records[i].command)
                    notifications++;
            if (notifications != 3 || probe.submit_records.size() != 10)
                ret = -1;
            for (int i = 0; i < 3; i++)
            {
                const int compute = probe.dispatch_submissions[i];
                // One preceding-compute wait and one ready wait for all three weights.
                // The last compute additionally drains the upload-chain tail.
                if (compute < 0 || probe.submit_records[compute].waits.size() != (i == 2 ? 3u : 2u))
                    ret = -1;
                ncnn::Mat expected(64);
                for (int j = 0; j < 64; j++)
                    expected[j] = input[i * 3][j] + input[i * 3 + 1][j] + input[i * 3 + 2][j];
                if (CompareMat(expected, output[i], 0.001f))
                    ret = -1;
            }
            if (staged)
                for (int i = 0; i < 9; i++)
                    if (!probe.copy_regions[i].source_destroyed)
                        ret = -1;
            if (cmd.reset())
                ret = -1;
        }
    }
    for (int i = 0; i < 9; i++)
        weight[i].data->storage_type = storage_types[i];
    if (probe.semaphores != probe.destroyed_semaphores || probe.fences != probe.destroyed_fences || probe.command_buffers != probe.freed_command_buffers)
        ret = -1;
    if (ret != 0)
        fprintf(stderr, "multiple weight uploads did not share one submission and ready signal per compute staged=%d\n", staged);
    return ret;
}

class FailingFlushAllocator : public ncnn::VkStagingAllocator
{
public:
    explicit FailingFlushAllocator(const ncnn::VulkanDevice* vkdev)
        : ncnn::VkStagingAllocator(vkdev), flushes(0), fail(true)
    {
    }
    virtual int flush(ncnn::VkBufferMemory* memory)
    {
        flushes++;
        return fail ? -1 : ncnn::VkStagingAllocator::flush(memory);
    }
    int flushes;
    bool fail;
};

static int test_weight_storage_staging_flush_failure()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkBlobAllocator blobs(vkdev);
    FailingFlushAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    const ncnn::Mat input = RandomMat(256);
    for (int upload = 0; upload < 2; upload++)
    {
        ncnn::VkMat device;
        ncnn::Mat output;
        ncnn::VkCompute cmd(vkdev);
        staging.fail = true;
        staging.flushes = 0;
        WeightRecordingProbe probe(vkdev);
        if (upload)
            cmd.record_upload(input, device, opt);
        else
            cmd.record_clone(input, device, opt);
        if (staging.flushes != 1 || !device.empty() || probe.copies != 0 || probe.dispatches != 0 || probe.submissions != 0)
        {
            fprintf(stderr, "staging flush failure did not stop recording upload=%d\n", upload);
            return -1;
        }
        if (cmd.reset())
            return -1;
        staging.fail = false;
        cmd.record_clone(input, device, opt);
        cmd.record_clone(device, output, opt);
        if (staging.flushes != 2 || cmd.submit_and_wait() || staging.flushes != 2 || CompareMat(input, output, 0.f))
        {
            fprintf(stderr, "staging flush failure prevented reuse after reset\n");
            return -1;
        }
    }
    for (int mapped = 0; mapped < 2; mapped++)
    {
        UnmappedWeightAllocator weights(vkdev);
        opt.blob_vkallocator = mapped ? (ncnn::VkAllocator*)&staging : (ncnn::VkAllocator*)&weights;
        staging.fail = true;
        staging.flushes = 0;
        ncnn::VkMat weight;
        ncnn::VkTransfer cmd(vkdev);
        WeightRecordingProbe probe(vkdev);
        cmd.record_upload(input, weight, opt);
        if (staging.flushes != 1 || cmd.pending_upload_total() != 0 || cmd.submit_and_wait() != 0 || probe.submissions != 0 || probe.copies != 0)
        {
            fprintf(stderr, "weight flush failure recorded or submitted GPU work\n");
            return -1;
        }
        staging.fail = false;
        cmd.record_upload(input, weight, opt);
        if (staging.flushes != 2 || cmd.submit_and_wait() != 0 || staging.flushes != 2)
        {
            fprintf(stderr, "weight flush failure prevented later recording\n");
            return -1;
        }
    }
    return 0;
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

static int test_weight_storage_transfer_wait_failure(bool multiple_weights)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    UnmappedWeightAllocator weights(vkdev);
    CountingWeightStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.staging_vkallocator = &staging;
    opt.blob_vkallocator = &weights;
    ncnn::Mat input = RandomMat(256);
    const int count = multiple_weights ? 3 : 1;
    ncnn::VkMat targets[3];
    {
        VulkanFailure fail(VulkanFailure::FENCE_WAIT);
        {
            ncnn::VkTransfer upload(vkdev);
            for (int i = 0; i < count; i++)
                upload.record_upload(input, targets[i], opt);
            if (fail.triggered)
            {
                fprintf(stderr, "weight recording unexpectedly waited for a submission\n");
                return -1;
            }
            const int ret = upload.submit_and_wait();
            // The command does not retry a failed wait during reset or destruction.
            // Complete the actual work before exercising ordinary reset cleanup.
            ncnn::vkDeviceWaitIdle(vkdev->vkdevice());
            if (ret == 0 || !fail.triggered || staging.frees != 0 || upload.reset() != 0 || staging.frees != count)
            {
                fprintf(stderr, "transfer wait error or staging cleanup failed\n");
                return -1;
            }
        }
    }
    return staging.frees == count ? 0 : -1;
}

static int test_weight_storage_workspace_wait_failure(bool deferred_weight)
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
    ncnn::VkCompute* cmd = new ncnn::VkCompute(vkdev);
    cmd->set_use_weight_upload_online(deferred_weight);
    ncnn::VkMat ordinary;
    ncnn::Mat ordinary_input = RandomMat(256);
    if (deferred_weight)
        cmd->record_clone(weight, output, opt);
    else
    {
        cmd->record_clone(ordinary_input, ordinary, opt);
        cmd->record_clone(ordinary, output, opt);
    }
    const int frees_before = staging.frees;
    VulkanFailure fail(VulkanFailure::FENCE_WAIT, 2);
    const int result = cmd->submit_and_wait();
    const bool retained = result != 0 && fail.triggered && cmd->reset() != 0 && staging.frees == frees_before;
    // The submit and reset failures retain resources until destruction waits
    // for the pending commands without another injected error.
    delete cmd;
    if (!retained || fail.wait_errors != 2 || staging.frees <= frees_before)
    {
        fprintf(stderr, "compute fence wait failure lost pending execution or staging lifetime deferred=%d\n", deferred_weight);
        return -1;
    }
    return 0;
}

static int test_weight_storage_upload_wait_failure(bool submit, bool staged = false)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator weights(vkdev, true);
    ncnn::VkBlobAllocator blobs(vkdev);
    CountingWeightStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    ncnn::VkTransfer upload(vkdev);
    ncnn::VkMat weight;
    upload.record_upload(RandomMat(256), weight, opt);
    if (upload.submit_and_wait() != 0 || weights.finalize_weights() != 0)
        return -1;

    // Exercise the upload queue even on unified-memory devices.
    const int storage_type = weight.data->storage_type;
    weight.data->storage_type = staged ? 2 : 1;
    opt.blob_vkallocator = &blobs;
    ncnn::Mat output;
    WeightRecordingProbe probe(vkdev, staged);
    ncnn::VkCompute* cmd = new ncnn::VkCompute(vkdev);
    cmd->set_use_weight_upload_online(true);
    VulkanFailure fail(VulkanFailure::UPLOAD_FENCE_WAIT, 2);
    cmd->record_clone(weight, output, opt);
    const int frees_before = staging.frees;

    // Compute has completed when the upload wait fails, both after final submit
    // and when resetting an unsubmitted compute recording.
    const int result = submit ? cmd->submit_and_wait() : cmd->reset();
    bool retained = result != 0 && fail.triggered && cmd->reset() != 0 && staging.frees == frees_before;
    if (staged)
        retained = retained && !probe.copy_regions.empty() && !probe.copy_regions[0].source_destroyed;
    delete cmd;
    if (staged)
        retained = retained && !probe.copy_regions.empty() && probe.copy_regions[0].source_destroyed;
    weight.data->storage_type = storage_type;
    if (!retained || fail.wait_errors != 2 || staging.frees <= frees_before)
    {
        fprintf(stderr, "upload fence wait failure lost pending work or staging lifetime submit=%d\n", submit);
        return -1;
    }
    return 0;
}

static int test_weight_storage_execution_resources(VulkanFailure::Failure failure)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator weights(vkdev, true);
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
    // Exercise resident-to-workspace transfers even on unified-memory test devices.
    std::vector<int> storage_types(count);
    for (int i = 0; i < count; i++)
    {
        storage_types[i] = weight[i].data->storage_type;
        weight[i].data->storage_type = 1;
    }
    opt.blob_vkallocator = &blobs;

    int ret = 0;
    {
        ncnn::VkCompute cmd(vkdev);
        cmd.set_use_weight_upload_online(true);
        {
            // Hundreds of logical batches reuse two already allocated command buffers.
            VulkanFailure check(failure, 5);
            int recorded = 0;
            for (; recorded < count; recorded++)
            {
                cmd.record_clone(weight[recorded], output[recorded], opt);
                if (check.triggered)
                    break;
            }
            if (check.submission_count() == 0 || check.command_allocations != 0)
            {
                fprintf(stderr, "recording allocated command buffers or did not stream work\n");
                return -1;
            }
            if (failure == VulkanFailure::TRACK_COMMANDS)
            {
                ret = cmd.submit_and_wait();
                if (ret != 0 || check.triggered || recorded != count || !check.reused_command || check.unsafe_command_reset || check.command_allocations != 0 || check.submission_count() <= count + 4)
                {
                    fprintf(stderr, "execution resources did not safely reuse commands or insufficient task coverage count=%d reused=%d\n", check.submission_count(), check.reused_command);
                    ret = -1;
                }
            }
            else
            {
                ret = check.triggered && recorded < count && check.command_allocations == 0 ? 0 : -1;
            }
        }
        if (ret == 0 && failure == VulkanFailure::TRACK_COMMANDS)
            for (int i = 0; i < count; i++)
                if (CompareMat(input[i], output[i], 0.f) != 0)
                    ret = -1;
        if (cmd.reset() != 0)
            ret = -1;

        // Reuse the same command across ordinary and paged recordings, including
        // recovery after interrupted command recording.
        for (int round = 0; round < 3 && ret == 0; round++)
        {
            const int output_count = round == 1 ? count : 1;
            ncnn::VkMat ordinary;
            if (round == 1)
            {
                for (int i = 0; i < count; i++)
                    cmd.record_clone(weight[i], output[i], opt);
            }
            else
            {
                cmd.record_clone(input[0], ordinary, opt);
                cmd.record_clone(ordinary, output[0], opt);
            }
            if (cmd.submit_and_wait() != 0)
                ret = -1;
            for (int i = 0; i < output_count && ret == 0; i++)
                if (CompareMat(input[i], output[i], 0.f) != 0)
                    ret = -1;
            if (cmd.reset() != 0)
                ret = -1;
            if (ret != 0)
                fprintf(stderr, "compute resource reuse failed failure=%d round=%d\n", failure, round);
        }
    }
    for (int i = 0; i < count; i++)
    {
        weight[i].data->storage_type = storage_types[i];
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
                    || test_weight_storage_allocator_failure(VulkanFailure::ALLOCATE_MEMORY)
                    || test_weight_storage_allocator_failure(VulkanFailure::BIND_MEMORY)
                    || test_weight_storage_staging_flush_failure()
                    || test_weight_storage_transfer_wait_failure(false)
                    || test_weight_storage_transfer_wait_failure(true)
                    || test_weight_storage_workspace_wait_failure(true)
                    || test_weight_storage_workspace_wait_failure(false)
                    || test_weight_storage_upload_wait_failure(true)
                    || test_weight_storage_upload_wait_failure(false)
                    || test_weight_storage_upload_wait_failure(true, true)
                    || test_weight_storage_upload_wait_failure(false, true)
                    || test_weight_storage_execution_resources(VulkanFailure::TRACK_COMMANDS)
                    || test_weight_storage_execution_resources(VulkanFailure::RESET_COMMAND)
                    || test_weight_storage_sync_init_failure(VulkanFailure::ALLOCATE_COMMAND, 1)
                    || test_weight_storage_sync_init_failure(VulkanFailure::ALLOCATE_COMMAND, 2)
                    || test_weight_storage_sync_init_failure(VulkanFailure::CREATE_FENCE, 1)
                    || test_weight_storage_sync_init_failure(VulkanFailure::CREATE_FENCE, 2)
                    || test_weight_storage_sync_init_failure(VulkanFailure::CREATE_SEMAPHORE, 1)
                    || test_weight_storage_sync_init_failure(VulkanFailure::CREATE_SEMAPHORE, 2)
                    || test_weight_storage_sync_init_failure(VulkanFailure::CREATE_SEMAPHORE, 3)
                    || test_weight_storage_sync_init_failure(VulkanFailure::CREATE_SEMAPHORE, 4)
                    || test_weight_storage_workspace_failure(VulkanFailure::ALLOCATE_MEMORY)
                    || test_weight_storage_workspace_failure(VulkanFailure::SUBMIT)
                    || test_weight_storage_staging_allocator_failure(VulkanFailure::CREATE_BUFFER)
                    || test_weight_storage_staging_allocator_failure(VulkanFailure::ALLOCATE_MEMORY)
                    || test_weight_storage_staging_allocator_failure(VulkanFailure::BIND_MEMORY)
                    || test_weight_storage_staging_allocator_failure(VulkanFailure::MAP_MEMORY)
                    || test_weight_storage_transfer_failure(VulkanFailure::SUBMIT)
                    || test_weight_storage_transfer_failure(VulkanFailure::SECOND_SUBMIT)
                    || test_weight_storage_repeated_weights()
                    || test_weight_storage_dispatch_pinning()
                    || test_weight_storage_record_time_preparation()
                    || test_weight_storage_workspace_allocation()
                    || test_weight_storage_private_pool()
                    || test_weight_storage_upload_overlap()
                    || test_weight_storage_upload_overlap(1)
                    || test_weight_storage_upload_overlap(2)
                    || test_weight_storage_upload_overlap(3)
                    || test_weight_storage_upload_overlap(4)
                    || test_weight_storage_upload_overlap(1, VulkanFailure::RESET_COMMAND)
                    || test_weight_storage_upload_overlap(3, VulkanFailure::RESET_COMMAND)
                    || test_weight_storage_upload_overlap(1, VulkanFailure::RESET_FENCE)
                    || test_weight_storage_upload_overlap(3, VulkanFailure::RESET_FENCE)
                    || test_weight_storage_group_signal(false)
                    || test_weight_storage_group_signal(true)
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
