// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_weight_storage.h"
#include "pipeline.h"

#if NCNN_WEIGHT_STORAGE_NATIVE_PROBE
// Inspect native uploads without exposing the command's private weight pool.
// On unified devices, hide one successful target mapping to exercise staging;
// the allocator still owns and unmaps the real mapping when it clears its pool.
class WeightStorageUploadProbe;
static WeightStorageUploadProbe* g_weight_storage_upload_probe = 0;
class WeightStorageUploadProbe
{
public:
    explicit WeightStorageUploadProbe(const ncnn::VulkanDevice* vkdev)
        : hide_next_map(vkdev->info.type() == 1 || vkdev->info.type() == 3), hidden_maps(0), map(ncnn::vkMapMemory), copy(ncnn::vkCmdCopyBuffer), create(ncnn::vkCreateBuffer), barrier(ncnn::vkCmdPipelineBarrier)
    {
        g_weight_storage_upload_probe = this;
        ncnn::vkMapMemory = map_memory;
        ncnn::vkCmdCopyBuffer = record_copy;
        ncnn::vkCreateBuffer = create_buffer;
        ncnn::vkCmdPipelineBarrier = record_barrier;
    }
    ~WeightStorageUploadProbe()
    {
        ncnn::vkMapMemory = map;
        ncnn::vkCmdCopyBuffer = copy;
        ncnn::vkCreateBuffer = create;
        ncnn::vkCmdPipelineBarrier = barrier;
        g_weight_storage_upload_probe = 0;
    }
    struct Copy
    {
        VkCommandBuffer command;
        VkBuffer source;
        VkBuffer target;
        VkBufferCopy region;
    };
    std::vector<Copy> copies;
    struct Buffer
    {
        VkBuffer handle;
        VkSharingMode sharing_mode;
    };
    std::vector<Buffer> buffers;
    struct Barrier
    {
        VkCommandBuffer command;
        VkPipelineStageFlags src_stage;
        VkPipelineStageFlags dst_stage;
        VkBufferMemoryBarrier buffer_barrier;
    };
    std::vector<Barrier> barriers;
    bool hide_next_map;
    int hidden_maps;

private:
    WeightStorageUploadProbe(const WeightStorageUploadProbe&);
    WeightStorageUploadProbe& operator=(const WeightStorageUploadProbe&);
    static VKAPI_ATTR VkResult VKAPI_CALL map_memory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** data)
    {
        WeightStorageUploadProbe* p = g_weight_storage_upload_probe;
        const VkResult ret = p->map(device, memory, offset, size, flags, data);
        if (ret == VK_SUCCESS && p->hide_next_map)
        {
            *data = 0;
            p->hide_next_map = false;
            p->hidden_maps++;
        }
        return ret;
    }
    static VKAPI_ATTR void VKAPI_CALL record_copy(VkCommandBuffer command, VkBuffer source, VkBuffer target, uint32_t count, const VkBufferCopy* regions)
    {
        WeightStorageUploadProbe* p = g_weight_storage_upload_probe;
        for (uint32_t i = 0; i < count; i++)
        {
            Copy entry;
            entry.command = command;
            entry.source = source;
            entry.target = target;
            entry.region = regions[i];
            p->copies.push_back(entry);
        }
        p->copy(command, source, target, count, regions);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice device, const VkBufferCreateInfo* info, const VkAllocationCallbacks* callbacks, VkBuffer* buffer)
    {
        WeightStorageUploadProbe* p = g_weight_storage_upload_probe;
        VkResult ret = p->create(device, info, callbacks, buffer);
        if (ret == VK_SUCCESS)
        {
            Buffer entry;
            entry.handle = *buffer;
            entry.sharing_mode = info->sharingMode;
            p->buffers.push_back(entry);
        }
        return ret;
    }
    static VKAPI_ATTR void VKAPI_CALL record_barrier(VkCommandBuffer command, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage, VkDependencyFlags flags,
            uint32_t memory_count, const VkMemoryBarrier* memory_barriers, uint32_t buffer_count, const VkBufferMemoryBarrier* buffer_barriers,
            uint32_t image_count, const VkImageMemoryBarrier* image_barriers)
    {
        WeightStorageUploadProbe* p = g_weight_storage_upload_probe;
        for (uint32_t i = 0; i < buffer_count; i++)
        {
            Barrier entry;
            entry.command = command;
            entry.src_stage = src_stage;
            entry.dst_stage = dst_stage;
            entry.buffer_barrier = buffer_barriers[i];
            p->barriers.push_back(entry);
        }
        p->barrier(command, src_stage, dst_stage, flags, memory_count, memory_barriers, buffer_count, buffer_barriers, image_count, image_barriers);
    }
    PFN_vkMapMemory map;
    PFN_vkCmdCopyBuffer copy;
    PFN_vkCreateBuffer create;
    PFN_vkCmdPipelineBarrier barrier;
};
#endif

static int test_weight_storage_upload_mode()
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
    ncnn::Mat input = RandomMat(64);
    ncnn::VkMat weight;
    ncnn::VkTransfer upload(vkdev);
    upload.record_upload(input, weight, opt);
    if (upload.submit_and_wait() || weights.finalize_weights())
        return -1;
    if (weight.data->storage_type != 2)
        return 0;
    opt.blob_vkallocator = &blobs;

    ncnn::VkCompute cmd(vkdev);
    ncnn::Mat output;
    output.release();
    cmd.record_clone(weight, output, opt);
    if (!output.empty() || cmd.submit_and_wait() || cmd.reset())
    {
        fprintf(stderr, "deferred weight unexpectedly enabled online upload\n");
        return -1;
    }

    cmd.set_use_weight_upload_online(true);
    for (int round = 0; round < 2; round++)
    {
        // Empty and ordinary recordings must retain the selected online mode.
        if (cmd.submit_and_wait() || cmd.reset())
            return -1;
        ncnn::VkMat ordinary;
        cmd.record_clone(input, ordinary, opt);
        cmd.record_clone(ordinary, output, opt);
        if (cmd.submit_and_wait() || CompareMat(input, output, 0.f) || cmd.reset())
            return -1;
        cmd.record_clone(weight, output, opt);
        if (cmd.submit_and_wait() || CompareMat(input, output, 0.f) || cmd.reset())
        {
            fprintf(stderr, "online weight upload setting was lost after reset\n");
            return -1;
        }
    }

    // The caller can select the ordinary mode again before a new recording.
    cmd.set_use_weight_upload_online(false);
    if (cmd.submit_and_wait() || cmd.reset())
        return -1;
    ncnn::VkMat ordinary;
    cmd.record_clone(input, ordinary, opt);
    cmd.record_clone(ordinary, output, opt);
    if (cmd.submit_and_wait() || CompareMat(input, output, 0.f) || cmd.reset())
        return -1;
    output.release();
    cmd.record_clone(weight, output, opt);
    if (!output.empty() || cmd.submit_and_wait() || cmd.reset())
    {
        fprintf(stderr, "disabled online weight upload was changed by recording\n");
        return -1;
    }

    return 0;
}

static int test_weight_storage_bytes(const ncnn::Option& opt, int elempack, bool flatten, bool host)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkWeightAllocator disk(vkdev);
    ncnn::VkWeightAllocator resident(vkdev, host);
    ncnn::VkWeightStagingAllocator staging(vkdev);
    disk.set_weights_in_disk(true);
    disk.set_weight_file_directory(weight_storage_test_directory());

    ncnn::Mat src(7, 3, 2, 2, (size_t)4u * elempack, elempack);
    float* p = src;
    for (size_t i = 0; i < src.total() * elempack; i++)
        p[i] = (float)((int)(i % 53) - 26) * 0.125f;

    ncnn::Option upload_opt = opt;
    upload_opt.blob_vkallocator = &disk;
    upload_opt.staging_vkallocator = &staging;
    ncnn::VkMat file_weight;
    ncnn::VkTransfer upload(vkdev);
    upload.record_upload(src, file_weight, upload_opt, flatten);
    if (file_weight.empty() || upload.submit_and_wait() != 0 || disk.finalize_weights() != 0 || upload.reset() != 0)
    {
        fprintf(stderr, "test_weight_storage_bytes disk upload failed\n");
        return -1;
    }
    upload_opt.blob_vkallocator = &resident;
    ncnn::VkMat resident_weight;
    upload.record_upload(src, resident_weight, upload_opt, flatten);
    if (resident_weight.empty() || upload.submit_and_wait() != 0 || resident.finalize_weights() != 0)
    {
        fprintf(stderr, "test_weight_storage_bytes resident upload failed\n");
        return -1;
    }
    src.release();

    ncnn::Option download_opt = weight_storage_option();
    download_opt.blob_vkallocator = &staging;
    download_opt.staging_vkallocator = &staging;
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    ncnn::Mat expected;
    ncnn::Mat actual;
    cmd.record_clone(resident_weight, expected, download_opt);
    cmd.record_clone(file_weight, actual, download_opt);
    if (cmd.submit_and_wait() != 0)
    {
        fprintf(stderr, "test_weight_storage_bytes download failed\n");
        return -1;
    }
    for (int q = 0; q < expected.c; q++)
    {
        size_t size = (size_t)expected.w * expected.h * expected.d * expected.elemsize;
        size_t offset = q * expected.cstep * expected.elemsize;
        if (memcmp((const unsigned char*)actual.data + offset, (const unsigned char*)expected.data + offset, size) != 0)
        {
            fprintf(stderr, "test_weight_storage_bytes failed pack=%d flatten=%d host=%d fp16=%d bf16=%d\n", elempack, flatten, host, opt.use_fp16_storage, opt.use_bf16_storage);
            return -1;
        }
    }
    return 0;
}

static int test_weight_storage_layouts(const ncnn::Option& opt)
{
    for (int elempack = 1; elempack <= 4; elempack *= 4)
    {
        if (test_weight_storage_bytes(opt, elempack, true, false) != 0
                || test_weight_storage_bytes(opt, elempack, false, false) != 0
                || test_weight_storage_bytes(opt, elempack, false, true) != 0)
            return -1;
    }
    return 0;
}

static int test_weight_storage_precisions()
{
    ncnn::Option opt = weight_storage_option();
    if (test_weight_storage_layouts(opt) != 0)
        return -1;
    const ncnn::GpuInfo& info = ncnn::get_gpu_info(ncnn::get_default_gpu_index());
    if (info.support_fp16_storage())
    {
        opt.use_fp16_storage = true;
        if (test_weight_storage_layouts(opt) != 0)
            return -1;
        opt.use_fp16_storage = false;
    }
    if (info.support_bf16_storage())
    {
        opt.use_bf16_storage = true;
        if (test_weight_storage_layouts(opt) != 0)
            return -1;
    }
    return 0;
}

static int test_weight_storage_workspace_reuse()
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
    ncnn::Mat input[18];
    ncnn::VkMat weight[18];
    ncnn::VkTransfer upload(vkdev);
    for (int i = 0; i < 18; i++)
    {
        input[i] = RandomMat(256);
        upload.record_upload(input[i], weight[i], opt);
    }
    if (upload.submit_and_wait() != 0 || weights.finalize_weights() != 0)
    {
        fprintf(stderr, "test_weight_storage_workspace_reuse upload failed\n");
        return -1;
    }

    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;

    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    for (int round = 0; round < 3; round++)
    {
        ncnn::Mat output[18];
        for (int i = 0; i < 18; i++)
            cmd.record_clone(weight[i], output[i], opt);
        if (cmd.submit_and_wait() != 0)
        {
            fprintf(stderr, "test_weight_storage_workspace_reuse submit failed round=%d\n", round);
            return -1;
        }
        for (int i = 0; i < 18; i++)
        {
            if (CompareMat(input[i], output[i], 0.f) != 0)
            {
                fprintf(stderr, "test_weight_storage_workspace_reuse output failed round=%d weight=%d\n", round, i);
                return -1;
            }
        }
        if (cmd.reset() != 0)
        {
            fprintf(stderr, "test_weight_storage_workspace_reuse reset failed\n");
            return -1;
        }
    }
    ncnn::Mat output;
    cmd.record_clone(weight[0], output, opt);
    if (cmd.submit_and_wait() != 0 || CompareMat(input[0], output, 0.f) != 0)
    {
        fprintf(stderr, "test_weight_storage_workspace_reuse pending recording failed\n");
        return -1;
    }
    return 0;
}

static int test_weight_storage_residents()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkWeightStagingAllocator upload_staging(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    if (weights.set_weight_memory_limit(16384, 16384))
        return -1;
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &upload_staging;
    opt.weight_host_memory_limit = 16384;
    ncnn::Mat input[4];
    ncnn::VkMat weight[4];
    ncnn::VkTransfer upload(vkdev);
    for (int i = 0; i < 4; i++)
    {
        input[i] = RandomMat(i < 2 ? 1024 : 2048);
        upload.record_upload(input[i], weight[i], opt);
    }
    if (upload.submit_and_wait() != 0 || weights.finalize_weights() != 0 || upload.reset() != 0)
    {
        fprintf(stderr, "test_weight_storage_residents prepare failed\n");
        return -1;
    }
    uint64_t device_bytes, host_bytes;
    weights.get_weight_memory_usage(device_bytes, host_bytes);
    if (device_bytes > 16384 || host_bytes > 16384)
    {
        fprintf(stderr, "test_weight_storage_residents allocation budget exceeded\n");
        return -1;
    }

    opt.weight_device_memory_limit = 16384 - (size_t)device_bytes;
    opt.weight_host_memory_limit = 16384 - (size_t)host_bytes;
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    ncnn::Mat output[4];
    for (int i = 0; i < 4; i++)
        cmd.record_clone(weight[i], output[i], opt);
    if (cmd.submit_and_wait() != 0)
    {
        fprintf(stderr, "test_weight_storage_residents output failed\n");
        return -1;
    }
    for (int i = 0; i < 4; i++)
        if (CompareMat(input[i], output[i], 0.f) != 0)
            return -1;
    return 0;
}

static int test_weight_storage_automatic_workspace()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkWeightStagingAllocator upload_staging(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    ncnn::Mat input[3];
    ncnn::VkMat weight[3];
    {
        ncnn::VkTransfer upload(vkdev);
        ncnn::Option upload_opt = opt;
        upload_opt.blob_vkallocator = &weights;
        upload_opt.staging_vkallocator = &upload_staging;
        for (int i = 0; i < 3; i++)
        {
            input[i] = RandomMat(256);
            upload.record_upload(input[i], weight[i], upload_opt);
        }
        if (upload.submit_and_wait())
        {
            fprintf(stderr, "test_weight_storage_automatic_workspace failed at line %d\n", __LINE__);
            return -1;
        }
    }
    // automatic sizing also supports direct allocator uploads
    if (weights.finalize_weights())
    {
        fprintf(stderr, "test_weight_storage_automatic_workspace failed at line %d\n", __LINE__);
        return -1;
    }
    const char shader[] = "#version 450\n"
                          "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
                          "layout(binding = 0) readonly buffer A { float a[]; };\n"
                          "layout(binding = 1) readonly buffer B { float b[]; };\n"
                          "layout(binding = 2) readonly buffer C { float c[]; };\n"
                          "layout(binding = 3) writeonly buffer D { float d[]; };\n"
                          "void main() { uint i = gl_GlobalInvocationID.x; if (i < 256) d[i] = a[i] + b[i] + c[i]; }\n";
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(shader, opt, spirv))
    {
        fprintf(stderr, "test_weight_storage_automatic_workspace failed at line %d\n", __LINE__);
        return -1;
    }
    ncnn::Pipeline pipeline(vkdev);
    pipeline.set_local_size_xyz(1, 1, 1);
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>()))
    {
        fprintf(stderr, "test_weight_storage_automatic_workspace failed at line %d\n", __LINE__);
        return -1;
    }
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    ncnn::Mat warm_output;
    cmd.record_clone(weight[0], warm_output, opt);
    if (cmd.submit_and_wait() || CompareMat(input[0], warm_output, 0.f) || cmd.reset())
    {
        fprintf(stderr, "test_weight_storage_automatic_workspace failed at line %d\n", __LINE__);
        return -1;
    }
    ncnn::VkMat result(256, (size_t)4u, &blobs);
    if (result.empty())
    {
        fprintf(stderr, "test_weight_storage_automatic_workspace failed at line %d\n", __LINE__);
        return -1;
    }
    std::vector<ncnn::VkMat> bindings(4);
    for (int i = 0; i < 3; i++)
        bindings[i] = weight[i];
    bindings[3] = result;
    const std::vector<ncnn::vk_constant_type> constants;
    cmd.record_pipeline(&pipeline, bindings, constants, result);
    ncnn::Mat output;
    cmd.record_clone(result, output, opt);
    if (cmd.submit_and_wait())
    {
        fprintf(stderr, "test_weight_storage_automatic_workspace failed at line %d\n", __LINE__);
        return -1;
    }
    ncnn::Mat expected(256);
    for (int i = 0; i < 256; i++)
        ((float*)expected)[i] = ((const float*)input[0])[i] + ((const float*)input[1])[i] + ((const float*)input[2])[i];
    if (CompareMat(expected, output, 0.0001f))
    {
        fprintf(stderr, "test_weight_storage_automatic_workspace failed at line %d\n", __LINE__);
        return -1;
    }
    // Upload the same weights again after reset.
    if (cmd.reset())
    {
        fprintf(stderr, "test_weight_storage_automatic_workspace failed at line %d\n", __LINE__);
        return -1;
    }
    cmd.record_clone(weight[0], warm_output, opt);
    if (cmd.submit_and_wait() || CompareMat(input[0], warm_output, 0.f))
    {
        fprintf(stderr, "test_weight_storage_automatic_workspace failed at line %d\n", __LINE__);
        return -1;
    }
    if (opt.weight_device_memory_limit || opt.weight_host_memory_limit)
    {
        fprintf(stderr, "test_weight_storage_automatic_workspace failed at line %d\n", __LINE__);
        return -1;
    }

    return 0;
}

#if NCNN_BENCHMARK
static bool weight_storage_has_timestamps(const ncnn::VulkanDevice* vkdev)
{
    uint32_t count = 0;
    ncnn::vkGetPhysicalDeviceQueueFamilyProperties(vkdev->info.physical_device(), &count, 0);
    std::vector<VkQueueFamilyProperties> properties(count);
    ncnn::vkGetPhysicalDeviceQueueFamilyProperties(vkdev->info.physical_device(), &count, properties.data());
    const uint32_t family = vkdev->info.compute_queue_family_index();
    return family < count && properties[family].timestampValidBits != 0;
}

static int check_weight_storage_timestamps(ncnn::VkCompute& cmd)
{
    std::vector<uint64_t> timestamps(2, (uint64_t)-1);
    if (cmd.get_query_pool_results(0, 2, timestamps) || timestamps[0] == (uint64_t)-1 || timestamps[1] == (uint64_t)-1)
    {
        fprintf(stderr, "test_weight_storage_mixed_recording timestamp results unavailable\n");
        return -1;
    }
    return 0;
}
#endif

static int test_weight_storage_mixed_recording()
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
    ncnn::Mat input = RandomMat(64);
    ncnn::Mat weight_values = RandomMat(64);
    ncnn::VkMat weight;
    ncnn::VkTransfer upload(vkdev);
    upload.record_upload(weight_values, weight, opt);
    if (upload.submit_and_wait() || weights.finalize_weights())
        return -1;
    opt.blob_vkallocator = &blobs;
    const char shader[] = "#version 450\n"
                          "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
                          "layout(binding = 0) readonly buffer A { float a[]; };\n"
                          "layout(binding = 1) readonly buffer B { float b[]; };\n"
                          "layout(binding = 2) writeonly buffer C { float c[]; };\n"
                          "void main() { uint i = gl_GlobalInvocationID.x; if (i < 64) c[i] = a[i] + b[i]; }\n";
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(shader, opt, spirv))
        return -1;
    ncnn::Pipeline pipeline(vkdev);
    pipeline.set_local_size_xyz(1, 1, 1);
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>()))
        return -1;
    ncnn::Mat expected_prefix(64);
    ncnn::Mat expected_result(64);
    for (int i = 0; i < 64; i++)
    {
        ((float*)expected_prefix.data)[i] = ((const float*)input.data)[i] * 2.f;
        ((float*)expected_result.data)[i] = ((const float*)expected_prefix.data)[i] + ((const float*)weight_values.data)[i];
    }
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
#if NCNN_BENCHMARK
    const bool timestamps = weight_storage_has_timestamps(vkdev);
    if (timestamps && cmd.create_query_pool(2))
        return -1;
#endif
    // No weights, a late weight after ordinary copies/dispatch, a weight first,
    // and ordinary work again on the same reset command.
    const int modes[] = {0, 1, 2, 0};
    for (int round = 0; round < 4; round++)
    {
        blobs.clear();
#if NCNN_BENCHMARK
        if (timestamps)
            cmd.record_write_timestamp(0);
#endif
        ncnn::Mat early_weight;
        if (modes[round] == 2)
            cmd.record_clone(weight, early_weight, opt);
        ncnn::VkMat x;
        ncnn::VkMat copied;
        ncnn::VkMat prefix(64, (size_t)4u, &blobs);
        ncnn::VkMat result(64, (size_t)4u, &blobs);
        ncnn::VkMat ordinary_weight;
        if (prefix.empty() || result.empty())
            return -1;
        cmd.record_clone(input, x, opt);
        cmd.record_clone(x, copied, opt);
        std::vector<ncnn::VkMat> bindings(3);
        bindings[0] = x;
        bindings[1] = copied;
        bindings[2] = prefix;
        cmd.record_pipeline(&pipeline, bindings, std::vector<ncnn::vk_constant_type>(), prefix);
        ncnn::Mat prefix_output;
        cmd.record_clone(prefix, prefix_output, opt);
        if (modes[round] == 0)
            cmd.record_clone(weight_values, ordinary_weight, opt);
        bindings[0] = prefix;
        bindings[1] = modes[round] == 0 ? ordinary_weight : weight;
        bindings[2] = result;
        cmd.record_pipeline(&pipeline, bindings, std::vector<ncnn::vk_constant_type>(), result);
        // Ordinary uploads continue in their own pool after the weight dispatch.
        // They must not affect the command's independent temporary weight storage.
        ncnn::Mat replacement = RandomMat(64);
        ncnn::VkMat reused;
        ncnn::Mat replaced_output;
        cmd.record_clone(replacement, reused, opt);
        cmd.record_clone(reused, replaced_output, opt);
        ncnn::Mat output;
        cmd.record_clone(result, output, opt);
#if NCNN_BENCHMARK
        if (timestamps)
            cmd.record_write_timestamp(1);
#endif
        if (cmd.submit_and_wait() || CompareMat(replacement, replaced_output, 0.f) || CompareMat(expected_prefix, prefix_output, 0.0001f) || CompareMat(expected_result, output, 0.0001f)
                || (modes[round] == 2 && CompareMat(weight_values, early_weight, 0.f)))
        {
            fprintf(stderr, "test_weight_storage_mixed_recording failed round=%d mode=%d push_descriptors=%d\n", round, modes[round], vkdev->info.support_VK_KHR_push_descriptor());
            return -1;
        }
#if NCNN_BENCHMARK
        if (timestamps && check_weight_storage_timestamps(cmd))
            return -1;
#endif
        if (cmd.reset())
            return -1;
#if NCNN_BENCHMARK
        std::vector<uint64_t> stale_timestamps(2);
        if (timestamps && cmd.get_query_pool_results(0, 2, stale_timestamps) == 0)
        {
            fprintf(stderr, "test_weight_storage_mixed_recording reset left timestamp results ready\n");
            return -1;
        }
#endif
    }
#if NCNN_BENCHMARK
    if (timestamps)
    {
        // Query-only recording must not be mistaken for an empty command.
        cmd.record_write_timestamp(0);
        cmd.record_write_timestamp(1);
        if (cmd.submit_and_wait() || check_weight_storage_timestamps(cmd) || cmd.reset())
            return -1;
    }
#endif
    return 0;
}

static int test_weight_storage_recorded_weight_lifetime()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator store(vkdev);
    store.set_weights_in_disk(true);
    store.set_weight_file_directory(weight_storage_test_directory());
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    ncnn::Mat original = RandomMat(128);
    ncnn::VkMat weight(128, (size_t)4u, &store);
    if (weight.empty())
        return -1;
    memcpy(weight.mapped_ptr(), original.data, original.total() * original.elemsize);
    if (store.finalize_weights())
        return -1;
    ncnn::Mat output;
    cmd.record_clone(weight, output, opt);
    weight.release(); // the allocator keeps the recorded source backing alive
    if (cmd.submit_and_wait() || CompareMat(original, output, 0.f) || cmd.reset())
    {
        fprintf(stderr, "test_weight_storage_recorded_weight_lifetime failed\n");
        return -1;
    }
    return 0;
}

static int test_weight_storage_recorded_subview()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator store(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    store.set_weights_in_disk(true);
    store.set_weight_file_directory(weight_storage_test_directory());
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &store;
    opt.staging_vkallocator = &staging;
    int first = 0;
#if NCNN_BATCH
    first = std::max(4, (int)(vkdev->info.buffer_offset_alignment() / sizeof(float)));
#endif
    ncnn::Mat input = RandomMat(first + 257);
    ncnn::Mat bias = RandomMat(7);
    ncnn::VkMat weight;
    ncnn::VkMat bias_gpu;
    ncnn::VkTransfer upload(vkdev);
    upload.record_upload(input, weight, opt);
    upload.record_upload(bias, bias_gpu, opt);
    if (upload.submit_and_wait() || store.finalize_weights())
    {
        fprintf(stderr, "test_weight_storage_recorded_subview upload/finalization failed\n");
        return -1;
    }

    // A legitimate borrowed tensor view is smaller than its backing source.
    // Recording it first must not truncate the temporary allocation or its padding.
    ncnn::VkMat view(7, weight.data, (size_t)4u, &store);
#if NCNN_BATCH
    view.offset = first * sizeof(float);
#endif
    const size_t logical_bytes = (size_t)input.w * input.elemsize;
    if (view.buffer_capacity() >= logical_bytes || weight.data->capacity <= logical_bytes || !weight.mapped_ptr())
    {
        fprintf(stderr, "test_weight_storage_recorded_subview invalid fixture view=%zu capacity=%zu logical=%zu\n", view.buffer_capacity(), weight.data->capacity, logical_bytes);
        return -1;
    }
    const unsigned char* mapped_ptr = (const unsigned char*)weight.mapped_ptr();
    for (size_t i = weight.buffer_capacity(); i < weight.data->capacity; i++)
    {
        if (mapped_ptr[i] != 0)
        {
            fprintf(stderr, "test_weight_storage_recorded_subview nonzero allocation padding at %zu\n", i);
            return -1;
        }
    }
    ncnn::Mat expected_view(7);
    memcpy(expected_view.data, (const float*)input.data + first, 7 * sizeof(float));
    opt.blob_vkallocator = &blobs;
    const char shader[] = "#version 450\n"
                          "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n"
                          "layout(binding = 0) readonly buffer A { float a[]; };\n"
                          "layout(binding = 1) readonly buffer B { float b[]; };\n"
                          "layout(binding = 2) writeonly buffer C { float c[]; };\n"
                          "void main() { uint i = gl_GlobalInvocationID.x; if (i < 7) c[i] = a[i] + b[i]; }\n";
    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(shader, opt, spirv))
        return -1;
    ncnn::Pipeline pipeline(vkdev);
    pipeline.set_local_size_xyz(1, 1, 1);
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>()))
        return -1;
    ncnn::VkMat result(7, (size_t)4u, &blobs);
    if (result.empty())
        return -1;

#if NCNN_WEIGHT_STORAGE_NATIVE_PROBE
    WeightStorageUploadProbe probe(vkdev);
#endif
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    ncnn::Mat small_output;
    ncnn::Mat full_output;
    cmd.record_clone(view, small_output, opt);
    cmd.record_clone(weight, full_output, opt);
#if NCNN_WEIGHT_STORAGE_NATIVE_PROBE
    // The preceding full allocation remains leased while the next view is
    // resolved. Adjacent groups may share a backing buffer, but not its range.
    if (probe.copies.empty())
        return -1;
    const WeightStorageUploadProbe::Copy& first_upload = probe.copies[0];
    int uploads = 0;
    for (size_t i = 0; i < probe.copies.size(); i++)
    {
        const WeightStorageUploadProbe::Copy& copy = probe.copies[i];
        if (copy.source != first_upload.source)
            continue;
        uploads++;
        if (copy.region.size != weight.data->capacity
                || (i != 0 && copy.target == first_upload.target
                    && copy.region.dstOffset < first_upload.region.dstOffset + first_upload.region.size
                    && first_upload.region.dstOffset < copy.region.dstOffset + copy.region.size))
        {
            fprintf(stderr, "adjacent resolved groups overlapped or lost their complete source extent\n");
            return -1;
        }
    }
    if (uploads != 2)
    {
        fprintf(stderr, "adjacent subview resolves did not record two complete uploads\n");
        return -1;
    }
#endif
    if (cmd.submit_and_wait() || CompareMat(expected_view, small_output, 0.f) || CompareMat(input, full_output, 0.f) || cmd.reset())
    {
        fprintf(stderr, "test_weight_storage_recorded_subview lost source extent or view offset\n");
        return -1;
    }
    // recorded source addresses remain valid after caller handles are released
    ncnn::VkMat owning_view = weight;
    owning_view.w = view.w;
    owning_view.cstep = view.cstep;
#if NCNN_BATCH
    owning_view.nstep = view.nstep;
    owning_view.offset = view.offset;
#endif
    std::vector<ncnn::VkMat> bindings(3);
    bindings[0] = owning_view;
    bindings[1] = bias_gpu;
    bindings[2] = result;
    cmd.record_pipeline(&pipeline, bindings, std::vector<ncnn::vk_constant_type>(), result);
    cmd.record_clone(result, small_output, opt);
    bindings.clear();
    ncnn::Mat expected(7);
    for (int i = 0; i < 7; i++)
        ((float*)expected)[i] = ((const float*)input)[first + i] + ((const float*)bias)[i];
    owning_view.release();
    weight.release();
    bias_gpu.release();
    view.release();
    if (cmd.submit_and_wait() || CompareMat(expected, small_output, 0.0001f) || cmd.reset())
    {
        fprintf(stderr, "test_weight_storage_recorded_subview lost recorded source data\n");
        return -1;
    }
    return 0;
}

class FailingAllocator : public ncnn::Allocator
{
public:
    virtual void* fastMalloc(size_t)
    {
        return 0;
    }
    virtual void fastFree(void*)
    {
    }
};

class FailingStagingAllocator : public ncnn::VkStagingAllocator
{
public:
    FailingStagingAllocator(const ncnn::VulkanDevice* vkdev, bool unmapped)
        : ncnn::VkStagingAllocator(vkdev), unmapped(unmapped), allocations(0), mapping(0)
    {
    }

    virtual ncnn::VkBufferMemory* fastMalloc(size_t size)
    {
        allocations++;
        if (!unmapped)
            return 0;
        ncnn::VkBufferMemory* memory = ncnn::VkStagingAllocator::fastMalloc(size);
        if (memory)
        {
            mapping = memory->mapped_ptr;
            memory->mapped_ptr = 0;
        }
        return memory;
    }
    virtual void fastFree(ncnn::VkBufferMemory* memory)
    {
        memory->mapped_ptr = mapping;
        ncnn::VkStagingAllocator::fastFree(memory);
    }

    bool unmapped;
    int allocations;
    void* mapping;
};

static int check_weight_storage_failure(ncnn::VkCompute& cmd, const ncnn::VkMat& weight, const ncnn::Mat& input, const ncnn::Option& opt, const char* operation, bool output_empty, int allocations = 0)
{
    if (!output_empty || allocations > 1)
    {
        fprintf(stderr, "test_weight_storage_allocation_failure output not empty operation=%s\n", operation);
        return -1;
    }
    if (cmd.reset() != 0)
    {
        fprintf(stderr, "test_weight_storage_allocation_failure reset failed operation=%s\n", operation);
        return -1;
    }
    ncnn::Mat output;
    cmd.record_clone(weight, output, opt);
    if (cmd.submit_and_wait() != 0 || CompareMat(input, output, 0.001f) != 0 || cmd.reset() != 0)
    {
        fprintf(stderr, "test_weight_storage_allocation_failure recovery failed operation=%s\n", operation);
        return -1;
    }
    return 0;
}

static int test_weight_storage_allocation_failure(bool disk)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    FailingAllocator cpu;
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    const ncnn::Mat input = RandomMat(127);
    ncnn::VkMat native;
    {
        ncnn::VkCompute upload(vkdev);
        upload.record_clone(input, native, opt);
        if (upload.submit_and_wait() != 0)
        {
            fprintf(stderr, "test_weight_storage_allocation_failure native upload failed\n");
            return -1;
        }
    }
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkWeightStagingAllocator weight_staging(vkdev);
    ncnn::VkMat file;
    weights.set_weights_in_disk(true);
    weights.set_weight_file_directory(weight_storage_test_directory());
    {
        ncnn::Option upload_opt = opt;
        upload_opt.blob_vkallocator = &weights;
        upload_opt.staging_vkallocator = &weight_staging;
        ncnn::VkTransfer upload(vkdev);
        upload.record_upload(input, file, upload_opt);
        if (upload.submit_and_wait() != 0 || weights.finalize_weights() != 0)
        {
            fprintf(stderr, "test_weight_storage_allocation_failure disk upload failed\n");
            return -1;
        }
    }
    const ncnn::VkMat& src = disk ? file : native;
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    ncnn::Mat output;
    ncnn::VkMat device_output;

    ncnn::Option bad = opt;
    bad.blob_allocator = &cpu;
    cmd.record_clone(src, output, bad);
    if (check_weight_storage_failure(cmd, src, input, opt, "clone cpu allocation", output.empty()) != 0)
        return -1;
    cmd.record_download(src, output, bad);
    if (check_weight_storage_failure(cmd, src, input, opt, "download cpu allocation", output.empty()) != 0)
        return -1;

    {
        FailingStagingAllocator failing(vkdev, false);
        bad = opt;
        bad.staging_vkallocator = &failing;
        cmd.record_upload(input, device_output, bad);
        if (check_weight_storage_failure(cmd, src, input, opt, "upload staging allocation", device_output.empty(), failing.allocations) != 0)
            return -1;
    }
    {
        FailingStagingAllocator failing(vkdev, false);
        bad = opt;
        bad.staging_vkallocator = &failing;
        // exercise device-to-host copies even when the source is mapped
        void* mapping = src.data->mapped_ptr;
        src.data->mapped_ptr = 0;
        cmd.record_clone(src, output, bad);
        src.data->mapped_ptr = mapping;
        if (check_weight_storage_failure(cmd, src, input, opt, "clone staging allocation", output.empty(), failing.allocations) != 0)
            return -1;
    }
    {
        FailingStagingAllocator failing(vkdev, false);
        bad = opt;
        bad.blob_vkallocator = &failing;
        cmd.record_clone(src, device_output, bad);
        if (check_weight_storage_failure(cmd, src, input, opt, "clone device allocation", device_output.empty(), failing.allocations) != 0)
            return -1;
    }
    {
        FailingStagingAllocator failing(vkdev, false);
        bad = opt;
        bad.blob_vkallocator = &failing;
        bad.staging_vkallocator = &failing;
        bad.use_packing_layout = true;
        ncnn::VkMat packable(128, (size_t)4u, &blobs);
        cmd.record_download(packable, output, bad);
        if (check_weight_storage_failure(cmd, src, input, opt, "download packing allocation", output.empty(), failing.allocations) != 0)
            return -1;
    }
    return 0;
}

static int test_weight_storage_workspace_chunks()
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
    // Cross the staging chunk boundary; the native probe forces staging on unified devices.
    ncnn::Mat input(16 * 1024 * 1024 / 4 + 17);
    for (int i = 0; i < input.w; i++)
        ((float*)input)[i] = (float)(i % 257);
    ncnn::VkMat weight;
    ncnn::VkTransfer upload(vkdev);
    upload.record_upload(input, weight, opt);
    if (upload.submit_and_wait() || weights.finalize_weights())
        return -1;
    opt.blob_vkallocator = &blobs;
#if NCNN_WEIGHT_STORAGE_NATIVE_PROBE
    WeightStorageUploadProbe probe(vkdev);
#endif
    ncnn::VkCompute cmd(vkdev);
    cmd.set_use_weight_upload_online(true);
    ncnn::Mat output;
    cmd.record_clone(weight, output, opt);
#if NCNN_WEIGHT_STORAGE_NATIVE_PROBE
    if (probe.copies.size() < 2)
    {
        fprintf(stderr, "temporary weight did not record both staging chunks\n");
        return -1;
    }
    const WeightStorageUploadProbe::Copy first = probe.copies[0];
    const WeightStorageUploadProbe::Copy second = probe.copies[1];
    const VkDeviceSize chunk_size = (VkDeviceSize)16 * 1024 * 1024;
    if (first.source != second.source || first.target != second.target
            || first.region.srcOffset != second.region.srcOffset
            || first.region.size != chunk_size || second.region.size != weight.data->capacity - (size_t)chunk_size
            || second.region.dstOffset != first.region.dstOffset + chunk_size)
    {
        fprintf(stderr, "temporary weight staging chunks lost their complete allocation or reusable staging range\n");
        return -1;
    }
#endif
    if (cmd.submit_and_wait() || CompareMat(input, output, 0.f))
    {
        fprintf(stderr, "temporary weight staging chunks corrupted the complete allocation\n");
        return -1;
    }
#if NCNN_WEIGHT_STORAGE_NATIVE_PROBE
    bool exclusive = false;
    for (size_t i = 0; i < probe.buffers.size(); i++)
        if (probe.buffers[i].handle == first.target)
            exclusive = probe.buffers[i].sharing_mode == VK_SHARING_MODE_EXCLUSIVE;
    int releases = 0;
    int acquires = 0;
    for (size_t i = 0; i < probe.barriers.size(); i++)
    {
        const WeightStorageUploadProbe::Barrier& entry = probe.barriers[i];
        const VkBufferMemoryBarrier& barrier = entry.buffer_barrier;
        if (barrier.buffer != first.target || barrier.srcQueueFamilyIndex == barrier.dstQueueFamilyIndex)
            continue;
        if (barrier.srcQueueFamilyIndex != vkdev->info.transfer_queue_family_index()
                || barrier.dstQueueFamilyIndex != vkdev->info.compute_queue_family_index()
                || barrier.offset != first.region.dstOffset || barrier.size != weight.data->capacity)
        {
            fprintf(stderr, "temporary weight ownership transfer lost its complete allocation\n");
            return -1;
        }
        if (entry.command == second.command && barrier.srcAccessMask == VK_ACCESS_TRANSFER_WRITE_BIT && barrier.dstAccessMask == 0
                && entry.src_stage == VK_PIPELINE_STAGE_TRANSFER_BIT && entry.dst_stage == VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
            releases++;
        else if (entry.command != second.command && barrier.srcAccessMask == 0
                 && barrier.dstAccessMask == (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT)
                 && entry.src_stage == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                 && entry.dst_stage == (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT))
            acquires++;
    }
    const int transfers = vkdev->info.unified_compute_transfer_queue() ? 0 : 1;
    if (!exclusive || releases != transfers || acquires != transfers)
    {
        fprintf(stderr, "temporary weight requires exclusive sharing and matching upload release / compute acquire\n");
        return -1;
    }
#endif
    return 0;
}

static int test_weight_storage_upload_recording()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    UnmappedWeightAllocator weights(vkdev);
    ncnn::VkWeightAllocator host(vkdev, true);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.staging_vkallocator = &staging;
    ncnn::VkAllocator* allocators[3] = {&weights, &staging, &host};
    for (int i = 0; i < 3; i++)
    {
        opt.blob_vkallocator = allocators[i];
        ncnn::Mat source(7, (size_t)1u);
        memset(source.data, 0, source.total() * source.elemsize);
        memset(source.data, 37, 7);
        ncnn::VkMat target;
        ncnn::VkTransfer upload(vkdev);
        upload.record_upload(source, target, opt);
        if (target.empty())
            return -1;
        if ((i == 1 && !target.mapped_ptr()) || (target.mapped_ptr() && memcmp(target.mapped_ptr(), source.data, 7) != 0))
        {
            fprintf(stderr, "mapped weight was not filled while recording mode=%d\n", i);
            return -1;
        }
        memset(source.data, 91, 7);
        source.release();
        if ((i == 0 && !upload.pending_upload_total()) || upload.submit_and_wait() || upload.reset())
            return -1;
        ncnn::VkCompute download(vkdev);
        ncnn::Mat result;
        download.record_clone(target, result, opt);
        if (download.submit_and_wait())
            return -1;
        for (int j = 0; j < 7; j++)
            if (((const unsigned char*)result.data)[j] != 37)
                return -1;
        for (size_t j = 7; j < target.buffer_capacity(); j++)
            if (((const unsigned char*)result.data)[j] != 0)
                return -1;
    }
    opt.blob_vkallocator = &weights;
    for (int unmapped = 0; unmapped < 2; unmapped++)
    {
        FailingStagingAllocator failing(vkdev, unmapped != 0);
        ncnn::Option bad = opt;
        bad.staging_vkallocator = &failing;
        ncnn::Mat bytes(7, (size_t)1u);
        bytes.fill((unsigned char)19);
        ncnn::VkMat target;
        ncnn::VkTransfer rejected(vkdev);
        rejected.record_upload(bytes, target, bad);
        if (rejected.pending_upload_total() != 0 || rejected.submit_and_wait() != 0 || rejected.reset() != 0 || failing.allocations != 1)
        {
            fprintf(stderr, "test_weight_storage_upload_recording staging failure missing unmapped=%d\n", unmapped);
            return -1;
        }
        rejected.record_upload(bytes, target, opt);
        if (!rejected.pending_upload_total() || rejected.submit_and_wait() != 0)
            return -1;
        ncnn::VkCompute download(vkdev);
        ncnn::Mat result;
        download.record_clone(target, result, opt);
        if (download.submit_and_wait() != 0 || memcmp(result.data, bytes.data, 7) != 0)
            return -1;
    }
    return 0;
}

static int test_weight_storage_upload_chunks(bool custom_staging)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    UnmappedWeightAllocator weights(vkdev);
    ncnn::VkWeightStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = custom_staging ? &staging : 0;
    opt.weight_host_memory_limit = 16384;
    ncnn::VkTransfer upload(vkdev);
#if NCNN_WEIGHT_STORAGE_NATIVE_PROBE
    WeightStorageUploadProbe probe(vkdev);
    probe.hide_next_map = false;
    VkBuffer source_buffer = 0;
#endif
    const int sizes[2] = {7, 65543};
    for (int round = 0; round < 2; round++)
    {
        ncnn::VkMat targets[2];
#if NCNN_WEIGHT_STORAGE_NATIVE_PROBE
        const size_t first_copy = probe.copies.size();
#endif
        for (int i = 0; i < 2; i++)
        {
            ncnn::Mat source(sizes[i], (size_t)1u);
            memset(source.data, 0, source.total() * source.elemsize);
            for (int j = 0; j < sizes[i]; j++)
                ((unsigned char*)source.data)[j] = (unsigned char)(j * 17 + i + round);
            upload.record_upload(source, targets[i], opt);
            if (targets[i].empty())
                return -1;
            memset(source.data, 91, sizes[i]);
        }
#if NCNN_WEIGHT_STORAGE_NATIVE_PROBE
        if (probe.copies.size() - first_copy < 3)
        {
            fprintf(stderr, "weight copies were not recorded before submit\n");
            return -1;
        }
        if (!source_buffer)
            source_buffer = probe.copies[first_copy].source;
        for (size_t i = first_copy; i < probe.copies.size(); i++)
        {
            if (probe.copies[i].source != source_buffer || probe.copies[i].region.size > opt.weight_host_memory_limit)
            {
                fprintf(stderr, "weight upload did not reuse bounded staging\n");
                return -1;
            }
        }
#endif
        if (upload.submit_and_wait() || upload.reset())
        {
            fprintf(stderr, "chunked weight upload failed custom=%d round=%d\n", custom_staging, round);
            return -1;
        }
        ncnn::VkCompute download(vkdev);
        ncnn::Mat results[2];
        ncnn::Option download_opt = opt;
        download_opt.staging_vkallocator = &staging;
        for (int i = 0; i < 2; i++)
            download.record_clone(targets[i], results[i], download_opt);
        if (download.submit_and_wait())
        {
            fprintf(stderr, "chunked weight download failed custom=%d round=%d\n", custom_staging, round);
            return -1;
        }
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < sizes[i]; j++)
                if (((const unsigned char*)results[i].data)[j] != (unsigned char)(j * 17 + i + round))
                {
                    fprintf(stderr, "chunked weight value mismatch custom=%d round=%d weight=%d byte=%d actual=%d\n", custom_staging, round, i, j, ((const unsigned char*)results[i].data)[j]);
                    return -1;
                }
            for (size_t j = sizes[i]; j < targets[i].buffer_capacity(); j++)
                if (((const unsigned char*)results[i].data)[j] != 0)
                {
                    fprintf(stderr, "chunked weight padding mismatch custom=%d round=%d weight=%d byte=%lu\n", custom_staging, round, i, (unsigned long)j);
                    return -1;
                }
        }
    }
    return 0;
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
                    || test_weight_storage_upload_mode()
                    || test_weight_storage_workspace_chunks()
                    || test_weight_storage_upload_recording()
                    || test_weight_storage_upload_chunks(false)
                    || test_weight_storage_upload_chunks(true)
                    || test_weight_storage_precisions()
                    || test_weight_storage_workspace_reuse()

                    || test_weight_storage_allocation_failure(false)
                    || test_weight_storage_allocation_failure(true)
                    || test_weight_storage_residents()
                    || test_weight_storage_automatic_workspace()
                    || test_weight_storage_mixed_recording()
                    || test_weight_storage_recorded_weight_lifetime()
                    || test_weight_storage_recorded_subview();
    if (directory.remove() != 0)
    {
        fprintf(stderr, "weight storage temporary files leaked after tests\n");
        return 1;
    }
    return ret;
}
