// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_weight_storage.h"
#include "pipeline.h"

static int test_weight_storage_bytes(const ncnn::Option& opt, int elempack, bool flatten, int batch, bool host)
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkWeightAllocator disk(vkdev);
    ncnn::VkWeightAllocator resident(vkdev, host);
    ncnn::VkWeightStagingAllocator staging(vkdev);
    disk.set_weights_in_disk(true);
    disk.set_weight_file_directory(weight_storage_test_directory());

    ncnn::Mat src(7, 3, 2, 2, (size_t)4u * elempack, elempack, batch);
    for (int b = 0; b < batch; b++)
    {
        ncnn::Mat m = src.batch(b);
        float* p = m;
        for (size_t i = 0; i < m.total() * elempack; i++)
            p[i] = (float)((int)(i % 53) - 26) * 0.125f + b;
    }

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
    if (cmd.set_weight_memory_limit(download_opt.weight_device_memory_limit, download_opt.weight_host_memory_limit, download_opt.weight_device_cache_size, download_opt.weight_host_cache_size) != 0)
        return -1;
    ncnn::Mat expected;
    ncnn::Mat actual;
    cmd.record_clone(resident_weight, expected, download_opt);
    cmd.record_clone(file_weight, actual, download_opt);
    if (cmd.submit_and_wait() != 0)
    {
        fprintf(stderr, "test_weight_storage_bytes download failed\n");
        return -1;
    }
    for (int b = 0; b < batch; b++)
    {
        const ncnn::Mat expected_b = expected.batch(b);
        const ncnn::Mat actual_b = actual.batch(b);
        for (int q = 0; q < expected_b.c; q++)
        {
            size_t size = (size_t)expected_b.w * expected_b.h * expected_b.d * expected_b.elemsize;
            size_t offset = q * expected_b.cstep * expected_b.elemsize;
            if (memcmp((const unsigned char*)actual_b.data + offset, (const unsigned char*)expected_b.data + offset, size) != 0)
            {
                fprintf(stderr, "test_weight_storage_bytes failed pack=%d flatten=%d batch=%d host=%d fp16=%d bf16=%d\n", elempack, flatten, batch, host, opt.use_fp16_storage, opt.use_bf16_storage);
                return -1;
            }
        }
    }
    return 0;
}

static int test_weight_storage_layouts(const ncnn::Option& opt)
{
    for (int elempack = 1; elempack <= 4; elempack *= 4)
    {
        if (test_weight_storage_bytes(opt, elempack, true, 1, false) != 0
                || test_weight_storage_bytes(opt, elempack, false, 1, false) != 0
                || test_weight_storage_bytes(opt, elempack, false, 1, true) != 0)
            return -1;
#if NCNN_BATCH
        if (test_weight_storage_bytes(opt, elempack, true, 2, false) != 0
                || test_weight_storage_bytes(opt, elempack, false, 2, true) != 0)
            return -1;
#endif // NCNN_BATCH
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

static int test_weight_storage_cache()
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
        fprintf(stderr, "test_weight_storage_cache upload failed\n");
        return -1;
    }

    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_device_cache_size = 2048;
    opt.weight_host_cache_size = 256;
    ncnn::VkCompute cmd(vkdev);
    if (cmd.set_weight_memory_limit(opt.weight_device_memory_limit, opt.weight_host_memory_limit, opt.weight_device_cache_size, opt.weight_host_cache_size) != 0)
        return -1;
    for (int round = 0; round < 3; round++)
    {
        ncnn::Mat output[18];
        for (int i = 0; i < 18; i++)
            cmd.record_clone(weight[i], output[i], opt);
        if (cmd.submit_and_wait() != 0)
        {
            fprintf(stderr, "test_weight_storage_cache submit failed round=%d\n", round);
            return -1;
        }
        for (int i = 0; i < 18; i++)
        {
            if (CompareMat(input[i], output[i], 0.f) != 0)
            {
                fprintf(stderr, "test_weight_storage_cache output failed round=%d weight=%d\n", round, i);
                return -1;
            }
        }
        if (cmd.reset() != 0)
        {
            fprintf(stderr, "test_weight_storage_cache reset failed\n");
            return -1;
        }
    }
    if (cmd.clear_weight_cache() != 0)
    {
        fprintf(stderr, "test_weight_storage_cache clear failed\n");
        return -1;
    }
    ncnn::Mat output;
    cmd.record_clone(weight[0], output, opt);
    if (cmd.clear_weight_cache() == 0 || cmd.submit_and_wait() != 0 || CompareMat(input[0], output, 0.f) != 0)
    {
        fprintf(stderr, "test_weight_storage_cache pending recording failed\n");
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
    if (weights.set_weight_memory_limit(16384, 16384, 8192, 4096))
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
    if (device_bytes >= 16384 || host_bytes > 16384 || !weights.needs_weight_cache())
    {
        fprintf(stderr, "test_weight_storage_residents allocation budget exceeded\n");
        return -1;
    }
    opt.weight_device_cache_size = 8192;
    opt.weight_host_cache_size = 4096;
    opt.weight_device_memory_limit = 16384 - (size_t)device_bytes;
    opt.weight_host_memory_limit = 16384 - (size_t)host_bytes;
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    ncnn::VkCompute cmd(vkdev);
    if (cmd.set_weight_memory_limit(opt.weight_device_memory_limit, opt.weight_host_memory_limit, opt.weight_device_cache_size, opt.weight_host_cache_size) != 0)
        return -1;
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

static int test_weight_storage_automatic_cache()
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
            fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
            return -1;
        }
    }
    {
        ncnn::VkCompute early(vkdev);
        if (early.set_weight_memory_limit(opt.weight_device_memory_limit, opt.weight_host_memory_limit, opt.weight_device_cache_size, opt.weight_host_cache_size) != 0)
            return -1;
        ncnn::Mat output;
        early.record_clone(weight[0], output, opt);
        if (early.submit_and_wait() == 0 || early.reset())
        {
            fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
            return -1;
        }
        early.record_clone(weight[0], output, opt);
        if (early.submit_and_wait() == 0)
        {
            fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
            return -1;
        }
    }
    // automatic sizing also supports direct allocator uploads
    if (weights.finalize_weights())
    {
        fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
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
        fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
        return -1;
    }
    ncnn::Pipeline pipeline(vkdev);
    pipeline.set_local_size_xyz(1, 1, 1);
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), std::vector<ncnn::vk_specialization_type>()))
    {
        fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
        return -1;
    }
    ncnn::VkCompute cmd(vkdev);
    if (cmd.set_weight_memory_limit(opt.weight_device_memory_limit, opt.weight_host_memory_limit, opt.weight_device_cache_size, opt.weight_host_cache_size) != 0)
        return -1;
    ncnn::Mat warm_output;
    cmd.record_clone(weight[0], warm_output, opt);
    if (cmd.submit_and_wait() || CompareMat(input[0], warm_output, 0.f) || cmd.reset())
    {
        fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
        return -1;
    }
    ncnn::VkMat result(256, (size_t)4u, &blobs);
    if (result.empty())
    {
        fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
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
        fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
        return -1;
    }
    ncnn::Mat expected(256);
    for (int i = 0; i < 256; i++)
        ((float*)expected)[i] = ((const float*)input[0])[i] + ((const float*)input[1])[i] + ((const float*)input[2])[i];
    if (CompareMat(expected, output, 0.0001f))
    {
        fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
        return -1;
    }
    // reuse cached weights after automatic growth
    if (cmd.reset())
    {
        fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
        return -1;
    }
    cmd.record_clone(weight[0], warm_output, opt);
    if (cmd.submit_and_wait() || CompareMat(input[0], warm_output, 0.f))
    {
        fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
        return -1;
    }
    if (opt.weight_device_cache_size || opt.weight_host_cache_size || opt.weight_device_memory_limit || opt.weight_host_memory_limit)
    {
        fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
        return -1;
    }
    ncnn::Option fixed_opt = opt;
    fixed_opt.weight_device_cache_size = 2048;
    ncnn::VkCompute fixed(vkdev);
    if (fixed.set_weight_memory_limit(fixed_opt.weight_device_memory_limit, fixed_opt.weight_host_memory_limit, fixed_opt.weight_device_cache_size, fixed_opt.weight_host_cache_size) != 0)
        return -1;
    fixed.record_pipeline(&pipeline, bindings, constants, result);
    if (fixed.submit_and_wait() == 0)
    {
        fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
        return -1;
    }
    // A rejected whole-dispatch capacity check must not poison later executions.
    ncnn::Mat recovered;
    if (fixed.reset())
        return -1;
    fixed.record_clone(weight[0], recovered, fixed_opt);
    if (fixed.set_weight_memory_limit(1, 1, 1, 1) == 0)
    {
        fprintf(stderr, "weight cache settings changed while recording\n");
        return -1;
    }
    if (fixed.submit_and_wait() || CompareMat(input[0], recovered, 0.f))
    {
        fprintf(stderr, "test_weight_storage_automatic_cache fixed capacity recovery failed\n");
        return -1;
    }
    if (fixed.reset() || fixed.set_weight_memory_limit(0, 0, 4096, 256) == 0)
    {
        fprintf(stderr, "weight cache settings changed without clearing retained backing\n");
        return -1;
    }
    if (fixed.clear_weight_cache() || fixed.set_weight_memory_limit(0, 0, 4096, 256))
        return -1;
    fixed.record_pipeline(&pipeline, bindings, constants, result);
    fixed.record_clone(result, recovered, opt);
    if (fixed.submit_and_wait() || CompareMat(expected, recovered, 0.0001f))
    {
        fprintf(stderr, "weight cache reconfiguration failed\n");
        return -1;
    }
    // Individual uploads do not predeclare a group: the actual three-input
    // dispatch must still reject a hard memory budget that cannot hold it.
    ncnn::Option limited_opt = opt;
    limited_opt.weight_device_memory_limit = 2048;
    limited_opt.weight_host_memory_limit = 2048;
    ncnn::VkCompute limited(vkdev);
    if (limited.set_weight_memory_limit(limited_opt.weight_device_memory_limit, limited_opt.weight_host_memory_limit, limited_opt.weight_device_cache_size, limited_opt.weight_host_cache_size) != 0)
        return -1;
    limited.record_pipeline(&pipeline, bindings, constants, result);
    if (limited.submit_and_wait() == 0)
    {
        fprintf(stderr, "test_weight_storage_automatic_cache ignored simultaneous binding budget\n");
        return -1;
    }
    if (limited.reset())
        return -1;
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
    if (cmd.set_weight_memory_limit(opt.weight_device_memory_limit, opt.weight_host_memory_limit, opt.weight_device_cache_size, opt.weight_host_cache_size) != 0)
        return -1;
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
        ncnn::Mat output;
        cmd.record_clone(result, output, opt);
#if NCNN_BENCHMARK
        if (timestamps)
            cmd.record_write_timestamp(1);
#endif
        if (cmd.submit_and_wait() || CompareMat(expected_prefix, prefix_output, 0.0001f) || CompareMat(expected_result, output, 0.0001f)
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
    if (cmd.set_weight_memory_limit(opt.weight_device_memory_limit, opt.weight_host_memory_limit, opt.weight_device_cache_size, opt.weight_host_cache_size) != 0)
        return -1;
    ncnn::Mat original = RandomMat(128);
    ncnn::VkMat weight(128, (size_t)4u, &store);
    if (weight.empty())
        return -1;
    memcpy(weight.data->source_ptr, original.data, original.total() * original.elemsize);
    if (store.finalize_weights())
        return -1;
    ncnn::Mat output;
    cmd.record_clone(weight, output, opt);
    weight.release(); // The command retains its source; the allocator remains alive.
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
    ncnn::Mat input = RandomMat(257);
    ncnn::VkMat weight;
    ncnn::VkTransfer upload(vkdev);
    upload.record_upload(input, weight, opt);
    if (upload.submit_and_wait() || store.finalize_weights())
    {
        fprintf(stderr, "test_weight_storage_recorded_subview upload/finalization failed\n");
        return -1;
    }

    // A legitimate borrowed tensor view is smaller than its backing source.
    // Recording it first must not truncate the cached allocation or its padding.
    ncnn::VkMat view(7, weight.data, (size_t)4u, &store);
    int first = 0;
#if NCNN_BATCH
    first = 4;
    view.offset = first * sizeof(float);
#endif
    const size_t logical_bytes = (size_t)input.w * input.elemsize;
    if (view.buffer_capacity() >= weight.data->source_size || weight.data->source_size < logical_bytes || weight.data->capacity <= logical_bytes)
    {
        fprintf(stderr, "test_weight_storage_recorded_subview invalid fixture view=%zu source=%zu capacity=%zu logical=%zu\n", view.buffer_capacity(), weight.data->source_size, weight.data->capacity, logical_bytes);
        return -1;
    }
    ncnn::Mat expected_view(7);
    memcpy(expected_view.data, (const float*)input.data + first, 7 * sizeof(float));
    opt.blob_vkallocator = &blobs;
    ncnn::VkCompute cmd(vkdev);
    if (cmd.set_weight_memory_limit(opt.weight_device_memory_limit, opt.weight_host_memory_limit, opt.weight_device_cache_size, opt.weight_host_cache_size) != 0)
        return -1;
    ncnn::Mat small_output;
    ncnn::Mat full_output;
    cmd.record_clone(view, small_output, opt);
    cmd.record_clone(weight, full_output, opt);
    if (cmd.submit_and_wait() || CompareMat(expected_view, small_output, 0.f) || CompareMat(input, full_output, 0.f) || cmd.reset())
    {
        fprintf(stderr, "test_weight_storage_recorded_subview lost source extent or view offset\n");
        return -1;
    }
    // Unlike the borrowed view, this narrowed view retains the allocation.
    // It must remain valid after both caller-owned handles are released.
    ncnn::VkMat owning_view = weight;
    owning_view.w = view.w;
    owning_view.cstep = view.cstep;
#if NCNN_BATCH
    owning_view.nstep = view.nstep;
    owning_view.offset = view.offset;
#endif
    cmd.record_clone(owning_view, small_output, opt);
    cmd.record_clone(weight, full_output, opt);
    owning_view.release();
    weight.release();
    view.release();
    if (cmd.submit_and_wait() || CompareMat(expected_view, small_output, 0.f) || CompareMat(input, full_output, 0.f) || cmd.reset())
    {
        fprintf(stderr, "test_weight_storage_recorded_subview failed to retain owning views\n");
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

static int check_weight_storage_failure(ncnn::VkCompute& cmd, const ncnn::VkMat& weight, const ncnn::Mat& input, const ncnn::Option& opt, const char* operation, int allocations = 0)
{
    if (cmd.submit_and_wait() == 0 || allocations > 1)
    {
        fprintf(stderr, "test_weight_storage_allocation_failure error missing operation=%s\n", operation);
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
    ncnn::Mat output;
    ncnn::VkMat device_output;

    cmd.record_clone(ncnn::VkMat(), output, opt);
    if (check_weight_storage_failure(cmd, src, input, opt, "clone empty") != 0)
        return -1;
    cmd.record_download(ncnn::VkMat(), output, opt);
    if (check_weight_storage_failure(cmd, src, input, opt, "download empty") != 0)
        return -1;

    ncnn::Option bad = opt;
    bad.blob_allocator = &cpu;
    cmd.record_clone(src, output, bad);
    if (check_weight_storage_failure(cmd, src, input, opt, "clone cpu allocation") != 0)
        return -1;
    cmd.record_download(src, output, bad);
    if (check_weight_storage_failure(cmd, src, input, opt, "download cpu allocation") != 0)
        return -1;

    {
        FailingStagingAllocator failing(vkdev, false);
        bad = opt;
        bad.staging_vkallocator = &failing;
        cmd.record_upload(input, device_output, bad);
        if (check_weight_storage_failure(cmd, src, input, opt, "upload staging allocation", failing.allocations) != 0)
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
        if (check_weight_storage_failure(cmd, src, input, opt, "clone staging allocation", failing.allocations) != 0)
            return -1;
    }
    {
        FailingStagingAllocator failing(vkdev, true);
        bad = opt;
        bad.staging_vkallocator = &failing;
        cmd.record_upload(input, device_output, bad);
        if (check_weight_storage_failure(cmd, src, input, opt, "upload staging mapping", failing.allocations) != 0)
            return -1;
    }
    {
        FailingStagingAllocator failing(vkdev, true);
        bad = opt;
        bad.staging_vkallocator = &failing;
        // exercise device-to-host copies even when the source is mapped
        void* mapping = src.data->mapped_ptr;
        src.data->mapped_ptr = 0;
        cmd.record_clone(src, output, bad);
        src.data->mapped_ptr = mapping;
        if (check_weight_storage_failure(cmd, src, input, opt, "clone staging mapping", failing.allocations) != 0)
            return -1;
    }
    {
        FailingStagingAllocator failing(vkdev, false);
        bad = opt;
        bad.blob_vkallocator = &failing;
        cmd.record_clone(src, device_output, bad);
        if (check_weight_storage_failure(cmd, src, input, opt, "clone device allocation", failing.allocations) != 0)
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
        if (check_weight_storage_failure(cmd, src, input, opt, "download packing allocation", failing.allocations) != 0)
            return -1;
    }
    return 0;
}

static int test_weight_storage_cache_limit(size_t capacity, bool expect_error)
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
    ncnn::Mat input = RandomMat(12288);
    ncnn::VkMat weight;
    ncnn::VkTransfer upload(vkdev);
    upload.record_upload(input, weight, opt);
    if (upload.submit_and_wait() != 0 || weights.finalize_weights() != 0)
    {
        fprintf(stderr, "test_weight_storage_cache_limit prepare failed\n");
        return -1;
    }
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    opt.weight_device_memory_limit = 65536;
    opt.weight_host_memory_limit = 65536;
    opt.weight_device_cache_size = capacity;
    ncnn::VkCompute cmd(vkdev);
    if (cmd.set_weight_memory_limit(opt.weight_device_memory_limit, opt.weight_host_memory_limit, opt.weight_device_cache_size, opt.weight_host_cache_size) != 0)
        return -1;
    ncnn::Mat output;
    cmd.record_clone(weight, output, opt);
    int ret = cmd.submit_and_wait();
    if (expect_error)
    {
        if (ret == 0)
        {
            fprintf(stderr, "test_weight_storage_cache_limit error missing capacity=%zu\n", capacity);
            return -1;
        }
        return 0;
    }
    if (ret != 0 || CompareMat(input, output, 0.f) != 0)
    {
        fprintf(stderr, "test_weight_storage_cache_limit output failed capacity=%zu\n", capacity);
        return -1;
    }
    return 0;
}

static int test_weight_storage_upload_snapshot()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    UnmappedWeightAllocator weights(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    ncnn::Mat source(7, (size_t)1u);
    memset(source.data, 0, source.total() * source.elemsize);
    memset(source.data, 37, 7);
    ncnn::VkMat snapshot;
    ncnn::VkTransfer upload(vkdev);
    upload.record_upload(source, snapshot, opt);
    memset(source.data, 91, 7);
    source.release();
    if (!upload.pending_upload_total() || upload.submit_and_wait() || upload.reset())
        return -1;
    ncnn::VkCompute download(vkdev);
    ncnn::Mat result;
    download.record_clone(snapshot, result, opt);
    if (download.submit_and_wait())
        return -1;
    for (int i = 0; i < 7; i++)
        if (((const unsigned char*)result.data)[i] != 37)
            return -1;
    for (size_t i = 7; i < snapshot.buffer_capacity(); i++)
        if (((const unsigned char*)result.data)[i] != 0)
            return -1;
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
        if (rejected.submit_and_wait() == 0 || rejected.reset() == 0 || failing.allocations != 1)
        {
            fprintf(stderr, "test_weight_storage_upload_snapshot staging failure missing unmapped=%d\n", unmapped);
            return -1;
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
                    || test_weight_storage_upload_snapshot()
                    || test_weight_storage_precisions()
                    || test_weight_storage_cache()
                    || test_weight_storage_cache_limit(0, false)
                    || test_weight_storage_cache_limit(65536, false)
                    || test_weight_storage_cache_limit(98304, true)
                    || test_weight_storage_cache_limit(32768, true)
                    || test_weight_storage_allocation_failure(false)
                    || test_weight_storage_allocation_failure(true)
                    || test_weight_storage_residents()
                    || test_weight_storage_automatic_cache()
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
