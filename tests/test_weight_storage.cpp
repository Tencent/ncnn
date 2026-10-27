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
    ncnn::VkCompute cmd(vkdev, download_opt);
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
    ncnn::VkCompute cmd(vkdev, opt);
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
    weights.set_weight_memory_limit(16384, 16384);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &upload_staging;
    opt.weight_host_memory_limit = 16384;
    ncnn::Mat input = RandomMat(2048);
    ncnn::VkMat weight;
    ncnn::VkTransfer upload(vkdev);
    upload.record_upload(input, weight, opt);
    if (upload.submit_and_wait() != 0 || weights.finalize_weights() != 0 || upload.reset() != 0)
    {
        fprintf(stderr, "test_weight_storage_residents prepare failed\n");
        return -1;
    }
    if (upload_weight_storage_residents(weights, upload, opt) != 0)
    {
        fprintf(stderr, "test_weight_storage_residents upload failed\n");
        return -1;
    }
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    ncnn::VkCompute cmd(vkdev);
    ncnn::Mat output;
    cmd.record_clone(weight, output, opt);
    if (cmd.submit_and_wait() != 0 || CompareMat(input, output, 0.f) != 0)
    {
        fprintf(stderr, "test_weight_storage_residents output failed\n");
        return -1;
    }
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
        ncnn::VkCompute early(vkdev, opt);
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
    ncnn::VkCompute cmd(vkdev, opt);
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
    ncnn::VkCompute fixed(vkdev, fixed_opt);
    fixed.record_pipeline(&pipeline, bindings, constants, result);
    if (fixed.submit_and_wait() == 0)
    {
        fprintf(stderr, "test_weight_storage_automatic_cache failed at line %d\n", __LINE__);
        return -1;
    }
    return 0;
}

static int test_weight_storage_recorded_weight_lifetime()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &blobs;
    opt.staging_vkallocator = &staging;
    ncnn::VkCompute cmd(vkdev, opt);
    ncnn::Mat original = RandomMat(128);
    ncnn::Mat output;
    for (int round = 0; round < 3; round++)
    {
        {
            ncnn::VkWeightAllocator store(vkdev);
            store.set_weights_in_disk(true);
            store.set_weight_file_directory(weight_storage_test_directory());
            ncnn::VkMat weight(128, (size_t)4u, &store);
            if (weight.empty())
            {
                fprintf(stderr, "test_weight_storage_recorded_weight_lifetime failed at line %d\n", __LINE__);
                return -1;
            }
            memcpy(weight.data->source_ptr, original.data, original.total() * original.elemsize);
            if (store.finalize_weights())
            {
                fprintf(stderr, "test_weight_storage_recorded_weight_lifetime failed at line %d\n", __LINE__);
                return -1;
            }
            cmd.record_clone(weight, output, opt);
            if (round == 0)
            {
                weight.release();
                store.clear();
                if (cmd.submit_and_wait() == 0 || cmd.reset())
                {
                    fprintf(stderr, "test_weight_storage_recorded_weight_lifetime failed at line %d\n", __LINE__);
                    return -1;
                }
            }
            if (round == 2 && (cmd.submit_and_wait() || CompareMat(original, output, 0.f) || cmd.reset()))
            {
                fprintf(stderr, "test_weight_storage_recorded_weight_lifetime failed at line %d\n", __LINE__);
                return -1;
            }
        }
        // reject a destroyed allocator before reading its weights
        if (round == 1 && (cmd.submit_and_wait() == 0 || cmd.reset()))
        {
            fprintf(stderr, "test_weight_storage_recorded_weight_lifetime failed at line %d\n", __LINE__);
            return -1;
        }
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
    ncnn::VkCompute cmd(vkdev, opt);
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

static int test_weight_storage_upload_views()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    ncnn::Mat source(7, (size_t)1u);
    memset(source.data, 37, 7);
    ncnn::VkMat snapshot;
    ncnn::VkTransfer upload(vkdev);
    upload.record_upload(source, snapshot, opt);
    // The existing allocating API snapshots data before the caller reuses it.
    memset(source.data, 91, 7);
    if (upload.pending_upload_total() == 0 || upload.submit_and_wait() != 0 || upload.reset() != 0)
        return -1;
    ncnn::VkMat padded(11, (size_t)1u, &weights);
    void* mapping = padded.data->mapped_ptr;
    padded.data->mapped_ptr = 0;
    const ncnn::VkMat& destination = padded;
    upload.record_upload(source, destination, opt);
    // Drop the caller's references; transfer records retain both region handles.
    ncnn::VkMat result = padded;
    padded.release();
    source.release();
    if (upload.submit_and_wait() != 0 || upload.reset() != 0)
        return -1;
    result.data->mapped_ptr = mapping;
    ncnn::VkCompute download(vkdev);
    ncnn::Mat first;
    ncnn::Mat second;
    download.record_clone(snapshot, first, opt);
    download.record_clone(result, second, opt);
    if (download.submit_and_wait() != 0)
        return -1;
    for (int i = 0; i < 7; i++)
        if (((const unsigned char*)first.data)[i] != 37 || ((const unsigned char*)second.data)[i] != 91)
            return -1;
    for (int i = 7; i < 11; i++)
        if (((const unsigned char*)second.data)[i] != 0)
            return -1;
    for (int unmapped = 0; unmapped < 2; unmapped++)
    {
        FailingStagingAllocator failing(vkdev, unmapped != 0);
        ncnn::Option bad = opt;
        bad.staging_vkallocator = &failing;
        ncnn::Mat bytes(7, (size_t)1u);
        bytes.fill((unsigned char)19);
        ncnn::VkMat target(7, (size_t)1u, &weights);
        target.data->mapped_ptr = 0;
        ncnn::VkTransfer rejected(vkdev);
        rejected.record_upload(bytes, (const ncnn::VkMat&)target, bad);
        if (rejected.submit_and_wait() == 0 || rejected.error() == 0 || rejected.reset() == 0 || failing.allocations != 1)
        {
            fprintf(stderr, "test_weight_storage_upload_views staging failure missing unmapped=%d\n", unmapped);
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
                    || test_weight_storage_upload_views()
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
                    || test_weight_storage_recorded_weight_lifetime();
    if (directory.remove() != 0)
    {
        fprintf(stderr, "weight storage temporary files leaked after tests\n");
        return 1;
    }
    return ret;
}
