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
    cmd.record_clone(weight, output, opt);
    {
        VulkanFailure fail(failure);
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
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    ncnn::Mat input[12];
    ncnn::VkMat weight[12];
    ncnn::VkTransfer upload(vkdev);
    for (int i = 0; i < 12; i++)
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
    ncnn::Mat output[12];
    for (int i = 0; i < 12; i++)
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
    for (int i = 0; i < 12; i++)
        cmd.record_clone(weight[i], output[i], opt);
    if (cmd.submit_and_wait() != 0)
        return -1;
    for (int i = 0; i < 12; i++)
        if (CompareMat(input[i], output[i], 0.f) != 0)
            return -1;
    return 0;
}

static int test_weight_storage_resident_batches()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkWeightAllocator ordinary(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
    weights.set_weight_memory_limit(65536, 65536);
    ncnn::Option opt = weight_storage_option();
    opt.blob_vkallocator = &weights;
    opt.staging_vkallocator = &staging;
    opt.weight_host_memory_limit = 1024;
    ncnn::Mat input[24];
    ncnn::VkMat weight[24];
    ncnn::VkTransfer upload(vkdev);
    for (int i = 0; i < 24; i++)
    {
        input[i] = RandomMat(i % 3 == 0 ? 4 : i % 3 == 1 ? 256 : 1024);
        upload.record_upload(input[i], weight[i], opt);
    }
    if (upload.submit_and_wait() != 0 || weights.finalize_weights() != 0 || upload.reset() != 0)
        return -1;
    // A freed record must not be dereferenced while allocating or publishing residents.
    weight[2].release();
    opt.blob_vkallocator = &ordinary;
    ncnn::Mat extra_input[2] = {RandomMat(64), RandomMat(128)};
    ncnn::VkMat extra[2];
    upload.record_upload(extra_input[0], extra[0], opt);
    {
        VulkanFailure count(VulkanFailure::SUBMIT, 1000000);
        upload.record_upload(&weights, opt);
        if (count.submission_count() != 0 || weight[0].data->storage_type != 2)
        {
            fprintf(stderr, "resident record_upload submitted or published weights early\n");
            return -1;
        }
        upload.record_upload(extra_input[1], extra[1], opt);
        if (upload.submit_and_wait() != 0 || count.submission_count() > 100)
        {
            fprintf(stderr, "resident upload did not batch small weights: %d submits\n", count.submission_count());
            return -1;
        }
    }
    opt.blob_vkallocator = &blobs;
    ncnn::VkCompute cmd(vkdev);
    ncnn::Mat output[24];
    ncnn::Mat extra_output[2];
    for (int i = 0; i < 24; i++)
        if (i != 2)
            cmd.record_clone(weight[i], output[i], opt);
    for (int i = 0; i < 2; i++)
        cmd.record_clone(extra[i], extra_output[i], opt);
    if (cmd.submit_and_wait() != 0)
        return -1;
    for (int i = 0; i < 24; i++)
        if (i != 2 && CompareMat(input[i], output[i], 0.f) != 0)
            return -1;
    for (int i = 0; i < 2; i++)
        if (CompareMat(extra_input[i], extra_output[i], 0.f) != 0)
            return -1;
    return 0;
}

static int test_weight_storage_resident_failure()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkWeightAllocator first(vkdev);
    ncnn::VkWeightAllocator second(vkdev);
    ncnn::VkWeightAllocator* weights[2] = {&first, &second};
    ncnn::VkMat weight[2];
    ncnn::VkTransfer upload(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.weight_host_memory_limit = 1024;
    for (int i = 0; i < 2; i++)
    {
        weights[i]->set_weights_in_disk(true);
        weights[i]->set_weight_memory_limit(8192, 8192);
        opt.blob_vkallocator = weights[i];
        upload.record_upload(RandomMat(1024), weight[i], opt);
        if (upload.error() != 0 || weights[i]->finalize_weights() != 0)
            return -1;
        upload.record_upload(weights[i], opt);
    }
    {
        VulkanFailure fail(VulkanFailure::SECOND_SUBMIT);
        if (upload.submit_and_wait() == 0 || !fail.triggered || upload.error() == 0 || upload.reset() == 0)
        {
            fprintf(stderr, "resident upload midflight failure was lost\n");
            return -1;
        }
    }
    // Publication must follow successful upload, and the allocator retains its error.
    if (first.finalize_weights() == 0 && second.finalize_weights() == 0)
        return -1;
    return 0;
}

static int test_weight_storage_cached_batch()
{
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(ncnn::get_default_gpu_index());
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    weights.set_weights_in_disk(true);
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
    ncnn::Mat output[48];
    for (int round = 0; round < 2; round++)
    {
        for (int i = 0; i < 48; i++)
            cmd.record_clone(weight[i], output[i], opt);
        {
            VulkanFailure count(VulkanFailure::SUBMIT, 1000000);
            if (cmd.submit_and_wait() != 0)
                return -1;
            if (round == 1 && count.submission_count() != 1)
            {
                fprintf(stderr, "cached weights split into %d submissions\n", count.submission_count());
                return -1;
            }
        }
        for (int i = 0; i < 48; i++)
            if (CompareMat(input[i], output[i], 0.f) != 0)
                return -1;
        if (cmd.reset() != 0)
            return -1;
    }
    return 0;
}

int main()
{
    SRAND(7767517);
    if (ncnn::get_gpu_count() == 0)
        return 0;

    return 0
           || test_weight_storage_cache_failure(VulkanFailure::ALLOCATE_MEMORY)
           || test_weight_storage_cache_failure(VulkanFailure::BIND_MEMORY)
           || test_weight_storage_cache_failure(VulkanFailure::MAP_MEMORY)
           || test_weight_storage_cache_failure(VulkanFailure::SUBMIT)
           || test_weight_storage_transfer_failure(VulkanFailure::SUBMIT)
           || test_weight_storage_transfer_failure(VulkanFailure::SECOND_SUBMIT)
           || test_weight_storage_cached_batch()
           || test_weight_storage_resident_batches()
           || test_weight_storage_resident_failure()
           || test_weight_storage_midflight_failure(2)
           || test_weight_storage_midflight_failure(5)
           || test_weight_storage_midflight_failure(9);
}
