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
           || test_weight_storage_transfer_failure(VulkanFailure::SECOND_SUBMIT);
}
