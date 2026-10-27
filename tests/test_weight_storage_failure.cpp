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
    weights.set_weight_file_directory(weight_storage_test_directory());
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
    weights.set_weight_file_directory(weight_storage_test_directory());
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
        if (record_weight_storage_resident_tier(weights, upload, 0, opt, true) != 0)
            return -1;
        if (count.submission_count() != 0 || weight[0].data->storage_type != 2)
        {
            fprintf(stderr, "resident record_upload submitted or published weights early\n");
            return -1;
        }
        upload.record_upload(extra_input[1], extra[1], opt);
        const int ret = upload.submit_and_wait();
        weights.finish_resident_uploads(0, ret);
        if (ret != 0 || count.submission_count() > 100)
        {
            fprintf(stderr, "resident upload did not batch small weights: %d submits\n", count.submission_count());
            return -1;
        }
    }
    if (upload.reset() != 0 || record_weight_storage_resident_tier(weights, upload, 1, opt) != 0)
        return -1;
    const int host_ret = upload.submit_and_wait();
    weights.finish_resident_uploads(1, host_ret);
    if (host_ret != 0)
        return -1;
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
        weights[i]->set_weight_file_directory(weight_storage_test_directory());
        weights[i]->set_weight_memory_limit(8192, 8192);
        opt.blob_vkallocator = weights[i];
        upload.record_upload(RandomMat(1024), weight[i], opt);
        if (upload.error() != 0 || weights[i]->finalize_weights() != 0)
            return -1;
        if (record_weight_storage_resident_tier(*weights[i], upload, 0, opt, true) != 0)
            return -1;
    }
    {
        VulkanFailure fail(VulkanFailure::SECOND_SUBMIT);
        const int ret = upload.submit_and_wait();
        for (int i = 0; i < 2; i++)
            weights[i]->finish_resident_uploads(0, ret);
        if (ret == 0 || !fail.triggered || upload.error() == 0 || upload.reset() == 0)
        {
            fprintf(stderr, "resident upload midflight failure was lost\n");
            return -1;
        }
    }
    // Publication must follow successful upload, and the allocator retains its error.
    for (int i = 0; i < 2; i++)
    {
        weights[i]->finish_resident_uploads(0, 0);
        weights[i]->finish_resident_uploads(1, 0);
        if (weight[i].data->storage_type != 2)
            return -1;
    }
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
    ncnn::Mat output[48];
    for (int round = 0; round < 2; round++)
    {
        for (int i = 0; i < 48; i++)
            cmd.record_clone(weight[i], output[i], opt);
        {
            VulkanFailure count(VulkanFailure::SUBMIT, 1000000);
            if (cmd.submit_and_wait() != 0)
                return -1;
            // Cached weights fit in one batch without an artificial prefix submission.
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
    ncnn::VkWeightAllocator weights(vkdev);
    CountingWeightStagingAllocator staging(vkdev);
    ncnn::Option opt = weight_storage_option();
    opt.staging_vkallocator = &staging;
    ncnn::Mat input = RandomMat(256);
    ncnn::VkMat target(256, (size_t)4u, &weights);
    if (target.empty())
        return -1;
    void* mapping = target.data->mapped_ptr;
    target.data->mapped_ptr = 0;
    {
        VulkanFailure fail(VulkanFailure::FENCE_WAIT, 3);
        {
            ncnn::VkTransfer upload(vkdev);
            upload.record_upload(input, (const ncnn::VkMat&)target, opt);
            if (upload.submit_and_wait() == 0 || !fail.triggered || staging.frees != 0 || upload.reset() == 0)
            {
                fprintf(stderr, "failed fence wait prematurely released transfer staging\n");
                return -1;
            }
        }
        // The destructor retries two more transient errors before the real wait.
    }
    target.data->mapped_ptr = mapping;
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
        for (int i = 0; i < count; i++)
            cmd.record_clone(weight[i], output[i], opt);
        {
            VulkanFailure check(failure, 5);
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
