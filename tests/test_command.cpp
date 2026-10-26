// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "command.h"
#include "gpu.h"
#include "mat.h"
#include "testutil.h"

static int test_command_upload_download(const ncnn::Mat& a)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.blob_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;

    if (!vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
    if (!vkdev->info.support_bf16_packed()) opt.use_bf16_packed = false;
    if (!vkdev->info.support_bf16_storage()) opt.use_bf16_storage = false;

    ncnn::Mat c;
    {
        ncnn::VkCompute cmd(vkdev);

        ncnn::VkMat b;
        cmd.record_upload(a, b, opt);
        cmd.record_download(b, c, opt);

        cmd.submit_and_wait();
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);

    if (CompareMat(a, c, 0.001) != 0)
    {
        fprintf(stderr, "test_command_upload_download image failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
        return -1;
    }

    return 0;
}

static int test_command_clone(const ncnn::Mat& a)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.blob_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;

    if (!vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
    if (!vkdev->info.support_bf16_packed()) opt.use_bf16_packed = false;
    if (!vkdev->info.support_bf16_storage()) opt.use_bf16_storage = false;

    const bool support_image_storage = vkdev->info.support_image_storage();

    ncnn::Mat d;
    ncnn::Mat e;
    {
        ncnn::VkCompute cmd(vkdev);

        ncnn::VkMat b1;
        ncnn::VkMat b2;
        ncnn::VkImageMat b3;
        ncnn::VkImageMat c1;
        ncnn::VkImageMat c2;
        ncnn::VkMat c3;
        cmd.record_clone(a, b1, opt);
        cmd.record_clone(b1, b2, opt);

        if (support_image_storage)
        {
            cmd.record_clone(a, c1, opt);
            cmd.record_clone(b2, b3, opt);
            cmd.record_clone(c1, c2, opt);
            cmd.record_clone(b3, d, opt);
            cmd.record_clone(c2, c3, opt);
            cmd.record_clone(c3, e, opt);
        }
        else
        {
            cmd.record_clone(b2, d, opt);
        }

        cmd.submit_and_wait();
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);

    if (CompareMat(a, d, 0.001) != 0)
    {
        fprintf(stderr, "test_command_clone buffer failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
        return -1;
    }

    if (support_image_storage && CompareMat(a, e, 0.001) != 0)
    {
        fprintf(stderr, "test_command_clone image failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
        return -1;
    }

    return 0;
}

static int test_command_transfer(const ncnn::Mat& a)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkAllocator* blob_allocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_allocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.blob_vkallocator = blob_allocator;
    opt.staging_vkallocator = staging_allocator;

    if (!vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;
    if (!vkdev->info.support_bf16_packed()) opt.use_bf16_packed = false;
    if (!vkdev->info.support_bf16_storage()) opt.use_bf16_storage = false;

    ncnn::Mat c;
    {
        ncnn::VkTransfer cmd1(vkdev);

        ncnn::VkMat b;
        cmd1.record_upload(a, b, opt, false);

        cmd1.submit_and_wait();

        ncnn::VkCompute cmd2(vkdev);

        cmd2.record_download(b, c, opt);

        cmd2.submit_and_wait();
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);

    if (CompareMat(a, c, 0.001) != 0)
    {
        fprintf(stderr, "test_command_transfer buffer failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
        return -1;
    }

    return 0;
}

class FailingAllocator : public ncnn::Allocator
{
public:
    virtual void* fastMalloc(size_t) { return 0; }
    virtual void fastFree(void*) {}
};

class FailingStagingAllocator : public ncnn::VkStagingAllocator
{
public:
    FailingStagingAllocator(const ncnn::VulkanDevice* vkdev, bool unmapped)
        : ncnn::VkStagingAllocator(vkdev), unmapped(unmapped), allocations(0), mapping(0) {}

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

static int test_command_allocation_failure()
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
        if (upload.submit_and_wait())
            return -1;
    }
    ncnn::VkWeightAllocator weights(vkdev);
    ncnn::VkWeightStagingAllocator weight_staging(vkdev);
    ncnn::VkMat file;
    if (weights.set_weights_in_host_disk(true))
        return -1;
    {
        ncnn::Option upload_opt = opt;
        upload_opt.blob_vkallocator = &weights;
        upload_opt.staging_vkallocator = &weight_staging;
        ncnn::VkTransfer upload(vkdev);
        upload.record_upload(input, file, upload_opt);
        if (upload.submit_and_wait() || weights.finalize_weights())
            return -1;
    }
    // Empty sources, failed CPU allocations, failed/missing staging mappings,
    // and packing allocations all report failure through the command boundary.
    for (int storage = 0; storage < 2; storage++)
    {
        const ncnn::VkMat& source = storage == 0 ? native : file;
        for (int mode = 0; mode < 10; mode++)
        {
            FailingStagingAllocator failing(vkdev, mode == 5 || mode == 6);
            ncnn::Option bad = opt;
            ncnn::VkCompute cmd(vkdev);
            ncnn::Mat output;
            ncnn::VkMat device_output;
            if (mode == 0)
                cmd.record_clone(ncnn::VkMat(), output, bad);
            else if (mode == 1)
                cmd.record_download(ncnn::VkMat(), output, bad);
            else if (mode == 2 || mode == 3)
            {
                bad.blob_allocator = &cpu;
                if (mode == 2)
                    cmd.record_clone(source, output, bad);
                else
                    cmd.record_download(source, output, bad);
            }
            else if (mode == 4 || mode == 5)
            {
                bad.staging_vkallocator = &failing;
                cmd.record_upload(input, device_output, bad);
            }
            else if (mode == 6 || mode == 7)
            {
                // Hide only the source mapping to exercise the device-to-host path.
                void* mapping = source.data->mapped_ptr;
                source.data->mapped_ptr = 0;
                bad.staging_vkallocator = &failing;
                cmd.record_clone(source, output, bad);
                source.data->mapped_ptr = mapping;
            }
            else if (mode == 8)
            {
                bad.blob_vkallocator = &failing;
                cmd.record_clone(source, device_output, bad);
            }
            else
            {
                bad.blob_vkallocator = &failing;
                bad.staging_vkallocator = &failing;
                bad.use_packing_layout = true;
                ncnn::VkMat packable(128, (size_t)4u, &blobs);
                cmd.record_download(packable, output, bad);
            }
            if (cmd.submit_and_wait() == 0 || failing.allocations > 1)
            {
                fprintf(stderr, "command allocation failure storage %d mode %d was not propagated\n", storage, mode);
                return -1;
            }
            if (cmd.reset())
                return -1;
            cmd.record_clone(source, output, opt);
            if (cmd.submit_and_wait() || CompareMat(input, output, 0.001f))
                return -1;
        }
    }
    return 0;
}

static int test_command_0()
{
    return 0
           || test_command_upload_download(RandomMat(5, 7, 24))
           || test_command_upload_download(RandomMat(7, 9, 12))
           || test_command_upload_download(RandomMat(3, 5, 13))
           || test_command_upload_download(RandomMat(15, 24))
           || test_command_upload_download(RandomMat(19, 12))
           || test_command_upload_download(RandomMat(17, 15))
           || test_command_upload_download(RandomMat(128))
           || test_command_upload_download(RandomMat(124))
           || test_command_upload_download(RandomMat(127));
}

static int test_command_1()
{
    return 0
           || test_command_clone(RandomMat(5, 7, 24))
           || test_command_clone(RandomMat(7, 9, 12))
           || test_command_clone(RandomMat(3, 5, 13))
           || test_command_clone(RandomMat(15, 24))
           || test_command_clone(RandomMat(19, 12))
           || test_command_clone(RandomMat(17, 15))
           || test_command_clone(RandomMat(128))
           || test_command_clone(RandomMat(124))
           || test_command_clone(RandomMat(127));
}

static int test_command_2()
{
    return 0
           || test_command_transfer(RandomMat(5, 7, 24))
           || test_command_transfer(RandomMat(7, 9, 12))
           || test_command_transfer(RandomMat(3, 5, 13))
           || test_command_transfer(RandomMat(15, 24))
           || test_command_transfer(RandomMat(19, 12))
           || test_command_transfer(RandomMat(17, 15))
           || test_command_transfer(RandomMat(128))
           || test_command_transfer(RandomMat(124))
           || test_command_transfer(RandomMat(127));
}

int main()
{
    SRAND(7767517);

    const int ret = test_command_0() || test_command_1() || test_command_2() || test_command_allocation_failure();
    ncnn::destroy_gpu_instance();
    return ret;
}
