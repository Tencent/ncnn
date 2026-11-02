// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "command.h"
#include "gpu.h"
#include "mat.h"
#include "testutil.h"

static int test_command_upload_download(const ncnn::Mat& a, bool online)
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
        cmd.set_use_weight_upload_online(online);

        ncnn::VkMat b;
        cmd.record_upload(a, b, opt);
        cmd.record_download(b, c, opt);

        if (cmd.submit_and_wait())
            return -1;
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

static int test_command_clone(const ncnn::Mat& a, bool online)
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
        cmd.set_use_weight_upload_online(online);

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

        if (cmd.submit_and_wait())
            return -1;
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

static int test_command_transfer(const ncnn::Mat& a, bool online)
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
        cmd2.set_use_weight_upload_online(online);

        cmd2.record_download(b, c, opt);

        if (cmd2.submit_and_wait())
            return -1;
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

static int test_command_0(bool online)
{
    return 0
           || test_command_upload_download(RandomMat(5, 7, 24), online)
           || test_command_upload_download(RandomMat(7, 9, 12), online)
           || test_command_upload_download(RandomMat(3, 5, 13), online)
           || test_command_upload_download(RandomMat(15, 24), online)
           || test_command_upload_download(RandomMat(19, 12), online)
           || test_command_upload_download(RandomMat(17, 15), online)
           || test_command_upload_download(RandomMat(128), online)
           || test_command_upload_download(RandomMat(124), online)
           || test_command_upload_download(RandomMat(127), online);
}

static int test_command_1(bool online)
{
    return 0
           || test_command_clone(RandomMat(5, 7, 24), online)
           || test_command_clone(RandomMat(7, 9, 12), online)
           || test_command_clone(RandomMat(3, 5, 13), online)
           || test_command_clone(RandomMat(15, 24), online)
           || test_command_clone(RandomMat(19, 12), online)
           || test_command_clone(RandomMat(17, 15), online)
           || test_command_clone(RandomMat(128), online)
           || test_command_clone(RandomMat(124), online)
           || test_command_clone(RandomMat(127), online);
}

static int test_command_2(bool online)
{
    return 0
           || test_command_transfer(RandomMat(5, 7, 24), online)
           || test_command_transfer(RandomMat(7, 9, 12), online)
           || test_command_transfer(RandomMat(3, 5, 13), online)
           || test_command_transfer(RandomMat(15, 24), online)
           || test_command_transfer(RandomMat(19, 12), online)
           || test_command_transfer(RandomMat(17, 15), online)
           || test_command_transfer(RandomMat(128), online)
           || test_command_transfer(RandomMat(124), online)
           || test_command_transfer(RandomMat(127), online);
}

int main()
{
    SRAND(7767517);

    return test_command_0(false) || test_command_1(false) || test_command_2(false)
           || test_command_0(true) || test_command_1(true) || test_command_2(true);
}
