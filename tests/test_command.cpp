// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "command.h"
#include "gpu.h"
#include "mat.h"
#include "pipeline.h"
#include "testutil.h"

#include <string>

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
    ncnn::VkBlobAllocator reuse_allocator(vkdev, 0);

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

    ncnn::Option opt_reuse = opt;
    opt_reuse.blob_vkallocator = &reuse_allocator;
    ncnn::Mat new_data = a.clone();
    new_data.fill(7.f);
    ncnn::Mat new_result;
    bool reused = false;

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
        ncnn::VkMat source;
        ncnn::VkMat prime;
        // Leave source transfer-read before recording the old destination's last read.
        cmd.record_clone(new_data, source, opt);
        cmd.record_clone(source, prime, opt);
        cmd.record_clone(a, b1, opt_reuse);
        cmd.record_clone(b1, b2, opt);

        const VkBuffer buffer = b1.buffer();
        const size_t offset = b1.buffer_offset();
        b1.release();
        cmd.record_clone(source, b1, opt_reuse);
        reused = b1.buffer() == buffer && b1.buffer_offset() == offset;

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

        cmd.record_clone(b1, new_result, opt);
        cmd.submit_and_wait();
    }

    vkdev->reclaim_blob_allocator(blob_allocator);
    vkdev->reclaim_staging_allocator(staging_allocator);

    if (!reused || CompareMat(a, d, 0.001) != 0 || CompareMat(new_data, new_result, 0.001) != 0)
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

static int test_command_virtual_dispatch(bool use_push_constant)
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkBlobAllocator blob_allocator(vkdev);
    ncnn::VkStagingAllocator staging_allocator(vkdev);

    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_subgroup_ops = false;
    opt.use_cooperative_matrix = false;
    opt.blob_vkallocator = &blob_allocator;
    opt.staging_vkallocator = &staging_allocator;

    std::string source = "#version 450\n"
                         "layout(constant_id = 0) const uint sx = 1;\n"
                         "layout(constant_id = 1) const uint sy = 1;\n"
                         "layout(constant_id = 2) const uint sz = 1;\n"
                         "layout(binding = 0) writeonly buffer top_blob { float data[]; };\n";
    if (use_push_constant)
        source += "layout(push_constant) uniform parameter { float bias; } p;\n"
                  "#define BIAS p.bias\n";
    else
        source += "#define BIAS 0.f\n";

    source += "void main()\n"
              "{\n"
              "    if (gl_LocalInvocationIndex != 0u) return;\n"
              "    uvec3 wg = gl_WorkGroupID;\n"
              "    uvec3 count = gl_NumWorkGroups;\n"
              "    uvec3 gid = uvec3(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y, gl_GlobalInvocationID.z);\n"
              "    if (any(notEqual(gid, wg * uvec3(sx, sy, sz)))) return;\n"
              "    uint i = (wg.z * count.y + wg.y) * count.x + wg.x;\n"
              "    data[i] = float(i + count.x + count.y + count.z) + BIAS;\n"
              "}\n";

    std::vector<uint32_t> spirv;
    if (ncnn::compile_spirv_module(source.c_str(), opt, spirv) != 0)
        return -1;

    ncnn::Pipeline pipeline(vkdev);
    pipeline.set_local_size_xyz(2, 2, 2);

    std::vector<ncnn::vk_specialization_type> specializations(3);
    specializations[0].u32 = pipeline.local_size_x();
    specializations[1].u32 = pipeline.local_size_y();
    specializations[2].u32 = pipeline.local_size_z();
    if (pipeline.create(spirv.data(), spirv.size() * sizeof(uint32_t), specializations) != 0)
        return -1;

    std::vector<ncnn::vk_constant_type> constants(use_push_constant ? 1 : 0);
    if (use_push_constant)
        constants[0].f = 7.f;

    const uint32_t limits[3] = {vkdev->info.max_workgroup_count_x(), vkdev->info.max_workgroup_count_y(), vkdev->info.max_workgroup_count_z()};

    // exercise an oversized grid along the first axis with a small enough device limit
    int large_axis = -1;
    for (int i = 0; i < 3; i++)
    {
        if (limits[i] <= 65535)
        {
            large_axis = i;
            break;
        }
    }

    // repeat a normal dispatch after an oversized one to verify the base is reset
    ncnn::VkCompute cmd(vkdev);
    for (int i = 0; i < 3; i++)
    {
        int groups[3] = {2, 2, 2};
        if (i == 1)
        {
            if (large_axis == -1)
                continue;
            groups[large_axis] = (int)limits[large_axis] + 1;
            groups[(large_axis + 2) % 3] = 1;
        }

        const int n = groups[0] * groups[1] * groups[2];
        ncnn::Mat a(n);
        a.fill(-1.f);

        ncnn::VkMat output;
        cmd.record_clone(a, output, opt);

        std::vector<ncnn::VkMat> bindings(1, output);

        ncnn::VkMat dispatcher;
        dispatcher.w = groups[0] * pipeline.local_size_x();
        dispatcher.h = groups[1] * pipeline.local_size_y();
        dispatcher.c = groups[2] * pipeline.local_size_z();
        cmd.record_pipeline(&pipeline, bindings, constants, dispatcher);

        ncnn::Mat b;
        cmd.record_clone(output, b, opt);
        if (cmd.submit_and_wait() != 0)
            return -1;

        for (int j = 0; j < n; j++)
            a[j] = (float)(j + groups[0] + groups[1] + groups[2]) + (use_push_constant ? 7.f : 0.f);

        if (CompareMat(a, b, 0.f) != 0)
        {
            fprintf(stderr, "test_command_virtual_dispatch failed groups=(%d %d %d) use_push_constant=%d\n", groups[0], groups[1], groups[2], use_push_constant);
            return -1;
        }

        cmd.reset();
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_command_0() || test_command_1() || test_command_2() || test_command_virtual_dispatch(false) || test_command_virtual_dispatch(true);
}
