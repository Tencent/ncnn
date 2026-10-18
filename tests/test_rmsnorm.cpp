// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_rmsnorm(const ncnn::Mat& a, int affine_size, float eps, int affine, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, affine_size);
    pd.set(1, eps);
    pd.set(2, affine);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(affine_size);

    int ret = test_layer("RMSNorm", pd, weights, a, 1e-4f, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_rmsnorm failed a.dims=%d a=(%d %d %d %d) affine_size=%d eps=%f affine=%d\n", a.dims, a.w, a.h, a.d, a.c, affine_size, eps, affine);
    }

    return ret;
}

static int test_rmsnorm_0()
{
    return 0
           || test_rmsnorm(RandomMat(6, 4, 2), 6, 0.01f, 0)
           || test_rmsnorm(RandomMat(4, 5, 6), 4, 0.01f, 0)
           || test_rmsnorm(RandomMat(3, 3, 8), 3, 0.002f, 0)
           || test_rmsnorm(RandomMat(5, 6, 12), 5, 0.02f, 0)
           || test_rmsnorm(RandomMat(4, 7, 16), 4, 0.02f, 0)
           || test_rmsnorm(RandomMat(6, 7, 24), 6, 0.001f, 0)
           || test_rmsnorm(RandomMat(5, 8, 32), 5, 0.001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_rmsnorm(RandomMat(6, 4, 2), 6, 0.01f, 1)
           || test_rmsnorm(RandomMat(4, 5, 6), 4, 0.01f, 1)
           || test_rmsnorm(RandomMat(3, 3, 8), 3, 0.002f, 1)
           || test_rmsnorm(RandomMat(5, 6, 12), 5, 0.02f, 1)
           || test_rmsnorm(RandomMat(4, 7, 16), 4, 0.02f, 1)
           || test_rmsnorm(RandomMat(6, 7, 24), 6, 0.001f, 1)
           || test_rmsnorm(RandomMat(5, 8, 32), 5, 0.001f, 1);
}

static int test_rmsnorm_1()
{
    return 0
           || test_rmsnorm(RandomMat(6, 4, 2), 24, 0.01f, 0)
           || test_rmsnorm(RandomMat(4, 5, 6), 20, 0.01f, 0)
           || test_rmsnorm(RandomMat(3, 3, 8), 9, 0.002f, 0)
           || test_rmsnorm(RandomMat(5, 6, 12), 30, 0.02f, 0)
           || test_rmsnorm(RandomMat(4, 7, 16), 28, 0.02f, 0)
           || test_rmsnorm(RandomMat(6, 7, 24), 42, 0.001f, 0)
           || test_rmsnorm(RandomMat(5, 8, 32), 40, 0.001f, 0)
           || test_rmsnorm(RandomMat(6, 4, 2), 24, 0.01f, 1)
           || test_rmsnorm(RandomMat(4, 5, 6), 20, 0.01f, 1)
           || test_rmsnorm(RandomMat(3, 3, 8), 9, 0.002f, 1)
           || test_rmsnorm(RandomMat(5, 6, 12), 30, 0.02f, 1)
           || test_rmsnorm(RandomMat(4, 7, 16), 28, 0.02f, 1)
           || test_rmsnorm(RandomMat(6, 7, 24), 42, 0.001f, 1)
           || test_rmsnorm(RandomMat(5, 8, 32), 40, 0.001f, 1);
}

static int test_rmsnorm_2()
{
    return 0
           || test_rmsnorm(RandomMat(4, 2), 4, 0.01f, 0)
           || test_rmsnorm(RandomMat(5, 6), 5, 0.01f, 0)
           || test_rmsnorm(RandomMat(3, 8), 3, 0.002f, 0)
           || test_rmsnorm(RandomMat(6, 12), 6, 0.02f, 0)
           || test_rmsnorm(RandomMat(4, 16), 4, 0.02f, 0)
           || test_rmsnorm(RandomMat(7, 24), 7, 0.001f, 0)
           || test_rmsnorm(RandomMat(8, 32), 8, 0.001f, 0)
           || test_rmsnorm(RandomMat(4, 2), 4, 0.01f, 1)
           || test_rmsnorm(RandomMat(5, 6), 5, 0.01f, 1)
           || test_rmsnorm(RandomMat(3, 8), 3, 0.002f, 1)
           || test_rmsnorm(RandomMat(6, 12), 6, 0.02f, 1)
           || test_rmsnorm(RandomMat(4, 16), 4, 0.02f, 1)
           || test_rmsnorm(RandomMat(7, 24), 7, 0.001f, 1)
           || test_rmsnorm(RandomMat(8, 32), 8, 0.001f, 1);
}

static int test_rmsnorm_3()
{
    return 0
           || test_rmsnorm(RandomMat(2), 2, 0.01f, 0)
           || test_rmsnorm(RandomMat(6), 6, 0.01f, 0)
           || test_rmsnorm(RandomMat(8), 8, 0.002f, 0)
           || test_rmsnorm(RandomMat(12), 12, 0.02f, 0)
           || test_rmsnorm(RandomMat(16), 16, 0.02f, 0)
           || test_rmsnorm(RandomMat(24), 24, 0.001f, 0)
           || test_rmsnorm(RandomMat(32), 32, 0.001f, 0)
           || test_rmsnorm(RandomMat(2), 2, 0.01f, 1)
           || test_rmsnorm(RandomMat(6), 6, 0.01f, 1)
           || test_rmsnorm(RandomMat(8), 8, 0.002f, 1)
           || test_rmsnorm(RandomMat(12), 12, 0.02f, 1)
           || test_rmsnorm(RandomMat(16), 16, 0.02f, 1)
           || test_rmsnorm(RandomMat(24), 24, 0.001f, 1)
           || test_rmsnorm(RandomMat(32), 32, 0.001f, 1);
}

static int test_rmsnorm_4()
{
    return 0
           || test_rmsnorm(RandomMat(5, 3, 2, 8), 5, 0.01f, 0)
           || test_rmsnorm(RandomMat(4, 3, 2, 5), 4, 0.01f, 1)
           || test_rmsnorm(RandomMat(5, 3, 2, 8), 15, 0.002f, 0)
           || test_rmsnorm(RandomMat(4, 3, 2, 5), 12, 0.002f, 1)
           || test_rmsnorm(RandomMat(5, 3, 2, 8), 30, 0.001f, 0)
           || test_rmsnorm(RandomMat(4, 3, 2, 5), 24, 0.001f, 1);
}

static int test_rmsnorm_5()
{
    return 0
           || test_rmsnorm(RandomMat(4095), 4095, 0.001f, 0)
           || test_rmsnorm(RandomMat(4096), 4096, 0.001f, 0)
           || test_rmsnorm(RandomMat(4097), 4097, 0.001f, 1)
           || test_rmsnorm(RandomMat(16384), 16384, 0.001f, 1)
           || test_rmsnorm(RandomMat(4095, 3), 4095, 0.001f, 1)
           || test_rmsnorm(RandomMat(4096, 4), 4096, 0.001f, 0)
           || test_rmsnorm(RandomMat(4097, 4), 4097, 0.001f, 1)
           || test_rmsnorm(RandomMat(4096, 2, 1), 4096, 0.001f, 0)
           || test_rmsnorm(RandomMat(4097, 2, 4), 4097, 0.001f, 1)
           || test_rmsnorm(RandomMat(513, 3, 2, 4), 1539, 0.001f, 1)
           || test_rmsnorm(RandomMat(64, 65536, 1), 64, 0.001f, 1)
           || test_rmsnorm(RandomMat(64, 65537, 1), 64, 0.001f, 1)
           || test_rmsnorm(RandomMat(16, 65537, 4), 16, 0.001f, 1);
}

#if NCNN_VULKAN
static int test_rmsnorm_gpu_fp16a(const ncnn::Mat& a, bool use_subgroup_ops)
{
    ncnn::ParamDict pd;
    pd.set(0, a.w);
    pd.set(1, 0.001f);
    pd.set(2, 0);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_subgroup_ops = use_subgroup_ops;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev->info.support_subgroup_ops()) opt.use_subgroup_ops = false;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    ncnn::Layer* op = ncnn::create_layer_vulkan("RMSNorm");
    op->vkdev = vkdev;
    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);
    op->create_pipeline(opt);

    ncnn::Mat expected;
    test_layer_naive(ncnn::layer_to_index("RMSNorm"), pd, weights, a, expected, 0);

    ncnn::Mat out;
    int ret;
    {
        ncnn::VkCompute cmd(vkdev);
        ncnn::VkMat a_gpu;
        cmd.record_upload(a, a_gpu, opt);
        ret = op->forward_inplace(a_gpu, cmd, opt);
        if (ret == 0)
        {
            cmd.record_download(a_gpu, out, opt);
            ret = cmd.submit_and_wait();
        }
    }

    op->destroy_pipeline(opt);
    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    if (ret != 0 || CompareMat(expected, out, 0.001f) != 0)
    {
        fprintf(stderr, "test_rmsnorm_gpu_fp16a failed a.dims=%d a=(%d %d %d %d) subgroup=%d ret=%d\n", a.dims, a.w, a.h, a.d, a.c, use_subgroup_ops, ret);
        return -1;
    }

    return 0;
}

static int test_rmsnorm_gpu_fp16a()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    const ncnn::GpuInfo& info = ncnn::get_gpu_info();
    if (!info.support_fp16_storage() || !info.support_fp16_arithmetic())
        return 0;

    ncnn::Mat a(4096);
    a.fill(8.f);
    ncnn::Mat b(4096, 1, 4);
    b.fill(8.f);

    return test_rmsnorm_gpu_fp16a(a, false)
           || test_rmsnorm_gpu_fp16a(a, true)
           || test_rmsnorm_gpu_fp16a(b, false)
           || test_rmsnorm_gpu_fp16a(b, true);
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

    return 0
           || test_rmsnorm_0()
           || test_rmsnorm_1()
           || test_rmsnorm_2()
           || test_rmsnorm_3()
           || test_rmsnorm_4()
           || test_rmsnorm_5()
#if NCNN_VULKAN
           || test_rmsnorm_gpu_fp16a()
#endif // NCNN_VULKAN
           ;
}
