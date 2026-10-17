// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"

#include <math.h>
#endif

static int test_layernorm(const ncnn::Mat& a, int affine_size, float eps, int affine, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, affine_size);
    pd.set(1, eps);
    pd.set(2, affine);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(affine_size);
    weights[1] = RandomMat(affine_size);

    int ret = test_layer("LayerNorm", pd, weights, a, 1e-4f, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_layernorm failed a.dims=%d a=(%d %d %d %d) affine_size=%d eps=%f affine=%d\n", a.dims, a.w, a.h, a.d, a.c, affine_size, eps, affine);
    }

    return ret;
}

static int test_layernorm_0()
{
    return 0
           || test_layernorm(RandomMat(6, 4, 2), 6, 0.01f, 0)
           || test_layernorm(RandomMat(4, 5, 6), 4, 0.01f, 0)
           || test_layernorm(RandomMat(3, 3, 8), 3, 0.002f, 0)
           || test_layernorm(RandomMat(5, 6, 12), 5, 0.02f, 0)
           || test_layernorm(RandomMat(4, 7, 16), 4, 0.02f, 0)
           || test_layernorm(RandomMat(6, 7, 24), 6, 0.001f, 0)
           || test_layernorm(RandomMat(5, 8, 32), 5, 0.001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_layernorm(RandomMat(6, 4, 2), 6, 0.01f, 1)
           || test_layernorm(RandomMat(4, 5, 6), 4, 0.01f, 1)
           || test_layernorm(RandomMat(3, 3, 8), 3, 0.002f, 1)
           || test_layernorm(RandomMat(5, 6, 12), 5, 0.02f, 1)
           || test_layernorm(RandomMat(4, 7, 16), 4, 0.02f, 1)
           || test_layernorm(RandomMat(6, 7, 24), 6, 0.001f, 1)
           || test_layernorm(RandomMat(5, 8, 32), 5, 0.001f, 1);
}

static int test_layernorm_1()
{
    return 0
           || test_layernorm(RandomMat(6, 4, 2), 24, 0.01f, 0)
           || test_layernorm(RandomMat(4, 5, 6), 20, 0.01f, 0)
           || test_layernorm(RandomMat(3, 3, 8), 9, 0.002f, 0)
           || test_layernorm(RandomMat(5, 6, 12), 30, 0.02f, 0)
           || test_layernorm(RandomMat(4, 7, 16), 28, 0.02f, 0)
           || test_layernorm(RandomMat(6, 7, 24), 42, 0.001f, 0)
           || test_layernorm(RandomMat(5, 8, 32), 40, 0.001f, 0)
           || test_layernorm(RandomMat(6, 4, 2), 24, 0.01f, 1)
           || test_layernorm(RandomMat(4, 5, 6), 20, 0.01f, 1)
           || test_layernorm(RandomMat(3, 3, 8), 9, 0.002f, 1)
           || test_layernorm(RandomMat(5, 6, 12), 30, 0.02f, 1)
           || test_layernorm(RandomMat(4, 7, 16), 28, 0.02f, 1)
           || test_layernorm(RandomMat(6, 7, 24), 42, 0.001f, 1)
           || test_layernorm(RandomMat(5, 8, 32), 40, 0.001f, 1);
}

static int test_layernorm_2()
{
    return 0
           || test_layernorm(RandomMat(4, 2), 4, 0.01f, 0)
           || test_layernorm(RandomMat(5, 6), 5, 0.01f, 0)
           || test_layernorm(RandomMat(3, 8), 3, 0.002f, 0)
           || test_layernorm(RandomMat(6, 12), 6, 0.02f, 0)
           || test_layernorm(RandomMat(4, 16), 4, 0.02f, 0)
           || test_layernorm(RandomMat(7, 24), 7, 0.001f, 0)
           || test_layernorm(RandomMat(8, 32), 8, 0.001f, 0)
           || test_layernorm(RandomMat(4, 2), 4, 0.01f, 1)
           || test_layernorm(RandomMat(5, 6), 5, 0.01f, 1)
           || test_layernorm(RandomMat(3, 8), 3, 0.002f, 1)
           || test_layernorm(RandomMat(6, 12), 6, 0.02f, 1)
           || test_layernorm(RandomMat(4, 16), 4, 0.02f, 1)
           || test_layernorm(RandomMat(7, 24), 7, 0.001f, 1)
           || test_layernorm(RandomMat(8, 32), 8, 0.001f, 1);
}

static int test_layernorm_3()
{
    return 0
           || test_layernorm(RandomMat(2), 2, 0.01f, 0)
           || test_layernorm(RandomMat(6), 6, 0.01f, 0)
           || test_layernorm(RandomMat(8), 8, 0.002f, 0)
           || test_layernorm(RandomMat(12), 12, 0.02f, 0)
           || test_layernorm(RandomMat(16), 16, 0.02f, 0)
           || test_layernorm(RandomMat(24), 24, 0.001f, 0)
           || test_layernorm(RandomMat(32), 32, 0.001f, 0)
           || test_layernorm(RandomMat(2), 2, 0.01f, 1)
           || test_layernorm(RandomMat(6), 6, 0.01f, 1)
           || test_layernorm(RandomMat(8), 8, 0.002f, 1)
           || test_layernorm(RandomMat(12), 12, 0.02f, 1)
           || test_layernorm(RandomMat(16), 16, 0.02f, 1)
           || test_layernorm(RandomMat(24), 24, 0.001f, 1)
           || test_layernorm(RandomMat(32), 32, 0.001f, 1);
}

static int test_layernorm_4()
{
    return 0
           || test_layernorm(RandomMat(5, 3, 2, 8), 5, 0.01f, 0)
           || test_layernorm(RandomMat(4, 3, 2, 5), 4, 0.01f, 1)
           || test_layernorm(RandomMat(5, 3, 2, 8), 15, 0.002f, 0)
           || test_layernorm(RandomMat(4, 3, 2, 5), 12, 0.002f, 1)
           || test_layernorm(RandomMat(5, 3, 2, 8), 30, 0.001f, 0)
           || test_layernorm(RandomMat(4, 3, 2, 5), 24, 0.001f, 1);
}

static int test_layernorm_5()
{
    return 0
           || test_layernorm(RandomMat(4095), 4095, 0.001f, 0)
           || test_layernorm(RandomMat(4096), 4096, 0.001f, 0)
           || test_layernorm(RandomMat(4097), 4097, 0.001f, 1)
           || test_layernorm(RandomMat(16384), 16384, 0.001f, 1)
           || test_layernorm(RandomMat(4095, 3), 4095, 0.001f, 1)
           || test_layernorm(RandomMat(4096, 4), 4096, 0.001f, 0)
           || test_layernorm(RandomMat(4097, 4), 4097, 0.001f, 1)
           || test_layernorm(RandomMat(4096, 2, 1), 4096, 0.001f, 0)
           || test_layernorm(RandomMat(4097, 2, 4), 4097, 0.001f, 1)
           || test_layernorm(RandomMat(513, 3, 2, 4), 1539, 0.001f, 1)
           || test_layernorm(RandomMat(32, 65535, 1), 32, 0.001f, 0)
           || test_layernorm(RandomMat(32, 65536, 1), 32, 0.001f, 1)
           || test_layernorm(RandomMat(8, 65537, 4), 8, 0.001f, 1);
}

#if NCNN_VULKAN
static ncnn::Mat layernorm_reference(const ncnn::Mat& a, int affine_size, float eps)
{
    // use double precision instead of the generic CPU layer's serial float sums
    ncnn::Mat expected = a.clone();
    const int groups = a.w * a.h * a.d / affine_size;
    for (int q = 0; q < a.c; q++)
    {
        const float* ptr = a.channel(q);
        float* outptr = expected.channel(q);
        for (int g = 0; g < groups; g++)
        {
            double sum = 0;
            for (int i = 0; i < affine_size; i++)
                sum += ptr[g * affine_size + i];
            const double mean = sum / affine_size;

            double sqsum = 0;
            for (int i = 0; i < affine_size; i++)
            {
                const double v = ptr[g * affine_size + i] - mean;
                sqsum += v * v;
            }
            const double inv_stddev = 1.0 / sqrt(sqsum / affine_size + eps);

            for (int i = 0; i < affine_size; i++)
                outptr[g * affine_size + i] = (float)((ptr[g * affine_size + i] - mean) * inv_stddev);
        }
    }

    return expected;
}

static int test_layernorm_gpu_reference(const ncnn::Mat& a, int affine_size)
{
    const float eps = 1e-5f;
    ncnn::ParamDict pd;
    pd.set(0, affine_size);
    pd.set(1, eps);
    pd.set(2, 0);

    ncnn::Mat expected = layernorm_reference(a, affine_size, eps);

    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;

    for (int subgroup = 0; subgroup < 2; subgroup++)
    {
        opt.use_subgroup_ops = subgroup != 0;
        ncnn::Mat b;
        int ret = test_layer_gpu(ncnn::layer_to_index("LayerNorm"), pd, std::vector<ncnn::Mat>(), opt, a, b, ncnn::Mat(), 0);
        // allow rounding in the existing float coeffs/norm shaders, including FMA
        if (ret != 233 && (ret != 0 || CompareMat(expected, b, 0.005f) != 0))
        {
            fprintf(stderr, "test_layernorm_gpu_reference failed a=(%d %d %d %d) affine_size=%d subgroup=%d\n", a.w, a.h, a.d, a.c, affine_size, subgroup);
            return -1;
        }
    }

    return 0;
}

static int test_layernorm_gpu_numerics()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    const int subgroup_size = std::max(4, std::min(128, (int)ncnn::get_gpu_device()->info.subgroup_size()));
    const int sizes[] = {subgroup_size - 1, subgroup_size, subgroup_size + 1, subgroup_size * 2 + 1, 4096, 16384};
    const int channels[] = {1, 4};
    for (int c = 0; c < 2; c++)
    {
        for (int s = 0; s < 6; s++)
        {
            ncnn::Mat a(sizes[s], 1, channels[c]);
            for (int q = 0; q < a.c; q++)
            {
                float* ptr = a.channel(q);
                for (int i = 0; i < a.w; i++)
                    ptr[i] = 100.1f + q * 0.5f;
            }
            if (test_layernorm_gpu_reference(a, a.w) != 0)
                return -1;

            for (int q = 0; q < a.c; q++)
            {
                float* ptr = a.channel(q);
                for (int i = 0; i < a.w; i++)
                    ptr[i] = 100.f + q * 0.5f + (i % 7 - 3) * 0.125f;
            }
            if (test_layernorm_gpu_reference(a, a.w) != 0)
                return -1;
        }
    }

    return 0;
}

static int test_layernorm_gpu_fp16a(const ncnn::Mat& a, bool use_subgroup_ops)
{
    const float eps = 1e-5f;
    ncnn::ParamDict pd;
    pd.set(0, a.w);
    pd.set(1, eps);
    pd.set(2, 0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_subgroup_ops = use_subgroup_ops;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev->info.support_subgroup_ops()) opt.use_subgroup_ops = false;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    ncnn::Layer* op = ncnn::create_layer_vulkan("LayerNorm");
    op->vkdev = vkdev;
    op->load_param(pd);

    std::vector<ncnn::Mat> weights;
    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);
    int ret = op->create_pipeline(opt);

    ncnn::Mat out;
    if (ret == 0)
    {
        // exercise fp16 arithmetic directly because test_layer_gpu disables it
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

    ncnn::Mat expected = layernorm_reference(a, a.w, eps);
    if (ret != 0 || CompareMat(expected, out, 0.002f) != 0)
    {
        fprintf(stderr, "test_layernorm_gpu_fp16a failed a=(%d %d %d %d) subgroup=%d ret=%d\n", a.w, a.h, a.d, a.c, use_subgroup_ops, ret);
        return -1;
    }

    return 0;
}

static int test_layernorm_gpu_fp16a()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    const ncnn::GpuInfo& info = ncnn::get_gpu_info();
    if (!info.support_fp16_storage() || !info.support_fp16_arithmetic())
        return 0;

    const int sizes[] = {4096, 4097, 16384};
    const int channels[] = {1, 4};
    for (int c = 0; c < 2; c++)
    {
        for (int s = 0; s < 3; s++)
        {
            ncnn::Mat a(sizes[s], 2, channels[c]);
            for (int q = 0; q < a.c; q++)
            {
                float* ptr = a.channel(q);
                for (int i = 0; i < a.w; i++)
                {
                    // use exactly representable inputs and a nonconstant row with a large square sum
                    ptr[i] = 8.f + q * 0.5f;
                    ptr[a.w + i] = 8.f + q * 0.5f + (i % 7 - 3) * 8.f;
                }
            }
            if (test_layernorm_gpu_fp16a(a, false) || test_layernorm_gpu_fp16a(a, true))
                return -1;
        }
    }

    return 0;
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

    return 0
           || test_layernorm_0()
           || test_layernorm_1()
           || test_layernorm_2()
           || test_layernorm_3()
           || test_layernorm_4()
           || test_layernorm_5()
#if NCNN_VULKAN
           || test_layernorm_gpu_numerics()
           || test_layernorm_gpu_fp16a()
#endif // NCNN_VULKAN
           ;
}
