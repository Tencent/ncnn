// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif

static int test_softmax(const ncnn::Mat& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); // axis
    pd.set(1, 1);    // fixbug0

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Softmax", pd, weights, a, 0.001);
    if (ret != 0)
    {
        fprintf(stderr, "test_softmax failed a.dims=%d a=(%d %d %d %d) axis=%d\n", a.dims, a.w, a.h, a.d, a.c, axis);
    }

    return ret;
}

static int test_softmax_axes(const ncnn::Mat& a)
{
    for (int axis = 0; axis < a.dims; axis++)
    {
        if (test_softmax(a, axis) != 0)
            return -1;
    }

    return 0;
}

static int test_softmax_negative_axes(const ncnn::Mat& a)
{
    for (int axis = -a.dims; axis < 0; axis++)
    {
        if (test_softmax(a, axis) != 0)
            return -1;
    }

    return 0;
}

static int test_softmax_0()
{
    // channel counts exercise pack16, pack8, pack4 and scalar tails
    return 0
           || test_softmax_negative_axes(RandomMat(9, 5, 7, 32))
           || test_softmax_axes(RandomMat(10, 7, 6, 40))
           || test_softmax_axes(RandomMat(8, 7, 5, 28))
           || test_softmax_axes(RandomMat(9, 7, 5, 31));
}

static int test_softmax_1()
{
    return 0
           || test_softmax_negative_axes(RandomMat(25, 27, 32))
           || test_softmax_axes(RandomMat(22, 19, 40))
           || test_softmax_axes(RandomMat(27, 29, 28))
           || test_softmax_axes(RandomMat(23, 25, 31));
}

static int test_softmax_2()
{
    return 0
           || test_softmax_negative_axes(RandomMat(125, 32))
           || test_softmax_axes(RandomMat(147, 40))
           || test_softmax_axes(RandomMat(127, 28))
           || test_softmax_axes(RandomMat(129, 31));
}

static int test_softmax_3()
{
    return 0
           || test_softmax_negative_axes(RandomMat(128))
           || test_softmax_axes(RandomMat(120))
           || test_softmax_axes(RandomMat(124))
           || test_softmax_axes(RandomMat(127));
}

static int test_softmax_large()
{
    // large allocations exercise staging reuse and multi-workgroup reductions
    return 0
           || test_softmax(RandomMat(23, 25, 27, 32), -4)
           || test_softmax(RandomMat(24, 27, 29, 28), 0);
}

#if NCNN_VULKAN
static int test_softmax_gpu_fp16a(const ncnn::Mat& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); // axis
    pd.set(1, 1);    // fixbug0

    std::vector<ncnn::Mat> weights(0);

    ncnn::Mat expected;
    int ret = test_layer_naive(ncnn::LayerType::Softmax, pd, weights, a, expected, 0);
    if (ret != 0)
        return ret;

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    ncnn::Layer* op = ncnn::create_layer_vulkan("Softmax");
    op->vkdev = vkdev;
    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);
    ret = op->create_pipeline(opt);

    ncnn::Mat b;
    if (ret == 0)
    {
        // exercise fp16 arithmetic directly because test_layer_gpu disables it
        // the upload packs to elempack 4 when the outermost dim is a multiple of 4
        ncnn::VkCompute cmd(vkdev);
        ncnn::VkMat a_gpu;
        cmd.record_upload(a, a_gpu, opt);
        ret = op->forward_inplace(a_gpu, cmd, opt);
        if (ret == 0)
        {
            cmd.record_download(a_gpu, b, opt);
            ret = cmd.submit_and_wait();
        }
    }

    op->destroy_pipeline(opt);
    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    ncnn::Mat b1;
    if (ret == 0)
    {
        ncnn::Option opt_cpu;
        opt_cpu.num_threads = 1;
        ncnn::convert_packing(b, b1, 1, opt_cpu);
    }

    // the probabilities are small, so check the relative error
    // CompareMat accepts any absolute error below its epsilon
    bool ok = ret == 0 && b1.dims == expected.dims && b1.w == expected.w && b1.h == expected.h && b1.d == expected.d && b1.c == expected.c;
    for (int q = 0; ok && q < expected.c; q++)
    {
        const float* pe = expected.channel(q);
        const float* pb = b1.channel(q);
        for (int i = 0; i < expected.w * expected.h * expected.d; i++)
        {
            if (!(fabs(pb[i] - pe[i]) <= 0.001f * fabs(pe[i])))
            {
                fprintf(stderr, "value not match at c:%d i:%d expect %g but got %g\n", q, i, pe[i], pb[i]);
                ok = false;
                break;
            }
        }
    }

    if (!ok)
    {
        fprintf(stderr, "test_softmax_gpu_fp16a failed a.dims=%d a=(%d %d %d %d) axis=%d ret=%d\n", a.dims, a.w, a.h, a.d, a.c, axis, ret);
        return -1;
    }

    return 0;
}

static ncnn::Mat ConstantMat(ncnn::Mat m)
{
    m.fill(0.5f);
    return m;
}

static int test_softmax_gpu_fp16a()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    const ncnn::GpuInfo& info = ncnn::get_gpu_info();
    if (!info.support_fp16_storage() || !info.support_fp16_arithmetic())
        return 0;

    // every softmax slice sums 4096 or more ones, an fp16 sum stops growing at 2048
    // the outermost dim picks pack4 (multiple of 4) or pack1, and the packed axis needs 4x the length
    return 0
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(16384)), 0)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(4097)), 0)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(3, 16384)), 0)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(3, 4097)), 0)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(4096, 4)), 1)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(4096, 3)), 1)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(2, 2, 16384)), 0)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(2, 2, 4097)), 0)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(2, 4096, 4)), 1)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(2, 4096, 3)), 1)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(4096, 2, 4)), 2)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(4096, 2, 3)), 2)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(2, 1, 2, 16384)), 0)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(2, 1, 2, 4097)), 0)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(2, 2, 4096, 4)), 1)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(2, 2, 4096, 3)), 1)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(2, 4096, 2, 4)), 2)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(2, 4096, 2, 3)), 2)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(4096, 2, 2, 4)), 3)
           || test_softmax_gpu_fp16a(ConstantMat(ncnn::Mat(4096, 2, 2, 3)), 3);
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

    return 0
           || test_softmax_0()
           || test_softmax_1()
           || test_softmax_2()
           || test_softmax_3()
           || test_softmax_large()
#if NCNN_VULKAN
           || test_softmax_gpu_fp16a()
#endif
           // pack8 rows include a full avx512 block, an eight-element tail and a scalar tail
           || test_softmax(RandomMat(25, 24), 0);
}
