// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

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
// record_upload casts to fp16 and packs to 4 when the element count allows it
static ncnn::Mat softmax_fp16_shape_hint(const ncnn::Mat& a)
{
    const int dims = a.dims;
    int elempack = 1;
    if (dims == 1) elempack = a.w % 4 == 0 ? 4 : 1;
    if (dims == 2) elempack = a.h % 4 == 0 ? 4 : 1;
    if (dims == 3 || dims == 4) elempack = a.c % 4 == 0 ? 4 : 1;

    const size_t elemsize = elempack * 2u;
    if (dims == 1) return ncnn::Mat(a.w / elempack, (void*)0, elemsize, elempack);
    if (dims == 2) return ncnn::Mat(a.w, a.h / elempack, (void*)0, elemsize, elempack);
    if (dims == 3) return ncnn::Mat(a.w, a.h, a.c / elempack, (void*)0, elemsize, elempack);
    return ncnn::Mat(a.w, a.h, a.d, a.c / elempack, (void*)0, elemsize, elempack);
}

// exercise fp16 arithmetic directly because test_layer_gpu disables it
static int softmax_gpu_forward_fp16a(const ncnn::Mat& a, int axis, bool use_subgroup_ops, ncnn::Mat& out)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); // axis
    pd.set(1, 1);    // fixbug0

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_subgroup_ops = use_subgroup_ops;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev->info.support_subgroup_ops())
        opt.use_subgroup_ops = false;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    ncnn::Layer* op = ncnn::create_layer_vulkan("Softmax");
    op->vkdev = vkdev;
    op->load_param(pd);

    std::vector<ncnn::Mat> weights(0);
    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);

    // softmax keeps the shape, and the hint must match what record_upload feeds the layer:
    // fp16 storage, packed to 4
    op->top_shapes.resize(1);
    op->top_shapes[0] = softmax_fp16_shape_hint(a);

    int ret = op->create_pipeline(opt);

    if (ret == 0)
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

    return ret;
}

static int test_softmax_gpu_subgroup(const ncnn::Mat& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); // axis
    pd.set(1, 1);    // fixbug0

    std::vector<ncnn::Mat> weights(0);

    // include fp16 arithmetic so that the afp conversion on the way out of the subgroup
    // reduce is covered as well
    const bool fp16_ps[4] = {false, true, true, false};
    const bool fp16_arith[4] = {false, false, true, false};
    const bool bf16[4] = {false, false, false, true};
    const char* names[4] = {"fp32", "fp16ps", "fp16psa", "bf16ps"};
    const float epsilons[4] = {0.001f, 0.002f, 0.002f, 0.01f};

    const ncnn::GpuInfo& info = ncnn::get_gpu_device()->info;

    for (int i = 0; i < 4; i++)
    {
        if (fp16_ps[i] && !info.support_fp16_storage())
            continue;
        if (bf16[i] && !info.support_bf16_storage())
            continue;

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_vulkan_compute = true;
        opt.use_fp16_packed = fp16_ps[i];
        opt.use_fp16_storage = fp16_ps[i];
        opt.use_fp16_arithmetic = fp16_arith[i];
        opt.use_bf16_packed = bf16[i];
        opt.use_bf16_storage = bf16[i];

        opt.use_subgroup_ops = false;
        ncnn::Mat b0;
        int ret = test_layer_gpu(ncnn::layer_to_index("Softmax"), pd, weights, opt, a, b0, a, 0);
        if (ret == 233)
        {
            // test_layer_gpu turns fp16 arithmetic off, that configuration is covered below
            continue;
        }
        if (ret != 0)
        {
            fprintf(stderr, "test_softmax_gpu_subgroup fallback failed a.dims=%d a=(%d %d %d %d) axis=%d %s ret=%d\n", a.dims, a.w, a.h, a.d, a.c, axis, names[i], ret);
            return -1;
        }

        opt.use_subgroup_ops = true;
        ncnn::Mat b1;
        ret = test_layer_gpu(ncnn::layer_to_index("Softmax"), pd, weights, opt, a, b1, a, 0);
        if (ret != 0)
        {
            fprintf(stderr, "test_softmax_gpu_subgroup subgroup failed a.dims=%d a=(%d %d %d %d) axis=%d %s ret=%d\n", a.dims, a.w, a.h, a.d, a.c, axis, names[i], ret);
            return -1;
        }

        if (CompareMat(b0, b1, epsilons[i]) != 0)
        {
            fprintf(stderr, "test_softmax_gpu_subgroup mismatch a.dims=%d a=(%d %d %d %d) axis=%d %s\n", a.dims, a.w, a.h, a.d, a.c, axis, names[i]);
            return -1;
        }
    }

    // fp16 storage together with fp16 arithmetic, the configuration that needs the afp
    // conversion when the subgroup reduce writes its result back
    if (info.support_fp16_storage() && info.support_fp16_arithmetic())
    {
        ncnn::Mat b0;
        int ret = softmax_gpu_forward_fp16a(a, axis, false, b0);
        if (ret != 0)
        {
            fprintf(stderr, "test_softmax_gpu_subgroup fallback failed a.dims=%d a=(%d %d %d %d) axis=%d fp16psa ret=%d\n", a.dims, a.w, a.h, a.d, a.c, axis, ret);
            return -1;
        }

        ncnn::Mat b1;
        ret = softmax_gpu_forward_fp16a(a, axis, true, b1);
        if (ret != 0)
        {
            fprintf(stderr, "test_softmax_gpu_subgroup subgroup failed a.dims=%d a=(%d %d %d %d) axis=%d fp16psa ret=%d\n", a.dims, a.w, a.h, a.d, a.c, axis, ret);
            return -1;
        }

        if (CompareMat(b0, b1, 0.01f) != 0)
        {
            fprintf(stderr, "test_softmax_gpu_subgroup mismatch a.dims=%d a=(%d %d %d %d) axis=%d fp16psa\n", a.dims, a.w, a.h, a.d, a.c, axis);
            return -1;
        }
    }

    return 0;
}

static int test_softmax_subgroup()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    // every contiguous innermost axis, plus widths around the subgroup size
    return 0
           || test_softmax_gpu_subgroup(RandomMat(1000), 0)
           || test_softmax_gpu_subgroup(RandomMat(100000), 0)
           || test_softmax_gpu_subgroup(RandomMat(129, 33), 1)
           || test_softmax_gpu_subgroup(RandomMat(64, 17, 12), 2)
           || test_softmax_gpu_subgroup(RandomMat(96, 5, 4, 8), 3)
           || test_softmax_gpu_subgroup(RandomMat(56, 56, 64), 2)
           || test_softmax_gpu_subgroup(RandomMat(1, 1, 1000), 2)
           || test_softmax_gpu_subgroup(RandomMat(31, 8), 1)
           || test_softmax_gpu_subgroup(RandomMat(33, 9, 5), 2)
           || test_softmax_gpu_subgroup(RandomMat(4097, 3), 1);
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
           // pack8 rows include a full avx512 block, an eight-element tail and a scalar tail
           || test_softmax(RandomMat(25, 24), 0)
#if NCNN_VULKAN
           || test_softmax_subgroup()
#endif
           ;
}
