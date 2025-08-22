// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int packing_cpu_naive(const ncnn::Mat& a, ncnn::Mat& b, int out_elempack)
{
    ncnn::ParamDict pd;
    pd.set(0, out_elempack);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;

    ncnn::Layer* op = ncnn::create_layer_naive("Packing");

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    op->forward(a, b, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

static int test_packing_cpu_fp32(const ncnn::Mat& a, int in_elempack, int out_elempack)
{
    ncnn::ParamDict pd;
    pd.set(0, out_elempack);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_int8_inference = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("Packing");

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat ap;
    ncnn::convert_packing(a, ap, in_elempack, opt);

    ncnn::Mat b;
    packing_cpu_naive(ap, b, out_elempack);

    ncnn::Mat c;
    op->forward(ap, c, opt);

    op->destroy_pipeline(opt);

    delete op;

    if (CompareMat(b, c, 0.001) != 0)
    {
        fprintf(stderr, "test_packing_cpu_fp32 failed a.dims=%d a=(%d %d %d %d) in_elempack=%d out_elempack=%d\n", a.dims, a.w, a.h, a.d, a.c, in_elempack, out_elempack);
        return -1;
    }

    return 0;
}

static int test_packing_cpu_fp16(const ncnn::Mat& a, int in_elempack, int out_elempack)
{
    ncnn::ParamDict pd;
    pd.set(0, out_elempack);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_int8_inference = false;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("Packing");

    if (!op->support_fp16_storage)
    {
        delete op;
        return 0;
    }

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat a16;
    ncnn::cast_float32_to_float16(a, a16, opt);

    ncnn::Mat ap;
    ncnn::convert_packing(a16, ap, in_elempack, opt);

    ncnn::Mat b;
    packing_cpu_naive(ap, b, out_elempack);

    ncnn::Mat c;
    op->forward(ap, c, opt);

    op->destroy_pipeline(opt);

    delete op;

    ncnn::Mat c32;
    ncnn::cast_float16_to_float32(c, c32, opt);

    if (CompareMat(b, c32, 0.001) != 0)
    {
        fprintf(stderr, "test_packing_cpu_fp16 failed a.dims=%d a=(%d %d %d %d) in_elempack=%d out_elempack=%d\n", a.dims, a.w, a.h, a.d, a.c, in_elempack, out_elempack);
        return -1;
    }

    return 0;
}

static int test_packing_cpu_int8(const ncnn::Mat& a, int in_elempack, int out_elempack)
{
    ncnn::ParamDict pd;
    pd.set(0, out_elempack);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_int8_inference = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("Packing");

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat a8;
    if (a.dims == 1) a8 = RandomS8Mat(a.w);
    if (a.dims == 2) a8 = RandomS8Mat(a.w, a.h);
    if (a.dims == 3) a8 = RandomS8Mat(a.w, a.h, a.c);
    if (a.dims == 4) a8 = RandomS8Mat(a.w, a.h, a.d, a.c);

    ncnn::Mat ap;
    ncnn::convert_packing(a8, ap, in_elempack, opt);

    ncnn::Mat b;
    packing_cpu_naive(ap, b, out_elempack);

    ncnn::Mat c;
    op->forward(ap, c, opt);

    op->destroy_pipeline(opt);

    delete op;

    ncnn::Mat b32;
    ncnn::cast_int8_to_float32(b, b32, opt);

    ncnn::Mat c32;
    ncnn::cast_int8_to_float32(c, c32, opt);

    if (CompareMat(b32, c32, 0.001) != 0)
    {
        fprintf(stderr, "test_packing_cpu_int8 failed a.dims=%d a=(%d %d %d %d) in_elempack=%d out_elempack=%d\n", a.dims, a.w, a.h, a.d, a.c, in_elempack, out_elempack);
        return -1;
    }

    return 0;
}

static int test_packing_cpu(const ncnn::Mat& a, int in_elempack, int out_elempack)
{
    return 0
           || test_packing_cpu_fp32(a, in_elempack, out_elempack)
           || test_packing_cpu_fp16(a, in_elempack, out_elempack)
           || test_packing_cpu_int8(a, in_elempack, out_elempack);
}

#if NCNN_VULKAN
static int test_packing_gpu_fp32(const ncnn::Mat& a, int in_elempack, int out_elempack)
{
    ncnn::ParamDict pd;
    pd.set(0, out_elempack);
    pd.set(2, 1); // cast_type_from
    pd.set(3, 1); // cast_type_to

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_packing_layout = true;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    if (!vkdev->info.support_fp16_packed()) opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage()) opt.use_fp16_storage = false;

    ncnn::Layer* op = ncnn::create_layer_vulkan("Packing");

    op->vkdev = vkdev;

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat ap;
    ncnn::convert_packing(a, ap, in_elempack, opt);

    ncnn::Mat b;
    packing_cpu_naive(ap, b, out_elempack);

    ncnn::Mat d;

    // forward
    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat a_gpu;
    cmd.record_clone(ap, a_gpu, opt);

    ncnn::VkMat d_gpu;
    op->forward(a_gpu, d_gpu, cmd, opt);

    // download
    cmd.record_clone(d_gpu, d, opt);

    cmd.submit_and_wait();

    op->destroy_pipeline(opt);

    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    if (CompareMat(b, d, 0.001) != 0)
    {
        fprintf(stderr, "test_packing_gpu failed a.dims=%d a=(%d %d %d %d) in_elempack=%d out_elempack=%d\n", a.dims, a.w, a.h, a.d, a.c, in_elempack, out_elempack);
        return -1;
    }

    return 0;
}

static int test_packing_gpu_int8(const ncnn::Mat& a, int in_elempack, int out_elempack)
{
    ncnn::ParamDict pd;
    pd.set(0, out_elempack);
    pd.set(2, 4); // cast_type_from
    pd.set(3, 4); // cast_type_to

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_packing_layout = true;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    if (!vkdev->info.support_int8_packed()) opt.use_int8_packed = false;
    if (!vkdev->info.support_int8_storage()) opt.use_int8_storage = false;

    ncnn::Layer* op = ncnn::create_layer_vulkan("Packing");

    op->vkdev = vkdev;

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    op->create_pipeline(opt);

    ncnn::Mat a8;
    if (a.dims == 1) a8 = RandomS8Mat(a.w);
    if (a.dims == 2) a8 = RandomS8Mat(a.w, a.h);
    if (a.dims == 3) a8 = RandomS8Mat(a.w, a.h, a.c);
    if (a.dims == 4) a8 = RandomS8Mat(a.w, a.h, a.d, a.c);

    ncnn::Mat ap;
    ncnn::convert_packing(a8, ap, in_elempack, opt);

    ncnn::Mat b;
    packing_cpu_naive(ap, b, out_elempack);

    ncnn::Mat c;

    // forward
    ncnn::VkCompute cmd(vkdev);

    // upload
    ncnn::VkMat a_gpu;
    cmd.record_clone(ap, a_gpu, opt);

    ncnn::VkMat c_gpu;
    op->forward(a_gpu, c_gpu, cmd, opt);

    // download
    cmd.record_clone(c_gpu, c, opt);

    cmd.submit_and_wait();

    op->destroy_pipeline(opt);

    delete op;

    ncnn::Mat b32;
    ncnn::cast_int8_to_float32(b, b32, opt);

    ncnn::Mat c32;
    ncnn::cast_int8_to_float32(c, c32, opt);

    if (CompareMat(b32, c32, 0.001) != 0)
    {
        fprintf(stderr, "test_packing_gpu_int8 failed a.dims=%d a=(%d %d %d %d) in_elempack=%d out_elempack=%d\n", a.dims, a.w, a.h, a.d, a.c, in_elempack, out_elempack);
        return -1;
    }

    return 0;
}

static int test_packing_gpu(const ncnn::Mat& a, int in_elempack, int out_elempack)
{
    return 0
           || test_packing_gpu_fp32(a, in_elempack, out_elempack)
           || test_packing_gpu_int8(a, in_elempack, out_elempack);
}
#endif

static int test_packing_cpu(const ncnn::Mat& a)
{
    return 0
           || test_packing_cpu(a, 1, 1)
           || test_packing_cpu(a, 4, 4)
           || test_packing_cpu(a, 4, 8)
           || test_packing_cpu(a, 1, 4)
           || test_packing_cpu(a, 4, 1)
           || test_packing_cpu(a, 1, 8)
           || test_packing_cpu(a, 8, 1)
           || test_packing_cpu(a, 4, 8)
           || test_packing_cpu(a, 8, 4)
           || test_packing_cpu(a, 1, 16)
           || test_packing_cpu(a, 16, 1)
           || test_packing_cpu(a, 4, 16)
           || test_packing_cpu(a, 16, 4)
           || test_packing_cpu(a, 8, 16)
           || test_packing_cpu(a, 16, 8);
}

#if NCNN_VULKAN
static int test_packing_gpu(const ncnn::Mat& a)
{
    return 0
           || test_packing_gpu(a, 1, 1)
           || test_packing_gpu(a, 4, 4)
           || test_packing_gpu(a, 1, 4)
           || test_packing_gpu(a, 4, 1);
}
#endif // NCNN_VULKAN

static int test_packing_0()
{
    ncnn::Mat a = RandomMat(9, 7, 10, 16);
    ncnn::Mat b = RandomMat(9, 7, 10, 3);
    return 0
           || test_packing_cpu(a)
           || test_packing_cpu(b)
#if NCNN_VULKAN
           || test_packing_gpu(a)
#endif
           ;
}

static int test_packing_1()
{
    ncnn::Mat a = RandomMat(9, 10, 16);
    ncnn::Mat b = RandomMat(9, 10, 3);
    return 0
           || test_packing_cpu(a)
           || test_packing_cpu(b)
#if NCNN_VULKAN
           || test_packing_gpu(a)
#endif
           ;
}

static int test_packing_2()
{
    ncnn::Mat a = RandomMat(19, 16);
    return 0
           || test_packing_cpu(a)
#if NCNN_VULKAN
           || test_packing_gpu(a)
#endif
           ;
}

static int test_packing_3()
{
    ncnn::Mat a = RandomMat(80);
    return 0
           || test_packing_cpu(a)
#if NCNN_VULKAN
           || test_packing_gpu(a)
#endif
           ;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_packing_0()
           || test_packing_1()
           || test_packing_2()
           || test_packing_3();
}
