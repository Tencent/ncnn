// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif // NCNN_VULKAN

static int test_gemm_dynamic_a(int m, int trans_a)
{
    const int n = 32;
    const int k = 16;

    ncnn::ParamDict pd;
    pd.set(2, trans_a);
    pd.set(3, 1);
    pd.set(5, 1);
    pd.set(6, 1);
    pd.set(7, 8);
    pd.set(8, n);
    pd.set(9, k);
    pd.set(10, 4);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(n * k);
    weights[1] = RandomMat(n);

    ncnn::Mat a = trans_a ? RandomMat(m, k) : RandomMat(k, m);

    int ret = test_layer("Gemm", pd, weights, a, 0.005f);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_dynamic_a failed m=%d trans_a=%d\n", m, trans_a);
    }

    return ret;
}

static int test_gemm_dynamic_a_0()
{
    return 0
           || test_gemm_dynamic_a(8, 0)
           || test_gemm_dynamic_a(9, 0)
           || test_gemm_dynamic_a(9, 1);
}

#if NCNN_VULKAN
static int test_gemm_dynamic_a_gpu_fp16a(int m, int trans_a)
{
    const int n = 32;
    const int k = 16;

    ncnn::ParamDict pd;
    pd.set(2, trans_a);
    pd.set(3, 1);
    pd.set(5, 1);
    pd.set(6, 1);
    pd.set(7, 8);
    pd.set(8, n);
    pd.set(9, k);
    pd.set(10, 4);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(n * k);
    weights[1] = RandomMat(n);

    ncnn::Mat a = trans_a ? RandomMat(m, k) : RandomMat(k, m);

    ncnn::Mat expected;
    test_layer_naive(ncnn::LayerType::Gemm, pd, weights, a, expected, 0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkBlobAllocator blob_vkallocator(vkdev);
    ncnn::VkStagingAllocator staging_vkallocator(vkdev);
    ncnn::VkWeightAllocator weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator weight_staging_vkallocator(vkdev);

    opt.blob_vkallocator = &blob_vkallocator;
    opt.workspace_vkallocator = &blob_vkallocator;
    opt.staging_vkallocator = &staging_vkallocator;

    ncnn::Layer* op = ncnn::create_layer_vulkan("Gemm");
    op->vkdev = vkdev;
    op->load_param(pd);
    op->load_model(ncnn::ModelBinFromMatArray(weights.data()));
    int ret = op->create_pipeline(opt);

    if (ret == 0)
    {
        ncnn::VkTransfer cmd(vkdev);
        ncnn::Option opt_upload = opt;
        opt_upload.blob_vkallocator = &weight_vkallocator;
        opt_upload.workspace_vkallocator = &weight_vkallocator;
        opt_upload.staging_vkallocator = &weight_staging_vkallocator;
        ret = op->upload_model(cmd, opt_upload);
        if (ret == 0)
            ret = cmd.submit_and_wait();
        weight_vkallocator.finalize_weights();
    }

    ncnn::Mat out;
    if (ret == 0)
    {
        ncnn::VkCompute cmd(vkdev);
        ncnn::VkMat a_gpu;
        ncnn::VkMat out_gpu;
        cmd.record_upload(a, a_gpu, opt);
        ret = op->forward(a_gpu, out_gpu, cmd, opt);
        if (ret == 0)
        {
            cmd.record_download(out_gpu, out, opt);
            ret = cmd.submit_and_wait();
        }
    }

    op->destroy_pipeline(opt);
    delete op;

    if (ret != 0 || CompareMat(expected, out, 0.01f) != 0)
    {
        fprintf(stderr, "test_gemm_dynamic_a_gpu_fp16a failed m=%d trans_a=%d ret=%d\n", m, trans_a, ret);
        return -1;
    }

    return 0;
}

static int test_gemm_dynamic_a_gpu_fp16a_0()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    const ncnn::GpuInfo& info = ncnn::get_gpu_info();
    if (!info.support_fp16_packed() || !info.support_fp16_storage() || !info.support_fp16_arithmetic())
        return 0;

    return 0
           || test_gemm_dynamic_a_gpu_fp16a(8, 0)
           || test_gemm_dynamic_a_gpu_fp16a(9, 0)
           || test_gemm_dynamic_a_gpu_fp16a(9, 1);
}
#endif // NCNN_VULKAN

#if NCNN_VALIDATION
static int test_gemm_load_param()
{
    ncnn::ParamDict base;
    base.set(4, 1);
    base.set(7, 8);
    base.set(9, 8);
    if (test_layer_param(ncnn::LayerType::Gemm, base, 0) != 0)
        return -1;

    return test_layer_param(ncnn::LayerType::Gemm, base, 7, -1, -1);
}
#endif // NCNN_VALIDATION

int main()
{
    SRAND(7767517);

    return 0
           || test_gemm_dynamic_a_0()
#if NCNN_VULKAN
           || test_gemm_dynamic_a_gpu_fp16a_0()
#endif // NCNN_VULKAN
#if NCNN_VALIDATION
           || test_gemm_load_param()
#endif // NCNN_VALIDATION
           ;
}
