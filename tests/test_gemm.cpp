// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"
#include "modelbin.h"

static int test_gemm_empty_bottom(int M, int N, int K)
{
    // the base Gemm::forward must reject empty input blobs loudly
    // instead of silently producing empty/garbage output
    // NOTE the arch-optimized Gemm_x86 / Gemm_arm override forward,
    // so use create_layer_naive to exercise the base implementation

    ncnn::ParamDict pd;
    pd.set(2, 0);  // transA
    pd.set(3, 1);  // transB
    pd.set(4, 0);  // constantA
    pd.set(5, 0);  // constantB
    pd.set(6, 0);  // constantC
    pd.set(7, M);  // constantM
    pd.set(8, N);  // constantN
    pd.set(9, K);  // constantK
    pd.set(10, -1);
    pd.set(11, 0); // output_N1M
    pd.set(14, 0); // output_transpose

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.lightmode = false;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;

    ncnn::Mat A = RandomMat(K, M);
    ncnn::Mat B = RandomMat(K, N);

    // dynamic A and B, empty A
    {
        ncnn::Layer* op = ncnn::create_layer_naive(ncnn::LayerType::Gemm);
        op->load_param(pd);
        op->create_pipeline(opt);

        std::vector<ncnn::Mat> bottom_blobs(2);
        bottom_blobs[1] = B;
        std::vector<ncnn::Mat> top_blobs(1);

        int ret = op->forward(bottom_blobs, top_blobs, opt);

        op->destroy_pipeline(opt);
        delete op;

        if (ret == 0)
        {
            fprintf(stderr, "test_gemm_empty_bottom dynamic AB with empty A not rejected M=%d N=%d K=%d\n", M, N, K);
            return -1;
        }
    }

    // dynamic A and B, empty B
    {
        ncnn::Layer* op = ncnn::create_layer_naive(ncnn::LayerType::Gemm);
        op->load_param(pd);
        op->create_pipeline(opt);

        std::vector<ncnn::Mat> bottom_blobs(2);
        bottom_blobs[0] = A;
        std::vector<ncnn::Mat> top_blobs(1);

        int ret = op->forward(bottom_blobs, top_blobs, opt);

        op->destroy_pipeline(opt);
        delete op;

        if (ret == 0)
        {
            fprintf(stderr, "test_gemm_empty_bottom dynamic AB with empty B not rejected M=%d N=%d K=%d\n", M, N, K);
            return -1;
        }
    }

    // constant B, empty dynamic A
    {
        ncnn::ParamDict pd2 = pd;
        pd2.set(5, 1); // constantB

        ncnn::Mat B_data = RandomMat(K, N);
        std::vector<ncnn::Mat> weights(1, B_data);

        ncnn::Layer* op = ncnn::create_layer_naive(ncnn::LayerType::Gemm);
        op->load_param(pd2);

        ncnn::ModelBinFromMatArray mb(weights.data());
        op->load_model(mb);
        op->create_pipeline(opt);

        std::vector<ncnn::Mat> bottom_blobs(1);
        std::vector<ncnn::Mat> top_blobs(1);

        int ret = op->forward(bottom_blobs, top_blobs, opt);

        op->destroy_pipeline(opt);
        delete op;

        if (ret == 0)
        {
            fprintf(stderr, "test_gemm_empty_bottom constant B with empty A not rejected M=%d N=%d K=%d\n", M, N, K);
            return -1;
        }
    }

    // sanity: valid inputs still succeed
    {
        ncnn::Layer* op = ncnn::create_layer_naive(ncnn::LayerType::Gemm);
        op->load_param(pd);
        op->create_pipeline(opt);

        std::vector<ncnn::Mat> bottom_blobs(2);
        bottom_blobs[0] = A;
        bottom_blobs[1] = B;
        std::vector<ncnn::Mat> top_blobs(1);

        int ret = op->forward(bottom_blobs, top_blobs, opt);

        op->destroy_pipeline(opt);
        delete op;

        if (ret != 0 || top_blobs[0].dims != 2 || top_blobs[0].w != N || top_blobs[0].h != M)
        {
            fprintf(stderr, "test_gemm_empty_bottom valid inputs failed ret=%d M=%d N=%d K=%d\n", ret, M, N, K);
            return -1;
        }
    }

    return 0;
}

#if NCNN_VULKAN
static int test_gemm_empty_bottom_vulkan(int M, int N, int K)
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
    {
        // no vulkan device, skip
        return 0;
    }

    // Gemm_vulkan::forward must reject an empty bottom VkMat gracefully
    // instead of SIGFPE on the elemsize division with elempack == 0

    ncnn::ParamDict pd;
    pd.set(2, 0);  // transA
    pd.set(3, 1);  // transB
    pd.set(4, 0);  // constantA
    pd.set(5, 1);  // constantB
    pd.set(6, 0);  // constantC
    pd.set(7, M);  // constantM
    pd.set(8, N);  // constantN
    pd.set(9, K);  // constantK
    pd.set(10, -1);
    pd.set(11, 0); // output_N1M
    pd.set(14, 0); // output_transpose

    ncnn::Mat B_data = RandomMat(K, N);
    std::vector<ncnn::Mat> weights(1, B_data);

    ncnn::VkWeightAllocator weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator weight_staging_vkallocator(vkdev);

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.lightmode = false;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    ncnn::Layer* op = ncnn::create_layer_vulkan(ncnn::LayerType::Gemm);
    if (!op || !op->support_vulkan)
    {
        delete op;
        vkdev->reclaim_blob_allocator(blob_vkallocator);
        vkdev->reclaim_staging_allocator(staging_vkallocator);
        return 0;
    }

    op->vkdev = vkdev;

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);

    op->create_pipeline(opt);

    {
        ncnn::VkTransfer cmd(vkdev);

        ncnn::Option opt_upload = opt;
        opt_upload.blob_vkallocator = &weight_vkallocator;
        opt_upload.workspace_vkallocator = &weight_vkallocator;
        opt_upload.staging_vkallocator = &weight_staging_vkallocator;

        op->upload_model(cmd, opt_upload);

        cmd.submit_and_wait();
    }

    int ret;
    {
        ncnn::VkCompute cmd(vkdev);

        // empty bottom blob, like a released shared input blob after the first extract
        std::vector<ncnn::VkMat> bottom_blobs(1);
        std::vector<ncnn::VkMat> top_blobs(1);

        ret = op->forward(bottom_blobs, top_blobs, cmd, opt);

        cmd.submit_and_wait();
    }

    op->destroy_pipeline(opt);

    delete op;

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    if (ret == 0)
    {
        fprintf(stderr, "test_gemm_empty_bottom_vulkan empty bottom not rejected M=%d N=%d K=%d\n", M, N, K);
        return -1;
    }

    return 0;
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

    int ret = 0
              || test_gemm_empty_bottom(11, 12, 13)
              || test_gemm_empty_bottom(1, 2, 3)
              || test_gemm_empty_bottom(4, 1, 6);

#if NCNN_VULKAN
    ret = ret
          || test_gemm_empty_bottom_vulkan(11, 12, 13);
#endif

    return ret;
}
