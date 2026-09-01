// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"
#include "modelbin.h"

// Defense-in-depth companion of tests/test_gemm.cpp empty-bottom cases:
// the arch-optimized Gemm layers (Gemm_x86 / Gemm_arm / Gemm_mips /
// Gemm_loongarch / Gemm_riscv) override forward(vector) and bypass the base
// Gemm::forward empty-bottom guard, so they must reject empty input blobs on
// their own. Use the default create_layer path so the arch-optimized layer
// is exercised (on x86 that is Gemm_x86 or one of its isa variants).

static int test_gemm_empty_dynamic_ab(int M, int N, int K)
{
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
    opt.lightmode = true;
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
        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Gemm);
        op->load_param(pd);
        op->create_pipeline(opt);

        std::vector<ncnn::Mat> bottom_blobs(2);
        bottom_blobs[1] = B;
        std::vector<ncnn::Mat> top_blobs(1);

        int ret = op->forward(bottom_blobs, top_blobs, opt);

        op->destroy_pipeline(opt);
        delete op;

        if (ret != -1)
        {
            fprintf(stderr, "test_gemm_empty dynamic AB with empty A not rejected with -1 ret=%d M=%d N=%d K=%d\n", ret, M, N, K);
            return -1;
        }
    }

    // dynamic A and B, empty B
    {
        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Gemm);
        op->load_param(pd);
        op->create_pipeline(opt);

        std::vector<ncnn::Mat> bottom_blobs(2);
        bottom_blobs[0] = A;
        std::vector<ncnn::Mat> top_blobs(1);

        int ret = op->forward(bottom_blobs, top_blobs, opt);

        op->destroy_pipeline(opt);
        delete op;

        if (ret != -1)
        {
            fprintf(stderr, "test_gemm_empty dynamic AB with empty B not rejected with -1 ret=%d M=%d N=%d K=%d\n", ret, M, N, K);
            return -1;
        }
    }

    // sanity: valid dynamic A and B still succeed
    {
        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Gemm);
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
            fprintf(stderr, "test_gemm_empty dynamic AB valid inputs failed ret=%d M=%d N=%d K=%d\n", ret, M, N, K);
            return -1;
        }
    }

    return 0;
}

static int test_gemm_empty_constant_b(int M, int N, int K)
{
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

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.lightmode = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;

    ncnn::Mat A = RandomMat(K, M);

    ncnn::Mat B_data = RandomMat(K, N);
    std::vector<ncnn::Mat> weights(1, B_data);

    // constant B, empty dynamic A
    {
        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Gemm);
        op->load_param(pd);

        ncnn::ModelBinFromMatArray mb(weights.data());
        op->load_model(mb);
        op->create_pipeline(opt);

        std::vector<ncnn::Mat> bottom_blobs(1);
        std::vector<ncnn::Mat> top_blobs(1);

        int ret = op->forward(bottom_blobs, top_blobs, opt);

        op->destroy_pipeline(opt);
        delete op;

        if (ret != -1)
        {
            fprintf(stderr, "test_gemm_empty constant B with empty A not rejected with -1 ret=%d M=%d N=%d K=%d\n", ret, M, N, K);
            return -1;
        }
    }

    // sanity: constant B with valid dynamic A still succeeds
    {
        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Gemm);
        op->load_param(pd);

        ncnn::ModelBinFromMatArray mb(weights.data());
        op->load_model(mb);
        op->create_pipeline(opt);

        std::vector<ncnn::Mat> bottom_blobs(1);
        bottom_blobs[0] = A;
        std::vector<ncnn::Mat> top_blobs(1);

        int ret = op->forward(bottom_blobs, top_blobs, opt);

        op->destroy_pipeline(opt);
        delete op;

        if (ret != 0 || top_blobs[0].dims != 2 || top_blobs[0].w != N || top_blobs[0].h != M)
        {
            fprintf(stderr, "test_gemm_empty constant B with valid A failed ret=%d M=%d N=%d K=%d\n", ret, M, N, K);
            return -1;
        }
    }

    return 0;
}

static int test_gemm_empty_constant_a(int M, int N, int K)
{
    ncnn::ParamDict pd;
    pd.set(2, 0);  // transA
    pd.set(3, 1);  // transB
    pd.set(4, 1);  // constantA
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
    opt.lightmode = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;

    ncnn::Mat B = RandomMat(K, N);

    ncnn::Mat A_data = RandomMat(K, M);
    std::vector<ncnn::Mat> weights(1, A_data);

    // constant A, empty dynamic B
    {
        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Gemm);
        op->load_param(pd);

        ncnn::ModelBinFromMatArray mb(weights.data());
        op->load_model(mb);
        op->create_pipeline(opt);

        std::vector<ncnn::Mat> bottom_blobs(1);
        std::vector<ncnn::Mat> top_blobs(1);

        int ret = op->forward(bottom_blobs, top_blobs, opt);

        op->destroy_pipeline(opt);
        delete op;

        if (ret != -1)
        {
            fprintf(stderr, "test_gemm_empty constant A with empty B not rejected with -1 ret=%d M=%d N=%d K=%d\n", ret, M, N, K);
            return -1;
        }
    }

    // sanity: constant A with valid dynamic B still succeeds
    {
        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Gemm);
        op->load_param(pd);

        ncnn::ModelBinFromMatArray mb(weights.data());
        op->load_model(mb);
        op->create_pipeline(opt);

        std::vector<ncnn::Mat> bottom_blobs(1);
        bottom_blobs[0] = B;
        std::vector<ncnn::Mat> top_blobs(1);

        int ret = op->forward(bottom_blobs, top_blobs, opt);

        op->destroy_pipeline(opt);
        delete op;

        if (ret != 0 || top_blobs[0].dims != 2 || top_blobs[0].w != N || top_blobs[0].h != M)
        {
            fprintf(stderr, "test_gemm_empty constant A with valid B failed ret=%d M=%d N=%d K=%d\n", ret, M, N, K);
            return -1;
        }
    }

    return 0;
}

static int test_gemm_empty_constant_ab(int M, int N, int K)
{
    // constant A and B: bottom_blobs may be an empty vector, the guard must
    // not dereference it nor reject the valid packed constant data
    ncnn::ParamDict pd;
    pd.set(2, 0);  // transA
    pd.set(3, 1);  // transB
    pd.set(4, 1);  // constantA
    pd.set(5, 1);  // constantB
    pd.set(6, 0);  // constantC
    pd.set(7, M);  // constantM
    pd.set(8, N);  // constantN
    pd.set(9, K);  // constantK
    pd.set(10, -1);
    pd.set(11, 0); // output_N1M
    pd.set(14, 0); // output_transpose

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.lightmode = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(K, M); // A_data
    weights[1] = RandomMat(K, N); // B_data

    ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Gemm);
    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);
    op->create_pipeline(opt);

    std::vector<ncnn::Mat> bottom_blobs;
    std::vector<ncnn::Mat> top_blobs(1);

    int ret = op->forward(bottom_blobs, top_blobs, opt);

    op->destroy_pipeline(opt);
    delete op;

    if (ret != 0 || top_blobs[0].dims != 2 || top_blobs[0].w != N || top_blobs[0].h != M)
    {
        fprintf(stderr, "test_gemm_empty constant AB with empty bottom_blobs failed ret=%d M=%d N=%d K=%d\n", ret, M, N, K);
        return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    int mnk[][3] = {
        {5, 7, 11},
        {1, 2, 3},
        {13, 1, 6}
    };

    int mnk_count = sizeof(mnk) / sizeof(int) / 3;

    for (int i = 0; i < mnk_count; i++)
    {
        int M = mnk[i][0];
        int N = mnk[i][1];
        int K = mnk[i][2];

        int ret = 0
                  || test_gemm_empty_dynamic_ab(M, N, K)
                  || test_gemm_empty_constant_b(M, N, K)
                  || test_gemm_empty_constant_a(M, N, K)
                  || test_gemm_empty_constant_ab(M, N, K);

        if (ret != 0)
            return ret;
    }

    return 0;
}
