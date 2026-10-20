// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

// Runs one layer through load_param, load_model, create_pipeline and, when
// inputs are given, forward, and returns the first non-zero return code, or 0.
static int run_layer(int naive, const char* type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& inputs, int top_count)
{
    ncnn::Layer* op = naive ? ncnn::create_layer_naive(type) : ncnn::create_layer_cpu(type);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    int ret = op->load_param(pd);
    if (ret == 0)
    {
        ncnn::ModelBinFromMatArray mb(weights.data());
        ret = op->load_model(mb);
    }

    if (ret == 0)
    {
        ret = op->create_pipeline(opt);
        if (ret == 0 && !inputs.empty())
        {
            std::vector<ncnn::Mat> tops(top_count);
            ret = op->forward(inputs, tops, opt);
        }
        op->destroy_pipeline(opt);
    }

    delete op;
    return ret;
}

// A is M x KA (runtime, or constant M x K), B is KB x N (runtime, or constant K x N)
static int test_gemm_shape(int naive, int constantA, int constantB, int constantC, int M, int N, int K, int a_k, int b_k, int c_m, int expected_ret)
{
    ncnn::ParamDict pd;
    pd.set(4, constantA);
    pd.set(5, constantB);
    pd.set(6, constantC);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    if (constantC)
        pd.set(10, 3);

    std::vector<ncnn::Mat> weights;
    if (constantA)
        weights.push_back(RandomMat(K, M));
    if (constantB)
        weights.push_back(RandomMat(N, K));
    if (constantC)
        weights.push_back(RandomMat(N, M));
    weights.push_back(ncnn::Mat());

    std::vector<ncnn::Mat> inputs;
    if (!constantA)
        inputs.push_back(RandomMat(a_k, c_m));
    if (!constantB)
        inputs.push_back(RandomMat(N, b_k));

    int ret = run_layer(naive, "Gemm", pd, weights, inputs, 1);
    if (ret != expected_ret)
    {
        fprintf(stderr, "test_gemm_shape failed naive=%d constantA=%d constantB=%d constantC=%d M=%d N=%d K=%d a_k=%d b_k=%d c_m=%d ret=%d expected=%d\n", naive, constantA, constantB, constantC, M, N, K, a_k, b_k, c_m, ret, expected_ret);
        return -1;
    }

    return 0;
}

#if NCNN_INT8
// the int8 scale of a constant B is read by forward, so a model without it must not load
static int test_gemm_int8_scale(int naive, int with_scale, int expected_ret)
{
    const int N = 8;
    const int K = 16;

    ncnn::ParamDict pd;
    pd.set(4, 0);
    pd.set(5, 1);
    pd.set(6, 0);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(18, 2);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomS8Mat(N, K);
    if (with_scale)
        weights[1] = RandomMat(1, 10.f, 20.f);

    std::vector<ncnn::Mat> inputs;

    int ret = run_layer(naive, "Gemm", pd, weights, inputs, 1);
    if ((ret == 0) != (expected_ret == 0))
    {
        fprintf(stderr, "test_gemm_int8_scale failed naive=%d with_scale=%d ret=%d expected=%d\n", naive, with_scale, ret, expected_ret);
        return -1;
    }

    return 0;
}
#endif // NCNN_INT8

static int test_gemm_invalid_0(int naive)
{
    return 0
           || test_gemm_shape(naive, 0, 1, 0, 4, 8, 16, 16, 0, 4, 0)
           || test_gemm_shape(naive, 0, 1, 0, 4, 8, 16, 4096, 0, 4, -1)
           || test_gemm_shape(naive, 0, 0, 0, 4, 8, 16, 16, 16, 4, 0)
           || test_gemm_shape(naive, 0, 0, 0, 4, 8, 16, 4096, 16, 4, -1)
           || test_gemm_shape(naive, 0, 0, 0, 4, 8, 16, 8, 16, 4, 0)
           || test_gemm_shape(naive, 0, 1, 0, 4, 8, 16, 8, 0, 4, -1)
           || test_gemm_shape(naive, 1, 0, 0, 4, 8, 16, 0, 16, 4, 0)
           || test_gemm_shape(naive, 1, 0, 0, 4, 8, 16, 0, 1, 4, -1)
           || test_gemm_shape(naive, 0, 1, 1, 4, 8, 16, 16, 0, 4, 0)
           || test_gemm_shape(naive, 0, 1, 1, 4, 8, 16, 16, 0, 1024, -1);
}

int main()
{
    SRAND(7767517);

#if NCNN_VALIDATION
    if (test_gemm_invalid_0(1) || test_gemm_invalid_0(0))
        return -1;
#endif // NCNN_VALIDATION

#if NCNN_INT8
    return 0
           || test_gemm_int8_scale(1, 1, 0)
           || test_gemm_int8_scale(0, 1, 0)
           || test_gemm_int8_scale(1, 0, -100)
           || test_gemm_int8_scale(0, 0, -100);
#else
    return 0;
#endif // NCNN_INT8
}
