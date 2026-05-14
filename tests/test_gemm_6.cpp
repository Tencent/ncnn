// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_gemm_fp16s(int M, int N, int K, int output_transpose)
{
    ncnn::ParamDict pd;
    pd.set(0, 1.25f); // alpha
    pd.set(1, 1.f);   // beta
    pd.set(2, 0);     // transA
    pd.set(3, 0);     // transB
    pd.set(4, 0);     // constantA
    pd.set(5, 0);     // constantB
    pd.set(6, 1);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, -1);
    pd.set(14, output_transpose);
    pd.set(20, 1);
    pd.set(21, N);
    pd.set(22, 8);

    std::vector<ncnn::Mat> weights;

    std::vector<ncnn::Mat> a;
    a.push_back(ncnn::Mat(K, M));
    a.push_back(ncnn::Mat(N, K));

    for (size_t i = 0; i < a.size(); i++)
    {
        Randomize(a[i], -0.5f, 0.5f);
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;

    int ret = test_layer_opt("Gemm", pd, weights, opt, a, 1, 0.001, TEST_LAYER_DISABLE_GPU_TESTING);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_fp16s failed M=%d N=%d K=%d output_transpose=%d\n", M, N, K, output_transpose);
        return ret;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_gemm_fp16s(15, 15, 9, 0)
           || test_gemm_fp16s(15, 8, 9, 0)
           || test_gemm_fp16s(15, 4, 9, 0)
           || test_gemm_fp16s(15, 15, 9, 1);
}
