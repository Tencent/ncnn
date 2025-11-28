// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_gemm_nt(int M, int N, int K, int transA, int transB, int output_transpose, int constantA, int constantB)
{
    ncnn::ParamDict pd;
    pd.set(2, transA);
    pd.set(3, transB);
    pd.set(4, constantA);
    pd.set(5, constantB);
    pd.set(6, 1);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, -1);
    pd.set(14, output_transpose);

    std::vector<ncnn::Mat> weights;
    if (constantA) weights.push_back(transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M));
    if (constantB) weights.push_back(transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K));

    std::vector<ncnn::Mat> a;
    if (!constantA) a.push_back(transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M));
    if (!constantB) a.push_back(transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K));

    for (size_t i = 0; i < weights.size(); i++)
    {
        Randomize(weights[i]);
    }

    for (size_t i = 0; i < a.size(); i++)
    {
        Randomize(a[i]);
    }

    float epsilon = 0.001;

    int ret = test_layer("Gemm", pd, weights, a, 1, epsilon, TEST_LAYER_ENABLE_THREADING);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_nt failed M=%d N=%d K=%d transA=%d transB=%d output_transpose=%d constantA=%d constantB=%d\n", M, N, K, transA, transB, output_transpose, constantA, constantB);
    }

    return ret;
}

static int test_gemm_0(int M, int N, int K)
{
    return 0
           || test_gemm_nt(M, N, K, 0, 0, 0, 0, 0)
           || test_gemm_nt(M, N, K, 0, 1, 0, 0, 0)
           || test_gemm_nt(M, N, K, 1, 0, 1, 0, 0)
           || test_gemm_nt(M, N, K, 1, 1, 1, 0, 0)

           || test_gemm_nt(M, N, K, 0, 0, 1, 1, 0)
           || test_gemm_nt(M, N, K, 0, 1, 1, 1, 0)
           || test_gemm_nt(M, N, K, 1, 0, 0, 1, 0)
           || test_gemm_nt(M, N, K, 1, 1, 0, 1, 0)

           || test_gemm_nt(M, N, K, 0, 0, 0, 0, 1)
           || test_gemm_nt(M, N, K, 0, 1, 1, 0, 1)
           || test_gemm_nt(M, N, K, 1, 0, 0, 0, 1)
           || test_gemm_nt(M, N, K, 1, 1, 1, 0, 1)

           || test_gemm_nt(M, N, K, 0, 0, 1, 1, 1)
           || test_gemm_nt(M, N, K, 0, 1, 0, 1, 1)
           || test_gemm_nt(M, N, K, 1, 0, 1, 1, 1)
           || test_gemm_nt(M, N, K, 1, 1, 0, 1, 1);
}

int main()
{
    SRAND(7767517);

    int mnk[][3] = {
        {1, 20, 40},
        {20, 2, 39},
        {3, 30, 13},
        {33, 1, 19}
    };

    int mnk_count = sizeof(mnk) / sizeof(int) / 3;

    for (int i = 0; i < mnk_count; i++)
    {
        int M = mnk[i][0];
        int N = mnk[i][1];
        int K = mnk[i][2];

        int ret = test_gemm_0(M, N, K);

        if (ret != 0)
            return ret;
    }

    return 0;
}
