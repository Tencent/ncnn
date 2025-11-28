// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_gemm_oom(int M, int N, int K, int transA, int transB, int output_transpose, int constantA, int constantB, int output_N1M = 0)
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
    pd.set(11, output_N1M);
    pd.set(14, output_transpose);

    std::vector<ncnn::Mat> weights;
    if (constantA) weights.push_back(transA ? (output_N1M ? ncnn::Mat(M, 1, K) : ncnn::Mat(M, K)) : (output_N1M ? ncnn::Mat(K, 1, M) : ncnn::Mat(K, M)));
    if (constantB) weights.push_back(transB ? (output_N1M ? ncnn::Mat(K, 1, N) : ncnn::Mat(K, N)) : (output_N1M ? ncnn::Mat(N, 1, K) : ncnn::Mat(N, K)));

    std::vector<ncnn::Mat> a;
    if (!constantA) a.push_back(transA ? (output_N1M ? ncnn::Mat(M, 1, K) : ncnn::Mat(M, K)) : (output_N1M ? ncnn::Mat(K, 1, M) : ncnn::Mat(K, M)));
    if (!constantB) a.push_back(transB ? (output_N1M ? ncnn::Mat(K, 1, N) : ncnn::Mat(K, N)) : (output_N1M ? ncnn::Mat(N, 1, K) : ncnn::Mat(N, K)));

    for (size_t i = 0; i < weights.size(); i++)
    {
        Randomize(weights[i]);
    }

    for (size_t i = 0; i < a.size(); i++)
    {
        Randomize(a[i]);
    }

    int ret = test_layer_oom("Gemm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm_oom failed M=%d N=%d K=%d transA=%d transB=%d output_transpose=%d constantA=%d constantB=%d output_N1M=%d\n", M, N, K, transA, transB, output_transpose, constantA, constantB, output_N1M);
    }

    return ret;
}

static int test_gemm_0(int M, int N, int K)
{
    return 0
           || test_gemm_oom(M, N, K, 0, 0, 0, 0, 0, 0)
           || test_gemm_oom(M, N, K, 1, 0, 1, 0, 0, 0)
           || test_gemm_oom(M, N, K, 0, 1, 1, 1, 0, 0)
           || test_gemm_oom(M, N, K, 1, 1, 0, 1, 0, 1)
           || test_gemm_oom(M, N, K, 0, 0, 0, 0, 1, 0)
           || test_gemm_oom(M, N, K, 0, 1, 1, 0, 1, 0)
           || test_gemm_oom(M, N, K, 1, 0, 1, 1, 1, 0)
           || test_gemm_oom(M, N, K, 1, 1, 0, 1, 1, 1);
}

int main()
{
    SRAND(7767517);

    int mnk[][3] = {
        {11, 12, 13},
        {1, 2, 3},
        {4, 1, 6},
        {9, 8, 7}
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
