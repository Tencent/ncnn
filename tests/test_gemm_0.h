// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_gemm(int M, int N, int K, float alpha, int transA, int transB, int output_transpose, int constantA, int constantB, int output_N1M = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, 1.f); // beta
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

    int ret = test_layer("Gemm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm failed M=%d N=%d K=%d alpha=%f transA=%d transB=%d output_transpose=%d constantA=%d constantB=%d output_N1M=%d\n", M, N, K, alpha, transA, transB, output_transpose, constantA, constantB, output_N1M);
    }

    return ret;
}

static int test_gemm_0(int M, int N, int K)
{
    return 0
           || test_gemm(M, N, K, 2.1f, 0, 0, 0, 0, 0)
           || test_gemm(M, N, K, 3.1f, 0, 1, 0, 0, 0)
           || test_gemm(M, N, K, 4.1f, 1, 0, 0, 0, 0)
           || test_gemm(M, N, K, 5.1f, 1, 1, 0, 0, 0)
           || test_gemm(M, N, K, 2.1f, 0, 0, 1, 0, 0)
           || test_gemm(M, N, K, 3.1f, 0, 1, 1, 0, 0)
           || test_gemm(M, N, K, 4.1f, 1, 0, 1, 0, 0)
           || test_gemm(M, N, K, 5.1f, 1, 1, 1, 0, 0)

           || test_gemm(M, N, K, 1.7f, 0, 1, 0, 0, 0, 1)
           || test_gemm(M, N, K, 1.7f, 1, 1, 0, 1, 0, 1)
           || test_gemm(M, N, K, 1.9f, 0, 0, 0, 0, 1, 1)
           || test_gemm(M, N, K, 1.9f, 1, 0, 0, 1, 1, 1)
           || test_gemm(M, N, K, 1.7f, 0, 1, 1, 1, 0, 1)
           || test_gemm(M, N, K, 1.7f, 1, 1, 1, 0, 1, 1)
           || test_gemm(M, N, K, 1.9f, 0, 0, 1, 1, 1, 1)
           || test_gemm(M, N, K, 1.9f, 1, 0, 1, 0, 0, 1)

           || test_gemm(M, N, K, 2.1f, 0, 0, 0, 1, 0)
           || test_gemm(M, N, K, 3.1f, 0, 1, 0, 1, 0)
           || test_gemm(M, N, K, 4.1f, 1, 0, 0, 1, 0)
           || test_gemm(M, N, K, 5.1f, 1, 1, 0, 1, 0)
           || test_gemm(M, N, K, 2.1f, 0, 0, 1, 1, 0)
           || test_gemm(M, N, K, 3.1f, 0, 1, 1, 1, 0)
           || test_gemm(M, N, K, 4.1f, 1, 0, 1, 1, 0)
           || test_gemm(M, N, K, 5.1f, 1, 1, 1, 1, 0)

           || test_gemm(M, N, K, 2.1f, 0, 0, 0, 0, 1)
           || test_gemm(M, N, K, 3.1f, 0, 1, 0, 0, 1)
           || test_gemm(M, N, K, 4.1f, 1, 0, 0, 0, 1)
           || test_gemm(M, N, K, 5.1f, 1, 1, 0, 0, 1)
           || test_gemm(M, N, K, 2.1f, 0, 0, 1, 0, 1)
           || test_gemm(M, N, K, 3.1f, 0, 1, 1, 0, 1)
           || test_gemm(M, N, K, 4.1f, 1, 0, 1, 0, 1)
           || test_gemm(M, N, K, 5.1f, 1, 1, 1, 0, 1)

           || test_gemm(M, N, K, 2.1f, 0, 0, 0, 1, 1)
           || test_gemm(M, N, K, 3.1f, 0, 1, 0, 1, 1)
           || test_gemm(M, N, K, 4.1f, 1, 0, 0, 1, 1)
           || test_gemm(M, N, K, 5.1f, 1, 1, 0, 1, 1)
           || test_gemm(M, N, K, 2.1f, 0, 0, 1, 1, 1)
           || test_gemm(M, N, K, 3.1f, 0, 1, 1, 1, 1)
           || test_gemm(M, N, K, 4.1f, 1, 0, 1, 1, 1)
           || test_gemm(M, N, K, 5.1f, 1, 1, 1, 1, 1);
}
