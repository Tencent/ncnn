// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_gemm(int M, int N, int K, int TILE_M, int TILE_N, int TILE_K, float alpha, int transA, int transB, int output_transpose)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, 1.f); // beta
    pd.set(2, transA);
    pd.set(3, transB);
    pd.set(14, output_transpose);

    pd.set(20, TILE_M);
    pd.set(21, TILE_N);
    pd.set(22, TILE_K);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a(2);
    a[0] = transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M);
    a[1] = transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K);

    Randomize(a[0]);
    Randomize(a[1]);

    int ret = test_layer("Gemm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm failed M=%d N=%d K=%d TILE_M=%d TILE_N=%d TILE_K=%d alpha=%f transA=%d transB=%d output_transpose=%d\n", M, N, K, TILE_M, TILE_N, TILE_K, alpha, transA, transB, output_transpose);
    }

    return ret;
}

static int test_gemm_0(int M, int N, int K, int TILE_M, int TILE_N, int TILE_K)
{
    return 0
           || test_gemm(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 0, 0, 0)
           || test_gemm(M, N, K, TILE_M, TILE_N, TILE_K, 3.1f, 0, 1, 0)
           || test_gemm(M, N, K, TILE_M, TILE_N, TILE_K, 4.1f, 1, 0, 0)
           || test_gemm(M, N, K, TILE_M, TILE_N, TILE_K, 5.1f, 1, 1, 0)
           || test_gemm(M, N, K, TILE_M, TILE_N, TILE_K, 2.1f, 0, 0, 1)
           || test_gemm(M, N, K, TILE_M, TILE_N, TILE_K, 3.1f, 0, 1, 1)
           || test_gemm(M, N, K, TILE_M, TILE_N, TILE_K, 4.1f, 1, 0, 1)
           || test_gemm(M, N, K, TILE_M, TILE_N, TILE_K, 5.1f, 1, 1, 1);
}
