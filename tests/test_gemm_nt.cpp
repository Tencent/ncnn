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

static int test_gemm_bias_nt(int M, int N, int K, const ncnn::Mat& C, float alpha, float beta, int transA, int transB, int output_transpose, int constantA, int constantB, int constantC)
{
    int broadcast_type_C = 0;
    if (C.dims == 1 && C.w == 1)
    {
        // scalar
        broadcast_type_C = 0;
    }
    if (C.dims == 1 && C.w == M)
    {
        // M
        // auto broadcast from h to w is the ncnn-style convention
        broadcast_type_C = 1;
    }
    if (C.dims == 1 && C.w == N)
    {
        // N
        broadcast_type_C = 4;
    }
    if (C.dims == 2 && C.w == 1 && C.h == M)
    {
        // Mx1
        broadcast_type_C = 2;
    }
    if (C.dims == 2 && C.w == N && C.h == M)
    {
        // MxN
        broadcast_type_C = 3;
    }
    if (C.dims == 2 && C.w == N && C.h == 1)
    {
        // 1xN
        broadcast_type_C = 4;
    }

    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, beta);
    pd.set(2, transA);
    pd.set(3, transB);
    pd.set(4, constantA);
    pd.set(5, constantB);
    pd.set(6, constantC);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, broadcast_type_C);
    pd.set(14, output_transpose);

    std::vector<ncnn::Mat> weights;
    if (constantA) weights.push_back(transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M));
    if (constantB) weights.push_back(transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K));
    if (constantC) weights.push_back(C);

    std::vector<ncnn::Mat> a;
    if (!constantA) a.push_back(transA ? ncnn::Mat(M, K) : ncnn::Mat(K, M));
    if (!constantB) a.push_back(transB ? ncnn::Mat(K, N) : ncnn::Mat(N, K));
    if (!constantC) a.push_back(C);

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
        fprintf(stderr, "test_gemm_bias_nt failed M=%d N=%d K=%d C.dims=%d C=(%d %d %d) alpha=%f beta=%f transA=%d transB=%d output_transpose=%d constantA=%d constantB=%d constantC=%d\n", M, N, K, C.dims, C.w, C.h, C.c, alpha, beta, transA, transB, output_transpose, constantA, constantB, constantC);
    }

    return ret;
}

static int test_gemm_1(int M, int N, int K)
{
    return 0
           || test_gemm_bias_nt(M, N, K, RandomMat(1), 2.1f, 0.5f, 0, 0, 0, 0, 0, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(M), 3.1f, 0.6f, 0, 1, 0, 0, 0, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(1, M), 4.1f, 0.7f, 1, 0, 1, 0, 0, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, M), 5.1f, 0.8f, 1, 1, 1, 0, 0, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, 1), 2.1f, 0.5f, 0, 0, 0, 0, 0, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(N), 3.1f, 0.6f, 0, 1, 0, 0, 0, 0)

           || test_gemm_bias_nt(M, N, K, RandomMat(1), 2.1f, 0.5f, 1, 0, 0, 1, 0, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(M), 3.1f, 0.6f, 1, 1, 0, 1, 0, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(1, M), 4.1f, 0.7f, 0, 0, 1, 1, 0, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, M), 5.1f, 0.8f, 0, 1, 1, 1, 0, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, 1), 2.1f, 0.5f, 1, 0, 0, 1, 0, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(N), 3.1f, 0.6f, 1, 1, 0, 1, 0, 0)

           || test_gemm_bias_nt(M, N, K, RandomMat(1), 2.1f, 0.5f, 0, 1, 0, 0, 1, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(M), 3.1f, 0.6f, 1, 0, 0, 0, 1, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(1, M), 4.1f, 0.7f, 1, 1, 1, 0, 1, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, M), 5.1f, 0.8f, 0, 0, 1, 0, 1, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, 1), 2.1f, 0.5f, 0, 1, 0, 0, 1, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(N), 3.1f, 0.6f, 1, 0, 0, 0, 1, 0)

           || test_gemm_bias_nt(M, N, K, RandomMat(1), 2.1f, 0.5f, 1, 1, 0, 1, 1, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(M), 3.1f, 0.6f, 0, 0, 0, 1, 1, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(1, M), 4.1f, 0.7f, 0, 1, 1, 1, 1, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, M), 5.1f, 0.8f, 1, 0, 1, 1, 1, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, 1), 2.1f, 0.5f, 1, 1, 0, 1, 1, 0)
           || test_gemm_bias_nt(M, N, K, RandomMat(N), 3.1f, 0.6f, 0, 0, 0, 1, 1, 0)

           || test_gemm_bias_nt(M, N, K, RandomMat(1), 2.1f, 0.5f, 1, 0, 0, 0, 0, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(M), 3.1f, 0.6f, 1, 1, 0, 0, 0, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(1, M), 4.1f, 0.7f, 0, 0, 1, 0, 0, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, M), 5.1f, 0.8f, 0, 1, 1, 0, 0, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, 1), 2.1f, 0.5f, 1, 0, 0, 0, 0, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(N), 3.1f, 0.6f, 1, 1, 0, 0, 0, 1)

           || test_gemm_bias_nt(M, N, K, RandomMat(1), 2.1f, 0.5f, 0, 0, 0, 1, 0, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(M), 3.1f, 0.6f, 0, 1, 0, 1, 0, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(1, M), 4.1f, 0.7f, 1, 0, 1, 1, 0, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, M), 5.1f, 0.8f, 1, 1, 1, 1, 0, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, 1), 2.1f, 0.5f, 0, 0, 0, 1, 0, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(N), 3.1f, 0.6f, 0, 1, 0, 1, 0, 1)

           || test_gemm_bias_nt(M, N, K, RandomMat(1), 2.1f, 0.5f, 1, 1, 0, 0, 1, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(M), 3.1f, 0.6f, 0, 0, 0, 0, 1, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(1, M), 4.1f, 0.7f, 0, 1, 1, 0, 1, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, M), 5.1f, 0.8f, 1, 0, 1, 0, 1, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, 1), 2.1f, 0.5f, 1, 1, 0, 0, 1, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(N), 3.1f, 0.6f, 0, 0, 0, 0, 1, 1)

           || test_gemm_bias_nt(M, N, K, RandomMat(1), 2.1f, 0.5f, 0, 1, 0, 1, 1, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(M), 3.1f, 0.6f, 1, 0, 0, 1, 1, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(1, M), 4.1f, 0.7f, 1, 1, 1, 1, 1, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, M), 5.1f, 0.8f, 0, 0, 1, 1, 1, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(N, 1), 2.1f, 0.5f, 0, 1, 0, 1, 1, 1)
           || test_gemm_bias_nt(M, N, K, RandomMat(N), 3.1f, 0.6f, 1, 0, 0, 1, 1, 1);
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

        int ret = test_gemm_0(M, N, K) || test_gemm_1(M, N, K);

        if (ret != 0)
            return ret;
    }

    return 0;
}
