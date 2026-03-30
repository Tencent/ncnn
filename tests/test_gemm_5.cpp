// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_gemm(int M, int N, int K, int output_transpose, int output_N1M = 0)
{
    // int ds = output_transpose ? N : M;
    // int output_elempack = ds % 8 == 0 ? 8 : 1;

    ncnn::ParamDict pd;
    pd.set(0, 1.f); // alpha
    pd.set(1, 1.f); // beta
    pd.set(2, 0);   // transA
    pd.set(3, 0);   // transB
    pd.set(4, 0);   // constantA
    pd.set(5, 0);   // constantB
    pd.set(6, 1);
    pd.set(7, M);
    pd.set(8, N);
    pd.set(9, K);
    pd.set(10, -1);
    pd.set(11, output_N1M);
    // pd.set(12, output_elempack);
    pd.set(13, 1); // output_elemtype = fp32
    pd.set(14, output_transpose);

    std::vector<ncnn::Mat> weights;

    std::vector<ncnn::Mat> a;
    a.push_back(output_N1M ? ncnn::Mat(K, 1, M) : ncnn::Mat(K, M));
    a.push_back(output_N1M ? ncnn::Mat(N, 1, K) : ncnn::Mat(N, K));

    for (size_t i = 0; i < a.size(); i++)
    {
        Randomize(a[i]);
    }

    int ret = test_layer("Gemm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gemm failed M=%d N=%d K=%d output_transpose=%d output_N1M=%d\n", M, N, K, output_transpose, output_N1M);
    }

    return ret;
}

static int test_gemm_0(int M, int N, int K)
{
    return 0
           || test_gemm(M, N, K, 0, 0)
           || test_gemm(M, N, K, 0, 1)
           || test_gemm(M, N, K, 1, 0)
           || test_gemm(M, N, K, 1, 1);
}

int main()
{
    SRAND(7767517);

    int mnk_scalar[][3] = {
        {1, 1, 1},
        {2, 2, 2},
        {3, 3, 3},
        {4, 4, 4},
        {5, 5, 5},
        {7, 7, 7},
        {8, 8, 8},
        {15, 15, 15},
        {16, 16, 16},
        {24, 24, 24},
        {31, 32, 31},
        {32, 32, 32},
        {47, 48, 47},
        {64, 64, 64},
    };

    for (int i = 0; i < 14; i++)
    {
        int ret = test_gemm_0(mnk_scalar[i][0], mnk_scalar[i][1], mnk_scalar[i][2]);
        if (ret != 0)
            return ret;
    }

    return 0;
}
