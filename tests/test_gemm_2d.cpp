// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_gemm_2.h"

// geometry rows exercise both input transposes and output layouts
static int test_gemm_geometry_transposes(int M, int N, int K)
{
    const int cases[][3] = {
        {0, 0, 0},
        {0, 1, 0},
        {1, 0, 0},
        {1, 1, 0},
        {0, 0, 1},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 1}
    };

    for (int i = 0; i < 8; i++)
    {
        int ret = test_gemm_bias(M, N, K, RandomMat(N, M), 2.1f, 0.5f, cases[i][0], cases[i][1], cases[i][2], 0, 0, 0);
        if (ret != 0)
            return ret;
    }

    return 0;
}

// bias layout rows cover packed output bodies and the final column tile
static int test_gemm_bias_tails(int M, int N, int K)
{
    return 0
           || test_gemm_bias(M, N, K, RandomMat(1), 2.1f, 0.5f, 0, 0, 0, 0, 0, 0)
           || test_gemm_bias(M, N, K, RandomMat(M), 3.1f, 0.6f, 0, 1, 0, 0, 0, 0)
           || test_gemm_bias(M, N, K, RandomMat(1), 2.1f, 0.5f, 0, 0, 1, 0, 0, 0)
           || test_gemm_bias(M, N, K, RandomMat(M), 3.1f, 0.6f, 0, 1, 1, 0, 0, 0)
           || test_gemm_bias(M, N, K, RandomMat(1, M), 4.1f, 0.7f, 1, 0, 1, 0, 0, 0)
           || test_gemm_bias(M, N, K, RandomMat(N, 1), 2.1f, 0.5f, 0, 0, 0, 0, 0, 0)
           || test_gemm_bias(M, N, K, RandomMat(N), 3.1f, 0.6f, 0, 1, 1, 0, 0, 0);
}

// constant b conversion retains the eight-column tile with a reduction tail
static int test_gemm_constant_b_tail()
{
    return 0
           || test_gemm_bias(1, 35, 47, RandomMat(35, 1), 2.1f, 0.5f, 0, 0, 0, 0, 1, 0)
           || test_gemm_bias(1, 35, 47, RandomMat(35, 1), 2.1f, 0.5f, 0, 1, 0, 0, 1, 0);
}

int main()
{
    SRAND(7767517);

    // the original constant operand packing groups retain their cpu-only policy
    int ret = 0
              || test_gemm_0(24, 24, 47, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_gemm_0(24, 35, 24, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_gemm_0(47, 24, 24, TEST_LAYER_DISABLE_GPU_TESTING);
    if (ret != 0)
        return ret;

    // one full parameter group covers constant operands and all six bias layouts
    ret = test_gemm_0(23, 31, 23);
    if (ret != 0)
        return ret;

    return 0
           || test_gemm_geometry_transposes(1, 35, 47)
           || test_gemm_geometry_transposes(23, 31, 1)
           || test_gemm_geometry_transposes(23, 1, 23)
           || test_gemm_bias_tails(1, 35, 47)
           || test_gemm_bias_tails(23, 31, 1)
           || test_gemm_bias_tails(23, 1, 23)
           || test_gemm_constant_b_tail();
}
