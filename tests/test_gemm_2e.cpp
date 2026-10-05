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

// constant operand rows retain pack16 bodies and all input transpose layouts
static int test_gemm_constant_pack16()
{
    return 0
           || test_gemm_bias(48, 35, 47, RandomMat(35, 48), 2.1f, 0.5f, 0, 0, 0, 1, 0, 0)
           || test_gemm_bias(48, 35, 47, RandomMat(35, 48), 2.1f, 0.5f, 1, 0, 0, 1, 0, 0)
           || test_gemm_bias(48, 35, 47, RandomMat(35, 48), 2.1f, 0.5f, 0, 0, 1, 1, 0, 0)
           || test_gemm_bias(48, 35, 47, RandomMat(35, 48), 2.1f, 0.5f, 1, 0, 1, 1, 0, 0)
           || test_gemm_bias(47, 48, 47, RandomMat(48, 47), 2.1f, 0.5f, 0, 0, 0, 0, 1, 0)
           || test_gemm_bias(47, 48, 47, RandomMat(48, 47), 2.1f, 0.5f, 0, 1, 0, 0, 1, 0)
           || test_gemm_bias(47, 48, 47, RandomMat(48, 47), 2.1f, 0.5f, 0, 0, 1, 0, 1, 0)
           || test_gemm_bias(47, 48, 47, RandomMat(48, 47), 2.1f, 0.5f, 0, 1, 1, 0, 1, 0)
           || test_gemm_bias(47, 35, 48, RandomMat(35, 47), 2.1f, 0.5f, 0, 0, 0, 1, 1, 0)
           || test_gemm_bias(47, 35, 48, RandomMat(35, 47), 2.1f, 0.5f, 0, 1, 0, 1, 1, 0)
           || test_gemm_bias(47, 35, 48, RandomMat(35, 47), 2.1f, 0.5f, 1, 0, 0, 1, 1, 0)
           || test_gemm_bias(47, 35, 48, RandomMat(35, 47), 2.1f, 0.5f, 1, 1, 0, 1, 1, 0);
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

// constant a conversion retains a partial row panel and reduction tail
static int test_gemm_constant_a_tail()
{
    return 0
           || test_gemm_bias(47, 48, 47, RandomMat(48, 47), 2.1f, 0.5f, 0, 0, 0, 1, 0, 0)
           || test_gemm_bias(47, 48, 47, RandomMat(48, 47), 2.1f, 0.5f, 1, 0, 0, 1, 0, 0);
}

int main()
{
    SRAND(7767517);

    // the original constant operand packing groups retain their cpu-only policy
    int ret = 0
              || test_gemm_0(31, 7, 3, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_gemm_0(32, 32, 9, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_gemm_0(44, 19, 7, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_gemm_0(32, 24, 5, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_gemm_0(20, 24, 5, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_gemm_0(32, 20, 5, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_gemm_0(24, 20, 5, TEST_LAYER_DISABLE_GPU_TESTING);
    if (ret != 0)
        return ret;

    // one full parameter group covers constant operands and all six bias layouts
    ret = test_gemm_0(28, 20, 7);
    if (ret != 0)
        return ret;

    return 0
           || test_gemm_geometry_transposes(47, 35, 48)
           || test_gemm_geometry_transposes(47, 48, 47)
           || test_gemm_geometry_transposes(48, 35, 47)
           || test_gemm_constant_pack16()
           || test_gemm_bias_tails(47, 35, 48)
           || test_gemm_bias_tails(47, 48, 47)
           || test_gemm_bias_tails(48, 35, 47)
           || test_gemm_constant_a_tail();
}
