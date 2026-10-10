// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_gemm_2.h"

int main()
{
    SRAND(7767517);

    // the flag applies to constant operands; dynamic operands retain every matrix size
    return 0
           || test_gemm_0(1, 1, 1)
           || test_gemm_0(2, 2, 2, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gemm_0(3, 3, 3, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gemm_0(4, 4, 4)
           || test_gemm_0(5, 5, 5, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gemm_0(6, 6, 6, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gemm_0(7, 7, 7, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gemm_0(8, 8, 8, TEST_LAYER_DISABLE_GPU_TESTING);
}
