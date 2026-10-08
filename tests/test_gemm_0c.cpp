// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_gemm_0.h"

int main()
{
    SRAND(7767517);

    // the flag applies to constant operands; dynamic operands retain every matrix size
    return 0
           || test_gemm_0(12, 12, 23)
           || test_gemm_0(12, 31, 12)
           || test_gemm_0(23, 12, 12)
           || test_gemm_0(1, 1, 47, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gemm_0(1, 35, 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gemm_0(47, 1, 1, TEST_LAYER_DISABLE_GPU_TESTING);
}
