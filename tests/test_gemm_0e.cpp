// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_gemm_0.h"

int main()
{
    SRAND(7767517);

    // the flag applies to constant operands; dynamic operands retain every matrix size
    return 0
           || test_gemm_0(1, 35, 47)
           || test_gemm_0(23, 31, 1)
           || test_gemm_0(23, 1, 23)
           || test_gemm_0(23, 31, 23)
           || test_gemm_0(31, 7, 3, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gemm_0(28, 20, 7);
}
