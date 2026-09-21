// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "test_gemm_0.h"

int main()
{
    SRAND(7767517);

    // the flag applies to constant operands; dynamic operands retain every matrix size
    return 0
           || test_gemm_0(32, 32, 9, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gemm_0(44, 19, 7, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gemm_0(47, 35, 48)
           || test_gemm_0(47, 48, 47)
           || test_gemm_0(48, 35, 47);
}
