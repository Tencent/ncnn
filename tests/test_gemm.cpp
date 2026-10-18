// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#if NCNN_VALIDATION
static int test_gemm_load_param()
{
    ncnn::ParamDict base;
    base.set(4, 1);
    base.set(7, 8);
    base.set(9, 8);
    if (test_layer_param(ncnn::LayerType::Gemm, base, 0) != 0)
        return -1;

    return test_layer_param(ncnn::LayerType::Gemm, base, 7, -1, -1);
}
#endif // NCNN_VALIDATION

int main()
{
    SRAND(7767517);

    return 0
#if NCNN_VALIDATION
           || test_gemm_load_param()
#endif // NCNN_VALIDATION
           ;
}
