// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

static int test_gemm_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::Gemm);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if (ret != (valid ? 0 : -1))
    {
        const int transA = pd.get(2, 0);
        const int transB = pd.get(3, 0);
        const int constantA = pd.get(4, 0);
        const int constantB = pd.get(5, 0);
        const int constantC = pd.get(6, 0);
        const int constantM = pd.get(7, 0);
        const int constantN = pd.get(8, 0);
        const int constantK = pd.get(9, 0);
        const int constant_broadcast_type_C = pd.get(10, 0);
        const int output_N1M = pd.get(11, 0);
        const int output_elempack = pd.get(12, 0);
        const int output_elemtype = pd.get(13, 0);
        const int output_transpose = pd.get(14, 0);
        const int quantize_term = pd.get(18, 0);

        fprintf(stderr, "test_gemm_load_param failed ret=%d expected=%d transA=%d transB=%d constantA=%d constantB=%d constantC=%d constantM=%d constantN=%d constantK=%d constant_broadcast_type_C=%d output_N1M=%d output_elempack=%d output_elemtype=%d output_transpose=%d quantize_term=%d\n", ret, valid ? 0 : -1, transA, transB, constantA, constantB, constantC, constantM, constantN, constantK, constant_broadcast_type_C, output_N1M, output_elempack, output_elemtype, output_transpose, quantize_term);
        return -1;
    }

    return 0;
}

static int test_gemm_load_param()
{
    ncnn::ParamDict pd;
    pd.set(4, 1);
    pd.set(7, 8);
    pd.set(9, 8);
    if (test_gemm_load_param_case(pd, true) != 0)
        return -1;

    pd.set(7, -1);
    if (test_gemm_load_param_case(pd, false) != 0)
        return -1;

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_gemm_load_param();
}
