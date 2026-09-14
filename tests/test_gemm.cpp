// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_gemm_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    for (int backend = 0; backend < 2; backend++)
    {
        ncnn::Layer* layer = backend == 0 ? ncnn::create_layer_naive(ncnn::LayerType::Gemm) : ncnn::create_layer_cpu(ncnn::LayerType::Gemm);
        if (!layer)
            return -1;

        int ret = layer->load_param(pd);
        delete layer;

        if ((ret == 0) != valid)
        {
            fprintf(stderr, "Gemm load_param backend=%d returned %d, expected %s\n", backend, ret, valid ? "success" : "failure");
            return -1;
        }
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
