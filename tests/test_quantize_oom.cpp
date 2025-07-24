// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_quantize_oom(const ncnn::Mat& a, float scale_low, float scale_high)
{
    ncnn::Mat scale_data;
    if (scale_low == scale_high)
    {
        scale_data.create(1);
        scale_data[0] = scale_low;
    }
    else
    {
        if (a.dims == 1) scale_data.create(1);
        if (a.dims == 2) scale_data.create(a.h);
        if (a.dims == 3) scale_data.create(a.c);
        Randomize(scale_data, scale_low, scale_high);
    }

    ncnn::ParamDict pd;
    pd.set(0, scale_data.w);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = scale_data;

    int ret = test_layer_oom("Quantize", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_quantize_oom failed a.dims=%d a=(%d %d %d) scale_low=%f scale_high=%f\n", a.dims, a.w, a.h, a.c, scale_low, scale_high);
    }

    return ret;
}

static int test_quantize_0()
{
    return 0
           || test_quantize_oom(RandomMat(5, 7, 24), 100.f, 100.f)
           || test_quantize_oom(RandomMat(7, 9, 12), 100.f, 100.f)
           || test_quantize_oom(RandomMat(3, 5, 13), 100.f, 100.f);
}

static int test_quantize_1()
{
    return 0
           || test_quantize_oom(RandomMat(15, 24), 100.f, 100.f)
           || test_quantize_oom(RandomMat(17, 12), 100.f, 100.f)
           || test_quantize_oom(RandomMat(19, 15), 100.f, 100.f);
}

static int test_quantize_2()
{
    return 0
           || test_quantize_oom(RandomMat(128), 120.f, 140.f)
           || test_quantize_oom(RandomMat(124), 120.f, 140.f)
           || test_quantize_oom(RandomMat(127), 120.f, 140.f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_quantize_0()
           || test_quantize_1()
           || test_quantize_2();
}
