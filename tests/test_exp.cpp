// Copyright 2026 ihb2032 <hebome@foxmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_exp(const ncnn::Mat& a, float base, float scale, float shift)
{
    ncnn::ParamDict pd;
    pd.set(0, base);
    pd.set(1, scale);
    pd.set(2, shift);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Exp", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_exp failed a.dims=%d a=(%d %d %d %d) base=%f scale=%f shift=%f\n", a.dims, a.w, a.h, a.d, a.c, base, scale, shift);
    }

    return ret;
}

static int test_exp_0()
{
    return 0
           || test_exp(RandomMat(5, 7, 24, -1.f, 1.f), -1.f, 1.f, 0.f)
           || test_exp(RandomMat(7, 9, 12, -1.f, 1.f), -1.f, 0.75f, -0.25f)
           || test_exp(RandomMat(3, 5, 13, -1.f, 1.f), 2.f, 0.5f, 0.125f);
}

static int test_exp_1()
{
    return 0
           || test_exp(RandomMat(15, 24, -1.f, 1.f), -1.f, 1.f, 0.f)
           || test_exp(RandomMat(17, 12, -1.f, 1.f), -1.f, 1.25f, 0.5f)
           || test_exp(RandomMat(19, 15, -1.f, 1.f), 2.f, 0.75f, -0.5f);
}

static int test_exp_2()
{
    return 0
           || test_exp(RandomMat(128, -1.f, 1.f), -1.f, 1.f, 0.f)
           || test_exp(RandomMat(124, -1.f, 1.f), -1.f, 0.5f, 0.25f)
           || test_exp(RandomMat(127, -1.f, 1.f), 2.f, 1.5f, -0.75f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_exp_0()
           || test_exp_1()
           || test_exp_2();
}
