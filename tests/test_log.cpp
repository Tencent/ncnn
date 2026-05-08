// Copyright 2026 ihb2032 <hebome@foxmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_log(const ncnn::Mat& a, float base, float scale, float shift)
{
    ncnn::ParamDict pd;
    pd.set(0, base);
    pd.set(1, scale);
    pd.set(2, shift);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Log", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_log failed a.dims=%d a=(%d %d %d %d) base=%f scale=%f shift=%f\n", a.dims, a.w, a.h, a.d, a.c, base, scale, shift);
    }

    return ret;
}

static int test_log_0()
{
    return 0
           || test_log(RandomMat(5, 7, 24, 0.001f, 2.f), -1.f, 1.f, 0.f)
           || test_log(RandomMat(7, 9, 12, 0.001f, 2.f), -1.f, 0.75f, 0.25f)
           || test_log(RandomMat(3, 5, 13, 0.001f, 2.f), 2.f, 0.5f, 0.125f);
}

static int test_log_1()
{
    return 0
           || test_log(RandomMat(15, 24, 0.001f, 2.f), -1.f, 1.f, 0.f)
           || test_log(RandomMat(17, 12, 0.001f, 2.f), -1.f, 1.25f, 0.5f)
           || test_log(RandomMat(19, 15, 0.001f, 2.f), 2.f, 0.75f, 0.25f);
}

static int test_log_2()
{
    return 0
           || test_log(RandomMat(128, 0.001f, 2.f), -1.f, 1.f, 0.f)
           || test_log(RandomMat(124, 0.001f, 2.f), -1.f, 0.5f, 0.25f)
           || test_log(RandomMat(127, 0.001f, 2.f), 2.f, 1.5f, 0.125f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_log_0()
           || test_log_1()
           || test_log_2();
}
