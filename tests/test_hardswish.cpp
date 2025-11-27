// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_hardswish(const ncnn::Mat& a, float alpha, float beta)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, beta);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("HardSwish", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_hardswish failed a.dims=%d a=(%d %d %d) alpha=%f beta=%f\n", a.dims, a.w, a.h, a.c, alpha, beta);
    }

    return ret;
}

static int test_hardswish_0()
{
    return 0
           || test_hardswish(RandomMat(5, 7, 24), 0.2f, 0.5f)
           || test_hardswish(RandomMat(7, 9, 12), 0.2f, 0.5f)
           || test_hardswish(RandomMat(3, 5, 13), 0.2f, 0.5f);
}

static int test_hardswish_1()
{
    return 0
           || test_hardswish(RandomMat(15, 24), 0.2f, 0.5f)
           || test_hardswish(RandomMat(17, 12), 0.2f, 0.5f)
           || test_hardswish(RandomMat(19, 15), 0.2f, 0.5f);
}

static int test_hardswish_2()
{
    return 0
           || test_hardswish(RandomMat(128), 0.2f, 0.5f)
           || test_hardswish(RandomMat(124), 0.2f, 0.5f)
           || test_hardswish(RandomMat(127), 0.2f, 0.5f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_hardswish_0()
           || test_hardswish_1()
           || test_hardswish_2();
}
