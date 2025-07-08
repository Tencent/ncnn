// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_power(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;
    pd.set(0, 1.1f);
    pd.set(1, 1.5f);
    pd.set(2, 2.0f);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Power", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_power failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
    }

    return ret;
}

static int test_power_0()
{
    return 0
           || test_power(RandomMat(5, 7, 24))
           || test_power(RandomMat(7, 9, 12))
           || test_power(RandomMat(3, 5, 13));
}

static int test_power_1()
{
    return 0
           || test_power(RandomMat(15, 24))
           || test_power(RandomMat(19, 12))
           || test_power(RandomMat(17, 15));
}

static int test_power_2()
{
    return 0
           || test_power(RandomMat(128))
           || test_power(RandomMat(124))
           || test_power(RandomMat(127));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_power_0()
           || test_power_1()
           || test_power_2();
}
