// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static void set_input_params(ncnn::ParamDict& pd, const ncnn::Mat& a)
{
    pd.set(0, a.w);
    if (a.dims >= 2)
        pd.set(1, a.h);
    if (a.dims == 4)
        pd.set(11, a.d);
    if (a.dims >= 3)
        pd.set(2, a.c);
}

static int test_input(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;
    set_input_params(pd, a);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Input", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_input failed a.dims=%d a=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
    }

    return ret;
}

static int test_input_4d()
{
    return 0
           || test_input(RandomMat(5, 7, 3, 16))
           || test_input(RandomMat(3, 5, 4, 13));
}

static int test_input_0()
{
    return 0
           || test_input(RandomMat(5, 7, 16))
           || test_input(RandomMat(3, 5, 13));
}

static int test_input_1()
{
    return 0
           || test_input(RandomMat(6, 16))
           || test_input(RandomMat(7, 15));
}

static int test_input_2()
{
    return 0
           || test_input(RandomMat(128))
           || test_input(RandomMat(127));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_input_4d()
           || test_input_0()
           || test_input_1()
           || test_input_2();
}
