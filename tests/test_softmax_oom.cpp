// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_softmax_oom(const ncnn::Mat& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); // axis
    pd.set(1, 1);    // fixbug0

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer_oom("Softmax", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_softmax_oom failed a.dims=%d a=(%d %d %d %d) axis=%d\n", a.dims, a.w, a.h, a.d, a.c, axis);
    }

    return ret;
}

static int test_softmax_0()
{
    ncnn::Mat a = RandomMat(18, 17, 19, 32);
    return test_softmax_oom(a, 0) || test_softmax_oom(a, 1) || test_softmax_oom(a, 2) || test_softmax_oom(a, 3);
}

static int test_softmax_1()
{
    ncnn::Mat a = RandomMat(25, 27, 32);
    return test_softmax_oom(a, 0) || test_softmax_oom(a, 1) || test_softmax_oom(a, 2);
}

static int test_softmax_2()
{
    ncnn::Mat a = RandomMat(25, 32);
    return test_softmax_oom(a, 0) || test_softmax_oom(a, 1);
}

static int test_softmax_3()
{
    ncnn::Mat a = RandomMat(128);
    return test_softmax_oom(a, 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_softmax_0()
           || test_softmax_1()
           || test_softmax_2()
           || test_softmax_3();
}
