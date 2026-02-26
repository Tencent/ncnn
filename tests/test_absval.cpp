// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_absval(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("AbsVal", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_absval failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
    }

    return ret;
}

static int test_absval_0()
{
    return 0
           || test_absval(RandomMat(1, 1, 13))
           || test_absval(RandomMat(5, 7, 24))
           || test_absval(RandomMat(7, 9, 12))
           || test_absval(RandomMat(3, 5, 13));
}

static int test_absval_1()
{
    return 0
           || test_absval(RandomMat(1, 13))
           || test_absval(RandomMat(15, 24))
           || test_absval(RandomMat(19, 12))
           || test_absval(RandomMat(17, 15));
}

static int test_absval_2()
{
    return 0
           || test_absval(RandomMat(1))
           || test_absval(RandomMat(128))
           || test_absval(RandomMat(124))
           || test_absval(RandomMat(127));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_absval_0()
           || test_absval_1()
           || test_absval_2();
}
