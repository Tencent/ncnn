// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_noop(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(1);
    as[0] = a;

    int ret = test_layer("Noop", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_noop failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
    }

    return ret;
}

static int test_noop_0()
{
    return 0
           || test_noop(RandomMat(5, 7, 24))
           || test_noop(RandomMat(7, 9, 12))
           || test_noop(RandomMat(3, 5, 13));
}

static int test_noop_1()
{
    return 0
           || test_noop(RandomMat(15, 24))
           || test_noop(RandomMat(17, 12))
           || test_noop(RandomMat(19, 15));
}

static int test_noop_2()
{
    return 0
           || test_noop(RandomMat(128))
           || test_noop(RandomMat(124))
           || test_noop(RandomMat(127));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_noop_0()
           || test_noop_1()
           || test_noop_2();
}
