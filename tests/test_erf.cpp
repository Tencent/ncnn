// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_erf(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Erf", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_erf failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
    }

    return ret;
}

static int test_erf_0()
{
    return 0
           || test_erf(RandomMat(10, 12, 5))
           || test_erf(RandomMat(3, 6, 18))
           || test_erf(RandomMat(12, 4, 7));
}

static int test_erf_1()
{
    return 0
           || test_erf(RandomMat(12, 28))
           || test_erf(RandomMat(20, 15))
           || test_erf(RandomMat(19, 13));
}

static int test_erf_2()
{
    return 0
           || test_erf(RandomMat(64))
           || test_erf(RandomMat(156))
           || test_erf(RandomMat(243));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_erf_0()
           || test_erf_1()
           || test_erf_2();
}
