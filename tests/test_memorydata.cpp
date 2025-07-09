// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_memorydata(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;
    pd.set(0, a.w);
    pd.set(1, a.h);
    pd.set(2, a.c);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = a;

    std::vector<ncnn::Mat> as(0);

    int ret = test_layer("MemoryData", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_memorydata failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
    }

    return ret;
}

static int test_memorydata_0()
{
    return 0
           || test_memorydata(RandomMat(5, 7, 16))
           || test_memorydata(RandomMat(3, 5, 13));
}

static int test_memorydata_1()
{
    return 0
           || test_memorydata(RandomMat(6, 16))
           || test_memorydata(RandomMat(7, 15));
}

static int test_memorydata_2()
{
    return 0
           || test_memorydata(RandomMat(128))
           || test_memorydata(RandomMat(127));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_memorydata_0()
           || test_memorydata_1()
           || test_memorydata_2();
}
