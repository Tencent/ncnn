// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_layer_oom(const ncnn::Mat& a, int outw, int outh, int outd, int outc)
{
    ncnn::ParamDict pd;
    pd.set(0, outw);  // w
    pd.set(1, outh);  // h
    pd.set(11, outd); // d
    pd.set(2, outc);  // c

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer_oom("Reshape", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_layer_oom failed a.dims=%d a=(%d %d %d %d) outw=%d outh=%d outd=%d outc=%d\n", a.dims, a.w, a.h, a.d, a.c, outw, outh, outd, outc);
    }

    return ret;
}

static int test_reshape_0()
{
    ncnn::Mat a = RandomMat(3, 8, 5, 32);

    return 0
           || test_layer_oom(a, 5, 8, 3, 32)
           || test_layer_oom(a, 3, 8, 32, 5)
           || test_layer_oom(a, 32, 5, 8, 3)
           || test_layer_oom(a, 4, 3, 16, -1)
           || test_layer_oom(a, 6, 16, -1, 2)
           || test_layer_oom(a, 4, -1, 8, 8)
           || test_layer_oom(a, -1, 16, 6, 2)
           || test_layer_oom(a, 16, 6, -233, -1)
           || test_layer_oom(a, 8, -1, -233, 8)
           || test_layer_oom(a, -1, 6, -233, 16)
           || test_layer_oom(a, 8, -1, -233, -233)
           || test_layer_oom(a, -1, 6, -233, -233)
           || test_layer_oom(a, -1, -233, -233, -233);
}

static int test_reshape_1()
{
    ncnn::Mat a = RandomMat(6, 7, 32);

    return 0
           || test_layer_oom(a, 6, 8, 4, 7)
           || test_layer_oom(a, 2, 6, 7, 16)
           || test_layer_oom(a, 7, 6, -233, 32)
           || test_layer_oom(a, 6, 32, -233, 7)
           || test_layer_oom(a, 32, 7, -233, 6)
           || test_layer_oom(a, 2, 6, -233, -1)
           || test_layer_oom(a, -1, 8, -233, 2)
           || test_layer_oom(a, -1, 4, -233, -233)
           || test_layer_oom(a, 8, -1, -233, -233)
           || test_layer_oom(a, 32, 42, -233, -233)
           || test_layer_oom(a, -1, -233, -233, -233);
}

static int test_reshape_2()
{
    ncnn::Mat a = RandomMat(14, 32);

    return 0
           || test_layer_oom(a, 2, 7, 2, 16)
           || test_layer_oom(a, 16, 1, 7, 4)
           || test_layer_oom(a, 7, 2, -233, 32)
           || test_layer_oom(a, 4, 16, -233, 7)
           || test_layer_oom(a, 16, 14, -233, 2)
           || test_layer_oom(a, 2, 4, -233, -1)
           || test_layer_oom(a, -1, 8, -233, 2)
           || test_layer_oom(a, 28, 16, -233, -233)
           || test_layer_oom(a, -1, 14, -233, -233)
           || test_layer_oom(a, 16, -1, -233, -233)
           || test_layer_oom(a, -1, -233, -233, -233);
}

static int test_reshape_3()
{
    ncnn::Mat a = RandomMat(240);

    return 0
           || test_layer_oom(a, 1, 1, 1, 240)
           || test_layer_oom(a, 10, 1, 1, 24)
           || test_layer_oom(a, 3, 5, -233, 16)
           || test_layer_oom(a, 3, 8, -233, 10)
           || test_layer_oom(a, 8, 5, -233, 6)
           || test_layer_oom(a, 2, 5, -233, -1)
           || test_layer_oom(a, -1, 5, -233, 4)
           || test_layer_oom(a, 8, 30, -233, -233)
           || test_layer_oom(a, -1, 2, -233, -233)
           || test_layer_oom(a, 24, -1, -233, -233)
           || test_layer_oom(a, -1, -233, -233, -233);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_reshape_0()
           || test_reshape_1()
           || test_reshape_2()
           || test_reshape_3();
}
