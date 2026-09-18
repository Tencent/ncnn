// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_flatten_oom(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer_oom("Flatten", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_flatten_oom failed a.dims=%d a=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
    }

    return ret;
}

static int test_flatten_0()
{
    return 0
           || test_flatten_oom(RandomMat(3, 5, 7, 8))
           || test_flatten_oom(RandomMat(9, 10, 2, 16))
           || test_flatten_oom(RandomMat(6, 6, 6, 15));
}

static int test_flatten_1()
{
    return 0
           || test_flatten_oom(RandomMat(3, 5, 8))
           || test_flatten_oom(RandomMat(9, 10, 16))
           || test_flatten_oom(RandomMat(6, 6, 15));
}

static int test_flatten_2()
{
    return 0
           || test_flatten_oom(RandomMat(13, 13))
           || test_flatten_oom(RandomMat(16, 16))
           || test_flatten_oom(RandomMat(8, 12));
}

static int test_flatten_3()
{
    return 0
           || test_flatten_oom(RandomMat(32))
           || test_flatten_oom(RandomMat(17));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_flatten_0()
           || test_flatten_1()
           || test_flatten_2()
           || test_flatten_3();
}
