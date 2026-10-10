// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_split(const ncnn::Mat& a, int top_blob_count)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(1);
    as[0] = a;

    int ret = test_layer("Split", pd, weights, as, top_blob_count);
    if (ret != 0)
    {
        fprintf(stderr, "test_split failed a.dims=%d a=(%d %d %d %d) top_blob_count=%d\n", a.dims, a.w, a.h, a.d, a.c, top_blob_count);
    }

    return ret;
}

static int test_split_4d()
{
    return 0
           || test_split(RandomMat(5, 7, 3, 16), 2)
           || test_split(RandomMat(3, 5, 4, 13), 3);
}

static int test_split_0()
{
    return 0
           || test_split(RandomMat(5, 7, 16), 2)
           || test_split(RandomMat(3, 5, 13), 3);
}

static int test_split_1()
{
    return 0
           || test_split(RandomMat(6, 16), 2)
           || test_split(RandomMat(7, 15), 3);
}

static int test_split_2()
{
    return 0
           || test_split(RandomMat(128), 2)
           || test_split(RandomMat(127), 3);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_split_4d()
           || test_split_0()
           || test_split_1()
           || test_split_2();
}
