// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_deepcopy(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("DeepCopy", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_deepcopy failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
    }

    return ret;
}

static int test_deepcopy_0()
{
    return 0
           || test_deepcopy(RandomMat(5, 7, 24))
           || test_deepcopy(RandomMat(7, 9, 12))
           || test_deepcopy(RandomMat(3, 5, 13));
}

static int test_deepcopy_1()
{
    return 0
           || test_deepcopy(RandomMat(15, 24))
           || test_deepcopy(RandomMat(17, 12))
           || test_deepcopy(RandomMat(19, 15));
}

static int test_deepcopy_2()
{
    return 0
           || test_deepcopy(RandomMat(128))
           || test_deepcopy(RandomMat(124))
           || test_deepcopy(RandomMat(127));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_deepcopy_0()
           || test_deepcopy_1()
           || test_deepcopy_2();
}
