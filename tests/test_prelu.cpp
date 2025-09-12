// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_prelu(const ncnn::Mat& a, int num_slope)
{
    ncnn::ParamDict pd;
    pd.set(0, num_slope);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(num_slope);

    int ret = test_layer("PReLU", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_prelu failed a.dims=%d a=(%d %d %d) num_slope=%d\n", a.dims, a.w, a.h, a.c, num_slope);
    }

    return ret;
}

static int test_prelu_0()
{
    return 0
           || test_prelu(RandomMat(5, 7, 24), 24)
           || test_prelu(RandomMat(5, 7, 24), 1)
           || test_prelu(RandomMat(5, 7, 32), 32)
           || test_prelu(RandomMat(5, 7, 32), 1)
           || test_prelu(RandomMat(7, 9, 12), 12)
           || test_prelu(RandomMat(7, 9, 12), 1)
           || test_prelu(RandomMat(3, 5, 13), 13)
           || test_prelu(RandomMat(3, 5, 13), 1);
}

static int test_prelu_1()
{
    return 0
           || test_prelu(RandomMat(15, 24), 24)
           || test_prelu(RandomMat(15, 24), 1)
           || test_prelu(RandomMat(15, 32), 32)
           || test_prelu(RandomMat(15, 32), 1)
           || test_prelu(RandomMat(17, 12), 12)
           || test_prelu(RandomMat(17, 12), 1)
           || test_prelu(RandomMat(19, 15), 15)
           || test_prelu(RandomMat(19, 15), 1);
}

static int test_prelu_2()
{
    return 0
           || test_prelu(RandomMat(128), 128)
           || test_prelu(RandomMat(128), 1)
           || test_prelu(RandomMat(124), 124)
           || test_prelu(RandomMat(124), 1)
           || test_prelu(RandomMat(120), 120)
           || test_prelu(RandomMat(120), 1)
           || test_prelu(RandomMat(127), 127)
           || test_prelu(RandomMat(127), 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_prelu_0()
           || test_prelu_1()
           || test_prelu_2();
}
