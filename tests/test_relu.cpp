// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_relu(const ncnn::Mat& a, float slope)
{
    ncnn::ParamDict pd;
    pd.set(0, slope); //slope

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("ReLU", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_relu failed a.dims=%d a=(%d %d %d %d) slope=%f\n", a.dims, a.w, a.h, a.d, a.c, slope);
    }

    return ret;
}

static int test_relu_0()
{
    return 0
           || test_relu(RandomMat(5, 6, 7, 24), 0.f)
           || test_relu(RandomMat(5, 6, 7, 24), 0.1f)
           || test_relu(RandomMat(7, 8, 9, 12), 0.f)
           || test_relu(RandomMat(7, 8, 9, 12), 0.1f)
           || test_relu(RandomMat(3, 4, 5, 13), 0.f)
           || test_relu(RandomMat(3, 4, 5, 13), 0.1f);
}

static int test_relu_1()
{
    return 0
           || test_relu(RandomMat(5, 7, 24), 0.f)
           || test_relu(RandomMat(5, 7, 24), 0.1f)
           || test_relu(RandomMat(7, 9, 12), 0.f)
           || test_relu(RandomMat(7, 9, 12), 0.1f)
           || test_relu(RandomMat(3, 5, 13), 0.f)
           || test_relu(RandomMat(3, 5, 13), 0.1f);
}

static int test_relu_2()
{
    return 0
           || test_relu(RandomMat(15, 24), 0.f)
           || test_relu(RandomMat(15, 24), 0.1f)
           || test_relu(RandomMat(17, 12), 0.f)
           || test_relu(RandomMat(17, 12), 0.1f)
           || test_relu(RandomMat(19, 15), 0.f)
           || test_relu(RandomMat(19, 15), 0.1f);
}

static int test_relu_3()
{
    return 0
           || test_relu(RandomMat(128), 0.f)
           || test_relu(RandomMat(128), 0.1f)
           || test_relu(RandomMat(124), 0.f)
           || test_relu(RandomMat(124), 0.1f)
           || test_relu(RandomMat(127), 0.f)
           || test_relu(RandomMat(127), 0.1f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_relu_0()
           || test_relu_1()
           || test_relu_2()
           || test_relu_3();
}
