// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_softmax(const ncnn::Mat& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); // axis
    pd.set(1, 1);    // fixbug0

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Softmax", pd, weights, a, 0.001);
    if (ret != 0)
    {
        fprintf(stderr, "test_softmax failed a.dims=%d a=(%d %d %d %d) axis=%d\n", a.dims, a.w, a.h, a.d, a.c, axis);
    }

    return ret;
}

static int test_softmax_axes(const ncnn::Mat& a)
{
    for (int axis = 0; axis < a.dims; axis++)
    {
        if (test_softmax(a, axis) != 0)
            return -1;
    }

    return 0;
}

static int test_softmax_negative_axes(const ncnn::Mat& a)
{
    for (int axis = -a.dims; axis < 0; axis++)
    {
        if (test_softmax(a, axis) != 0)
            return -1;
    }

    return 0;
}

static int test_softmax_0()
{
    // channel counts exercise pack16, pack8, pack4 and scalar tails
    return 0
           || test_softmax_negative_axes(RandomMat(9, 5, 7, 32))
           || test_softmax_axes(RandomMat(10, 7, 6, 40))
           || test_softmax_axes(RandomMat(8, 7, 5, 28))
           || test_softmax_axes(RandomMat(9, 7, 5, 31));
}

static int test_softmax_1()
{
    return 0
           || test_softmax_negative_axes(RandomMat(25, 27, 32))
           || test_softmax_axes(RandomMat(22, 19, 40))
           || test_softmax_axes(RandomMat(27, 29, 28))
           || test_softmax_axes(RandomMat(23, 25, 31));
}

static int test_softmax_2()
{
    return 0
           || test_softmax_negative_axes(RandomMat(125, 32))
           || test_softmax_axes(RandomMat(147, 40))
           || test_softmax_axes(RandomMat(127, 28))
           || test_softmax_axes(RandomMat(129, 31));
}

static int test_softmax_3()
{
    return 0
           || test_softmax_negative_axes(RandomMat(128))
           || test_softmax_axes(RandomMat(120))
           || test_softmax_axes(RandomMat(124))
           || test_softmax_axes(RandomMat(127));
}

static int test_softmax_large()
{
    // large allocations exercise staging reuse and multi-workgroup reductions
    return 0
           || test_softmax(RandomMat(23, 25, 27, 32), -4)
           || test_softmax(RandomMat(24, 27, 29, 28), 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_softmax_0()
           || test_softmax_1()
           || test_softmax_2()
           || test_softmax_3()
           || test_softmax_large()
           // pack8 rows include a full avx512 block, an eight-element tail and a scalar tail
           || test_softmax(RandomMat(25, 24), 0);
}
