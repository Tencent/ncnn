// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_softmax(const ncnn::Mat& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); // axis
    pd.set(1, 1);    // fixbug0

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Softmax", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_softmax failed a.dims=%d a=(%d %d %d %d) axis=%d\n", a.dims, a.w, a.h, a.d, a.c, axis);
    }

    return ret;
}

static int test_softmax_nd(const ncnn::Mat& m)
{
    const int dims = m.dims;
    for (int i = -dims; i < dims; i++)
    {
        int ret = test_softmax(m, i);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_softmax_0()
{
    ncnn::Mat a = RandomMat(23, 25, 27, 32);
    ncnn::Mat b = RandomMat(21, 22, 19, 40);
    ncnn::Mat c = RandomMat(24, 27, 29, 28);
    ncnn::Mat d = RandomMat(25, 23, 25, 31);

    return 0
           || test_softmax_nd(a)
           || test_softmax_nd(b)
           || test_softmax_nd(c)
           || test_softmax_nd(d);
}

static int test_softmax_1()
{
    ncnn::Mat a = RandomMat(25, 27, 32);
    ncnn::Mat b = RandomMat(22, 19, 40);
    ncnn::Mat c = RandomMat(27, 29, 28);
    ncnn::Mat d = RandomMat(23, 25, 31);

    return 0
           || test_softmax_nd(a)
           || test_softmax_nd(b)
           || test_softmax_nd(c)
           || test_softmax_nd(d);
}

static int test_softmax_2()
{
    ncnn::Mat a = RandomMat(125, 32);
    ncnn::Mat b = RandomMat(147, 40);
    ncnn::Mat c = RandomMat(127, 28);
    ncnn::Mat d = RandomMat(129, 31);

    return 0
           || test_softmax_nd(a)
           || test_softmax_nd(b)
           || test_softmax_nd(c)
           || test_softmax_nd(d);
}

static int test_softmax_3()
{
    ncnn::Mat a = RandomMat(128);
    ncnn::Mat b = RandomMat(120);
    ncnn::Mat c = RandomMat(124);
    ncnn::Mat d = RandomMat(127);

    return 0
           || test_softmax_nd(a)
           || test_softmax_nd(b)
           || test_softmax_nd(c)
           || test_softmax_nd(d);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_softmax_0()
           || test_softmax_1()
           || test_softmax_2()
           || test_softmax_3();
}
