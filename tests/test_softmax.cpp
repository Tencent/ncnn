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

// Test with use_approximate_exp enabled, using a relaxed epsilon to account for
// the approximation error of fast_exp (< 0.02% per operation; accumulated over
// a softmax reduction the absolute output error stays well within 1e-2).
static int test_softmax_approx_exp(const ncnn::Mat& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);
    pd.set(1, 1);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_approximate_exp = true;

    int ret = test_layer_opt("Softmax", pd, weights, opt, a, 0.01f);
    if (ret != 0)
    {
        fprintf(stderr, "test_softmax_approx_exp failed a.dims=%d a=(%d %d %d %d) axis=%d\n", a.dims, a.w, a.h, a.d, a.c, axis);
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

static int test_softmax_approx_exp_nd(const ncnn::Mat& m)
{
    const int dims = m.dims;
    for (int i = -dims; i < dims; i++)
    {
        int ret = test_softmax_approx_exp(m, i);
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

// Tests for use_approximate_exp path: cover all dims/axis combinations with a
// relaxed epsilon since fast_exp introduces a small but non-zero approximation
// error compared to the reference expf().
static int test_softmax_approx_exp_0()
{
    ncnn::Mat a = RandomMat(23, 25, 27, 32);
    ncnn::Mat b = RandomMat(21, 22, 19, 40);
    ncnn::Mat c = RandomMat(24, 27, 29, 28);
    ncnn::Mat d = RandomMat(25, 23, 25, 31);

    return 0
           || test_softmax_approx_exp_nd(a)
           || test_softmax_approx_exp_nd(b)
           || test_softmax_approx_exp_nd(c)
           || test_softmax_approx_exp_nd(d);
}

static int test_softmax_approx_exp_1()
{
    ncnn::Mat a = RandomMat(25, 27, 32);
    ncnn::Mat b = RandomMat(22, 19, 40);
    ncnn::Mat c = RandomMat(27, 29, 28);
    ncnn::Mat d = RandomMat(23, 25, 31);

    return 0
           || test_softmax_approx_exp_nd(a)
           || test_softmax_approx_exp_nd(b)
           || test_softmax_approx_exp_nd(c)
           || test_softmax_approx_exp_nd(d);
}

static int test_softmax_approx_exp_2()
{
    ncnn::Mat a = RandomMat(125, 32);
    ncnn::Mat b = RandomMat(147, 40);
    ncnn::Mat c = RandomMat(127, 28);
    ncnn::Mat d = RandomMat(129, 31);

    return 0
           || test_softmax_approx_exp_nd(a)
           || test_softmax_approx_exp_nd(b)
           || test_softmax_approx_exp_nd(c)
           || test_softmax_approx_exp_nd(d);
}

static int test_softmax_approx_exp_3()
{
    ncnn::Mat a = RandomMat(128);
    ncnn::Mat b = RandomMat(120);
    ncnn::Mat c = RandomMat(124);
    ncnn::Mat d = RandomMat(127);

    return 0
           || test_softmax_approx_exp_nd(a)
           || test_softmax_approx_exp_nd(b)
           || test_softmax_approx_exp_nd(c)
           || test_softmax_approx_exp_nd(d);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_softmax_0()
           || test_softmax_1()
           || test_softmax_2()
           || test_softmax_3()
           || test_softmax_approx_exp_0()
           || test_softmax_approx_exp_1()
           || test_softmax_approx_exp_2()
           || test_softmax_approx_exp_3();
}
