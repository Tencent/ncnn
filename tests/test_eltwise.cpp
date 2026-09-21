// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static void print_float_array(const ncnn::Mat& a)
{
    fprintf(stderr, "[");
    for (int i = 0; i < a.w; i++)
    {
        fprintf(stderr, " %f", a[i]);
    }
    fprintf(stderr, " ]");
}

static int test_eltwise(const std::vector<ncnn::Mat>& a, int op_type, const ncnn::Mat& coeffs, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, coeffs);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Eltwise", pd, weights, a, 1, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_eltwise failed a[0].dims=%d a[0]=(%d %d %d %d) op_type=%d", a[0].dims, a[0].w, a[0].h, a[0].d, a[0].c, op_type);
        fprintf(stderr, " coeffs=");
        print_float_array(coeffs);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_eltwise_0()
{
    std::vector<ncnn::Mat> a(2);
    a[0] = RandomMat(16, 12, 12);
    a[1] = RandomMat(16, 12, 12);

    std::vector<ncnn::Mat> b(2);
    b[0] = RandomMat(15, 11, 24);
    b[1] = RandomMat(15, 11, 24);

    std::vector<ncnn::Mat> c(2);
    c[0] = RandomMat(9, 5, 7);
    c[1] = RandomMat(9, 5, 7);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())
           || test_eltwise(b, 0, ncnn::Mat(), TEST_LAYER_DISABLE_GPU_TESTING)
           || test_eltwise(b, 1, ncnn::Mat(), TEST_LAYER_DISABLE_GPU_TESTING)
           || test_eltwise(b, 2, ncnn::Mat(), TEST_LAYER_DISABLE_GPU_TESTING)
           || test_eltwise(c, 0, ncnn::Mat())
           || test_eltwise(c, 1, ncnn::Mat())
           || test_eltwise(c, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(2))
           || test_eltwise(a, 1, RandomMat(2))
           || test_eltwise(a, 2, RandomMat(2))
           || test_eltwise(b, 0, RandomMat(2), TEST_LAYER_DISABLE_GPU_TESTING)
           || test_eltwise(b, 1, RandomMat(2), TEST_LAYER_DISABLE_GPU_TESTING)
           || test_eltwise(b, 2, RandomMat(2), TEST_LAYER_DISABLE_GPU_TESTING)
           || test_eltwise(c, 0, RandomMat(2))
           || test_eltwise(c, 1, RandomMat(2))
           || test_eltwise(c, 2, RandomMat(2));
}

static int test_eltwise_1()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(15, 11, 16);
    a[1] = RandomMat(15, 11, 16);
    a[2] = RandomMat(15, 11, 16);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(3))
           || test_eltwise(a, 1, RandomMat(3))
           || test_eltwise(a, 2, RandomMat(3));
}

static int test_eltwise_2()
{
    std::vector<ncnn::Mat> a(4);
    a[0] = RandomMat(7, 3, 5);
    a[1] = RandomMat(7, 3, 5);
    a[2] = RandomMat(7, 3, 5);
    a[3] = RandomMat(7, 3, 5);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(4))
           || test_eltwise(a, 1, RandomMat(4))
           || test_eltwise(a, 2, RandomMat(4));
}

static int test_eltwise_3()
{
    std::vector<ncnn::Mat> a(5);
    a[0] = RandomMat(12, 4, 16);
    a[1] = RandomMat(12, 4, 16);
    a[2] = RandomMat(12, 4, 16);
    a[3] = RandomMat(12, 4, 16);
    a[4] = RandomMat(12, 4, 16);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(5))
           || test_eltwise(a, 1, RandomMat(5))
           || test_eltwise(a, 2, RandomMat(5));
}

static int test_eltwise_4()
{
    std::vector<ncnn::Mat> a(2);
    a[0] = RandomMat(16, 12);
    a[1] = RandomMat(16, 12);

    std::vector<ncnn::Mat> b(2);
    b[0] = RandomMat(11, 24);
    b[1] = RandomMat(11, 24);

    std::vector<ncnn::Mat> c(2);
    c[0] = RandomMat(9, 7);
    c[1] = RandomMat(9, 7);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())
           || test_eltwise(b, 0, ncnn::Mat())
           || test_eltwise(b, 1, ncnn::Mat())
           || test_eltwise(b, 2, ncnn::Mat())
           || test_eltwise(c, 0, ncnn::Mat())
           || test_eltwise(c, 1, ncnn::Mat())
           || test_eltwise(c, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(2))
           || test_eltwise(a, 1, RandomMat(2))
           || test_eltwise(a, 2, RandomMat(2))
           || test_eltwise(b, 0, RandomMat(2))
           || test_eltwise(b, 1, RandomMat(2))
           || test_eltwise(b, 2, RandomMat(2))
           || test_eltwise(c, 0, RandomMat(2))
           || test_eltwise(c, 1, RandomMat(2))
           || test_eltwise(c, 2, RandomMat(2));
}

static int test_eltwise_5()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(15, 16);
    a[1] = RandomMat(15, 16);
    a[2] = RandomMat(15, 16);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(3))
           || test_eltwise(a, 1, RandomMat(3))
           || test_eltwise(a, 2, RandomMat(3));
}

static int test_eltwise_6()
{
    std::vector<ncnn::Mat> a(4);
    a[0] = RandomMat(7, 5);
    a[1] = RandomMat(7, 5);
    a[2] = RandomMat(7, 5);
    a[3] = RandomMat(7, 5);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(4))
           || test_eltwise(a, 1, RandomMat(4))
           || test_eltwise(a, 2, RandomMat(4));
}

static int test_eltwise_7()
{
    std::vector<ncnn::Mat> a(5);
    a[0] = RandomMat(12, 16);
    a[1] = RandomMat(12, 16);
    a[2] = RandomMat(12, 16);
    a[3] = RandomMat(12, 16);
    a[4] = RandomMat(12, 16);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(5))
           || test_eltwise(a, 1, RandomMat(5))
           || test_eltwise(a, 2, RandomMat(5));
}

static int test_eltwise_8()
{
    std::vector<ncnn::Mat> a(2);
    a[0] = RandomMat(12);
    a[1] = RandomMat(12);

    std::vector<ncnn::Mat> b(2);
    b[0] = RandomMat(24);
    b[1] = RandomMat(24);

    std::vector<ncnn::Mat> c(2);
    c[0] = RandomMat(19);
    c[1] = RandomMat(19);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())
           || test_eltwise(b, 0, ncnn::Mat())
           || test_eltwise(b, 1, ncnn::Mat())
           || test_eltwise(b, 2, ncnn::Mat())
           || test_eltwise(c, 0, ncnn::Mat())
           || test_eltwise(c, 1, ncnn::Mat())
           || test_eltwise(c, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(2))
           || test_eltwise(a, 1, RandomMat(2))
           || test_eltwise(a, 2, RandomMat(2))
           || test_eltwise(b, 0, RandomMat(2))
           || test_eltwise(b, 1, RandomMat(2))
           || test_eltwise(b, 2, RandomMat(2))
           || test_eltwise(c, 0, RandomMat(2))
           || test_eltwise(c, 1, RandomMat(2))
           || test_eltwise(c, 2, RandomMat(2));
}

static int test_eltwise_9()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(36);
    a[1] = RandomMat(36);
    a[2] = RandomMat(36);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(3))
           || test_eltwise(a, 1, RandomMat(3))
           || test_eltwise(a, 2, RandomMat(3));
}

static int test_eltwise_10()
{
    std::vector<ncnn::Mat> a(4);
    a[0] = RandomMat(15);
    a[1] = RandomMat(15);
    a[2] = RandomMat(15);
    a[3] = RandomMat(15);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(4))
           || test_eltwise(a, 1, RandomMat(4))
           || test_eltwise(a, 2, RandomMat(4));
}

static int test_eltwise_11()
{
    std::vector<ncnn::Mat> a(5);
    a[0] = RandomMat(16);
    a[1] = RandomMat(16);
    a[2] = RandomMat(16);
    a[3] = RandomMat(16);
    a[4] = RandomMat(16);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(5))
           || test_eltwise(a, 1, RandomMat(5))
           || test_eltwise(a, 2, RandomMat(5));
}

static int test_eltwise_12()
{
    std::vector<ncnn::Mat> a(2);
    a[0] = RandomMat(31, 5, 3, 12);
    a[1] = RandomMat(31, 5, 3, 12);

    std::vector<ncnn::Mat> b(3);
    b[0] = RandomMat(32, 4, 5, 32);
    b[1] = RandomMat(32, 4, 5, 32);
    b[2] = RandomMat(32, 4, 5, 32);

    std::vector<ncnn::Mat> c(4);
    c[0] = RandomMat(33, 6, 7, 7);
    c[1] = RandomMat(33, 6, 7, 7);
    c[2] = RandomMat(33, 6, 7, 7);
    c[3] = RandomMat(33, 6, 7, 7);

    return 0
           || test_eltwise(a, 0, ncnn::Mat())
           || test_eltwise(a, 1, ncnn::Mat())
           || test_eltwise(a, 2, ncnn::Mat())
           || test_eltwise(b, 0, ncnn::Mat())
           || test_eltwise(b, 1, ncnn::Mat())
           || test_eltwise(b, 2, ncnn::Mat())
           || test_eltwise(c, 0, ncnn::Mat())
           || test_eltwise(c, 1, ncnn::Mat())
           || test_eltwise(c, 2, ncnn::Mat())

           || test_eltwise(a, 0, RandomMat(2))
           || test_eltwise(a, 1, RandomMat(2))
           || test_eltwise(a, 2, RandomMat(2))
           || test_eltwise(b, 0, RandomMat(3))
           || test_eltwise(b, 1, RandomMat(3))
           || test_eltwise(b, 2, RandomMat(3))
           || test_eltwise(c, 0, RandomMat(4))
           || test_eltwise(c, 1, RandomMat(4))
           || test_eltwise(c, 2, RandomMat(4));
}

#if NCNN_VALIDATION
static int test_eltwise_load_param()
{
    ncnn::ParamDict pd;
    pd.set(1, ncnn::Mat(2));
    if (test_layer_param(ncnn::LayerType::Eltwise, pd, 0) != 0)
        return -1;

    const ncnn::ParamDict base = pd;

    pd = base;
    pd.set(1, ncnn::Mat(0));
    if (test_layer_param(ncnn::LayerType::Eltwise, pd, 0) != 0)
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    pd = base;
    pd.set(1, missing_data);
    if (test_layer_param(ncnn::LayerType::Eltwise, pd, -1) != 0)
        return -1;

    pd = base;
    pd.set(1, ncnn::Mat(2, (size_t)1u));
    if (test_layer_param(ncnn::LayerType::Eltwise, pd, -1) != 0)
        return -1;

    pd.set(1, ncnn::Mat(2, 2));
    return test_layer_param(ncnn::LayerType::Eltwise, pd, -1);
}

static int test_eltwise_load_param_type()
{
    ncnn::ParamDict base;
    if (test_layer_param(ncnn::LayerType::Eltwise, base, 0) != 0)
        return -1;

    for (int i = 0; i <= 2; i++)
    {
        if (test_layer_param(ncnn::LayerType::Eltwise, base, 0, i, 0) != 0)
            return -1;
    }

    const int invalid[] = {-1, 3, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        if (test_layer_param(ncnn::LayerType::Eltwise, base, 0, invalid[i], -1) != 0)
            return -1;
    }

    return 0;
}
#endif // NCNN_VALIDATION

int main()
{
    SRAND(7767517);

    return 0
           || test_eltwise_0()
           || test_eltwise_1()
           || test_eltwise_2()
           || test_eltwise_3()
           || test_eltwise_4()
           || test_eltwise_5()
           || test_eltwise_6()
           || test_eltwise_7()
           || test_eltwise_8()
           || test_eltwise_9()
           || test_eltwise_10()
           || test_eltwise_11()
           || test_eltwise_12()
#if NCNN_VALIDATION
           || test_eltwise_load_param()
           || test_eltwise_load_param_type()
#endif // NCNN_VALIDATION
           ;
}
