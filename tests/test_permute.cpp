// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_permute(const ncnn::Mat& a, int order_type, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, order_type);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Permute", pd, weights, a, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_permute failed a.dims=%d a=(%d %d %d %d) order_type=%d\n", a.dims, a.w, a.h, a.d, a.c, order_type);
    }

    return ret;
}

static int test_permute_0()
{
    ncnn::Mat a = RandomMat(128);
    ncnn::Mat b = RandomMat(127);

    return 0
           || test_permute(a, 0)
           || test_permute(b, 0);
}

static int test_permute_1()
{
    ncnn::Mat a = RandomMat(12, 32);
    ncnn::Mat b = RandomMat(8, 15);
    ncnn::Mat c = RandomMat(11, 16);
    ncnn::Mat d = RandomMat(7, 9);

    for (int order_type = 0; order_type < 2; order_type++)
    {
        int ret = 0
                  || test_permute(a, order_type)
                  || test_permute(b, order_type)
                  || test_permute(c, order_type)
                  || test_permute(d, order_type);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_permute_nd(const ncnn::Mat* a, int order_type, int packed_flag = 0)
{
    // the flag applies to the second packed shape
    return 0
           || test_permute(a[0], order_type)
           || test_permute(a[1], order_type, packed_flag)
           || test_permute(a[2], order_type)
           || test_permute(a[3], order_type)
           || test_permute(a[4], order_type)
           || test_permute(a[5], order_type);
}

static int test_permute_2()
{
    ncnn::Mat a[] = {
        RandomMat(8, 16, 32),
        RandomMat(12, 8, 16),
        RandomMat(7, 14, 12),
        RandomMat(4, 4, 13),
        RandomMat(1, 2, 7),
        RandomMat(8, 5, 6)
    };

    int ret = 0
              || test_permute_nd(a, 0, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_permute_nd(a, 1, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_permute_nd(a, 2)
              || test_permute_nd(a, 3)
              || test_permute_nd(a, 4)
              || test_permute_nd(a, 5);
    if (ret != 0)
        return -1;

    return 0;
}

static int test_permute_3()
{
    ncnn::Mat a[] = {
        RandomMat(8, 12, 16, 32),
        RandomMat(12, 4, 8, 16),
        RandomMat(7, 8, 14, 12),
        RandomMat(4, 4, 4, 13),
        RandomMat(1, 2, 3, 7),
        RandomMat(8, 6, 5, 6)
    };

    int ret = 0
              || test_permute_nd(a, 0, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_permute_nd(a, 1, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_permute_nd(a, 2, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_permute_nd(a, 3, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_permute_nd(a, 4, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_permute_nd(a, 5, TEST_LAYER_DISABLE_GPU_TESTING)
              || test_permute_nd(a, 6)
              || test_permute_nd(a, 7)
              || test_permute_nd(a, 8)
              || test_permute_nd(a, 9)
              || test_permute_nd(a, 10)
              || test_permute_nd(a, 11)
              || test_permute_nd(a, 12)
              || test_permute_nd(a, 13)
              || test_permute_nd(a, 14)
              || test_permute_nd(a, 15)
              || test_permute_nd(a, 16)
              || test_permute_nd(a, 17)
              || test_permute_nd(a, 18)
              || test_permute_nd(a, 19)
              || test_permute_nd(a, 20)
              || test_permute_nd(a, 21)
              || test_permute_nd(a, 22)
              || test_permute_nd(a, 23);
    if (ret != 0)
        return -1;

    return 0;
}

#if NCNN_VALIDATION
static int test_permute_load_param()
{
    ncnn::ParamDict base;
    if (test_layer_param(ncnn::LayerType::Permute, base, 0) != 0)
        return -1;

    for (int i = 0; i <= 23; i++)
    {
        if (test_layer_param(ncnn::LayerType::Permute, base, 0, i, 0) != 0)
            return -1;
    }

    const int invalid[] = {-1, 24, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        if (test_layer_param(ncnn::LayerType::Permute, base, 0, invalid[i], -1) != 0)
            return -1;
    }

    return 0;
}
#endif // NCNN_VALIDATION

int main()
{
    SRAND(7767517);

    return 0
           || test_permute_0()
           || test_permute_1()
           || test_permute_2()
           || test_permute_3()
#if NCNN_VALIDATION
           || test_permute_load_param()
#endif // NCNN_VALIDATION
           ;
}
