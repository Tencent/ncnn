// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_shufflechannel(const ncnn::Mat& a, int group, int reverse)
{
    ncnn::ParamDict pd;
    pd.set(0, group);
    pd.set(1, reverse);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("ShuffleChannel", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_shufflechannel failed a.dims=%d a=(%d %d %d %d) group=%d reverse=%d\n", a.dims, a.w, a.h, a.d, a.c, group, reverse);
    }

    return ret;
}

static int test_shufflechannel(int w, int h, int c, int group, int reverse)
{
    return test_shufflechannel(RandomMat(w, h, c), group, reverse);
}

static int test_shufflechannel(int w, int h, int d, int c, int group, int reverse)
{
    return test_shufflechannel(RandomMat(w, h, d, c), group, reverse);
}

static int test_shufflechannel_0()
{
    return 0
           || test_shufflechannel(3, 7, 1, 1, 0)
           || test_shufflechannel(5, 7, 2, 2, 0)
           || test_shufflechannel(3, 9, 3, 3, 0)
           || test_shufflechannel(5, 7, 4, 2, 0)
           || test_shufflechannel(3, 7, 12, 3, 0)
           || test_shufflechannel(5, 9, 12, 4, 0)
           || test_shufflechannel(3, 7, 12, 6, 0)
           || test_shufflechannel(5, 7, 15, 3, 0)
           || test_shufflechannel(3, 9, 15, 5, 0)
           || test_shufflechannel(5, 7, 16, 2, 0)
           || test_shufflechannel(5, 9, 16, 4, 0)
           || test_shufflechannel(3, 7, 16, 8, 0)
           || test_shufflechannel(1, 1, 20, 2, 0)
           || test_shufflechannel(5, 7, 20, 2, 0)
           || test_shufflechannel(5, 7, 24, 2, 0)
           || test_shufflechannel(3, 7, 24, 3, 0)
           || test_shufflechannel(5, 9, 24, 4, 0)
           || test_shufflechannel(3, 7, 32, 2, 0)
           || test_shufflechannel(3, 7, 32, 8, 0)
           || test_shufflechannel(5, 7, 48, 2, 0)
           || test_shufflechannel(5, 7, 48, 3, 0)
           || test_shufflechannel(5, 9, 64, 4, 0);
}

static int test_shufflechannel_1()
{
    return 0
           || test_shufflechannel(3, 7, 1, 1, 1)
           || test_shufflechannel(5, 7, 2, 2, 1)
           || test_shufflechannel(3, 9, 3, 3, 1)
           || test_shufflechannel(5, 7, 4, 2, 1)
           || test_shufflechannel(3, 7, 12, 3, 1)
           || test_shufflechannel(5, 9, 12, 4, 1)
           || test_shufflechannel(3, 7, 12, 6, 1)
           || test_shufflechannel(5, 7, 15, 3, 1)
           || test_shufflechannel(3, 9, 15, 5, 1)
           || test_shufflechannel(5, 7, 16, 2, 1)
           || test_shufflechannel(5, 9, 16, 4, 1)
           || test_shufflechannel(3, 7, 16, 8, 1)
           || test_shufflechannel(1, 1, 20, 10, 1)
           || test_shufflechannel(5, 7, 20, 2, 1)
           || test_shufflechannel(5, 7, 24, 2, 1)
           || test_shufflechannel(3, 7, 24, 3, 1)
           || test_shufflechannel(5, 9, 24, 4, 1)
           || test_shufflechannel(3, 7, 32, 2, 1)
           || test_shufflechannel(3, 7, 32, 8, 1)
           || test_shufflechannel(5, 7, 48, 2, 1)
           || test_shufflechannel(5, 7, 48, 3, 1)
           || test_shufflechannel(3, 7, 64, 4, 1);
}

static int test_shufflechannel_2()
{
    return 0
           || test_shufflechannel(5, 3, 2, 24, 2, 0)
           || test_shufflechannel(3, 5, 3, 15, 3, 0)
           || test_shufflechannel(5, 3, 2, 16, 8, 0)
           || test_shufflechannel(5, 3, 2, 24, 2, 1)
           || test_shufflechannel(3, 5, 3, 15, 5, 1)
           || test_shufflechannel(5, 3, 2, 16, 8, 1);
}

static int test_shufflechannel_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    for (int backend = 0; backend < 2; backend++)
    {
        ncnn::Layer* layer = backend == 0 ? ncnn::create_layer_naive(ncnn::LayerType::ShuffleChannel) : ncnn::create_layer_cpu(ncnn::LayerType::ShuffleChannel);
        if (!layer)
            return -1;

        int ret = layer->load_param(pd);
        delete layer;

        if ((ret == 0) != valid)
        {
            fprintf(stderr, "ShuffleChannel load_param backend=%d returned %d, expected %s\n", backend, ret, valid ? "success" : "failure");
            return -1;
        }
    }

    return 0;
}

static int test_shufflechannel_load_param()
{
    ncnn::ParamDict pd;
    pd.set(0, 2);
    if (test_shufflechannel_load_param_case(pd, true) != 0)
        return -1;

    const int invalid[] = {0, -1, -8, INT_MIN};
    for (int i = 0; i < 4; i++)
    {
        pd.set(0, invalid[i]);
        if (test_shufflechannel_load_param_case(pd, false) != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_shufflechannel_2()
           || test_shufflechannel_0()
           || test_shufflechannel_1()
           || test_shufflechannel_load_param();
}
