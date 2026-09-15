// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_pooling1d(int w, int h, int pooling_type, int kernel, int stride, int pad, int global_pooling, int pad_mode, int avgpool_count_include_pad, int adaptive_pooling, int out_w)
{
    ncnn::Mat a = RandomMat(w, h);

    ncnn::ParamDict pd;
    pd.set(0, pooling_type);              // pooling_type
    pd.set(1, kernel);                    // kernel_w
    pd.set(2, stride);                    // stride_w
    pd.set(3, pad);                       // pad_w
    pd.set(4, global_pooling);            // global_pooling
    pd.set(5, pad_mode);                  // pad_mode
    pd.set(6, avgpool_count_include_pad); // avgpool_count_include_pad
    pd.set(7, adaptive_pooling);          // adaptive_pooling
    pd.set(8, out_w);                     // out_w

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Pooling1D", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_pooling1d failed w=%d h=%d pooling_type=%d kernel=%d stride=%d pad=%d global_pooling=%d pad_mode=%d avgpool_count_include_pad=%d adaptive_pooling=%d out_w=%d\n", w, h, pooling_type, kernel, stride, pad, global_pooling, pad_mode, avgpool_count_include_pad, adaptive_pooling, out_w);
    }

    return ret;
}

static int test_pooling1d_0()
{
    static const int ksp[11][3] = {
        {2, 1, 0},
        {2, 2, 0},
        {3, 1, 0},
        {3, 2, 1},
        {4, 1, 0},
        {4, 2, 1},
        {5, 1, 0},
        {5, 2, 2},
        {7, 1, 0},
        {7, 2, 1},
        {7, 3, 2},
    };

    for (int i = 0; i < 11; i++)
    {
        int ret = 0
                  || test_pooling1d(9, 1, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling1d(9, 2, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling1d(9, 3, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 2, 0, 0, 0)
                  || test_pooling1d(9, 4, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 3, 0, 0, 0)
                  || test_pooling1d(9, 7, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling1d(9, 8, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling1d(9, 15, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 2, 0, 0, 0)
                  || test_pooling1d(9, 16, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 3, 0, 0, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_pooling1d_1()
{
    static const int ksp[11][3] = {
        {2, 1, 0},
        {2, 2, 0},
        {3, 1, 0},
        {3, 2, 1},
        {4, 1, 0},
        {4, 2, 1},
        {5, 1, 0},
        {5, 2, 2},
        {7, 1, 0},
        {7, 2, 1},
        {7, 3, 2},
    };

    for (int i = 0; i < 11; i++)
    {
        int ret = 0
                  || test_pooling1d(9, 1, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling1d(9, 2, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling1d(9, 3, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 1, 0, 0)
                  || test_pooling1d(9, 4, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling1d(9, 7, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling1d(9, 8, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 1, 0, 0)
                  || test_pooling1d(9, 12, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 2, 1, 0, 0)
                  || test_pooling1d(9, 15, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling1d(9, 16, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling1d(9, 64, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 3, 1, 0, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_pooling1d_2()
{
    return 0
           || test_pooling1d(2, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(6, 3, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(7, 8, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling1d(48, 4, 0, 2, 2, 0, 0, 0, 0, 0, 0)
           || test_pooling1d(48, 15, 0, 2, 2, 1, 0, 0, 0, 0, 0);
}

// adaptive avg pool
static int test_pooling1d_3()
{
    return 0
           || test_pooling1d(2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(3, 3, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(3, 3, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(3, 3, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(3, 3, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(3, 3, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(3, 3, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 7)
           || test_pooling1d(4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 8)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(6, 4, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 7)
           || test_pooling1d(8, 8, 1, 1, 1, 0, 0, 0, 0, 1, 8)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 1)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 3)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 5)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 7)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 9)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 11)
           || test_pooling1d(11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 13)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 2)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 4)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 6)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 8)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 10)
           || test_pooling1d(13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 12);
}

// adaptive max pool
static int test_pooling1d_4()
{
    return 0
           || test_pooling1d(2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 7)
           || test_pooling1d(4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 8)
           || test_pooling1d(6, 4, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(6, 4, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(6, 4, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(6, 4, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(6, 4, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(6, 4, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 7)
           || test_pooling1d(8, 8, 0, 1, 1, 0, 0, 0, 0, 1, 8)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 1)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 3)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 5)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 7)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 9)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 11)
           || test_pooling1d(11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 13)
           || test_pooling1d(13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 2)
           || test_pooling1d(13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 4)
           || test_pooling1d(13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 6)
           || test_pooling1d(13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 8)
           || test_pooling1d(13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 10)
           || test_pooling1d(13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 12);
}

static int test_pooling1d_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::Pooling1D);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if (ret != (valid ? 0 : -1))
    {
        const int kernel_w = pd.get(1, 0);
        const int stride_w = pd.get(2, 1);
        const int global_pooling = pd.get(4, 0);
        const int adaptive_pooling = pd.get(7, 0);

        fprintf(stderr, "test_pooling1d_load_param failed ret=%d expected=%d kernel_w=%d stride_w=%d global_pooling=%d adaptive_pooling=%d\n", ret, valid ? 0 : -1, kernel_w, stride_w, global_pooling, adaptive_pooling);
        fprintf(stderr, "pooling_type=%d pad_mode=%d\n", pd.get(0, 0), pd.get(5, 0));
        fprintf(stderr, "out_w=%d\n", pd.get(8, 0));
        return -1;
    }

    return 0;
}

static int test_pooling1d_load_param()
{
    ncnn::ParamDict pd;
    pd.set(1, 3);
    if (test_pooling1d_load_param_case(pd, true) != 0)
        return -1;

    pd.set(2, 0);
    if (test_pooling1d_load_param_case(pd, false) != 0)
        return -1;

    pd.set(4, 1); // global pooling does not use the local stride
    if (test_pooling1d_load_param_case(pd, true) != 0)
        return -1;

    return 0;
}

static int test_pooling1d_load_param_type()
{
    ncnn::ParamDict base;
    base.set(1, 3);
    if (test_pooling1d_load_param_case(base, true) != 0)
        return -1;

    for (int i = 0; i <= 1; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(0, i);
        if (test_pooling1d_load_param_case(pd, true) != 0)
            return -1;
    }

    const int invalid[] = {-1, 2, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(0, invalid[i]);
        if (test_pooling1d_load_param_case(pd, false) != 0)
            return -1;
    }

    for (int i = 0; i <= 3; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(5, i);
        if (test_pooling1d_load_param_case(pd, true) != 0)
            return -1;
    }

    const int invalid_pad[] = {-1, 4, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(5, invalid_pad[i]);
        if (test_pooling1d_load_param_case(pd, false) != 0)
            return -1;

        pd.set(4, 1); // global pooling does not use pad_mode
        if (test_pooling1d_load_param_case(pd, true) != 0)
            return -1;

        pd.set(4, 0);
        pd.set(8, 1);
        pd.set(7, 1); // adaptive pooling does not use pad_mode
        if (test_pooling1d_load_param_case(pd, true) != 0)
            return -1;
    }

    return 0;
}

static int test_pooling1d_load_param_adaptive()
{
    ncnn::ParamDict base;
    base.set(1, 3);
    base.set(7, 1);
    if (test_pooling1d_load_param_case(base, false) != 0)
        return -1;

    base.set(8, 2);
    if (test_pooling1d_load_param_case(base, true) != 0)
        return -1;

    {
        ncnn::ParamDict pd = base;
        pd.set(8, 1);
        if (test_pooling1d_load_param_case(pd, true) != 0)
            return -1;
    }

    const int invalid[] = {0, -1, -233, INT_MIN};
    for (int i = 0; i < 4; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(8, invalid[i]);
        if (test_pooling1d_load_param_case(pd, false) != 0)
            return -1;

        pd.set(4, 1); // global pooling ignores the adaptive output size
        if (test_pooling1d_load_param_case(pd, true) != 0)
            return -1;

        pd.set(4, 0);
        pd.set(7, 0); // ordinary pooling ignores the adaptive output size
        if (test_pooling1d_load_param_case(pd, true) != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_pooling1d_0()
           || test_pooling1d_1()
           || test_pooling1d_2()
           || test_pooling1d_3()
           || test_pooling1d_4()
           || test_pooling1d_load_param()
           || test_pooling1d_load_param_type()
           || test_pooling1d_load_param_adaptive();
}
