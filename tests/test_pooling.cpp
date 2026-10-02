// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_pooling(int w, int h, int c, int pooling_type, int kernel, int stride, int pad, int global_pooling, int pad_mode, int avgpool_count_include_pad, int adaptive_pooling, int out_w, int flag = 0)
{
    ncnn::Mat a = RandomMat(w, h, c);

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

    int ret = test_layer("Pooling", pd, weights, a, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_pooling failed w=%d h=%d c=%d pooling_type=%d kernel=%d stride=%d pad=%d global_pooling=%d pad_mode=%d avgpool_count_include_pad=%d adaptive_pooling=%d out_w=%d\n", w, h, c, pooling_type, kernel, stride, pad, global_pooling, pad_mode, avgpool_count_include_pad, adaptive_pooling, out_w);
    }

    return ret;
}

static int test_pooling_0()
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
                  || test_pooling(9, 7, 1, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling(9, 7, 2, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling(9, 7, 3, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 2, 0, 0, 0)
                  || test_pooling(9, 7, 4, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 3, 0, 0, 0)
                  || test_pooling(9, 7, 7, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling(9, 7, 8, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling(9, 7, 15, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 2, 0, 0, 0)
                  || test_pooling(9, 7, 16, 0, ksp[i][0], ksp[i][1], ksp[i][2], 0, 3, 0, 0, 0, TEST_LAYER_DISABLE_GPU_TESTING);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_pooling_1()
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
                  || test_pooling(9, 7, 1, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling(9, 7, 2, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling(9, 7, 3, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 1, 0, 0)
                  || test_pooling(9, 7, 4, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0)
                  || test_pooling(9, 7, 7, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling(9, 7, 8, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 1, 0, 0)
                  || test_pooling(9, 7, 12, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 2, 1, 0, 0)
                  || test_pooling(9, 7, 15, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 0, 0, 0, 0)
                  || test_pooling(9, 7, 16, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 1, 0, 0, 0, TEST_LAYER_DISABLE_GPU_TESTING)
                  || test_pooling(9, 7, 64, 1, ksp[i][0], ksp[i][1], ksp[i][2], 0, 3, 1, 0, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_pooling_2()
{
    return 0
           || test_pooling(2, 5, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(5, 2, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(3, 6, 3, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(6, 3, 3, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(4, 4, 4, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(6, 4, 4, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(8, 7, 8, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(7, 8, 8, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(11, 13, 16, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(13, 11, 16, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(110, 103, 106, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(130, 101, 106, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(80, 93, 128, 0, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(80, 91, 128, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_pooling(48, 48, 4, 0, 2, 2, 0, 0, 0, 0, 0, 0)
           || test_pooling(48, 48, 15, 0, 2, 2, 1, 0, 0, 0, 0, 0);
}

// adaptive avg pool
static int test_pooling_3()
{
    return 0
           || test_pooling(2, 5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling(2, 5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling(2, 5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling(2, 5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling(2, 5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling(2, 5, 1, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling(5, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling(5, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling(5, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling(5, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling(5, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling(5, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling(3, 6, 3, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling(3, 6, 3, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling(3, 6, 3, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling(3, 6, 3, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling(3, 6, 3, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling(3, 6, 3, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling(4, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling(4, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling(4, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling(4, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling(4, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling(4, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling(4, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 7)
           || test_pooling(4, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 8)
           || test_pooling(6, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling(6, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling(6, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling(6, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling(6, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling(6, 4, 4, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling(8, 7, 8, 1, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling(8, 7, 8, 1, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling(8, 7, 8, 1, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling(8, 7, 8, 1, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling(8, 7, 8, 1, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling(8, 7, 8, 1, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling(8, 7, 8, 1, 1, 1, 0, 0, 0, 0, 1, 7)
           || test_pooling(8, 7, 8, 1, 1, 1, 0, 0, 0, 0, 1, 8)
           || test_pooling(8, 7, 8, 1, 1, 1, 0, 0, 0, 0, 1, 9)
           || test_pooling(11, 13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 1)
           || test_pooling(11, 13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 3)
           || test_pooling(11, 13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 5)
           || test_pooling(11, 13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 7)
           || test_pooling(11, 13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 9)
           || test_pooling(11, 13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 11)
           || test_pooling(11, 13, 16, 1, 1, 1, 0, 0, 0, 1, 0, 13)
           || test_pooling(13, 11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 2)
           || test_pooling(13, 11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 4)
           || test_pooling(13, 11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 6)
           || test_pooling(13, 11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 8)
           || test_pooling(13, 11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 10)
           || test_pooling(13, 11, 16, 1, 1, 1, 0, 0, 0, 1, 0, 12);
}

// adaptive max pool
static int test_pooling_4()
{
    return 0
           || test_pooling(2, 5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling(2, 5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling(2, 5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling(2, 5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling(2, 5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling(2, 5, 1, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling(5, 2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling(5, 2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling(5, 2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling(5, 2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling(5, 2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling(5, 2, 1, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling(3, 6, 3, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling(3, 6, 3, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling(3, 6, 3, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling(3, 6, 3, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling(3, 6, 3, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling(3, 6, 3, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling(4, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling(4, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling(4, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling(4, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling(4, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling(4, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling(4, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 7)
           || test_pooling(4, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 8)
           || test_pooling(6, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling(6, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling(6, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling(6, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling(6, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling(6, 4, 4, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling(8, 7, 8, 0, 1, 1, 0, 0, 0, 0, 1, 1)
           || test_pooling(8, 7, 8, 0, 1, 1, 0, 0, 0, 0, 1, 2)
           || test_pooling(8, 7, 8, 0, 1, 1, 0, 0, 0, 0, 1, 3)
           || test_pooling(8, 7, 8, 0, 1, 1, 0, 0, 0, 0, 1, 4)
           || test_pooling(8, 7, 8, 0, 1, 1, 0, 0, 0, 0, 1, 5)
           || test_pooling(8, 7, 8, 0, 1, 1, 0, 0, 0, 0, 1, 6)
           || test_pooling(8, 7, 8, 0, 1, 1, 0, 0, 0, 0, 1, 7)
           || test_pooling(8, 7, 8, 0, 1, 1, 0, 0, 0, 0, 1, 8)
           || test_pooling(8, 7, 8, 0, 1, 1, 0, 0, 0, 0, 1, 9)
           || test_pooling(11, 13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 1)
           || test_pooling(11, 13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 3)
           || test_pooling(11, 13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 5)
           || test_pooling(11, 13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 7)
           || test_pooling(11, 13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 9)
           || test_pooling(11, 13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 11)
           || test_pooling(11, 13, 16, 0, 1, 1, 0, 0, 0, 1, 0, 13)
           || test_pooling(13, 11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 2)
           || test_pooling(13, 11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 4)
           || test_pooling(13, 11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 6)
           || test_pooling(13, 11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 8)
           || test_pooling(13, 11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 10)
           || test_pooling(13, 11, 16, 0, 1, 1, 0, 0, 0, 1, 0, 12);
}

#if NCNN_VALIDATION
static int test_pooling_load_param()
{
    ncnn::ParamDict base;
    base.set(1, 3);
    if (test_layer_param(ncnn::LayerType::Pooling, base, 0) != 0)
        return -1;

    // global pooling does not use the local stride
    ncnn::ParamDict global = base;
    global.set(4, 1);

    return 0
           || test_layer_param(ncnn::LayerType::Pooling, base, 2, 0, -1)
           || test_layer_param(ncnn::LayerType::Pooling, global, 2, 0, 0);
}

static int test_pooling_load_param_type()
{
    ncnn::ParamDict base;
    base.set(1, 3);
    if (test_layer_param(ncnn::LayerType::Pooling, base, 0) != 0)
        return -1;

    for (int i = 0; i <= 1; i++)
    {
        if (test_layer_param(ncnn::LayerType::Pooling, base, 0, i, 0) != 0)
            return -1;
    }

    const int invalid[] = {-1, 2, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        if (test_layer_param(ncnn::LayerType::Pooling, base, 0, invalid[i], -1) != 0)
            return -1;
    }

    for (int i = 0; i <= 3; i++)
    {
        if (test_layer_param(ncnn::LayerType::Pooling, base, 5, i, 0) != 0)
            return -1;
    }

    // global and adaptive pooling do not use pad_mode
    ncnn::ParamDict global = base;
    global.set(4, 1);
    ncnn::ParamDict adaptive = base;
    adaptive.set(4, 0);
    adaptive.set(8, 1);
    adaptive.set(7, 1);

    const int invalid_pad[] = {-1, 4, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        int ret = 0
                  || test_layer_param(ncnn::LayerType::Pooling, base, 5, invalid_pad[i], -1)
                  || test_layer_param(ncnn::LayerType::Pooling, global, 5, invalid_pad[i], 0)
                  || test_layer_param(ncnn::LayerType::Pooling, adaptive, 5, invalid_pad[i], 0);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_pooling_load_param_adaptive()
{
    ncnn::ParamDict base;
    base.set(1, 3);
    base.set(7, 1);
    if (test_layer_param(ncnn::LayerType::Pooling, base, -1) != 0)
        return -1;

    base.set(8, 2);
    if (test_layer_param(ncnn::LayerType::Pooling, base, 0) != 0)
        return -1;

    base.set(18, 2);

    // global and ordinary pooling ignore adaptive output sizes
    ncnn::ParamDict global = base;
    global.set(4, 1);
    ncnn::ParamDict ordinary = base;
    ordinary.set(4, 0);
    ordinary.set(7, 0);

    const int ids[] = {8, 18};
    const int invalid[] = {0, -1, -234, INT_MIN};
    for (int i = 0; i < 2; i++)
    {
        if (test_layer_param(ncnn::LayerType::Pooling, base, ids[i], 1, 0) != 0)
            return -1;

        if (test_layer_param(ncnn::LayerType::Pooling, base, ids[i], -233, 0) != 0)
            return -1;

        for (int j = 0; j < 4; j++)
        {
            int ret = 0
                      || test_layer_param(ncnn::LayerType::Pooling, base, ids[i], invalid[j], -1)
                      || test_layer_param(ncnn::LayerType::Pooling, global, ids[i], invalid[j], 0)
                      || test_layer_param(ncnn::LayerType::Pooling, ordinary, ids[i], invalid[j], 0);
            if (ret != 0)
                return ret;
        }
    }

    return 0;
}
#endif // NCNN_VALIDATION

int main()
{
    SRAND(7767517);

    return 0
           || test_pooling_0()
           || test_pooling_1()
           || test_pooling_2()
           || test_pooling_3()
           || test_pooling_4()
#if NCNN_VALIDATION
           || test_pooling_load_param()
           || test_pooling_load_param_type()
           || test_pooling_load_param_adaptive()
#endif // NCNN_VALIDATION
           ;
}
