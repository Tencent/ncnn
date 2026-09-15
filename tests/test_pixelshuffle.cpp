// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_pixelshuffle(const ncnn::Mat& a, int upscale_factor, int mode)
{
    ncnn::ParamDict pd;
    pd.set(0, upscale_factor);
    pd.set(1, mode);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("PixelShuffle", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_pixelshuffle failed a.dims=%d a=(%d %d %d) upscale_factor=%d mode=%d\n", a.dims, a.w, a.h, a.c, upscale_factor, mode);
    }

    return ret;
}

static int test_pixelshuffle_0()
{
    return 0
           || test_pixelshuffle(RandomMat(7, 7, 1), 1, 0)
           || test_pixelshuffle(RandomMat(7, 7, 8), 2, 0)
           || test_pixelshuffle(RandomMat(7, 7, 12), 2, 0)
           || test_pixelshuffle(RandomMat(7, 7, 64), 4, 0)
           || test_pixelshuffle(RandomMat(7, 7, 32), 2, 0)
           || test_pixelshuffle(RandomMat(7, 7, 48), 2, 0)
           || test_pixelshuffle(RandomMat(7, 7, 36), 3, 0)
           || test_pixelshuffle(RandomMat(7, 7, 72), 3, 0)
           || test_pixelshuffle(RandomMat(7, 7, 90), 3, 0);
}

static int test_pixelshuffle_1()
{
    return 0
           || test_pixelshuffle(RandomMat(7, 7, 1), 1, 1)
           || test_pixelshuffle(RandomMat(7, 7, 8), 2, 1)
           || test_pixelshuffle(RandomMat(7, 7, 12), 2, 1)
           || test_pixelshuffle(RandomMat(7, 7, 64), 4, 1)
           || test_pixelshuffle(RandomMat(7, 7, 32), 2, 1)
           || test_pixelshuffle(RandomMat(7, 7, 48), 2, 1)
           || test_pixelshuffle(RandomMat(7, 7, 36), 3, 1)
           || test_pixelshuffle(RandomMat(7, 7, 90), 3, 1);
}

static int test_pixelshuffle_2()
{
    return 0
           || test_pixelshuffle(RandomMat(8, 3, 8), 2, 0)
           || test_pixelshuffle(RandomMat(12, 3, 8), 2, 0)
           || test_pixelshuffle(RandomMat(13, 3, 8), 2, 0)
           || test_pixelshuffle(RandomMat(8, 3, 8), 2, 1)
           || test_pixelshuffle(RandomMat(12, 3, 8), 2, 1)
           || test_pixelshuffle(RandomMat(13, 3, 8), 2, 1)
           || test_pixelshuffle(RandomMat(4, 3, 64), 4, 0)
           || test_pixelshuffle(RandomMat(5, 3, 64), 4, 0)
           || test_pixelshuffle(RandomMat(4, 3, 64), 4, 1)
           || test_pixelshuffle(RandomMat(5, 3, 64), 4, 1);
}

static int test_pixelshuffle_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    for (int backend = 0; backend < 2; backend++)
    {
        ncnn::Layer* layer = backend == 0 ? ncnn::create_layer_naive(ncnn::LayerType::PixelShuffle) : ncnn::create_layer_cpu(ncnn::LayerType::PixelShuffle);
        if (!layer)
            return -1;

        int ret = layer->load_param(pd);
        delete layer;

        if (ret != (valid ? 0 : -1))
        {
            fprintf(stderr, "PixelShuffle load_param backend=%d returned %d, expected %s\n", backend, ret, valid ? "success" : "failure");
            return -1;
        }
    }

    return 0;
}

static int test_pixelshuffle_load_param()
{
    ncnn::ParamDict pd;
    pd.set(0, 2);
    if (test_pixelshuffle_load_param_case(pd, true) != 0)
        return -1;

    const int invalid[] = {0, -1, -8, INT_MIN};
    for (int i = 0; i < 4; i++)
    {
        pd.set(0, invalid[i]);
        if (test_pixelshuffle_load_param_case(pd, false) != 0)
            return -1;
    }
    pd.set(0, 65536); // squaring this value wraps to zero on 32-bit int
    if (test_pixelshuffle_load_param_case(pd, false) != 0)
        return -1;
    return 0;
}

int main()
{
    SRAND(7767517);

    return test_pixelshuffle_0() || test_pixelshuffle_1() || test_pixelshuffle_2() || test_pixelshuffle_load_param();
}
