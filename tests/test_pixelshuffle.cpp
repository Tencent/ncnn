// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

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

int main()
{
    SRAND(7767517);

    return test_pixelshuffle_0() || test_pixelshuffle_1();
}
