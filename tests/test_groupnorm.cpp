// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_groupnorm(const ncnn::Mat& a, int group, float eps, int affine)
{
    int channels = a.c;
    if (a.dims == 1)
    {
        channels = a.w;
    }
    else if (a.dims == 2)
    {
        channels = a.h;
    }

    ncnn::ParamDict pd;
    pd.set(0, group);
    pd.set(1, channels);
    pd.set(2, eps);
    pd.set(3, affine);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(channels);
    weights[1] = RandomMat(channels);

    int ret = test_layer("GroupNorm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_groupnorm failed a.dims=%d a=(%d %d %d %d) group=%d eps=%f affine=%d\n", a.dims, a.w, a.h, a.d, a.c, group, eps, affine);
    }

    return ret;
}

static int test_groupnorm_0()
{
    return 0
           || test_groupnorm(RandomMat(3, 6, 4, 2), 1, 0.01f, 0)
           || test_groupnorm(RandomMat(2, 3, 3, 8), 2, 0.002f, 0)
           || test_groupnorm(RandomMat(3, 4, 5, 6), 3, 0.01f, 0)
           || test_groupnorm(RandomMat(4, 5, 6, 12), 4, 0.02f, 0)
           || test_groupnorm(RandomMat(5, 6, 7, 24), 2, 0.001f, 0)
           || test_groupnorm(RandomMat(2, 8, 9, 24), 3, 0.0001f, 0)
           || test_groupnorm(RandomMat(3, 6, 4, 2), 1, 0.01f, 1)
           || test_groupnorm(RandomMat(2, 3, 3, 8), 2, 0.002f, 1)
           || test_groupnorm(RandomMat(3, 4, 5, 6), 3, 0.01f, 1)
           || test_groupnorm(RandomMat(4, 5, 6, 12), 4, 0.02f, 1)
           || test_groupnorm(RandomMat(5, 6, 7, 24), 2, 0.001f, 1)
           || test_groupnorm(RandomMat(2, 8, 9, 24), 3, 0.0001f, 1);
}

static int test_groupnorm_1()
{
    return 0
           || test_groupnorm(RandomMat(6, 4, 2), 1, 0.01f, 0)
           || test_groupnorm(RandomMat(3, 3, 8), 2, 0.002f, 0)
           || test_groupnorm(RandomMat(4, 5, 6), 3, 0.01f, 0)
           || test_groupnorm(RandomMat(5, 6, 12), 4, 0.02f, 0)
           || test_groupnorm(RandomMat(6, 7, 24), 2, 0.001f, 0)
           || test_groupnorm(RandomMat(8, 9, 24), 3, 0.0001f, 0)
           || test_groupnorm(RandomMat(6, 4, 2), 1, 0.01f, 1)
           || test_groupnorm(RandomMat(3, 3, 8), 2, 0.002f, 1)
           || test_groupnorm(RandomMat(4, 5, 6), 3, 0.01f, 1)
           || test_groupnorm(RandomMat(5, 6, 12), 4, 0.02f, 1)
           || test_groupnorm(RandomMat(6, 7, 24), 2, 0.001f, 1)
           || test_groupnorm(RandomMat(8, 9, 24), 3, 0.0001f, 1);
}

static int test_groupnorm_2()
{
    return 0
           || test_groupnorm(RandomMat(24, 2), 1, 0.01f, 0)
           || test_groupnorm(RandomMat(23, 8), 2, 0.002f, 0)
           || test_groupnorm(RandomMat(25, 6), 3, 0.01f, 0)
           || test_groupnorm(RandomMat(26, 12), 4, 0.02f, 0)
           || test_groupnorm(RandomMat(27, 24), 2, 0.001f, 0)
           || test_groupnorm(RandomMat(29, 24), 3, 0.0001f, 0)
           || test_groupnorm(RandomMat(24, 2), 1, 0.01f, 1)
           || test_groupnorm(RandomMat(23, 8), 2, 0.002f, 1)
           || test_groupnorm(RandomMat(25, 6), 3, 0.01f, 1)
           || test_groupnorm(RandomMat(26, 12), 4, 0.02f, 1)
           || test_groupnorm(RandomMat(27, 24), 2, 0.001f, 1)
           || test_groupnorm(RandomMat(29, 24), 3, 0.0001f, 1);
}

static int test_groupnorm_3()
{
    return 0
           || test_groupnorm(RandomMat(12), 1, 0.01f, 0)
           || test_groupnorm(RandomMat(18), 2, 0.002f, 0)
           || test_groupnorm(RandomMat(36), 3, 0.01f, 0)
           || test_groupnorm(RandomMat(212), 4, 0.02f, 0)
           || test_groupnorm(RandomMat(124), 2, 0.001f, 0)
           || test_groupnorm(RandomMat(324), 3, 0.0001f, 0)
           || test_groupnorm(RandomMat(12), 1, 0.01f, 1)
           || test_groupnorm(RandomMat(18), 2, 0.002f, 1)
           || test_groupnorm(RandomMat(36), 3, 0.01f, 1)
           || test_groupnorm(RandomMat(212), 4, 0.02f, 1)
           || test_groupnorm(RandomMat(124), 2, 0.001f, 1)
           || test_groupnorm(RandomMat(324), 3, 0.0001f, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_groupnorm_0()
           || test_groupnorm_1()
           || test_groupnorm_2()
           || test_groupnorm_3();
}
