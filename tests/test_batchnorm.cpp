// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_batchnorm(const ncnn::Mat& a, float eps)
{
    int channels;
    if (a.dims == 1) channels = a.w;
    if (a.dims == 2) channels = a.h;
    if (a.dims == 3 || a.dims == 4) channels = a.c;

    ncnn::ParamDict pd;
    pd.set(0, channels); // channels
    pd.set(1, eps);      // eps

    std::vector<ncnn::Mat> weights(4);
    weights[0] = RandomMat(channels);
    weights[1] = RandomMat(channels);
    weights[2] = RandomMat(channels);
    weights[3] = RandomMat(channels);

    // var must be positive
    Randomize(weights[2], 0.001f, 2.f);

    int ret = test_layer("BatchNorm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_batchnorm failed a.dims=%d a=(%d %d %d %d) eps=%f\n", a.dims, a.w, a.h, a.d, a.c, eps);
    }

    return ret;
}

static int test_batchnorm_0()
{
    return 0
           || test_batchnorm(RandomMat(5, 6, 7, 24), 0.f)
           || test_batchnorm(RandomMat(5, 6, 7, 24), 0.01f)
           || test_batchnorm(RandomMat(7, 8, 9, 12), 0.f)
           || test_batchnorm(RandomMat(7, 8, 9, 12), 0.001f)
           || test_batchnorm(RandomMat(3, 4, 5, 13), 0.f)
           || test_batchnorm(RandomMat(3, 4, 5, 13), 0.f)
           || test_batchnorm(RandomMat(3, 4, 6, 32), 0.f)
           || test_batchnorm(RandomMat(3, 4, 5, 32), 0.001f);
}

static int test_batchnorm_1()
{
    return 0
           || test_batchnorm(RandomMat(5, 7, 24), 0.f)
           || test_batchnorm(RandomMat(5, 7, 24), 0.01f)
           || test_batchnorm(RandomMat(7, 9, 12), 0.f)
           || test_batchnorm(RandomMat(7, 9, 12), 0.001f)
           || test_batchnorm(RandomMat(3, 5, 13), 0.f)
           || test_batchnorm(RandomMat(3, 5, 13), 0.001f)
           || test_batchnorm(RandomMat(3, 5, 16), 0.001f)
           || test_batchnorm(RandomMat(3, 5, 32), 0.001f);
}

static int test_batchnorm_2()
{
    return 0
           || test_batchnorm(RandomMat(15, 24), 0.f)
           || test_batchnorm(RandomMat(15, 24), 0.01f)
           || test_batchnorm(RandomMat(17, 12), 0.f)
           || test_batchnorm(RandomMat(17, 12), 0.001f)
           || test_batchnorm(RandomMat(19, 15), 0.f)
           || test_batchnorm(RandomMat(19, 15), 0.001f)
           || test_batchnorm(RandomMat(128, 16), 0.f)
           || test_batchnorm(RandomMat(16, 128), 0.001f);
}

static int test_batchnorm_3()
{
    return 0
           || test_batchnorm(RandomMat(128), 0.f)
           || test_batchnorm(RandomMat(128), 0.001f)
           || test_batchnorm(RandomMat(124), 0.f)
           || test_batchnorm(RandomMat(124), 0.1f)
           || test_batchnorm(RandomMat(127), 0.f)
           || test_batchnorm(RandomMat(127), 0.1f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_batchnorm_0()
           || test_batchnorm_1()
           || test_batchnorm_2()
           || test_batchnorm_3();
}
