// Copyright 2026 ihb2032 <hebome@foxmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_threshold(const ncnn::Mat& a, float threshold)
{
    ncnn::ParamDict pd;
    pd.set(0, threshold);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Threshold", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_threshold failed a.dims=%d a=(%d %d %d %d) threshold=%f\n", a.dims, a.w, a.h, a.d, a.c, threshold);
    }

    return ret;
}

static int test_threshold_0()
{
    return 0
           || test_threshold(RandomMat(5, 6, 7, 24), 0.f)
           || test_threshold(RandomMat(5, 6, 7, 24), 0.3f)
           || test_threshold(RandomMat(7, 8, 9, 12), -0.1f)
           || test_threshold(RandomMat(7, 8, 9, 12), 0.7f)
           || test_threshold(RandomMat(3, 4, 5, 13), -0.5f)
           || test_threshold(RandomMat(3, 4, 5, 13), 1.1f);
}

static int test_threshold_1()
{
    return 0
           || test_threshold(RandomMat(5, 7, 24), 0.f)
           || test_threshold(RandomMat(5, 7, 24), 0.3f)
           || test_threshold(RandomMat(7, 9, 12), -0.1f)
           || test_threshold(RandomMat(7, 9, 12), 0.7f)
           || test_threshold(RandomMat(3, 5, 13), -0.5f)
           || test_threshold(RandomMat(3, 5, 13), 1.1f);
}

static int test_threshold_2()
{
    return 0
           || test_threshold(RandomMat(15, 24), 0.f)
           || test_threshold(RandomMat(15, 24), 0.3f)
           || test_threshold(RandomMat(17, 12), -0.1f)
           || test_threshold(RandomMat(17, 12), 0.7f)
           || test_threshold(RandomMat(19, 15), -0.5f)
           || test_threshold(RandomMat(19, 15), 1.1f);
}

static int test_threshold_3()
{
    return 0
           || test_threshold(RandomMat(128), 0.f)
           || test_threshold(RandomMat(128), 0.3f)
           || test_threshold(RandomMat(124), -0.1f)
           || test_threshold(RandomMat(124), 0.7f)
           || test_threshold(RandomMat(127), -0.5f)
           || test_threshold(RandomMat(127), 1.1f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_threshold_0()
           || test_threshold_1()
           || test_threshold_2()
           || test_threshold_3();
}
