// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_bias(const ncnn::Mat& a)
{
    int channels = a.c;

    ncnn::ParamDict pd;
    pd.set(0, channels);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(channels);

    int ret = test_layer("Bias", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_bias failed a.dims=%d a=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
    }

    return ret;
}

static int test_bias_0()
{
    return 0
           || test_bias(RandomMat(5, 6, 7, 24))
           || test_bias(RandomMat(7, 8, 9, 12))
           || test_bias(RandomMat(3, 4, 5, 13));
}

static int test_bias_1()
{
    return 0
           || test_bias(RandomMat(5, 7, 24))
           || test_bias(RandomMat(7, 9, 12))
           || test_bias(RandomMat(3, 5, 13));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_bias_0()
           || test_bias_1();
}
