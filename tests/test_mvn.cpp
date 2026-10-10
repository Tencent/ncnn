// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_mvn(const ncnn::Mat& a, int normalize_variance, int across_channels, float eps)
{
    ncnn::ParamDict pd;
    pd.set(0, normalize_variance);
    pd.set(1, across_channels);
    pd.set(2, eps);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("MVN", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_mvn failed a.dims=%d a=(%d %d %d %d) normalize_variance=%d across_channels=%d eps=%f\n", a.dims, a.w, a.h, a.d, a.c, normalize_variance, across_channels, eps);
    }

    return ret;
}

static int test_mvn_0()
{
    ncnn::Mat a = RandomMat(5, 4, 3, 3);
    ncnn::Mat b = RandomMat(3, 3, 2, 8);

    return 0
           || test_mvn(a, 0, 0, 0.0001f)
           || test_mvn(a, 1, 0, 0.001f)
           || test_mvn(b, 0, 1, 0.0001f)
           || test_mvn(b, 1, 1, 0.001f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_mvn_0();
}
