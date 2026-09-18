// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_statisticspooling(const ncnn::Mat& a, int include_stddev)
{
    ncnn::ParamDict pd;
    pd.set(0, include_stddev);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("StatisticsPooling", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_statisticspooling failed a.dims=%d a=(%d %d %d %d) include_stddev=%d\n", a.dims, a.w, a.h, a.d, a.c, include_stddev);
    }

    return ret;
}

static int test_statisticspooling_0()
{
    ncnn::Mat a = RandomMat(5, 4, 3, 3);
    ncnn::Mat b = RandomMat(3, 3, 2, 8);

    return 0
           || test_statisticspooling(a, 0)
           || test_statisticspooling(a, 1)
           || test_statisticspooling(b, 0)
           || test_statisticspooling(b, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_statisticspooling_0();
}
