// Copyright 2023 Xiaomi Corp.   (author: Fangjun Kuang)
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_cumulativesum(const ncnn::Mat& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("CumulativeSum", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_cumulativesum failed a.dims=%d a=(%d %d %d) axis=%d\n", a.dims, a.w, a.h, a.c, axis);
    }

    return ret;
}

static int test_cumulativesum_1d()
{
    return 0
           || test_cumulativesum(RandomMat(6), 0)
           || test_cumulativesum(RandomMat(10), 0)
           || test_cumulativesum(RandomMat(10), -1)
           || test_cumulativesum(RandomMat(10), -2)
           || test_cumulativesum(RandomMat(101), 0);
}

static int test_cumulativesum_2d()
{
    return 0
           || test_cumulativesum(RandomMat(6, 8), 0)
           || test_cumulativesum(RandomMat(20, 103), 1)
           || test_cumulativesum(RandomMat(106, 50), -1)
           || test_cumulativesum(RandomMat(106, 50), -2);
}

static int test_cumulativesum_3d()
{
    return 0
           || test_cumulativesum(RandomMat(10, 6, 8), 0)
           || test_cumulativesum(RandomMat(303, 20, 103), 1)
           || test_cumulativesum(RandomMat(106, 50, 99), 2)
           || test_cumulativesum(RandomMat(303, 200, 103), -1)
           || test_cumulativesum(RandomMat(303, 200, 103), -2)
           || test_cumulativesum(RandomMat(303, 200, 103), -2);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_cumulativesum_1d()
           || test_cumulativesum_2d()
           || test_cumulativesum_3d();
}
