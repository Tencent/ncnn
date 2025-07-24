// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_reorg(const ncnn::Mat& a, int stride, int mode)
{
    ncnn::ParamDict pd;
    pd.set(0, stride);
    pd.set(1, mode);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Reorg", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_reorg failed a.dims=%d a=(%d %d %d) stride=%d mode=%d\n", a.dims, a.w, a.h, a.c, stride, mode);
    }

    return ret;
}

static int test_reorg_0()
{
    return 0
           || test_reorg(RandomMat(6, 7, 1), 1, 0)
           || test_reorg(RandomMat(6, 6, 2), 2, 0)
           || test_reorg(RandomMat(6, 8, 3), 2, 0)
           || test_reorg(RandomMat(4, 4, 4), 4, 0)
           || test_reorg(RandomMat(8, 8, 8), 2, 0)
           || test_reorg(RandomMat(10, 10, 12), 2, 0)
           || test_reorg(RandomMat(9, 9, 4), 3, 0)
           || test_reorg(RandomMat(9, 9, 16), 3, 0);
}

static int test_reorg_1()
{
    return 0
           || test_reorg(RandomMat(6, 7, 1), 1, 1)
           || test_reorg(RandomMat(6, 6, 2), 2, 1)
           || test_reorg(RandomMat(6, 8, 3), 2, 1)
           || test_reorg(RandomMat(4, 4, 4), 4, 1)
           || test_reorg(RandomMat(8, 8, 8), 2, 1)
           || test_reorg(RandomMat(10, 10, 12), 2, 1)
           || test_reorg(RandomMat(9, 9, 4), 3, 1)
           || test_reorg(RandomMat(9, 9, 16), 3, 1);
}

int main()
{
    SRAND(7767517);

    return test_reorg_0() || test_reorg_1();
}
