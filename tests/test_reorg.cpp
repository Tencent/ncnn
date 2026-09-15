// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

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

static int test_reorg_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 2);
    if (test_layer_param(ncnn::LayerType::Reorg, base, 0) != 0)
        return -1;

    const int invalid[] = {0, -1, -8, INT_MIN};
    for (int i = 0; i < 4; i++)
    {
        if (test_layer_param(ncnn::LayerType::Reorg, base, 0, invalid[i], -1) != 0)
            return -1;
    }
    // the squared value exceeds the int range
    return test_layer_param(ncnn::LayerType::Reorg, base, 0, 65536, -1);
}

static int test_reorg_load_param_type()
{
    ncnn::ParamDict base;
    if (test_layer_param(ncnn::LayerType::Reorg, base, 0) != 0)
        return -1;

    for (int i = 0; i <= 1; i++)
    {
        if (test_layer_param(ncnn::LayerType::Reorg, base, 1, i, 0) != 0)
            return -1;
    }

    const int invalid[] = {-1, 2, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        if (test_layer_param(ncnn::LayerType::Reorg, base, 1, invalid[i], -1) != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_reorg_0() || test_reorg_1() || test_reorg_load_param() || test_reorg_load_param_type();
}
