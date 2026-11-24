// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include <float.h>

static int test_softplus(const ncnn::Mat& a, float threshold = FLT_MAX, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, threshold);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Softplus", pd, weights, a, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_softplus failed a.dims=%d a=(%d %d %d %d) threshold=%f\n", a.dims, a.w, a.h, a.d, a.c, threshold);
    }

    return ret;
}

// cpu pack8/pack16 cases reuse the Vulkan pack4 path covered by the pack4 cases
// keep the 1d sizes for dispatch boundary coverage
static int test_softplus_0()
{
    return 0
           || test_softplus(RandomMat(5, 6, 7, 24), FLT_MAX, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_softplus(RandomMat(7, 8, 9, 12))
           || test_softplus(RandomMat(3, 4, 5, 13))
           || test_softplus(RandomMat(5, 7, 24), FLT_MAX, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_softplus(RandomMat(7, 9, 12))
           || test_softplus(RandomMat(3, 5, 13));
}

static int test_softplus_1()
{
    return 0
           || test_softplus(RandomMat(15, 24), FLT_MAX, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_softplus(RandomMat(17, 12))
           || test_softplus(RandomMat(19, 15));
}

static int test_softplus_2()
{
    return 0
           || test_softplus(RandomMat(128))
           || test_softplus(RandomMat(124))
           || test_softplus(RandomMat(127));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_softplus_0()
           || test_softplus_1()
           || test_softplus_2()
           || test_softplus(RandomMat(128), 20.f)
           || test_softplus(RandomMat(124), 0.f)
           || test_softplus(RandomMat(127), -2.f)
           || test_softplus(RandomMat(129), 0.5f);
}
