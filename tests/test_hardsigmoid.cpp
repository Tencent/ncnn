// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_hardsigmoid(const ncnn::Mat& a, float alpha, float beta, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, beta);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("HardSigmoid", pd, weights, a, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_hardsigmoid failed a.dims=%d a=(%d %d %d %d) alpha=%f beta=%f\n", a.dims, a.w, a.h, a.d, a.c, alpha, beta);
    }

    return ret;
}

// cpu pack8/pack16 cases reuse the Vulkan pack4 path covered by the pack4 cases
// keep the 1d sizes for dispatch boundary coverage
static int test_hardsigmoid_0()
{
    return 0
           || test_hardsigmoid(RandomMat(5, 6, 7, 24), 0.5f, 0.5f, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_hardsigmoid(RandomMat(7, 8, 9, 12), 0.5f, 0.5f)
           || test_hardsigmoid(RandomMat(3, 4, 5, 13), 0.5f, 0.5f)
           || test_hardsigmoid(RandomMat(3, 3, 48), 0.5f, 0.5f, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_hardsigmoid(RandomMat(5, 7, 24), 0.5f, 0.5f, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_hardsigmoid(RandomMat(7, 9, 12), 0.5f, 0.5f)
           || test_hardsigmoid(RandomMat(3, 5, 13), 0.5f, 0.5f);
}

static int test_hardsigmoid_1()
{
    return 0
           || test_hardsigmoid(RandomMat(19, 48), 0.5f, 0.5f, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_hardsigmoid(RandomMat(15, 24), 0.5f, 0.5f, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_hardsigmoid(RandomMat(17, 12), 0.5f, 0.5f)
           || test_hardsigmoid(RandomMat(19, 15), 0.5f, 0.5f);
}

static int test_hardsigmoid_2()
{
    return 0
           || test_hardsigmoid(RandomMat(128), 0.5f, 0.5f)
           || test_hardsigmoid(RandomMat(124), 0.5f, 0.5f)
           || test_hardsigmoid(RandomMat(127), 0.5f, 0.5f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_hardsigmoid_0()
           || test_hardsigmoid_1()
           || test_hardsigmoid_2();
}
