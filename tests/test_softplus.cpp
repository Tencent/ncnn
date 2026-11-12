// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include <float.h>
#include <limits>

static int test_softplus(const ncnn::Mat& a, int flag = 0)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Softplus", pd, weights, a, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_softplus failed a.dims=%d a=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
    }

    return ret;
}

// cpu pack8/pack16 cases reuse the Vulkan pack4 path covered by the pack4 cases
// keep the 1d sizes for dispatch boundary coverage
static int test_softplus_0()
{
    return 0
           || test_softplus(RandomMat(5, 6, 7, 24), TEST_LAYER_DISABLE_GPU_TESTING)
           || test_softplus(RandomMat(7, 8, 9, 12))
           || test_softplus(RandomMat(3, 4, 5, 13))
           || test_softplus(RandomMat(5, 7, 24), TEST_LAYER_DISABLE_GPU_TESTING)
           || test_softplus(RandomMat(7, 9, 12))
           || test_softplus(RandomMat(3, 5, 13));
}

static int test_softplus_1()
{
    return 0
           || test_softplus(RandomMat(15, 24), TEST_LAYER_DISABLE_GPU_TESTING)
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

static int test_softplus_threshold(float threshold, int size, bool large_threshold = false)
{
    const float values[] = {-10.f, -3.f, -2.f, -1.f, 0.f, 0.5f, 1.f, 12.f, 20.f, 30.f, 100.f};
    ncnn::Mat a(size);
    ncnn::Mat expected(size);
    for (int i = 0; i < size; i++)
    {
        float x = values[i % 11];
        if (i % 11 == 0) x = large_threshold ? 88.5f : threshold;
        if (large_threshold && x > 88.5f) x = 88.5f;
        a[i] = x;
        expected[i] = x > threshold ? x : logf(expf(x) + 1.f);
    }

    ncnn::ParamDict pd;
    pd.set(0, threshold);
    std::vector<ncnn::Mat> weights;
    ncnn::Mat actual;
    int ret = test_layer_naive(ncnn::layer_to_index("Softplus"), pd, weights, a, actual, 0);
    if (ret == 0) ret = CompareMat(expected, actual, 0.001f);
    if (ret == 0) ret = test_layer("Softplus", pd, weights, a, 0.001f);
    if (ret != 0)
    {
        fprintf(stderr, "test_softplus_threshold failed threshold=%f size=%d\n", threshold, size);
    }
    return ret;
}

static int test_softplus_nonfinite()
{
    const float inf = std::numeric_limits<float>::infinity();
    ncnn::Mat a(4);
    a[0] = 100.f;
    a[1] = inf;
    a[2] = -inf;
    a[3] = std::numeric_limits<float>::quiet_NaN();

    ncnn::ParamDict pd;
    pd.set(0, FLT_MAX);
    std::vector<ncnn::Mat> weights;
    ncnn::Mat actual;
    int ret = test_layer_naive(ncnn::layer_to_index("Softplus"), pd, weights, a, actual, 0);
    if (ret != 0) return ret;
    if (actual[0] != inf || actual[1] != inf || actual[2] != 0.f || actual[3] == actual[3])
    {
        fprintf(stderr, "test_softplus_nonfinite failed\n");
        return -1;
    }
    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_softplus_0()
           || test_softplus_1()
           || test_softplus_2()
           || test_softplus_threshold(20.f, 128)
           || test_softplus_threshold(0.f, 124)
           || test_softplus_threshold(-2.f, 127)
           || test_softplus_threshold(0.5f, 129)
           || test_softplus_threshold(FLT_MAX, 129, true)
           || test_softplus_nonfinite();
}
