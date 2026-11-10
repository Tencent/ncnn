// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_gelu(const ncnn::Mat& a, bool fast_gelu, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, fast_gelu ? 1 : 0);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("GELU", pd, weights, a, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_gelu failed a.dims=%d a=(%d %d %d %d) fast_gelu=%s\n", a.dims, a.w, a.h, a.d, a.c, fast_gelu ? "true" : "false");
    }

    return ret;
}

// cpu pack8/pack16 cases reuse the Vulkan pack4 path covered by the pack4 cases
// keep the 1d sizes for dispatch boundary coverage
static int test_gelu_0()
{
    return 0
           || test_gelu(RandomMat(6, 7, 9, 32), false, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gelu(RandomMat(6, 7, 9, 32), true, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gelu(RandomMat(5, 6, 7, 24), false, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gelu(RandomMat(5, 6, 7, 24), true, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gelu(RandomMat(7, 8, 9, 12), false)
           || test_gelu(RandomMat(7, 8, 9, 12), true)
           || test_gelu(RandomMat(3, 4, 5, 13), false)
           || test_gelu(RandomMat(3, 4, 5, 13), true);
}

static int test_gelu_1()
{
    return 0
           || test_gelu(RandomMat(9, 7, 32), false, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gelu(RandomMat(9, 7, 32), true, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gelu(RandomMat(5, 7, 24), false, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gelu(RandomMat(5, 7, 24), true, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gelu(RandomMat(7, 9, 12), false)
           || test_gelu(RandomMat(7, 9, 12), true)
           || test_gelu(RandomMat(3, 5, 13), false)
           || test_gelu(RandomMat(3, 5, 13), true);
}

static int test_gelu_2()
{
    return 0
           || test_gelu(RandomMat(13, 32), false, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gelu(RandomMat(13, 32), true, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gelu(RandomMat(15, 24), false, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gelu(RandomMat(15, 24), true, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_gelu(RandomMat(17, 12), false)
           || test_gelu(RandomMat(17, 12), true)
           || test_gelu(RandomMat(19, 15), false)
           || test_gelu(RandomMat(19, 15), true);
}

static int test_gelu_3()
{
    return 0
           || test_gelu(RandomMat(128), false)
           || test_gelu(RandomMat(128), true)
           || test_gelu(RandomMat(124), false)
           || test_gelu(RandomMat(124), true)
           || test_gelu(RandomMat(127), false)
           || test_gelu(RandomMat(127), true)
           || test_gelu(RandomMat(120), false)
           || test_gelu(RandomMat(120), true);
}

static int test_gelu_4()
{
    // large magnitude inputs, gelu(x) must be 0 for x << 0 and x for x >> 0
    // fp16 arithmetic used to leak +0.022*|x| on the negative side (tanh_ps_f16 saturated at 1.044)
    return 0
           || test_gelu(RandomMat(6, 7, 9, 32, -100.f, 100.f), false)
           || test_gelu(RandomMat(6, 7, 9, 32, -100.f, 100.f), true)
           || test_gelu(RandomMat(9, 7, 32, -100.f, 100.f), false)
           || test_gelu(RandomMat(9, 7, 32, -100.f, 100.f), true)
           || test_gelu(RandomMat(13, 32, -100.f, 100.f), false)
           || test_gelu(RandomMat(13, 32, -100.f, 100.f), true)
           || test_gelu(RandomMat(128, -100.f, 100.f), false)
           || test_gelu(RandomMat(128, -100.f, 100.f), true)
           || test_gelu(RandomMat(128, -2000.f, 2000.f), true);
}

static int test_gelu_5()
{
    // finite fp16 inputs must not overflow when multiplying by 1 + tanh(...)
    const float values[] = {-65504.f, -40000.f, -32768.f, -32752.f, -1.f, -0.f, 0.f, 1.f, 32752.f, 32768.f, 40000.f, 65504.f};
    ncnn::Mat a(128);
    ncnn::Mat b(127);
    for (int i = 0; i < a.w; i++)
        a[i] = values[i % 12];
    for (int i = 0; i < b.w; i++)
        b[i] = values[i % 12];

    return 0
           || test_gelu(a, false)
           || test_gelu(a, true)
           || test_gelu(b, false)
           || test_gelu(b, true);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_gelu_0()
           || test_gelu_1()
           || test_gelu_2()
           || test_gelu_3()
           || test_gelu_4()
           || test_gelu_5();
}
