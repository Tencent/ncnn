// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_swiglu(const ncnn::Mat& a, const ncnn::Mat& b, float sigmoid_scale, float up_offset, float activation_limit, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, sigmoid_scale);
    pd.set(1, up_offset);
    pd.set(2, activation_limit);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = b;

    int ret = test_layer("SwiGLU", pd, weights, as, 1, 0.001f, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_swiglu failed a.dims=%d a=(%d %d %d %d) scale=%f offset=%f limit=%f\n", a.dims, a.w, a.h, a.d, a.c, sigmoid_scale, up_offset, activation_limit);
    }

    return ret;
}

static int test_swiglu_0()
{
    const ncnn::Mat a0 = RandomMat(5, 7, 3, 8);
    const ncnn::Mat b0 = RandomMat(5, 7, 3, 8);
    const ncnn::Mat a1 = RandomMat(5, 7, 3, 7, -12.f, 12.f);
    const ncnn::Mat b1 = RandomMat(5, 7, 3, 7, -12.f, 12.f);
    const ncnn::Mat a2 = RandomMat(4, 9, 2, 16, -12.f, 12.f);
    const ncnn::Mat b2 = RandomMat(4, 9, 2, 16, -12.f, 12.f);
    const ncnn::Mat a3 = RandomMat(3, 5, 7, 5);
    const ncnn::Mat b3 = RandomMat(3, 5, 7, 5);

    // small limits stay within the default random range and exercise the clamp
    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a0, b0, 1.f, 0.f, 0.5f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 0.f, 10.f)
           || test_swiglu(a3, b3, 1.f, 0.5f, 0.f)
           || test_swiglu(a3, b3, 2.f, 0.f, 0.f)
           || test_swiglu(a3, b3, 1.702f, 1.f, 0.5f);
}

static int test_swiglu_1()
{
    const ncnn::Mat a0 = RandomMat(7, 9, 12);
    const ncnn::Mat b0 = RandomMat(7, 9, 12);
    const ncnn::Mat a1 = RandomMat(7, 9, 7, -12.f, 12.f);
    const ncnn::Mat b1 = RandomMat(7, 9, 7, -12.f, 12.f);
    const ncnn::Mat a2 = RandomMat(33, 5, 16, -12.f, 12.f);
    const ncnn::Mat b2 = RandomMat(33, 5, 16, -12.f, 12.f);
    const ncnn::Mat a3 = RandomMat(129, 3, 4);
    const ncnn::Mat b3 = RandomMat(129, 3, 4);

    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 0.f, 10.f)
           || test_swiglu(a3, b3, 1.f, 1.f, 0.f)
           || test_swiglu(a3, b3, 1.f, 1.f, 0.5f);
}

static int test_swiglu_2()
{
    const ncnn::Mat a0 = RandomMat(10, 24);
    const ncnn::Mat b0 = RandomMat(10, 24);
    const ncnn::Mat a1 = RandomMat(7, 17, -12.f, 12.f);
    const ncnn::Mat b1 = RandomMat(7, 17, -12.f, 12.f);
    const ncnn::Mat a2 = RandomMat(33, 12, -12.f, 12.f);
    const ncnn::Mat b2 = RandomMat(33, 12, -12.f, 12.f);
    const ncnn::Mat a3 = RandomMat(17, 15);
    const ncnn::Mat b3 = RandomMat(17, 15);
    const ncnn::Mat a4 = RandomMat(256, 300);
    const ncnn::Mat b4 = RandomMat(256, 300);

    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 0.f, 10.f)
           || test_swiglu(a3, b3, 1.f, 1.f, 0.f)
           || test_swiglu(a3, b3, 1.f, 1.f, 0.5f)
           || test_swiglu(a4, b4, 1.f, 0.f, 0.f, TEST_LAYER_ENABLE_THREADING);
}

static int test_swiglu_3()
{
    const ncnn::Mat a0 = RandomMat(128);
    const ncnn::Mat b0 = RandomMat(128);
    const ncnn::Mat a1 = RandomMat(124, -12.f, 12.f);
    const ncnn::Mat b1 = RandomMat(124, -12.f, 12.f);
    const ncnn::Mat a2 = RandomMat(127, -12.f, 12.f);
    const ncnn::Mat b2 = RandomMat(127, -12.f, 12.f);
    const ncnn::Mat a3 = RandomMat(17);
    const ncnn::Mat b3 = RandomMat(17);

    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 0.f, 10.f)
           || test_swiglu(a3, b3, 1.f, 0.25f, 0.f)
           || test_swiglu(a3, b3, 1.f, 0.25f, 0.5f);
}

static int test_swiglu_4()
{
    // 1d / 2d shapes that do not divide the pack size
    const ncnn::Mat a0 = RandomMat(129);
    const ncnn::Mat b0 = RandomMat(129);
    const ncnn::Mat a1 = RandomMat(65, 7, -12.f, 12.f);
    const ncnn::Mat b1 = RandomMat(65, 7, -12.f, 12.f);
    const ncnn::Mat a2 = RandomMat(31, 33, -12.f, 12.f);
    const ncnn::Mat b2 = RandomMat(31, 33, -12.f, 12.f);

    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 1.f, 10.f)
           || test_swiglu(a2, b2, 1.f, 1.f, 0.5f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_swiglu_0()
           || test_swiglu_1()
           || test_swiglu_2()
           || test_swiglu_3()
           || test_swiglu_4();
}
