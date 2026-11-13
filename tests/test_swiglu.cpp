// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include <float.h>

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
    const ncnn::Mat a1 = RandomMat(5, 7, 3, 7);
    const ncnn::Mat b1 = RandomMat(5, 7, 3, 7);
    const ncnn::Mat a2 = RandomMat(4, 9, 2, 16);
    const ncnn::Mat b2 = RandomMat(4, 9, 2, 16);
    const ncnn::Mat a3 = RandomMat(3, 5, 7, 5);
    const ncnn::Mat b3 = RandomMat(3, 5, 7, 5);

    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 0.f, 10.f)
           || test_swiglu(a3, b3, 1.f, 0.5f, 0.f)
           || test_swiglu(a3, b3, 2.f, 0.f, 0.f);
}

static int test_swiglu_4()
{
    // 1d / 2d shapes that do not divide the pack size
    const ncnn::Mat a0 = RandomMat(129);
    const ncnn::Mat b0 = RandomMat(129);
    const ncnn::Mat a1 = RandomMat(65, 7);
    const ncnn::Mat b1 = RandomMat(65, 7);
    const ncnn::Mat a2 = RandomMat(31, 33);
    const ncnn::Mat b2 = RandomMat(31, 33);

    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 1.f, 10.f);
}

static int test_swiglu_1()
{
    const ncnn::Mat a0 = RandomMat(7, 9, 12);
    const ncnn::Mat b0 = RandomMat(7, 9, 12);
    const ncnn::Mat a1 = RandomMat(7, 9, 7);
    const ncnn::Mat b1 = RandomMat(7, 9, 7);
    const ncnn::Mat a2 = RandomMat(33, 5, 16);
    const ncnn::Mat b2 = RandomMat(33, 5, 16);
    const ncnn::Mat a3 = RandomMat(129, 3, 4);
    const ncnn::Mat b3 = RandomMat(129, 3, 4);

    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 0.f, 10.f)
           || test_swiglu(a3, b3, 1.f, 1.f, 0.f);
}

static int test_swiglu_2()
{
    const ncnn::Mat a0 = RandomMat(10, 24);
    const ncnn::Mat b0 = RandomMat(10, 24);
    const ncnn::Mat a1 = RandomMat(7, 17);
    const ncnn::Mat b1 = RandomMat(7, 17);
    const ncnn::Mat a2 = RandomMat(33, 12);
    const ncnn::Mat b2 = RandomMat(33, 12);
    const ncnn::Mat a3 = RandomMat(17, 15);
    const ncnn::Mat b3 = RandomMat(17, 15);
    const ncnn::Mat a4 = RandomMat(256, 300);
    const ncnn::Mat b4 = RandomMat(256, 300);

    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 0.f, 10.f)
           || test_swiglu(a3, b3, 1.f, 1.f, 0.f)
           || test_swiglu(a4, b4, 1.f, 0.f, 0.f, TEST_LAYER_ENABLE_THREADING);
}

static int test_swiglu_3()
{
    const ncnn::Mat a0 = RandomMat(128);
    const ncnn::Mat b0 = RandomMat(128);
    const ncnn::Mat a1 = RandomMat(124);
    const ncnn::Mat b1 = RandomMat(124);
    const ncnn::Mat a2 = RandomMat(127);
    const ncnn::Mat b2 = RandomMat(127);
    const ncnn::Mat a3 = RandomMat(17);
    const ncnn::Mat b3 = RandomMat(17);

    return 0
           || test_swiglu(a0, b0, 1.f, 0.f, 0.f)
           || test_swiglu(a1, b1, 1.702f, 1.f, 7.f)
           || test_swiglu(a2, b2, 1.f, 0.f, 10.f)
           || test_swiglu(a3, b3, 1.f, 0.25f, 0.f);
}

static int test_swiglu_oracle(const char* name, float scale, float offset, float limit)
{
    ncnn::ParamDict pd;
    pd.set(0, scale);
    pd.set(1, offset);
    pd.set(2, limit);

    ncnn::Layer* op = ncnn::create_layer_naive("SwiGLU");
    if (!op)
    {
        fprintf(stderr, "create swiglu layer failed\n");
        return -1;
    }

    op->load_param(pd);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;

    op->create_pipeline(opt);

    const int size = 14;
    const float gate_data[size] = {-10.f, -7.f, -1.f, 0.f, 1.f, 3.f, 7.f, 10.f, 20.f, -0.5f, 0.25f, 88.f, -100.f, 100.f};
    const float up_data[size] = {-7.f, -1.f, 0.f, 1.f, 7.f, 10.f, -10.f, 0.5f, -0.25f, 3.f, -3.f, 100.f, 1.5f, -1.5f};

    ncnn::Mat gate(size);
    ncnn::Mat up(size);
    float* gateptr = gate;
    float* upptr = up;
    for (int i = 0; i < size; i++)
    {
        gateptr[i] = gate_data[i];
        upptr[i] = up_data[i];
    }

    std::vector<ncnn::Mat> as(2);
    as[0] = gate;
    as[1] = up;

    std::vector<ncnn::Mat> tops(1);
    int ret = op->forward(as, tops, opt);

    op->destroy_pipeline(opt);
    delete op;

    if (ret != 0 || tops[0].empty())
    {
        fprintf(stderr, "swiglu oracle forward failed ret=%d\n", ret);
        return -1;
    }

    const ncnn::Mat& out = tops[0];
    const float climit = limit > 0.f ? limit : FLT_MAX;
    const float mlimit = -climit;

    for (int i = 0; i < size; i++)
    {
        float g = gate_data[i];
        float u = up_data[i];
        g = std::min(g, climit);
        u = std::min(std::max(u, mlimit), climit);
        const float expected = g / (1.f + expf(-scale * g)) * (u + offset);

        if (!NearlyEqual(out[i], expected, 0.001f))
        {
            fprintf(stderr, "swiglu oracle %s failed i=%d expect %f but got %f\n", name, i, expected, out[i]);
            return -1;
        }
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_swiglu_0()
           || test_swiglu_1()
           || test_swiglu_2()
           || test_swiglu_3()
           || test_swiglu_4()
           || test_swiglu_oracle("standard", 1.f, 0.f, 0.f)
           || test_swiglu_oracle("gpt-oss", 1.702f, 1.f, 7.f)
           || test_swiglu_oracle("deepseek", 1.f, 0.f, 10.f);
}
