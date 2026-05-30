// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_innerproduct_oom(const ncnn::Mat& a, int outch, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, outch); // num_output
    pd.set(1, bias);  // bias_term
    pd.set(2, outch * a.w * a.h * a.d * a.c);

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * a.w * a.h * a.d * a.c);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer_oom("InnerProduct", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_oom failed a.dims=%d a=(%d %d %d %d) outch=%d bias=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.d, a.c, outch, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_innerproduct_0()
{
    return 0
           || test_innerproduct_oom(RandomMat(9, 3, 8), 7, 1)
           || test_innerproduct_oom(RandomMat(6, 2, 16), 16, 0);
}

static int test_innerproduct_1()
{
    return 0
           || test_innerproduct_oom(RandomMat(9, 8), 7, 1)
           || test_innerproduct_oom(RandomMat(6, 16), 16, 0);
}

static int test_innerproduct_2()
{
    return 0
           || test_innerproduct_oom(RandomMat(15), 8, 1)
           || test_innerproduct_oom(RandomMat(16), 16, 0);
}

#if NCNN_INT8
static int test_innerproduct_oom_int8(const ncnn::Mat& a, int outch, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, outch); // num_output
    pd.set(1, bias);  // bias_term
    pd.set(2, outch * a.w * a.h * a.d * a.c);
    pd.set(8, 1); // int8_scale_term

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 4 : 3);
    const int k = a.w * a.h * a.d * a.c;
    weights[0] = RandomMat(outch * k);
    ncnn::Mat weight_scales = scales_mat(weights[0], outch, k, k);
    ncnn::Mat input_scales = scales_mat(a, 1, k, k);

    if (bias)
    {
        weights[1] = RandomMat(outch);
        weights[2] = weight_scales;
        weights[3] = input_scales;
    }
    else
    {
        weights[1] = weight_scales;
        weights[2] = input_scales;
    }

    int flag = TEST_LAYER_DISABLE_GPU_TESTING;
    int ret = test_layer_oom("InnerProduct", pd, weights, a, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_oom_int8 failed a.dims=%d a=(%d %d %d %d) outch=%d bias=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.d, a.c, outch, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_innerproduct_3()
{
    return 0
           || test_innerproduct_oom_int8(RandomMat(8, 3, 15), 15, 1)
           || test_innerproduct_oom_int8(RandomMat(6, 3, 16), 16, 1);
}
#endif // NCNN_INT8

static int test_innerproduct_gemm_oom(const ncnn::Mat& a, int outch, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, bias);
    pd.set(2, outch * a.w);

    int activation_type = RAND() % 7;
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * a.w);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer_oom("InnerProduct", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_gemm_oom failed a.dims=%d a=(%d %d %d) outch=%d bias=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.c, outch, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_innerproduct_4()
{
    return 0
           || test_innerproduct_gemm_oom(RandomMat(9, 8), 7, 1)
           || test_innerproduct_gemm_oom(RandomMat(16, 20), 16, 0);
}

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return 0
           || test_innerproduct_0()
           || test_innerproduct_1()
           || test_innerproduct_2()
           || test_innerproduct_3()
           || test_innerproduct_4();
#else
    return 0
           || test_innerproduct_0()
           || test_innerproduct_1()
           || test_innerproduct_2()
           || test_innerproduct_4();
#endif
}
