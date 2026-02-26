// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_innerproduct(const ncnn::Mat& a, int outch, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, outch); // num_output
    pd.set(1, bias);  // bias_term
    pd.set(2, outch * a.w * a.h * a.c);

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * a.w * a.h * a.c);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer("InnerProduct", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct failed a.dims=%d a=(%d %d %d) outch=%d bias=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.c, outch, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_innerproduct_0()
{
    return 0
           || test_innerproduct(RandomMat(1, 3, 1), 1, 1)
           || test_innerproduct(RandomMat(3, 2, 2), 2, 0)
           || test_innerproduct(RandomMat(9, 3, 8), 7, 1)
           || test_innerproduct(RandomMat(2, 2, 8), 8, 0)
           || test_innerproduct(RandomMat(4, 3, 15), 8, 1)
           || test_innerproduct(RandomMat(6, 2, 16), 16, 0)
           || test_innerproduct(RandomMat(6, 2, 16), 7, 1)
           || test_innerproduct(RandomMat(6, 2, 5), 16, 1);
}

static int test_innerproduct_1()
{
    return 0
           || test_innerproduct(RandomMat(1, 1), 1, 1)
           || test_innerproduct(RandomMat(3, 2), 2, 0)
           || test_innerproduct(RandomMat(9, 8), 7, 1)
           || test_innerproduct(RandomMat(2, 8), 8, 0)
           || test_innerproduct(RandomMat(4, 15), 8, 1)
           || test_innerproduct(RandomMat(6, 16), 16, 0)
           || test_innerproduct(RandomMat(6, 16), 7, 1)
           || test_innerproduct(RandomMat(6, 5), 16, 1);
}

static int test_innerproduct_2()
{
    return 0
           || test_innerproduct(RandomMat(1), 1, 1)
           || test_innerproduct(RandomMat(2), 2, 0)
           || test_innerproduct(RandomMat(8), 7, 1)
           || test_innerproduct(RandomMat(8), 8, 0)
           || test_innerproduct(RandomMat(15), 8, 1)
           || test_innerproduct(RandomMat(15), 15, 1)
           || test_innerproduct(RandomMat(16), 16, 0)
           || test_innerproduct(RandomMat(16), 7, 1)
           || test_innerproduct(RandomMat(5), 16, 0)
           || test_innerproduct(RandomMat(32), 16, 1)
           || test_innerproduct(RandomMat(12), 16, 0)
           || test_innerproduct(RandomMat(16), 12, 1)
           || test_innerproduct(RandomMat(24), 32, 1);
}

#if NCNN_INT8
static int test_innerproduct_int8(const ncnn::Mat& a, int outch, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, outch); // num_output
    pd.set(1, bias);  // bias_term
    pd.set(2, outch * a.w * a.h * a.c);
    pd.set(8, 1); // int8_scale_term

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 4 : 3);
    const int k = a.w * a.h * a.c;
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
    int ret = test_layer("InnerProduct", pd, weights, a, 0.001f, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_int8 failed a.dims=%d a=(%d %d %d) outch=%d bias=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.c, outch, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_innerproduct_3()
{
    return 0
           || test_innerproduct_int8(RandomMat(1, 3, 1), 1, 1)
           || test_innerproduct_int8(RandomMat(3, 2, 2), 2, 1)
           || test_innerproduct_int8(RandomMat(5, 3, 3), 3, 1)
           || test_innerproduct_int8(RandomMat(7, 2, 3), 12, 1)
           || test_innerproduct_int8(RandomMat(9, 3, 4), 4, 1)
           || test_innerproduct_int8(RandomMat(2, 2, 7), 7, 1)
           || test_innerproduct_int8(RandomMat(4, 3, 8), 3, 1)
           || test_innerproduct_int8(RandomMat(6, 2, 8), 8, 1)
           || test_innerproduct_int8(RandomMat(8, 3, 15), 15, 1)
           || test_innerproduct_int8(RandomMat(7, 2, 16), 4, 1)
           || test_innerproduct_int8(RandomMat(6, 3, 16), 16, 1);
}
#endif // NCNN_INT8

static int test_innerproduct_gemm(const ncnn::Mat& a, int outch, int bias)
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

    int ret = test_layer("InnerProduct", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_gemm failed a.dims=%d a=(%d %d %d) outch=%d bias=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.c, outch, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_innerproduct_4()
{
    return 0
           || test_innerproduct_gemm(RandomMat(1, 1), 1, 1)
           || test_innerproduct_gemm(RandomMat(48, 1), 11, 1)
           || test_innerproduct_gemm(RandomMat(1, 5), 1, 1)
           || test_innerproduct_gemm(RandomMat(3, 2), 2, 0)
           || test_innerproduct_gemm(RandomMat(9, 8), 7, 1)
           || test_innerproduct_gemm(RandomMat(2, 8), 8, 0)
           || test_innerproduct_gemm(RandomMat(13, 20), 8, 1)
           || test_innerproduct_gemm(RandomMat(16, 20), 16, 0)
           || test_innerproduct_gemm(RandomMat(11, 24), 8, 0)
           || test_innerproduct_gemm(RandomMat(13, 24), 12, 1)
           || test_innerproduct_gemm(RandomMat(15, 20), 20, 1)
           || test_innerproduct_gemm(RandomMat(16, 20), 11, 1)
           || test_innerproduct_gemm(RandomMat(19, 16), 16, 1)
           || test_innerproduct_gemm(RandomMat(15, 15), 15, 1)
           || test_innerproduct_gemm(RandomMat(14, 15), 8, 1)
           || test_innerproduct_gemm(RandomMat(17, 15), 12, 1)
           || test_innerproduct_gemm(RandomMat(12, 16), 7, 1)
           || test_innerproduct_gemm(RandomMat(11, 32), 32, 1)
           || test_innerproduct_gemm(RandomMat(12, 32), 24, 1)
           || test_innerproduct_gemm(RandomMat(13, 32), 12, 1)
           || test_innerproduct_gemm(RandomMat(14, 32), 14, 1)
           || test_innerproduct_gemm(RandomMat(15, 32), 32, 1)
           || test_innerproduct_gemm(RandomMat(16, 24), 32, 1)
           || test_innerproduct_gemm(RandomMat(17, 20), 32, 1)
           || test_innerproduct_gemm(RandomMat(18, 14), 32, 1);
}

#if NCNN_INT8
static int test_innerproduct_gemm_int8(const ncnn::Mat& a, int outch, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, bias);
    pd.set(2, outch * a.w);
    pd.set(8, 1); // int8_scale_term

    std::vector<ncnn::Mat> weights(bias ? 4 : 3);
    const int k = a.w;
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
    int ret = test_layer("InnerProduct", pd, weights, a, 0.001f, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_gemm_int8 failed a.dims=%d a=(%d %d %d) outch=%d bias=%d\n", a.dims, a.w, a.h, a.c, outch, bias);
    }

    return ret;
}

static int test_innerproduct_5()
{
    return 0
           || test_innerproduct_gemm_int8(RandomMat(1, 5), 1, 1)
           || test_innerproduct_gemm_int8(RandomMat(3, 2), 2, 0)
           || test_innerproduct_gemm_int8(RandomMat(9, 8), 7, 1)
           || test_innerproduct_gemm_int8(RandomMat(2, 8), 8, 0)
           || test_innerproduct_gemm_int8(RandomMat(13, 12), 8, 1)
           || test_innerproduct_gemm_int8(RandomMat(16, 12), 16, 0)
           || test_innerproduct_gemm_int8(RandomMat(4, 15), 8, 1)
           || test_innerproduct_gemm_int8(RandomMat(6, 16), 16, 0)
           || test_innerproduct_gemm_int8(RandomMat(12, 16), 7, 1);
}
#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return 0
           || test_innerproduct_0()
           || test_innerproduct_1()
           || test_innerproduct_2()
           || test_innerproduct_3()
           || test_innerproduct_4()
           || test_innerproduct_5();
#else
    return 0
           || test_innerproduct_0()
           || test_innerproduct_1()
           || test_innerproduct_2()
           || test_innerproduct_4();
#endif
}
