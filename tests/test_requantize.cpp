// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

static int test_requantize(const ncnn::Mat& a, int scale_in_data_size, int scale_out_data_size, int bias_data_size, int activation_type, float alpha, float beta)
{
    ncnn::ParamDict pd;
    pd.set(0, scale_in_data_size);
    pd.set(1, scale_out_data_size);
    pd.set(2, bias_data_size);

    ncnn::Mat activation_params(2);
    activation_params[0] = alpha;
    activation_params[1] = beta;
    pd.set(3, activation_type);
    pd.set(4, activation_params);

    std::vector<ncnn::Mat> weights(bias_data_size ? 3 : 2);
    weights[0] = RandomMat(scale_in_data_size);
    weights[1] = RandomMat(scale_out_data_size);
    if (bias_data_size)
        weights[2] = RandomMat(bias_data_size);

    Randomize(weights[0], 0.0001, 0.001);
    Randomize(weights[1], 10, 100);

    int flag = TEST_LAYER_DISABLE_AUTO_INPUT_CASTING;
    int ret = test_layer("Requantize", pd, weights, a, 1, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_requantize failed a.dims=%d a=(%d %d %d %d) scale_in_data_size=%d scale_out_data_size=%d bias_data_size=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.d, a.c, scale_in_data_size, scale_out_data_size, bias_data_size, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_requantize(const ncnn::Mat& a, int scale_in_data_size, int scale_out_data_size, int bias_data_size)
{
    return 0
           || test_requantize(a, scale_in_data_size, scale_out_data_size, bias_data_size, 0, 0.f, 0.f)
           || test_requantize(a, scale_in_data_size, scale_out_data_size, bias_data_size, 1, 0.f, 0.f)
           || test_requantize(a, scale_in_data_size, scale_out_data_size, bias_data_size, 2, RandomFloat(0, 1), 0.f)
           || test_requantize(a, scale_in_data_size, scale_out_data_size, bias_data_size, 3, RandomFloat(-1, 0), RandomFloat(0, 1))
           || test_requantize(a, scale_in_data_size, scale_out_data_size, bias_data_size, 4, 0.f, 0.f)
           || test_requantize(a, scale_in_data_size, scale_out_data_size, bias_data_size, 5, 0.f, 0.f);
}

static int test_requantize_relu_empty_activation_params(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;
    pd.set(0, 1);
    pd.set(1, 1);
    pd.set(2, 0);
    pd.set(3, 1);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(1);
    weights[1] = RandomMat(1);

    Randomize(weights[0], 0.0001, 0.001);
    Randomize(weights[1], 10, 100);

    int flag = TEST_LAYER_DISABLE_AUTO_INPUT_CASTING | TEST_LAYER_DISABLE_AUTO_INPUT_PACKING;
    int ret = test_layer("Requantize", pd, weights, a, 1, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_requantize_relu_empty_activation_params failed a.dims=%d a=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
    }

    return ret;
}

static int test_requantize_pack1(const ncnn::Mat& a, int scale_in_data_size, int scale_out_data_size, int bias_data_size, int activation_type, float alpha, float beta)
{
    ncnn::ParamDict pd;
    pd.set(0, scale_in_data_size);
    pd.set(1, scale_out_data_size);
    pd.set(2, bias_data_size);

    ncnn::Mat activation_params(2);
    activation_params[0] = alpha;
    activation_params[1] = beta;
    pd.set(3, activation_type);
    pd.set(4, activation_params);

    std::vector<ncnn::Mat> weights(bias_data_size ? 3 : 2);
    weights[0] = RandomMat(scale_in_data_size);
    weights[1] = RandomMat(scale_out_data_size);
    if (bias_data_size)
        weights[2] = RandomMat(bias_data_size);

    Randomize(weights[0], 0.0001, 0.001);
    Randomize(weights[1], 10, 100);

    int flag = TEST_LAYER_DISABLE_AUTO_INPUT_CASTING | TEST_LAYER_DISABLE_AUTO_INPUT_PACKING;
    int ret = test_layer("Requantize", pd, weights, a, 1, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_requantize_pack1 failed a.dims=%d a=(%d %d %d %d) scale_in_data_size=%d scale_out_data_size=%d bias_data_size=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.d, a.c, scale_in_data_size, scale_out_data_size, bias_data_size, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_requantize_pack1(const ncnn::Mat& a, int scale_in_data_size, int scale_out_data_size, int bias_data_size)
{
    return 0
           || test_requantize_pack1(a, scale_in_data_size, scale_out_data_size, bias_data_size, 0, 0.f, 0.f)
           || test_requantize_pack1(a, scale_in_data_size, scale_out_data_size, bias_data_size, 1, 0.f, 0.f)
           || test_requantize_pack1(a, scale_in_data_size, scale_out_data_size, bias_data_size, 2, RandomFloat(0, 1), 0.f)
           || test_requantize_pack1(a, scale_in_data_size, scale_out_data_size, bias_data_size, 3, RandomFloat(-1, 0), RandomFloat(0, 1))
           || test_requantize_pack1(a, scale_in_data_size, scale_out_data_size, bias_data_size, 4, 0.f, 0.f)
           || test_requantize_pack1(a, scale_in_data_size, scale_out_data_size, bias_data_size, 5, 0.f, 0.f);
}

static int test_requantize_pack8(const ncnn::Mat& a, int scale_in_data_size, int scale_out_data_size, int bias_data_size, int activation_type, float alpha, float beta)
{
    ncnn::ParamDict pd;
    pd.set(0, scale_in_data_size);
    pd.set(1, scale_out_data_size);
    pd.set(2, bias_data_size);

    ncnn::Mat activation_params(2);
    activation_params[0] = alpha;
    activation_params[1] = beta;
    pd.set(3, activation_type);
    pd.set(4, activation_params);

    std::vector<ncnn::Mat> weights(bias_data_size ? 3 : 2);
    weights[0] = RandomMat(scale_in_data_size);
    weights[1] = RandomMat(scale_out_data_size);
    if (bias_data_size)
        weights[2] = RandomMat(bias_data_size);

    Randomize(weights[0], 0.0001, 0.001);
    Randomize(weights[1], 10, 100);

    int flag = TEST_LAYER_DISABLE_AUTO_INPUT_CASTING;
    // ordinary cases cover dims=1/2/3; retain the packed dims=4 cases
    if (a.dims != 4)
        flag |= TEST_LAYER_DISABLE_GPU_TESTING;
#if !__riscv
    flag |= TEST_LAYER_ENABLE_FORCE_INPUT_PACK8;
#endif
    int ret = test_layer("Requantize", pd, weights, a, 1, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_requantize_pack8 failed a.dims=%d a=(%d %d %d %d) scale_in_data_size=%d scale_out_data_size=%d bias_data_size=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.d, a.c, scale_in_data_size, scale_out_data_size, bias_data_size, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_requantize_pack8(const ncnn::Mat& a, int scale_in_data_size, int scale_out_data_size, int bias_data_size)
{
    return 0
           || test_requantize_pack8(a, scale_in_data_size, scale_out_data_size, bias_data_size, 0, 0.f, 0.f)
           || test_requantize_pack8(a, scale_in_data_size, scale_out_data_size, bias_data_size, 1, 0.f, 0.f)
           || test_requantize_pack8(a, scale_in_data_size, scale_out_data_size, bias_data_size, 2, RandomFloat(0, 1), 0.f)
           || test_requantize_pack8(a, scale_in_data_size, scale_out_data_size, bias_data_size, 3, RandomFloat(-1, 0), RandomFloat(0, 1))
           || test_requantize_pack8(a, scale_in_data_size, scale_out_data_size, bias_data_size, 4, 0.f, 0.f)
           || test_requantize_pack8(a, scale_in_data_size, scale_out_data_size, bias_data_size, 5, 0.f, 0.f);
}

static int test_requantize_0()
{
    return 0
           || test_requantize_pack1(RandomIntMat(7, 9, 12), 1, 1, 12)
           || test_requantize_pack1(RandomIntMat(7, 9, 12), 1, 1, 1)
           || test_requantize_pack1(RandomIntMat(7, 9, 12), 1, 1, 0)
           || test_requantize_pack1(RandomIntMat(7, 9, 12), 12, 12, 12)
           || test_requantize_pack1(RandomIntMat(7, 9, 12), 12, 12, 1)
           || test_requantize_pack1(RandomIntMat(7, 9, 12), 12, 12, 0)
           || test_requantize_pack1(RandomIntMat(7, 9, 12), 1, 12, 12)
           || test_requantize_pack1(RandomIntMat(7, 9, 12), 1, 12, 1)
           || test_requantize_pack1(RandomIntMat(7, 9, 12), 1, 12, 0)
           || test_requantize_pack1(RandomIntMat(7, 9, 12), 12, 1, 12)
           || test_requantize_pack1(RandomIntMat(7, 9, 12), 12, 1, 1)
           || test_requantize_pack1(RandomIntMat(7, 9, 12), 12, 1, 0)
           || test_requantize_pack1(RandomIntMat(3, 5, 13), 1, 1, 13)
           || test_requantize_pack1(RandomIntMat(3, 5, 13), 1, 1, 1)
           || test_requantize_pack1(RandomIntMat(3, 5, 13), 1, 1, 0)
           || test_requantize_pack1(RandomIntMat(3, 5, 13), 13, 13, 13)
           || test_requantize_pack1(RandomIntMat(3, 5, 13), 13, 13, 1)
           || test_requantize_pack1(RandomIntMat(3, 5, 13), 13, 13, 0)
           || test_requantize_pack1(RandomIntMat(3, 5, 13), 1, 13, 13)
           || test_requantize_pack1(RandomIntMat(3, 5, 13), 1, 13, 1)
           || test_requantize_pack1(RandomIntMat(3, 5, 13), 1, 13, 0)
           || test_requantize_pack1(RandomIntMat(3, 5, 13), 13, 1, 13)
           || test_requantize_pack1(RandomIntMat(3, 5, 13), 13, 1, 1)
           || test_requantize_pack1(RandomIntMat(3, 5, 13), 13, 1, 0);
}

static int test_requantize_1()
{
    return 0
           || test_requantize_pack1(RandomIntMat(17, 12), 1, 1, 12)
           || test_requantize_pack1(RandomIntMat(17, 12), 1, 1, 1)
           || test_requantize_pack1(RandomIntMat(17, 12), 1, 1, 0)
           || test_requantize_pack1(RandomIntMat(17, 12), 12, 12, 12)
           || test_requantize_pack1(RandomIntMat(17, 12), 12, 12, 1)
           || test_requantize_pack1(RandomIntMat(17, 12), 12, 12, 0)
           || test_requantize_pack1(RandomIntMat(17, 12), 1, 12, 12)
           || test_requantize_pack1(RandomIntMat(17, 12), 1, 12, 1)
           || test_requantize_pack1(RandomIntMat(17, 12), 1, 12, 0)
           || test_requantize_pack1(RandomIntMat(17, 12), 12, 1, 12)
           || test_requantize_pack1(RandomIntMat(17, 12), 12, 1, 1)
           || test_requantize_pack1(RandomIntMat(17, 12), 12, 1, 0)
           || test_requantize_pack1(RandomIntMat(19, 15), 1, 1, 15)
           || test_requantize_pack1(RandomIntMat(19, 15), 1, 1, 1)
           || test_requantize_pack1(RandomIntMat(19, 15), 1, 1, 0)
           || test_requantize_pack1(RandomIntMat(19, 15), 15, 15, 15)
           || test_requantize_pack1(RandomIntMat(19, 15), 15, 15, 1)
           || test_requantize_pack1(RandomIntMat(19, 15), 15, 15, 0)
           || test_requantize_pack1(RandomIntMat(19, 15), 1, 15, 15)
           || test_requantize_pack1(RandomIntMat(19, 15), 1, 15, 1)
           || test_requantize_pack1(RandomIntMat(19, 15), 1, 15, 0)
           || test_requantize_pack1(RandomIntMat(19, 15), 15, 1, 15)
           || test_requantize_pack1(RandomIntMat(19, 15), 15, 1, 1)
           || test_requantize_pack1(RandomIntMat(19, 15), 15, 1, 0);
}

static int test_requantize_2()
{
    return 0
           || test_requantize_pack1(RandomIntMat(124), 1, 1, 1)
           || test_requantize_pack1(RandomIntMat(124), 1, 1, 0)
           || test_requantize_pack1(RandomIntMat(127), 1, 1, 1)
           || test_requantize_pack1(RandomIntMat(127), 1, 1, 0)
           || test_requantize_pack1(RandomIntMat(127), 1, 1, 0, 2, 0.f, 0.f)
           || test_requantize_pack1(RandomIntMat(127), 1, 1, 0, 2, RandomFloat(-1, 0), 0.f)
           || test_requantize_relu_empty_activation_params(RandomIntMat(127));
}

static int test_requantize_3()
{
    return 0
#ifndef __riscv
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 1, 1, 24)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 1, 1, 1)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 1, 1, 0)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 24, 24, 24)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 24, 24, 1)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 24, 24, 0)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 1, 24, 24)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 1, 24, 1)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 1, 24, 0)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 24, 1, 24)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 24, 1, 1)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 24, 1, 0)
           || test_requantize_pack8(RandomIntMat(15, 24), 1, 1, 24)
           || test_requantize_pack8(RandomIntMat(15, 24), 1, 1, 1)
           || test_requantize_pack8(RandomIntMat(15, 24), 1, 1, 0)
           || test_requantize_pack8(RandomIntMat(15, 24), 24, 24, 24)
           || test_requantize_pack8(RandomIntMat(15, 24), 24, 24, 1)
           || test_requantize_pack8(RandomIntMat(15, 24), 24, 24, 0)
           || test_requantize_pack8(RandomIntMat(15, 24), 1, 24, 24)
           || test_requantize_pack8(RandomIntMat(15, 24), 1, 24, 1)
           || test_requantize_pack8(RandomIntMat(15, 24), 1, 24, 0)
           || test_requantize_pack8(RandomIntMat(15, 24), 24, 1, 24)
           || test_requantize_pack8(RandomIntMat(15, 24), 24, 1, 1)
           || test_requantize_pack8(RandomIntMat(15, 24), 24, 1, 0)
           || test_requantize_pack8(RandomIntMat(128), 1, 1, 1)
           || test_requantize_pack8(RandomIntMat(128), 1, 1, 0);
#else
           || test_requantize(RandomIntMat(5, 7, 24), 1, 1, 24)
           || test_requantize(RandomIntMat(5, 7, 24), 1, 1, 1)
           || test_requantize(RandomIntMat(5, 7, 24), 1, 1, 0)
           || test_requantize(RandomIntMat(5, 7, 24), 24, 24, 24)
           || test_requantize(RandomIntMat(5, 7, 24), 24, 24, 1)
           || test_requantize(RandomIntMat(5, 7, 24), 24, 24, 0)
           || test_requantize(RandomIntMat(5, 7, 24), 1, 24, 24)
           || test_requantize(RandomIntMat(5, 7, 24), 1, 24, 1)
           || test_requantize(RandomIntMat(5, 7, 24), 1, 24, 0)
           || test_requantize(RandomIntMat(5, 7, 24), 24, 1, 24)
           || test_requantize(RandomIntMat(5, 7, 24), 24, 1, 1)
           || test_requantize(RandomIntMat(5, 7, 24), 24, 1, 0)
           || test_requantize(RandomIntMat(15, 24), 1, 1, 24)
           || test_requantize(RandomIntMat(15, 24), 1, 1, 1)
           || test_requantize(RandomIntMat(15, 24), 1, 1, 0)
           || test_requantize(RandomIntMat(15, 24), 24, 24, 24)
           || test_requantize(RandomIntMat(15, 24), 24, 24, 1)
           || test_requantize(RandomIntMat(15, 24), 24, 24, 0)
           || test_requantize(RandomIntMat(15, 24), 1, 24, 24)
           || test_requantize(RandomIntMat(15, 24), 1, 24, 1)
           || test_requantize(RandomIntMat(15, 24), 1, 24, 0)
           || test_requantize(RandomIntMat(15, 24), 24, 1, 24)
           || test_requantize(RandomIntMat(15, 24), 24, 1, 1)
           || test_requantize(RandomIntMat(15, 24), 24, 1, 0)
           || test_requantize(RandomIntMat(128), 1, 1, 1)
           || test_requantize(RandomIntMat(128), 1, 1, 0)
           || test_requantize(RandomIntMat(127), 1, 1, 0, 2, RandomFloat(1, 2), 0.f);
#endif // __riscv
}

static int test_requantize_4()
{
    return 0
           || test_requantize_pack1(RandomIntMat(5, 3, 2, 12), 1, 1, 12)
           || test_requantize_pack1(RandomIntMat(5, 3, 2, 12), 12, 12, 0)
           || test_requantize_pack1(RandomIntMat(3, 5, 3, 13), 1, 13, 13)
           || test_requantize_pack1(RandomIntMat(3, 5, 3, 13), 13, 1, 0)
           || test_requantize_pack8(RandomIntMat(5, 3, 2, 24), 1, 1, 24)
           || test_requantize_pack8(RandomIntMat(5, 3, 2, 24), 24, 24, 0);
}

static int test_requantize_activation_params_text()
{
#if NCNN_STRING
    const char* params[] = {"0=1 1=1 3=3 -23304=2,-1,2", "0=1 1=1 3=3 -23304=2,-1.0,2.0"};
    for (int i = 0; i < 2; i++)
    {
        TestParamDict pd;
        if (pd.load_param(params[i]) != 0 || pd.type(4) != 5 + i)
            return -1;

        std::vector<ncnn::Mat> weights(2);
        weights[0].create(1);
        weights[0][0] = 1.f;
        weights[1].create(1);
        weights[1][0] = 1.f;

        ncnn::Mat a(2);
        int* p = a;
        p[0] = -3;
        p[1] = 3;
        ncnn::Mat reference(2);
        reference[0] = -1.f;
        reference[1] = 2.f;
        ncnn::Mat b;
        int ret = test_layer_naive(ncnn::LayerType::Requantize, pd, weights, a, b, 0);
        if (ret == 0)
            ret = CompareMat(reference, b, 0.f);
        if (ret != 0)
        {
            fprintf(stderr, "test_requantize_activation_params_text failed params=%s ret=%d\n", params[i], ret);
            return ret;
        }

        const ncnn::Mat original = pd.get(4, ncnn::Mat());
        const int* q = original;
        if (i == 0 ? (q[0] != -1 || q[1] != 2) : (original[0] != -1.f || original[1] != 2.f))
        {
            fprintf(stderr, "test_requantize_activation_params_text modified params=%s\n", params[i]);
            return -1;
        }
    }
#endif
    return 0;
}

#if NCNN_VALIDATION
static int test_requantize_load_param_activation(const ncnn::ParamDict& base, int activation_type, const ncnn::Mat& activation_params, int expected_ret)
{
    ncnn::ParamDict pd = base;
    pd.set(3, activation_type);
    pd.set(4, activation_params);

    return test_layer_param(ncnn::LayerType::Requantize, pd, expected_ret);
}

static int test_requantize_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 1);
    base.set(1, 1);
    if (test_layer_param(ncnn::LayerType::Requantize, base, 0) != 0)
        return -1;

    ncnn::Mat params(2);
    params[0] = 0.1f;
    params[1] = 0.5f;

    int ret = 0
              || test_requantize_load_param_activation(base, 0, ncnn::Mat(), 0)
              || test_requantize_load_param_activation(base, 0, ncnn::Mat(0), 0)
              || test_requantize_load_param_activation(base, 0, params, 0)
              || test_requantize_load_param_activation(base, 0, params.range(0, 1), 0)
              || test_requantize_load_param_activation(base, 1, ncnn::Mat(), 0)
              || test_requantize_load_param_activation(base, 1, ncnn::Mat(0), 0)
              || test_requantize_load_param_activation(base, 1, params, 0)
              || test_requantize_load_param_activation(base, 1, params.range(0, 1), 0)
              || test_requantize_load_param_activation(base, 2, ncnn::Mat(), -1)
              || test_requantize_load_param_activation(base, 2, ncnn::Mat(0), -1)
              || test_requantize_load_param_activation(base, 2, params, 0)
              || test_requantize_load_param_activation(base, 2, params.range(0, 1), 0)
              || test_requantize_load_param_activation(base, 3, ncnn::Mat(), -1)
              || test_requantize_load_param_activation(base, 3, ncnn::Mat(0), -1)
              || test_requantize_load_param_activation(base, 3, params, 0)
              || test_requantize_load_param_activation(base, 3, params.range(0, 1), -1)
              || test_requantize_load_param_activation(base, 4, ncnn::Mat(), 0)
              || test_requantize_load_param_activation(base, 4, ncnn::Mat(0), 0)
              || test_requantize_load_param_activation(base, 4, params, 0)
              || test_requantize_load_param_activation(base, 4, params.range(0, 1), 0)
              || test_requantize_load_param_activation(base, 5, ncnn::Mat(), 0)
              || test_requantize_load_param_activation(base, 5, ncnn::Mat(0), 0)
              || test_requantize_load_param_activation(base, 5, params, 0)
              || test_requantize_load_param_activation(base, 5, params.range(0, 1), 0)
              || test_requantize_load_param_activation(base, 6, ncnn::Mat(), -1)
              || test_requantize_load_param_activation(base, 6, ncnn::Mat(0), -1)
              || test_requantize_load_param_activation(base, 6, params, 0)
              || test_requantize_load_param_activation(base, 6, params.range(0, 1), -1)
              || test_requantize_load_param_activation(base, 7, ncnn::Mat(), -1);
    if (ret != 0)
        return ret;

    if (test_layer_param(ncnn::LayerType::Requantize, base, 4, 1, -1)
            || test_layer_param(ncnn::LayerType::Requantize, base, 4, 1.f, -1))
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    const ncnn::Mat bad[] = {ncnn::Mat(2, (size_t)1u), ncnn::Mat(2, (size_t)2u), ncnn::Mat(2, 2), ncnn::Mat(2, (size_t)16u, 4), missing_data};
    for (int i = 0; i < 5; i++)
    {
        if (test_layer_param(ncnn::LayerType::Requantize, base, 4, bad[i], -1) != 0)
            return -1;
    }

    return 0;
}

static int test_requantize_load_param_text()
{
#if NCNN_STRING
    const char* params[] = {"0=1 1=1 3=3 -23304=2,-1,2", "0=1 1=1 3=3 -23304=2,-1.0,2.0"};
    for (int i = 0; i < 2; i++)
    {
        TestParamDict pd;
        if (pd.load_param(params[i]) != 0 || pd.type(4) != 5 + i)
            return -1;

        if (test_layer_param(ncnn::LayerType::Requantize, pd, 0) != 0)
            return -1;
    }
#endif
    return 0;
}
#endif // NCNN_VALIDATION

int main()
{
    SRAND(7767517);

    return 0
           || test_requantize_4()
           || test_requantize_0()
           || test_requantize_1()
           || test_requantize_2()
           || test_requantize_3()
           || test_requantize_activation_params_text()
#if NCNN_VALIDATION
           || test_requantize_load_param()
           || test_requantize_load_param_text()
#endif // NCNN_VALIDATION
           ;
}
