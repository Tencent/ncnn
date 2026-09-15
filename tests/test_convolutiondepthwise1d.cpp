// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_convolutiondepthwise1d(int w, int h, int outh, int kernel, int dilation, int stride, int pad, int bias, int group)
{
    ncnn::Mat a = RandomMat(w, h);

    ncnn::ParamDict pd;
    pd.set(0, outh);     // num_output
    pd.set(1, kernel);   // kernel_w
    pd.set(2, dilation); // dilation_w
    pd.set(3, stride);   // stride_w
    pd.set(4, pad);      // pad_w
    pd.set(5, bias);     // bias_term
    pd.set(6, outh / group * h / group * kernel * kernel * group);
    pd.set(7, group);

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(outh / group * h / group * kernel * kernel * group);
    weights[1] = RandomMat(outh);

    int ret = test_layer("ConvolutionDepthWise1D", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolutiondepthwise1d failed w=%d h=%d outh=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d act=%d actparams=[%f,%f]\n", w, h, outh, kernel, dilation, stride, pad, bias, group, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolutiondepthwise1d_0()
{
    static const int kdsp[16][4] = {
        {1, 1, 1, 0},
        {1, 1, 2, 0},
        {2, 1, 1, 1},
        {2, 1, 2, -233},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {3, 2, 1, 1},
        {4, 1, 1, 2},
        {4, 1, 2, -233},
        {4, 2, 1, -234},
        {5, 1, 1, -234},
        {5, 1, 2, 2},
        {5, 2, 2, 2},
        {7, 1, 1, 3},
        {7, 1, 2, 3},
        {7, 2, 1, -233},
    };

    for (int i = 0; i < 16; i++)
    {
        const int k = kdsp[i][0];
        const int d = kdsp[i][1];
        const int s = kdsp[i][2];
        const int p = kdsp[i][3];

        int ret = 0
                  || test_convolutiondepthwise1d(15, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise1d(15, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise1d(15, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise1d(15, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise1d(15, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise1d(15, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise1d(15, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise1d(15, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise1d(15, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise1d(15, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise1d(15, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise1d(15, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise1d(15, 16, 16, k, d, s, p, 1, 16)
                  || test_convolutiondepthwise1d(18, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise1d(18, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise1d(18, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise1d(18, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise1d(18, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise1d(18, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise1d(18, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise1d(18, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise1d(18, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise1d(18, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise1d(18, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise1d(18, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise1d(18, 16, 16, k, d, s, p, 1, 16)
                  || test_convolutiondepthwise1d(25, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise1d(25, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise1d(25, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise1d(25, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise1d(25, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise1d(25, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise1d(25, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise1d(25, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise1d(25, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise1d(25, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise1d(25, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise1d(25, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise1d(25, 16, 16, k, d, s, p, 1, 16);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_convolutiondepthwise1d_dynamic(int w, int h, int outh, int kernel, int dilation, int stride, int pad, int bias, int group)
{
    ncnn::Mat a = RandomMat(w, h);

    ncnn::ParamDict pd;
    pd.set(0, 0);
    pd.set(1, 0);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, 0);
    pd.set(7, group);
    pd.set(19, 1); // dynamic weight

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> as(bias ? 3 : 2);
    as[0] = a;
    as[1] = RandomMat(kernel, h / group, outh);
    if (bias)
        as[2] = RandomMat(outh);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("ConvolutionDepthWise1D", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolutiondepthwise1d_dynamic failed w=%d h=%d outh=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d act=%d actparams=[%f,%f]\n", w, h, outh, kernel, dilation, stride, pad, bias, group, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolutiondepthwise1d_1()
{
    static const int kdsp[7][4] = {
        {1, 1, 1, 0},
        {1, 1, 2, 0},
        {2, 1, 1, 1},
        {2, 1, 2, -233},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {3, 2, 1, -234},
    };

    for (int i = 0; i < 7; i++)
    {
        const int k = kdsp[i][0];
        const int d = kdsp[i][1];
        const int s = kdsp[i][2];
        const int p = kdsp[i][3];

        int ret = 0
                  || test_convolutiondepthwise1d_dynamic(11, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise1d_dynamic(11, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise1d_dynamic(11, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise1d_dynamic(11, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise1d_dynamic(11, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise1d_dynamic(11, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise1d_dynamic(11, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise1d_dynamic(11, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise1d_dynamic(11, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise1d_dynamic(11, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise1d_dynamic(11, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise1d_dynamic(11, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise1d_dynamic(11, 16, 16, k, d, s, p, 1, 16);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_convolutiondepthwise1d_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::ConvolutionDepthWise1D);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if (ret != (valid ? 0 : -1))
    {
        const int num_output = pd.get(0, 0);
        const int kernel_w = pd.get(1, 0);
        const int dilation_w = pd.get(2, 1);
        const int stride_w = pd.get(3, 1);
        const int weight_data_size = pd.get(6, 0);
        const int group = pd.get(7, 1);
        const int activation_type = pd.get(9, 0);
        const int dynamic_weight = pd.get(19, 0);

        fprintf(stderr, "test_convolutiondepthwise1d_load_param failed ret=%d expected=%d num_output=%d kernel_w=%d dilation_w=%d stride_w=%d weight_data_size=%d group=%d activation_type=%d dynamic_weight=%d\n", ret, valid ? 0 : -1, num_output, kernel_w, dilation_w, stride_w, weight_data_size, group, activation_type, dynamic_weight);

        const ncnn::Mat activation_params = pd.get(10, ncnn::Mat());
        fprintf(stderr, "activation_params type=%d dims=%d w=%d elemsize=%zu elempack=%d\n", pd.type(10), activation_params.dims, activation_params.w, activation_params.elemsize, activation_params.elempack);
        return -1;
    }

    return 0;
}

static int test_convolutiondepthwise1d_load_param_case(const ncnn::ParamDict& base, int id, int value, bool valid)
{
    ncnn::ParamDict pd = base;
    pd.set(id, value);

    int ret = test_convolutiondepthwise1d_load_param_case(pd, valid);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolutiondepthwise1d_load_param failed id=%d value=%d\n", id, value);
    }

    return ret;
}

static int test_convolutiondepthwise1d_load_param_activation(const ncnn::ParamDict& base, int activation_type, const ncnn::Mat& activation_params, bool valid)
{
    ncnn::ParamDict pd = base;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    return test_convolutiondepthwise1d_load_param_case(pd, valid);
}

static int test_convolutiondepthwise1d_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 8);
    base.set(1, 3);
    base.set(6, 216);
    base.set(7, 2);
    if (test_convolutiondepthwise1d_load_param_case(base, true) != 0)
        return -1;

    ncnn::Mat params(2);
    params[0] = 0.1f;
    params[1] = 0.5f;

    int ret = 0
              || test_convolutiondepthwise1d_load_param_activation(base, 0, ncnn::Mat(), true)
              || test_convolutiondepthwise1d_load_param_activation(base, 0, ncnn::Mat(0), true)
              || test_convolutiondepthwise1d_load_param_activation(base, 0, params, true)
              || test_convolutiondepthwise1d_load_param_activation(base, 0, params.range(0, 1), true)
              || test_convolutiondepthwise1d_load_param_activation(base, 1, ncnn::Mat(), true)
              || test_convolutiondepthwise1d_load_param_activation(base, 1, ncnn::Mat(0), true)
              || test_convolutiondepthwise1d_load_param_activation(base, 1, params, true)
              || test_convolutiondepthwise1d_load_param_activation(base, 1, params.range(0, 1), true)
              || test_convolutiondepthwise1d_load_param_activation(base, 2, ncnn::Mat(), false)
              || test_convolutiondepthwise1d_load_param_activation(base, 2, ncnn::Mat(0), false)
              || test_convolutiondepthwise1d_load_param_activation(base, 2, params, true)
              || test_convolutiondepthwise1d_load_param_activation(base, 2, params.range(0, 1), true)
              || test_convolutiondepthwise1d_load_param_activation(base, 3, ncnn::Mat(), false)
              || test_convolutiondepthwise1d_load_param_activation(base, 3, ncnn::Mat(0), false)
              || test_convolutiondepthwise1d_load_param_activation(base, 3, params, true)
              || test_convolutiondepthwise1d_load_param_activation(base, 3, params.range(0, 1), false)
              || test_convolutiondepthwise1d_load_param_activation(base, 4, ncnn::Mat(), true)
              || test_convolutiondepthwise1d_load_param_activation(base, 4, ncnn::Mat(0), true)
              || test_convolutiondepthwise1d_load_param_activation(base, 4, params, true)
              || test_convolutiondepthwise1d_load_param_activation(base, 4, params.range(0, 1), true)
              || test_convolutiondepthwise1d_load_param_activation(base, 5, ncnn::Mat(), true)
              || test_convolutiondepthwise1d_load_param_activation(base, 5, ncnn::Mat(0), true)
              || test_convolutiondepthwise1d_load_param_activation(base, 5, params, true)
              || test_convolutiondepthwise1d_load_param_activation(base, 5, params.range(0, 1), true)
              || test_convolutiondepthwise1d_load_param_activation(base, 6, ncnn::Mat(), false)
              || test_convolutiondepthwise1d_load_param_activation(base, 6, ncnn::Mat(0), false)
              || test_convolutiondepthwise1d_load_param_activation(base, 6, params, true)
              || test_convolutiondepthwise1d_load_param_activation(base, 6, params.range(0, 1), false)
              || test_convolutiondepthwise1d_load_param_activation(base, 7, ncnn::Mat(), false);
    if (ret != 0)
        return ret;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    const ncnn::Mat bad[] = {ncnn::Mat(2, (size_t)1u), ncnn::Mat(2, (size_t)2u), ncnn::Mat(2, 2), ncnn::Mat(2, (size_t)16u, 4), missing_data};
    for (int i = 0; i < 5; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(10, bad[i]);
        if (test_convolutiondepthwise1d_load_param_case(pd, false) != 0)
            return -1;
    }

    ret = 0
          || test_convolutiondepthwise1d_load_param_case(base, 0, 0, false)
          || test_convolutiondepthwise1d_load_param_case(base, 1, 0, false)
          || test_convolutiondepthwise1d_load_param_case(base, 2, 0, false)
          || test_convolutiondepthwise1d_load_param_case(base, 3, 0, false)
          || test_convolutiondepthwise1d_load_param_case(base, 1, INT_MAX, false)
          || test_convolutiondepthwise1d_load_param_case(base, 6, 217, false);
    if (ret != 0)
        return ret;

    const int groups[] = {0, -1, -8, INT_MIN, 3};
    for (int i = 0; i < 5; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(7, groups[i]);
        if (test_convolutiondepthwise1d_load_param_case(pd, false) != 0)
            return -1;
    }

    ncnn::ParamDict pd = base;
    pd.set(0, INT_MIN);
    pd.set(7, -1); // avoid evaluating INT_MIN % -1
    if (test_convolutiondepthwise1d_load_param_case(pd, false) != 0)
        return -1;

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_convolutiondepthwise1d_0() || test_convolutiondepthwise1d_1() || test_convolutiondepthwise1d_load_param();
}
