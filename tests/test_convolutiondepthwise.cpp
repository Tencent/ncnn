// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_convolutiondepthwise(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int group)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outch / group * c / group * kernel * kernel * group);
    pd.set(7, group);

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(outch / group * c / group * kernel * kernel * group);
    weights[1] = RandomMat(outch);

    int ret = test_layer("ConvolutionDepthWise", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolutiondepthwise failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, group, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolutiondepthwise_0()
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
                  || test_convolutiondepthwise(15, 7, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise(15, 7, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise(15, 7, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise(15, 7, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise(15, 7, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise(15, 7, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise(15, 7, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise(15, 7, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise(15, 7, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise(15, 7, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise(15, 7, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise(15, 7, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise(15, 7, 16, 16, k, d, s, p, 1, 16)
                  || test_convolutiondepthwise(18, 17, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise(18, 17, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise(18, 17, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise(18, 17, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise(18, 17, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise(18, 17, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise(18, 17, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise(18, 17, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise(18, 17, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise(18, 17, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise(18, 17, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise(18, 17, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise(18, 17, 16, 16, k, d, s, p, 1, 16)
                  || test_convolutiondepthwise(25, 33, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise(25, 33, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise(25, 33, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise(25, 33, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise(25, 33, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise(25, 33, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise(25, 33, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise(25, 33, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise(25, 33, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise(25, 33, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise(25, 33, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise(25, 33, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise(25, 33, 16, 16, k, d, s, p, 1, 16);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_convolutiondepthwise_load_param_activation(const ncnn::ParamDict& base, int activation_type, const ncnn::Mat& activation_params, int expected_ret)
{
    ncnn::ParamDict pd = base;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    return test_layer_param(ncnn::LayerType::ConvolutionDepthWise, pd, expected_ret);
}

static int test_convolutiondepthwise_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 8);
    base.set(1, 3);
    base.set(6, 216);
    base.set(7, 2);
    if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 0) != 0)
        return -1;

    ncnn::Mat params(2);
    params[0] = 0.1f;
    params[1] = 0.5f;

    int ret = 0
              || test_convolutiondepthwise_load_param_activation(base, 0, ncnn::Mat(), 0)
              || test_convolutiondepthwise_load_param_activation(base, 0, ncnn::Mat(0), 0)
              || test_convolutiondepthwise_load_param_activation(base, 0, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 0, params.range(0, 1), 0)
              || test_convolutiondepthwise_load_param_activation(base, 1, ncnn::Mat(), 0)
              || test_convolutiondepthwise_load_param_activation(base, 1, ncnn::Mat(0), 0)
              || test_convolutiondepthwise_load_param_activation(base, 1, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 1, params.range(0, 1), 0)
              || test_convolutiondepthwise_load_param_activation(base, 2, ncnn::Mat(), -1)
              || test_convolutiondepthwise_load_param_activation(base, 2, ncnn::Mat(0), -1)
              || test_convolutiondepthwise_load_param_activation(base, 2, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 2, params.range(0, 1), 0)
              || test_convolutiondepthwise_load_param_activation(base, 3, ncnn::Mat(), -1)
              || test_convolutiondepthwise_load_param_activation(base, 3, ncnn::Mat(0), -1)
              || test_convolutiondepthwise_load_param_activation(base, 3, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 3, params.range(0, 1), -1)
              || test_convolutiondepthwise_load_param_activation(base, 4, ncnn::Mat(), 0)
              || test_convolutiondepthwise_load_param_activation(base, 4, ncnn::Mat(0), 0)
              || test_convolutiondepthwise_load_param_activation(base, 4, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 4, params.range(0, 1), 0)
              || test_convolutiondepthwise_load_param_activation(base, 5, ncnn::Mat(), 0)
              || test_convolutiondepthwise_load_param_activation(base, 5, ncnn::Mat(0), 0)
              || test_convolutiondepthwise_load_param_activation(base, 5, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 5, params.range(0, 1), 0)
              || test_convolutiondepthwise_load_param_activation(base, 6, ncnn::Mat(), -1)
              || test_convolutiondepthwise_load_param_activation(base, 6, ncnn::Mat(0), -1)
              || test_convolutiondepthwise_load_param_activation(base, 6, params, 0)
              || test_convolutiondepthwise_load_param_activation(base, 6, params.range(0, 1), -1)
              || test_convolutiondepthwise_load_param_activation(base, 7, ncnn::Mat(), -1);
    if (ret != 0)
        return ret;

    if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 10, 1, -1)
            || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 10, 1.f, -1))
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    const ncnn::Mat bad[] = {ncnn::Mat(2, (size_t)1u), ncnn::Mat(2, (size_t)2u), ncnn::Mat(2, 2), ncnn::Mat(2, (size_t)16u, 4), missing_data};
    for (int i = 0; i < 5; i++)
    {
        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 10, bad[i], -1) != 0)
            return -1;
    }

    ret = 0
          || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 0, 0, -1)
          || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 1, 0, -1)
          || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 2, 0, -1)
          || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 3, 0, -1)
          || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 1, INT_MAX, -1)
          || test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 6, 217, -1);
    if (ret != 0)
        return ret;

    const int groups[] = {0, -1, -8, INT_MIN, 3};
    for (int i = 0; i < 5; i++)
    {
        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 7, groups[i], -1) != 0)
            return -1;
    }

    {
        ncnn::ParamDict pd = base;
        pd.set(0, INT_MIN);
        pd.set(7, -1); // avoid evaluating INT_MIN % -1
        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, pd, -1) != 0)
            return -1;
    }

    return 0;
}

static int test_convolutiondepthwise_load_param_int8()
{
    ncnn::ParamDict base;
    base.set(0, 1);
    base.set(1, 1);
    base.set(6, 1);
    if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 0) != 0)
        return -1;

    const int valid[] = {1, 2, 101, 102};
    for (int i = 0; i < 4; i++)
    {
#if NCNN_INT8
        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 8, valid[i], 0) != 0)
#else
        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 8, valid[i], -1) != 0)
#endif
            return -1;
    }

    const int invalid[] = {-1, 3, 100, 103, INT_MIN, INT_MAX};
    for (int i = 0; i < 6; i++)
    {
        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, base, 8, invalid[i], -1) != 0)
            return -1;
    }

    return 0;
}

static int test_convolutiondepthwise_load_param_text()
{
#if NCNN_STRING
    const char* params[] = {"0=1 1=1 6=1 9=3 -23310=2,-1,2", "0=1 1=1 6=1 9=3 -23310=2,-1.0,2.0"};
    for (int i = 0; i < 2; i++)
    {
        TestParamDict pd;
        if (pd.load_param(params[i]) != 0 || pd.type(10) != 5 + i)
            return -1;

        if (test_layer_param(ncnn::LayerType::ConvolutionDepthWise, pd, 0) != 0)
            return -1;

        std::vector<ncnn::Mat> weights(1);
        weights[0].create(1);
        weights[0][0] = 1.f;

        ncnn::Mat a(2, 1, 1);
        a[0] = -3.f;
        a[1] = 3.f;
        ncnn::Mat reference(2, 1, 1);
        reference[0] = -1.f;
        reference[1] = 2.f;
        ncnn::Mat b;
        int ret = test_layer_naive(ncnn::LayerType::ConvolutionDepthWise, pd, weights, a, b, 0);
        if (ret == 0)
            ret = CompareMat(reference, b, 0.f);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolutiondepthwise_load_param_text failed params=%s ret=%d\n", params[i], ret);
            return ret;
        }

        const ncnn::Mat original = pd.get(10, ncnn::Mat());
        const int* p = original;
        if (i == 0 ? (p[0] != -1 || p[1] != 2) : (original[0] != -1.f || original[1] != 2.f))
        {
            fprintf(stderr, "test_convolutiondepthwise_load_param_text modified params=%s\n", params[i]);
            return -1;
        }
    }
#endif
    return 0;
}

int main()
{
    SRAND(7767517);

    return test_convolutiondepthwise_0() || test_convolutiondepthwise_load_param() || test_convolutiondepthwise_load_param_text() || test_convolutiondepthwise_load_param_int8();
}
