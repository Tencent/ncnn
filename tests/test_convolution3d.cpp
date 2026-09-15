// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_convolution3d(int w, int h, int d, int c, int outch, int kernel, int dilation, int stride, int pad, int bias)
{
    ncnn::Mat a = RandomMat(w, h, d, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);    // num_output
    pd.set(1, kernel);   // kernel_w
    pd.set(2, dilation); // dilation_w
    pd.set(3, stride);   // stride_w
    pd.set(4, pad);      // pad_w
    pd.set(5, bias);     // bias_term
    pd.set(6, outch * c * kernel * kernel * kernel);

    int activation_type = RAND() % 6; // 0 1 2 3 4 5
    ncnn::Mat activation_params(2);
    activation_params[0] = RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * c * kernel * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer("Convolution3D", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution3d failed w=%d h=%d d=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, d, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolution3d_0()
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
                  || test_convolution3d(11, 10, 9, 1, 1, k, d, s, p, 1)
                  || test_convolution3d(11, 10, 9, 4, 13, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 13, 4, k, d, s, p, 1)
                  || test_convolution3d(11, 10, 9, 12, 12, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 8, 12, k, d, s, p, 1)
                  || test_convolution3d(11, 10, 9, 8, 13, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 13, 8, k, d, s, p, 1)
                  || test_convolution3d(11, 10, 9, 12, 16, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 15, 15, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 16, 16, k, d, s, p, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_convolution3d_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::Convolution3D);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if (ret != (valid ? 0 : -1))
    {
        const int num_output = pd.get(0, 0);
        const int kernel_w = pd.get(1, 0);
        const int kernel_h = pd.get(11, kernel_w);
        const int kernel_d = pd.get(21, kernel_w);
        const int dilation_w = pd.get(2, 1);
        const int dilation_h = pd.get(12, dilation_w);
        const int dilation_d = pd.get(22, dilation_w);
        const int stride_w = pd.get(3, 1);
        const int stride_h = pd.get(13, stride_w);
        const int stride_d = pd.get(23, stride_w);
        const int weight_data_size = pd.get(6, 0);
        const int activation_type = pd.get(9, 0);

        fprintf(stderr, "test_convolution3d_load_param failed ret=%d expected=%d num_output=%d kernel_w=%d kernel_h=%d kernel_d=%d dilation_w=%d dilation_h=%d dilation_d=%d stride_w=%d stride_h=%d stride_d=%d weight_data_size=%d activation_type=%d\n", ret, valid ? 0 : -1, num_output, kernel_w, kernel_h, kernel_d, dilation_w, dilation_h, dilation_d, stride_w, stride_h, stride_d, weight_data_size, activation_type);

        const ncnn::Mat activation_params = pd.get(10, ncnn::Mat());
        fprintf(stderr, "activation_params type=%d dims=%d w=%d elemsize=%zu elempack=%d\n", pd.type(10), activation_params.dims, activation_params.w, activation_params.elemsize, activation_params.elempack);
        return -1;
    }

    return 0;
}

static int test_convolution3d_load_param_case(const ncnn::ParamDict& base, int id, int value, bool valid)
{
    ncnn::ParamDict pd = base;
    pd.set(id, value);

    int ret = test_convolution3d_load_param_case(pd, valid);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution3d_load_param failed id=%d value=%d\n", id, value);
    }

    return ret;
}

static int test_convolution3d_load_param_activation(const ncnn::ParamDict& base, int activation_type, const ncnn::Mat& activation_params, bool valid)
{
    ncnn::ParamDict pd = base;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    return test_convolution3d_load_param_case(pd, valid);
}

static int test_convolution3d_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 8);
    base.set(1, 3);
    base.set(6, 216);
    if (test_convolution3d_load_param_case(base, true) != 0)
        return -1;

    ncnn::Mat params(2);
    params[0] = 0.1f;
    params[1] = 0.5f;

    int ret = 0
              || test_convolution3d_load_param_activation(base, 0, ncnn::Mat(), true)
              || test_convolution3d_load_param_activation(base, 0, ncnn::Mat(0), true)
              || test_convolution3d_load_param_activation(base, 0, params, true)
              || test_convolution3d_load_param_activation(base, 0, params.range(0, 1), true)
              || test_convolution3d_load_param_activation(base, 1, ncnn::Mat(), true)
              || test_convolution3d_load_param_activation(base, 1, ncnn::Mat(0), true)
              || test_convolution3d_load_param_activation(base, 1, params, true)
              || test_convolution3d_load_param_activation(base, 1, params.range(0, 1), true)
              || test_convolution3d_load_param_activation(base, 2, ncnn::Mat(), false)
              || test_convolution3d_load_param_activation(base, 2, ncnn::Mat(0), false)
              || test_convolution3d_load_param_activation(base, 2, params, true)
              || test_convolution3d_load_param_activation(base, 2, params.range(0, 1), true)
              || test_convolution3d_load_param_activation(base, 3, ncnn::Mat(), false)
              || test_convolution3d_load_param_activation(base, 3, ncnn::Mat(0), false)
              || test_convolution3d_load_param_activation(base, 3, params, true)
              || test_convolution3d_load_param_activation(base, 3, params.range(0, 1), false)
              || test_convolution3d_load_param_activation(base, 4, ncnn::Mat(), true)
              || test_convolution3d_load_param_activation(base, 4, ncnn::Mat(0), true)
              || test_convolution3d_load_param_activation(base, 4, params, true)
              || test_convolution3d_load_param_activation(base, 4, params.range(0, 1), true)
              || test_convolution3d_load_param_activation(base, 5, ncnn::Mat(), true)
              || test_convolution3d_load_param_activation(base, 5, ncnn::Mat(0), true)
              || test_convolution3d_load_param_activation(base, 5, params, true)
              || test_convolution3d_load_param_activation(base, 5, params.range(0, 1), true)
              || test_convolution3d_load_param_activation(base, 6, ncnn::Mat(), false)
              || test_convolution3d_load_param_activation(base, 6, ncnn::Mat(0), false)
              || test_convolution3d_load_param_activation(base, 6, params, true)
              || test_convolution3d_load_param_activation(base, 6, params.range(0, 1), false)
              || test_convolution3d_load_param_activation(base, 7, ncnn::Mat(), false);
    if (ret != 0)
        return ret;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    const ncnn::Mat bad[] = {ncnn::Mat(2, (size_t)1u), ncnn::Mat(2, (size_t)2u), ncnn::Mat(2, 2), ncnn::Mat(2, (size_t)16u, 4), missing_data};
    for (int i = 0; i < 5; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(10, bad[i]);
        if (test_convolution3d_load_param_case(pd, false) != 0)
            return -1;
    }

    ret = 0
          || test_convolution3d_load_param_case(base, 0, 0, false)
          || test_convolution3d_load_param_case(base, 1, 0, false)
          || test_convolution3d_load_param_case(base, 2, 0, false)
          || test_convolution3d_load_param_case(base, 3, 0, false)
          || test_convolution3d_load_param_case(base, 1, INT_MAX, false)
          || test_convolution3d_load_param_case(base, 6, 217, false);
    if (ret != 0)
        return ret;

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_convolution3d_0() || test_convolution3d_load_param();
}
