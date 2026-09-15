// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_deformableconv2d(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias)
{
    const int kernel_extent_w = dilation * (kernel - 1) + 1;
    const int kernel_extent_h = dilation * (kernel - 1) + 1;
    const int out_w = (w + pad + pad - kernel_extent_w) / stride + 1;
    const int out_h = (h + pad + pad - kernel_extent_h) / stride + 1;
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(w, h, c);
    a[1] = RandomMat(out_w, out_h, kernel * kernel * 2);
    a[2] = RandomMat(out_w, out_h, kernel * kernel);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outch * c * kernel * kernel);

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * c * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    float epsilon = 0.001;
    int ret = test_layer("DeformableConv2D", pd, weights, a, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_deformableconv2d failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
    }

    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_storage = false;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("DeformableConv2D", pd, weights, opt, a, 1, epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_deformableconv2d failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
        }
    }

    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = true;
        opt.use_fp16_storage = true;
        opt.use_fp16_arithmetic = true;
        opt.use_bf16_storage = true;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("DeformableConv2D", pd, weights, opt, a, 1, epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_deformableconv2d failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
        }
    }

    return ret;
}

static int test_deformableconv2d_0()
{
    static const int kdsp[10][4] = {
        {1, 1, 1, 0},
        {1, 1, 2, 0},
        {2, 1, 1, 1},
        {2, 1, 2, 0},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {3, 2, 1, 1},
        {4, 1, 2, 1},
        {5, 1, 2, 2},
        {5, 2, 2, 2},
    };

    for (int i = 0; i < 4; i++)
    {
        const int k = kdsp[i][0];
        const int d = kdsp[i][1];
        const int s = kdsp[i][2];
        const int p = kdsp[i][3];

        int ret = 0
                  || test_deformableconv2d(9, 7, 1, 1, k, d, s, p, 1)
                  || test_deformableconv2d(9, 7, 4, 13, k, d, s, p, 0)
                  || test_deformableconv2d(9, 7, 13, 4, k, d, s, p, 1)
                  || test_deformableconv2d(9, 7, 4, 8, k, d, s, p, 0)
                  || test_deformableconv2d(9, 7, 8, 4, k, d, s, p, 1)
                  || test_deformableconv2d(9, 7, 8, 13, k, d, s, p, 0)
                  || test_deformableconv2d(9, 7, 13, 8, k, d, s, p, 1)
                  || test_deformableconv2d(9, 7, 16, 16, k, d, s, p, 0)
                  || test_deformableconv2d(16, 16, 1 * 3, 1 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 1 * 3, 4 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 1 * 3, 8 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 1 * 3, 16 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 4 * 3, 1 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 4 * 3, 4 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 4 * 3, 8 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 4 * 3, 16 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 8 * 3, 1 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 8 * 3, 4 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 8 * 3, 8 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 8 * 3, 16 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 16 * 3, 1 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 16 * 3, 4 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 16 * 3, 8 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 16 * 3, 16 * 3, k, d, s, p, 1);

        if (ret != 0)
            return -1;
    }

    return 0
           || test_deformableconv2d(7, 5, 24, 32, 4, 2, 2, 2, 1)
           || test_deformableconv2d(7, 5, 32, 24, 4, 2, 2, 2, 1)
           || test_deformableconv2d(7, 5, 28, 32, 4, 2, 2, 2, 1)
           || test_deformableconv2d(7, 5, 32, 28, 4, 2, 2, 2, 1)
           || test_deformableconv2d(7, 5, 26, 32, 4, 2, 2, 2, 1)
           || test_deformableconv2d(7, 5, 32, 26, 4, 2, 2, 2, 1);
}

static int test_deformableconv2d_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::DeformableConv2D);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if (ret != (valid ? 0 : -1))
    {
        const int num_output = pd.get(0, 0);
        const int kernel_w = pd.get(1, 0);
        const int kernel_h = pd.get(11, kernel_w);
        const int dilation_w = pd.get(2, 1);
        const int dilation_h = pd.get(12, dilation_w);
        const int stride_w = pd.get(3, 1);
        const int stride_h = pd.get(13, stride_w);
        const int weight_data_size = pd.get(6, 0);
        const int activation_type = pd.get(9, 0);

        fprintf(stderr, "test_deformableconv2d_load_param failed ret=%d expected=%d num_output=%d kernel_w=%d kernel_h=%d dilation_w=%d dilation_h=%d stride_w=%d stride_h=%d weight_data_size=%d activation_type=%d\n", ret, valid ? 0 : -1, num_output, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, weight_data_size, activation_type);

        const ncnn::Mat activation_params = pd.get(10, ncnn::Mat());
        fprintf(stderr, "activation_params type=%d dims=%d w=%d elemsize=%zu elempack=%d\n", pd.type(10), activation_params.dims, activation_params.w, activation_params.elemsize, activation_params.elempack);
        return -1;
    }

    return 0;
}

static int test_deformableconv2d_load_param_case(const ncnn::ParamDict& base, int id, int value, bool valid)
{
    ncnn::ParamDict pd = base;
    pd.set(id, value);

    int ret = test_deformableconv2d_load_param_case(pd, valid);
    if (ret != 0)
    {
        fprintf(stderr, "test_deformableconv2d_load_param failed id=%d value=%d\n", id, value);
    }

    return ret;
}

static int test_deformableconv2d_load_param_activation(const ncnn::ParamDict& base, int activation_type, const ncnn::Mat& activation_params, bool valid)
{
    ncnn::ParamDict pd = base;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    return test_deformableconv2d_load_param_case(pd, valid);
}

static int test_deformableconv2d_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 8);
    base.set(1, 3);
    base.set(6, 216);
    if (test_deformableconv2d_load_param_case(base, true) != 0)
        return -1;

    ncnn::Mat params(2);
    params[0] = 0.1f;
    params[1] = 0.5f;

    int ret = 0
              || test_deformableconv2d_load_param_activation(base, 0, ncnn::Mat(), true)
              || test_deformableconv2d_load_param_activation(base, 0, ncnn::Mat(0), true)
              || test_deformableconv2d_load_param_activation(base, 0, params, true)
              || test_deformableconv2d_load_param_activation(base, 0, params.range(0, 1), true)
              || test_deformableconv2d_load_param_activation(base, 1, ncnn::Mat(), true)
              || test_deformableconv2d_load_param_activation(base, 1, ncnn::Mat(0), true)
              || test_deformableconv2d_load_param_activation(base, 1, params, true)
              || test_deformableconv2d_load_param_activation(base, 1, params.range(0, 1), true)
              || test_deformableconv2d_load_param_activation(base, 2, ncnn::Mat(), false)
              || test_deformableconv2d_load_param_activation(base, 2, ncnn::Mat(0), false)
              || test_deformableconv2d_load_param_activation(base, 2, params, true)
              || test_deformableconv2d_load_param_activation(base, 2, params.range(0, 1), true)
              || test_deformableconv2d_load_param_activation(base, 3, ncnn::Mat(), false)
              || test_deformableconv2d_load_param_activation(base, 3, ncnn::Mat(0), false)
              || test_deformableconv2d_load_param_activation(base, 3, params, true)
              || test_deformableconv2d_load_param_activation(base, 3, params.range(0, 1), false)
              || test_deformableconv2d_load_param_activation(base, 4, ncnn::Mat(), true)
              || test_deformableconv2d_load_param_activation(base, 4, ncnn::Mat(0), true)
              || test_deformableconv2d_load_param_activation(base, 4, params, true)
              || test_deformableconv2d_load_param_activation(base, 4, params.range(0, 1), true)
              || test_deformableconv2d_load_param_activation(base, 5, ncnn::Mat(), true)
              || test_deformableconv2d_load_param_activation(base, 5, ncnn::Mat(0), true)
              || test_deformableconv2d_load_param_activation(base, 5, params, true)
              || test_deformableconv2d_load_param_activation(base, 5, params.range(0, 1), true)
              || test_deformableconv2d_load_param_activation(base, 6, ncnn::Mat(), false)
              || test_deformableconv2d_load_param_activation(base, 6, ncnn::Mat(0), false)
              || test_deformableconv2d_load_param_activation(base, 6, params, true)
              || test_deformableconv2d_load_param_activation(base, 6, params.range(0, 1), false)
              || test_deformableconv2d_load_param_activation(base, 7, ncnn::Mat(), false);
    if (ret != 0)
        return ret;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    const ncnn::Mat bad[] = {ncnn::Mat(2, (size_t)1u), ncnn::Mat(2, (size_t)2u), ncnn::Mat(2, 2), ncnn::Mat(2, (size_t)16u, 4), missing_data};
    for (int i = 0; i < 5; i++)
    {
        ncnn::ParamDict pd = base;
        pd.set(10, bad[i]);
        if (test_deformableconv2d_load_param_case(pd, false) != 0)
            return -1;
    }

    ret = 0
          || test_deformableconv2d_load_param_case(base, 0, 0, false)
          || test_deformableconv2d_load_param_case(base, 1, 0, false)
          || test_deformableconv2d_load_param_case(base, 2, 0, false)
          || test_deformableconv2d_load_param_case(base, 3, 0, false)
          || test_deformableconv2d_load_param_case(base, 1, INT_MAX, false)
          || test_deformableconv2d_load_param_case(base, 6, 217, false);
    if (ret != 0)
        return ret;

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_deformableconv2d_0() || test_deformableconv2d_load_param();
}
