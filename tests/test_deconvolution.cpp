// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_deconvolution(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int output_pad_right, int output_pad_bottom, int output_w, int output_h)
{
    ncnn::Mat a = RandomMat(w, h, c);

    if (output_w > 0 && output_h > 0 && pad != -233 && pad != -234)
    {
        pad = -233;
    }

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outch * c * kernel * kernel);

    int activation_type = RAND() % 5; // 0 1 2 3 4
    ncnn::Mat activation_params(2);
    activation_params[0] = RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    pd.set(18, output_pad_right);
    pd.set(19, output_pad_bottom);
    pd.set(20, output_w);
    pd.set(21, output_h);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(outch * c * kernel * kernel);
    weights[1] = RandomMat(outch);

    int ret = test_layer("Deconvolution", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_deconvolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f] output_pad_right=%d output_pad_bottom=%d output_w=%d output_h=%d\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1], output_pad_right, output_pad_bottom, output_w, output_h);
        return ret;
    }

    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = false;
        opt.use_bf16_storage = false;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("Deconvolution", pd, weights, opt, a);
        if (ret != 0)
        {
            fprintf(stderr, "test_deconvolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f] output_pad_right=%d output_pad_bottom=%d output_w=%d output_h=%d\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1], output_pad_right, output_pad_bottom, output_w, output_h);
            return ret;
        }
    }

    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = true;
        opt.use_fp16_storage = true;
        opt.use_fp16_arithmetic = true;
        opt.use_bf16_packed = false;
        opt.use_bf16_storage = false;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("Deconvolution", pd, weights, opt, a);
        if (ret != 0)
        {
            fprintf(stderr, "test_deconvolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f] output_pad_right=%d output_pad_bottom=%d output_w=%d output_h=%d\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1], output_pad_right, output_pad_bottom, output_w, output_h);
            return ret;
        }
    }

    return ret;
}

static int test_deconvolution_0()
{
    static const int kdsp[16][4] = {
        {1, 1, 1, 0},
        {1, 1, 2, 0},
        {2, 1, 1, 1},
        {2, 1, 2, -233},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {3, 2, 1, 1},
        {4, 1, 1, -233},
        {4, 1, 2, -234},
        {4, 2, 1, -234},
        {5, 1, 1, 2},
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
                  || test_deconvolution(9, 7, 1, 1, k, d, s, p, 1, 0, 0, 0, 0)
                  || test_deconvolution(9, 7, 4, 13, k, d, s, p, 0, 1, 1, 7, 5)
                  || test_deconvolution(9, 7, 13, 4, k, d, s, p, 1, 1, 0, 0, 0)
                  || test_deconvolution(9, 7, 4, 8, k, d, s, p, 0, 0, 1, 0, 0)
                  || test_deconvolution(9, 7, 8, 4, k, d, s, p, 1, 0, 0, 7, 5)
                  || test_deconvolution(7, 7, 12, 12, k, d, s, p, 1, 0, 1, 0, 0)
                  || test_deconvolution(4, 5, 12, 11, k, d, s, p, 0, 0, 1, 1, 0)
                  || test_deconvolution(9, 7, 8, 13, k, d, s, p, 0, 2, 2, 0, 0)
                  || test_deconvolution(9, 7, 13, 8, k, d, s, p, 1, 2, 0, 0, 0)
                  || test_deconvolution(9, 7, 16, 16, k, d, s, p, 0, 0, 2, 7, 5);

        if (ret != 0)
            return -1;
    }

    // tier coverage for small outch and various elempack
    return 0
           || test_deconvolution(5, 4, 7, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           || test_deconvolution(6, 7, 2, 3, 2, 1, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution(5, 3, 1, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           || test_deconvolution(8, 6, 8, 2, 3, 2, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution(8, 4, 8, 15, 1, 1, 2, 0, 0, 0, 1, 0, 0)
           // kernel transform num_input < 4 paths
           || test_deconvolution(4, 5, 1, 8, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution(5, 4, 2, 7, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_deconvolution(5, 4, 2, 8, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution(4, 5, 2, 1, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution(5, 4, 1, 5, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           // elempack==1 large inch for inner tier cascading
           || test_deconvolution(3, 3, 14, 2, 3, 1, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution(3, 3, 14, 1, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           // avx512 tier coverage
           || test_deconvolution(5, 4, 1, 16, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           || test_deconvolution(5, 4, 2, 16, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution(5, 4, 8, 16, 3, 1, 2, 1, 1, 0, 0, 0, 0)
           || test_deconvolution(4, 3, 5, 16, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           || test_deconvolution(5, 4, 24, 8, 3, 1, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution(5, 4, 20, 8, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution(4, 5, 17, 8, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_deconvolution(5, 4, 7, 5, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution(4, 3, 24, 5, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution(5, 4, 20, 5, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution(4, 3, 17, 5, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution(4, 3, 24, 2, 3, 1, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution(5, 4, 17, 2, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution(5, 4, 20, 2, 3, 1, 2, 1, 0, 0, 0, 0, 0)
           || test_deconvolution(4, 3, 7, 1, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution(5, 4, 16, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           || test_deconvolution(4, 3, 24, 1, 3, 1, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution(5, 4, 20, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           || test_deconvolution(4, 3, 17, 1, 3, 1, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution(7, 5, 24, 32, 4, 2, 2, 2, 1, 0, 0, 0, 0)
           || test_deconvolution(7, 5, 32, 24, 4, 2, 2, 2, 1, 0, 0, 0, 0)
           || test_deconvolution(7, 5, 28, 32, 4, 2, 2, 2, 1, 0, 0, 0, 0)
           || test_deconvolution(7, 5, 32, 28, 4, 2, 2, 2, 1, 0, 0, 0, 0)
           || test_deconvolution(7, 5, 26, 32, 4, 2, 2, 2, 1, 0, 0, 0, 0)
           || test_deconvolution(7, 5, 32, 26, 4, 2, 2, 2, 1, 0, 0, 0, 0);
}

static int test_deconvolution_dynamic(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int output_pad_right, int output_pad_bottom, int output_w, int output_h)
{
    ncnn::Mat a = RandomMat(w, h, c);

    if (output_w > 0 && output_h > 0 && pad != -233 && pad != -234)
    {
        pad = -233;
    }

    ncnn::ParamDict pd;
    pd.set(0, 0);
    pd.set(1, 0);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, 0);
    pd.set(28, 1); // dynamic weight

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    pd.set(18, output_pad_right);
    pd.set(19, output_pad_bottom);
    pd.set(20, output_w);
    pd.set(21, output_h);

    std::vector<ncnn::Mat> as(bias ? 3 : 2);
    as[0] = a;
    as[1] = RandomMat(kernel, kernel, outch, c);
    if (bias)
        as[2] = RandomMat(outch);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Deconvolution", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_deconvolution_dynamic failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f] output_pad_right=%d output_pad_bottom=%d output_w=%d output_h=%d\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1], output_pad_right, output_pad_bottom, output_w, output_h);
    }

    return ret;
}

static int test_deconvolution_1()
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
                  || test_deconvolution_dynamic(9, 7, 1, 1, k, d, s, p, 1, 0, 0, 0, 0)
                  || test_deconvolution_dynamic(9, 7, 4, 13, k, d, s, p, 0, 1, 1, 7, 5)
                  || test_deconvolution_dynamic(9, 7, 13, 4, k, d, s, p, 1, 1, 0, 0, 0)
                  || test_deconvolution_dynamic(9, 7, 4, 8, k, d, s, p, 0, 0, 1, 0, 0)
                  || test_deconvolution_dynamic(9, 7, 8, 4, k, d, s, p, 1, 0, 0, 7, 5)
                  || test_deconvolution_dynamic(7, 7, 12, 12, k, d, s, p, 1, 0, 1, 0, 0)
                  || test_deconvolution_dynamic(4, 5, 12, 11, k, d, s, p, 0, 0, 1, 1, 0)
                  || test_deconvolution_dynamic(9, 7, 8, 13, k, d, s, p, 0, 2, 2, 0, 0)
                  || test_deconvolution_dynamic(9, 7, 13, 8, k, d, s, p, 1, 2, 0, 0, 0)
                  || test_deconvolution_dynamic(9, 7, 16, 16, k, d, s, p, 0, 0, 2, 7, 5);

        if (ret != 0)
            return -1;
    }

    // tier coverage for small outch and various elempack
    return 0
           || test_deconvolution_dynamic(5, 4, 7, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(6, 7, 2, 3, 2, 1, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 3, 1, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(8, 6, 8, 2, 3, 2, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution_dynamic(8, 4, 8, 15, 1, 1, 2, 0, 0, 0, 1, 0, 0)
           // kernel transform num_input < 4 paths
           || test_deconvolution_dynamic(4, 5, 1, 8, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 4, 2, 7, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 4, 2, 8, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(4, 5, 2, 1, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 4, 1, 5, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           // elempack==1 large inch for inner tier cascading
           || test_deconvolution_dynamic(3, 3, 14, 2, 3, 1, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution_dynamic(3, 3, 14, 1, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           // avx512 tier coverage
           || test_deconvolution_dynamic(5, 4, 1, 16, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 4, 2, 16, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 4, 8, 16, 3, 1, 2, 1, 1, 0, 0, 0, 0)
           || test_deconvolution_dynamic(4, 3, 5, 16, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 4, 24, 8, 3, 1, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 4, 20, 8, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(4, 5, 17, 8, 1, 1, 1, 0, 1, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 4, 7, 5, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(4, 3, 24, 5, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 4, 20, 5, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(4, 3, 17, 5, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(4, 3, 24, 2, 3, 1, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 4, 17, 2, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 4, 20, 2, 3, 1, 2, 1, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(4, 3, 7, 1, 3, 1, 1, 1, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 4, 16, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(4, 3, 24, 1, 3, 1, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution_dynamic(5, 4, 20, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0)
           || test_deconvolution_dynamic(4, 3, 17, 1, 3, 1, 1, 1, 1, 0, 0, 0, 0);
}

static int test_deconvolution_load_param_activation(const ncnn::ParamDict& base, int activation_type, const ncnn::Mat& activation_params, int expected_ret)
{
    ncnn::ParamDict pd = base;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    return test_layer_param(ncnn::LayerType::Deconvolution, pd, expected_ret);
}

static int test_deconvolution_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 8);
    base.set(1, 3);
    base.set(6, 216);
    if (test_layer_param(ncnn::LayerType::Deconvolution, base, 0) != 0)
        return -1;

    ncnn::Mat params(2);
    params[0] = 0.1f;
    params[1] = 0.5f;

    int ret = 0
              || test_deconvolution_load_param_activation(base, 0, ncnn::Mat(), 0)
              || test_deconvolution_load_param_activation(base, 0, ncnn::Mat(0), 0)
              || test_deconvolution_load_param_activation(base, 0, params, 0)
              || test_deconvolution_load_param_activation(base, 0, params.range(0, 1), 0)
              || test_deconvolution_load_param_activation(base, 1, ncnn::Mat(), 0)
              || test_deconvolution_load_param_activation(base, 1, ncnn::Mat(0), 0)
              || test_deconvolution_load_param_activation(base, 1, params, 0)
              || test_deconvolution_load_param_activation(base, 1, params.range(0, 1), 0)
              || test_deconvolution_load_param_activation(base, 2, ncnn::Mat(), -1)
              || test_deconvolution_load_param_activation(base, 2, ncnn::Mat(0), -1)
              || test_deconvolution_load_param_activation(base, 2, params, 0)
              || test_deconvolution_load_param_activation(base, 2, params.range(0, 1), 0)
              || test_deconvolution_load_param_activation(base, 3, ncnn::Mat(), -1)
              || test_deconvolution_load_param_activation(base, 3, ncnn::Mat(0), -1)
              || test_deconvolution_load_param_activation(base, 3, params, 0)
              || test_deconvolution_load_param_activation(base, 3, params.range(0, 1), -1)
              || test_deconvolution_load_param_activation(base, 4, ncnn::Mat(), 0)
              || test_deconvolution_load_param_activation(base, 4, ncnn::Mat(0), 0)
              || test_deconvolution_load_param_activation(base, 4, params, 0)
              || test_deconvolution_load_param_activation(base, 4, params.range(0, 1), 0)
              || test_deconvolution_load_param_activation(base, 5, ncnn::Mat(), 0)
              || test_deconvolution_load_param_activation(base, 5, ncnn::Mat(0), 0)
              || test_deconvolution_load_param_activation(base, 5, params, 0)
              || test_deconvolution_load_param_activation(base, 5, params.range(0, 1), 0)
              || test_deconvolution_load_param_activation(base, 6, ncnn::Mat(), -1)
              || test_deconvolution_load_param_activation(base, 6, ncnn::Mat(0), -1)
              || test_deconvolution_load_param_activation(base, 6, params, 0)
              || test_deconvolution_load_param_activation(base, 6, params.range(0, 1), -1)
              || test_deconvolution_load_param_activation(base, 7, ncnn::Mat(), -1);
    if (ret != 0)
        return ret;

    if (test_layer_param(ncnn::LayerType::Deconvolution, base, 10, 1, -1)
            || test_layer_param(ncnn::LayerType::Deconvolution, base, 10, 1.f, -1))
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    ncnn::Mat negative_length(0);
    negative_length.w = -1;

    const ncnn::Mat bad[] = {ncnn::Mat(2, (size_t)1u), ncnn::Mat(2, (size_t)2u), ncnn::Mat(2, 2), ncnn::Mat(2, (size_t)16u, 4), missing_data, negative_length};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++)
    {
        if (test_layer_param(ncnn::LayerType::Deconvolution, base, 10, bad[i], -1) != 0)
            return -1;
    }

    ret = 0
          || test_layer_param(ncnn::LayerType::Deconvolution, base, 0, 0, -1)
          || test_layer_param(ncnn::LayerType::Deconvolution, base, 1, 0, -1)
          || test_layer_param(ncnn::LayerType::Deconvolution, base, 2, 0, -1)
          || test_layer_param(ncnn::LayerType::Deconvolution, base, 3, 0, -1)
          || test_layer_param(ncnn::LayerType::Deconvolution, base, 1, INT_MAX, -1)
          || test_layer_param(ncnn::LayerType::Deconvolution, base, 6, 217, -1);
    if (ret != 0)
        return ret;

    return 0;
}

static int test_deconvolution_load_param_text()
{
#if NCNN_STRING
    const char* params[] = {"0=1 1=1 6=1 9=3 -23310=2,-1,2", "0=1 1=1 6=1 9=3 -23310=2,-1.0,2.0"};
    for (int i = 0; i < 2; i++)
    {
        TestParamDict pd;
        if (pd.load_param(params[i]) != 0 || pd.type(10) != 5 + i)
            return -1;

        if (test_layer_param(ncnn::LayerType::Deconvolution, pd, 0) != 0)
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
        int ret = test_layer_naive(ncnn::LayerType::Deconvolution, pd, weights, a, b, 0);
        if (ret == 0)
            ret = CompareMat(reference, b, 0.f);
        if (ret != 0)
        {
            fprintf(stderr, "test_deconvolution_load_param_text failed params=%s ret=%d\n", params[i], ret);
            return ret;
        }

        const ncnn::Mat original = pd.get(10, ncnn::Mat());
        const int* p = original;
        if (i == 0 ? (p[0] != -1 || p[1] != 2) : (original[0] != -1.f || original[1] != 2.f))
        {
            fprintf(stderr, "test_deconvolution_load_param_text modified params=%s\n", params[i]);
            return -1;
        }
    }
#endif
    return 0;
}

int main()
{
    SRAND(7767517);

    return test_deconvolution_0() || test_deconvolution_1() || test_deconvolution_load_param() || test_deconvolution_load_param_text();
}
