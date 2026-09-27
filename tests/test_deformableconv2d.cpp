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

#if NCNN_VALIDATION
static int test_deformableconv2d_load_param_activation(const ncnn::ParamDict& base, int activation_type, const ncnn::Mat& activation_params, int expected_ret)
{
    ncnn::ParamDict pd = base;
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    return test_layer_param(ncnn::LayerType::DeformableConv2D, pd, expected_ret);
}

static int test_deformableconv2d_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 8);
    base.set(1, 3);
    base.set(6, 216);
    if (test_layer_param(ncnn::LayerType::DeformableConv2D, base, 0) != 0)
        return -1;

    ncnn::Mat params(2);
    params[0] = 0.1f;
    params[1] = 0.5f;

    int ret = 0
              || test_deformableconv2d_load_param_activation(base, 0, ncnn::Mat(), 0)
              || test_deformableconv2d_load_param_activation(base, 0, ncnn::Mat(0), 0)
              || test_deformableconv2d_load_param_activation(base, 0, params, 0)
              || test_deformableconv2d_load_param_activation(base, 0, params.range(0, 1), 0)
              || test_deformableconv2d_load_param_activation(base, 1, ncnn::Mat(), 0)
              || test_deformableconv2d_load_param_activation(base, 1, ncnn::Mat(0), 0)
              || test_deformableconv2d_load_param_activation(base, 1, params, 0)
              || test_deformableconv2d_load_param_activation(base, 1, params.range(0, 1), 0)
              || test_deformableconv2d_load_param_activation(base, 2, ncnn::Mat(), -1)
              || test_deformableconv2d_load_param_activation(base, 2, ncnn::Mat(0), -1)
              || test_deformableconv2d_load_param_activation(base, 2, params, 0)
              || test_deformableconv2d_load_param_activation(base, 2, params.range(0, 1), 0)
              || test_deformableconv2d_load_param_activation(base, 3, ncnn::Mat(), -1)
              || test_deformableconv2d_load_param_activation(base, 3, ncnn::Mat(0), -1)
              || test_deformableconv2d_load_param_activation(base, 3, params, 0)
              || test_deformableconv2d_load_param_activation(base, 3, params.range(0, 1), -1)
              || test_deformableconv2d_load_param_activation(base, 4, ncnn::Mat(), 0)
              || test_deformableconv2d_load_param_activation(base, 4, ncnn::Mat(0), 0)
              || test_deformableconv2d_load_param_activation(base, 4, params, 0)
              || test_deformableconv2d_load_param_activation(base, 4, params.range(0, 1), 0)
              || test_deformableconv2d_load_param_activation(base, 5, ncnn::Mat(), 0)
              || test_deformableconv2d_load_param_activation(base, 5, ncnn::Mat(0), 0)
              || test_deformableconv2d_load_param_activation(base, 5, params, 0)
              || test_deformableconv2d_load_param_activation(base, 5, params.range(0, 1), 0)
              || test_deformableconv2d_load_param_activation(base, 6, ncnn::Mat(), -1)
              || test_deformableconv2d_load_param_activation(base, 6, ncnn::Mat(0), -1)
              || test_deformableconv2d_load_param_activation(base, 6, params, 0)
              || test_deformableconv2d_load_param_activation(base, 6, params.range(0, 1), -1)
              || test_deformableconv2d_load_param_activation(base, 7, ncnn::Mat(), -1);
    if (ret != 0)
        return ret;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    const ncnn::Mat bad[] = {ncnn::Mat(2, (size_t)1u), ncnn::Mat(2, (size_t)2u), ncnn::Mat(2, 2), ncnn::Mat(2, (size_t)16u, 4), missing_data};
    for (int i = 0; i < 5; i++)
    {
        if (test_layer_param(ncnn::LayerType::DeformableConv2D, base, 10, bad[i], -1) != 0)
            return -1;
    }

    ret = 0
          || test_layer_param(ncnn::LayerType::DeformableConv2D, base, 0, 0, -1)
          || test_layer_param(ncnn::LayerType::DeformableConv2D, base, 1, 0, -1)
          || test_layer_param(ncnn::LayerType::DeformableConv2D, base, 2, 0, -1)
          || test_layer_param(ncnn::LayerType::DeformableConv2D, base, 3, 0, -1)
          || test_layer_param(ncnn::LayerType::DeformableConv2D, base, 1, INT_MAX, -1)
          || test_layer_param(ncnn::LayerType::DeformableConv2D, base, 6, 217, -1);
    if (ret != 0)
        return ret;

    return 0;
}
#endif // NCNN_VALIDATION

int main()
{
    SRAND(7767517);

    return 0
           || test_deformableconv2d_0()
#if NCNN_VALIDATION
           || test_deformableconv2d_load_param()
#endif // NCNN_VALIDATION
           ;
}
