// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_convolution_vec(int w, int outch, int kernel, int dilation, int stride, int pad, int bias)
{
    ncnn::Mat a = RandomMat(w);

    ncnn::ParamDict pd;
    pd.set(0, outch);    // num_output
    pd.set(1, kernel);   // kernel_w
    pd.set(2, dilation); // dilation_w
    pd.set(3, stride);   // stride_w
    pd.set(4, pad);      // pad_w
    pd.set(5, bias);     // bias_term
    pd.set(6, outch * w * kernel * kernel);

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * w * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer("Convolution", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution_vec failed w=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolution_vec_activation(int activation_type)
{
    ncnn::Mat a(13);
    for (int i = 0; i < 13; i++)
        a[i] = (i % 3 - 1) * 4.f;

    ncnn::ParamDict pd;
    pd.set(0, 15);
    pd.set(1, 1);
    pd.set(2, 1);
    pd.set(3, 1);
    pd.set(4, 0);
    pd.set(5, 0);
    pd.set(6, 15 * 13);
    pd.set(9, activation_type);

    ncnn::Mat activation_params(2);
    activation_params[0] = activation_type == 6 ? 0.25f : -0.25f;
    activation_params[1] = 0.5f;
    pd.set(10, activation_params);

    // alternate negative, zero and positive outputs across packed groups and tails
    std::vector<ncnn::Mat> weights(1);
    weights[0].create(15 * 13);
    weights[0].fill(0.f);
    for (int q = 0; q < 15; q++)
        weights[0][q * 13 + q % 3] = 1.f;

    int ret = test_layer("Convolution", pd, weights, a);
    if (ret != 0)
        fprintf(stderr, "test_convolution_vec_activation failed act=%d\n", activation_type);

    return ret;
}

static int test_convolution_2()
{
    return 0
           || test_convolution_vec(1, 1, 1, 1, 1, 0, 1)
           || test_convolution_vec(11, 12, 1, 1, 1, 0, 0)
           || test_convolution_vec(20, 15, 1, 1, 1, 0, 1)
           || test_convolution_vec(12, 20, 1, 1, 1, 0, 0)
           || test_convolution_vec(3, 24, 1, 1, 1, 0, 1)
           || test_convolution_vec(24, 5, 1, 1, 1, 0, 0)
           || test_convolution_vec(32, 24, 1, 1, 1, 0, 1)
           || test_convolution_vec(12, 32, 1, 1, 1, 0, 0)
           || test_convolution_vec(64, 20, 1, 1, 1, 0, 1)
           || test_convolution_vec(64, 128, 1, 1, 1, 0, 0)
           || test_convolution_vec_activation(0)
           || test_convolution_vec_activation(1)
           || test_convolution_vec_activation(2)
           || test_convolution_vec_activation(6);
}

static int test_convolution_dynamic(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, 0);
    pd.set(1, 0);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, 0);
    pd.set(19, 1); // dynamic weight

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> as(bias ? 3 : 2);
    as[0] = a;
    as[1] = RandomMat(kernel, kernel, c, outch);
    if (bias)
        as[2] = RandomMat(outch);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Convolution", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution_dynamic failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolution_3()
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
                  || test_convolution_dynamic(11, 10, 1, 1, k, d, s, p, 1)
                  || test_convolution_dynamic(11, 10, 4, 13, k, d, s, p, 0)
                  || test_convolution_dynamic(11, 10, 13, 4, k, d, s, p, 1)
                  || test_convolution_dynamic(11, 10, 12, 12, k, d, s, p, 0)
                  || test_convolution_dynamic(11, 10, 8, 12, k, d, s, p, 1)
                  || test_convolution_dynamic(11, 10, 8, 13, k, d, s, p, 0)
                  || test_convolution_dynamic(11, 10, 13, 8, k, d, s, p, 1)
                  || test_convolution_dynamic(11, 10, 12, 16, k, d, s, p, 0)
                  || test_convolution_dynamic(11, 10, 15, 15, k, d, s, p, 0)
                  || test_convolution_dynamic(11, 10, 16, 16, k, d, s, p, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_convolution_2()
           || test_convolution_3();
}
