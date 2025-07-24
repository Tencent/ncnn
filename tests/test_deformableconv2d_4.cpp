// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

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

    return ret;
}

static int test_deformableconv2d_0()
{
    return 0
           || test_deformableconv2d(7, 5, 24, 32, 4, 2, 2, 2, 1)
           || test_deformableconv2d(7, 5, 32, 24, 4, 2, 2, 2, 1)
           || test_deformableconv2d(7, 5, 28, 32, 4, 2, 2, 2, 1)
           || test_deformableconv2d(7, 5, 32, 28, 4, 2, 2, 2, 1)
           || test_deformableconv2d(7, 5, 26, 32, 4, 2, 2, 2, 1)
           || test_deformableconv2d(7, 5, 32, 26, 4, 2, 2, 2, 1);
}

int main()
{
    SRAND(7767517);

    return test_deformableconv2d_0();
}
