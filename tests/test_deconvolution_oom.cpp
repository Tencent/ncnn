// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_deconvolution_oom(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int output_pad_right, int output_pad_bottom, int output_w, int output_h)
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

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * c * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer_oom("Deconvolution", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_deconvolution_oom failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f] output_pad_right=%d output_pad_bottom=%d output_w=%d output_h=%d\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1], output_pad_right, output_pad_bottom, output_w, output_h);
        return ret;
    }

    return ret;
}

static int test_deconvolution_0()
{
    return 0
           || test_deconvolution_oom(9, 7, 8, 8, 3, 1, 1, 1, 1, 0, 0, 0, 0)
           || test_deconvolution_oom(9, 7, 16, 16, 3, 1, 1, 1, 1, 0, 0, 0, 0);
}

int main()
{
    SRAND(7767517);

    return test_deconvolution_0();
}
