// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_convolution_oom(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int activation_type, ncnn::Mat activation_params)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outch * c * kernel * kernel);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * c * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer_oom("Convolution", pd, weights, a);
    if (ret != 0)
    {
        if (activation_type != 7)
            fprintf(stderr, "test_convolution_oom failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
        else
            fprintf(stderr, "test_convolution_oom failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%d]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params.row<int>(0)[0]);
        return ret;
    }

    return ret;
}

static int test_convolution_0()
{
    for (int j = 0; j < 11; j++)
    {
        ncnn::Mat activation_params;
        if (j != 7)
        {
            activation_params.create(2);
            activation_params[0] = (j == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
            activation_params[1] = RandomFloat(0, 1);
        }
        else
        {
            activation_params.create(1);
            activation_params.row<int>(0)[0] = RandomInt(0, 1); // fast==1
        }

        int ret = 0
                  || test_convolution_oom(9, 7, 31, 63, 1, 1, 1, 0, 1, j, activation_params)
                  || test_convolution_oom(9, 7, 31, 63, 3, 1, 1, 1, 1, j, activation_params);

        if (ret != 0)
            return -1;
    }

    return 0;
}

#if NCNN_INT8
static int test_convolution_oom_int8(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int activation_type, ncnn::Mat activation_params, bool requant = false)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outch * c * kernel * kernel);
    pd.set(8, requant ? 101 : 1); // int8_scale_term
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 5 : 4);
    weights[0] = RandomMat(outch * c * kernel * kernel);

    ncnn::Mat weight_scales = scales_mat(weights[0], outch, c * kernel * kernel, c * kernel * kernel);
    ncnn::Mat input_scales = scales_mat(a, 1, w * h * c, a.cstep);
    ncnn::Mat top_scales = requant ? scales_mat(a, 1, w * h * c, a.cstep) : ncnn::Mat();

    if (kernel == 3 && dilation == 1 && stride == 1)
    {
        // test for 6bit quant
        for (int i = 0; i < weight_scales.w; i++)
            weight_scales[i] = weight_scales[i] / 4.f;
    }

    if (bias)
    {
        weights[1] = RandomMat(outch);
        weights[2] = weight_scales;
        weights[3] = input_scales;
        weights[4] = top_scales;
    }
    else
    {
        weights[1] = weight_scales;
        weights[2] = input_scales;
        weights[3] = top_scales;
    }

    int flag = TEST_LAYER_DISABLE_GPU_TESTING;
    int ret = test_layer_oom("Convolution", pd, weights, a, flag);
    if (ret != 0)
    {
        if (activation_type != 7)
            fprintf(stderr, "test_convolution_oom_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d requant=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, requant, activation_type, activation_params[0], activation_params[1]);
        else
            fprintf(stderr, "test_convolution_oom_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d requant=%d act=%d actparams=[%d]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, requant, activation_type, activation_params.row<int>(0)[0]);
        return ret;
    }

    return ret;
}

static int test_convolution_1()
{
    for (int j = 0; j < 11; j++)
    {
        ncnn::Mat activation_params;
        if (j != 7)
        {
            activation_params.create(2);
            activation_params[0] = (j == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
            activation_params[1] = RandomFloat(0, 1);
        }
        else
        {
            activation_params.create(1);
            activation_params.row<int>(0)[0] = RandomInt(0, 1); // fast==1
        }

        int ret = 0
                  || test_convolution_oom_int8(9, 7, 31, 63, 1, 1, 1, 0, 1, j, activation_params)
                  || test_convolution_oom_int8(9, 7, 31, 63, 3, 1, 1, 1, 1, j, activation_params);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_convolution_2()
{
    for (int j = 0; j < 11; j++)
    {
        ncnn::Mat activation_params;
        if (j != 7)
        {
            activation_params.create(2);
            activation_params[0] = (j == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
            activation_params[1] = RandomFloat(0, 1);
        }
        else
        {
            activation_params.create(1);
            activation_params.row<int>(0)[0] = RandomInt(0, 1); // fast==1
        }

        int ret = 0
                  || test_convolution_oom_int8(9, 7, 31, 63, 1, 1, 1, 0, 1, j, activation_params, true)
                  || test_convolution_oom_int8(9, 7, 31, 63, 3, 1, 1, 1, 1, j, activation_params, true);

        if (ret != 0)
            return -1;
    }

    return 0;
}
#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

#if __mips__ || __loongarch64 || __riscv
    // TODO
    return 0;
#endif

#if NCNN_INT8
    return test_convolution_0() || test_convolution_1() || test_convolution_2();
#else
    return test_convolution_0();
#endif
}
