// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_convolutiondepthwise_oom(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int group)
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

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch / group * c / group * kernel * kernel * group);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer_oom("ConvolutionDepthWise", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolutiondepthwise_oom failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, group, activation_type, activation_params[0], activation_params[1]);
        return ret;
    }

    return ret;
}

static int test_convolutiondepthwise_0()
{
    return 0
           || test_convolutiondepthwise_oom(9, 7, 8, 8, 3, 1, 1, 1, 1, 8)
           || test_convolutiondepthwise_oom(9, 7, 16, 16, 3, 1, 1, 1, 1, 16)
           || test_convolutiondepthwise_oom(9, 7, 16, 8, 3, 1, 1, 1, 1, 2);
}

#if NCNN_INT8
static int test_convolutiondepthwise_oom_int8(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int group, bool requant = false)
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
    pd.set(8, requant ? 101 : 1); // int8_scale_term

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 5 : 4);
    weights[0] = RandomMat(outch / group * c / group * kernel * kernel * group);
    ncnn::Mat weight_scales = scales_mat(weights[0], group, c * kernel * kernel / group, c * kernel * kernel / group);
    ncnn::Mat input_scales = scales_mat(a, 1, w * h * c, a.cstep);
    ncnn::Mat top_scales = requant ? scales_mat(a, 1, w * h * c, a.cstep) : ncnn::Mat();
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
    int ret = test_layer_oom("ConvolutionDepthWise", pd, weights, a, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolutiondepthwise_oom_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d requant=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, group, requant, activation_type, activation_params[0], activation_params[1]);
        return ret;
    }

    return ret;
}

static int test_convolutiondepthwise_1()
{
    return 0
           || test_convolutiondepthwise_oom_int8(9, 7, 8, 8, 3, 1, 1, 1, 1, 8)
           || test_convolutiondepthwise_oom_int8(9, 7, 16, 16, 3, 1, 1, 1, 1, 16)
           || test_convolutiondepthwise_oom_int8(9, 7, 16, 8, 3, 1, 1, 1, 1, 2);
}

static int test_convolutiondepthwise_2()
{
    return 0
           || test_convolutiondepthwise_oom_int8(9, 7, 8, 8, 3, 1, 1, 1, 1, 8, true)
           || test_convolutiondepthwise_oom_int8(9, 7, 16, 16, 3, 1, 1, 1, 1, 16, true)
           || test_convolutiondepthwise_oom_int8(9, 7, 16, 8, 3, 1, 1, 1, 1, 2, true);
}
#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return test_convolutiondepthwise_0() || test_convolutiondepthwise_1() || test_convolutiondepthwise_2();
#else
    return test_convolutiondepthwise_0();
#endif
}
