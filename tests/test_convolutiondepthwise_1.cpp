// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_convolutiondepthwise_dynamic(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int group)
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
    pd.set(7, group);
    pd.set(19, 1); // dynamic weight

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> as(bias ? 3 : 2);
    as[0] = a;
    as[1] = RandomMat(kernel, kernel, c / group, outch);
    if (bias)
        as[2] = RandomMat(outch);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("ConvolutionDepthWise", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolutiondepthwise_dynamic failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, group, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolutiondepthwise_2()
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
                  || test_convolutiondepthwise_dynamic(11, 10, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise_dynamic(11, 10, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise_dynamic(11, 10, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise_dynamic(11, 10, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise_dynamic(11, 10, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise_dynamic(11, 10, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise_dynamic(11, 10, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise_dynamic(11, 10, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise_dynamic(11, 10, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise_dynamic(11, 10, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise_dynamic(11, 10, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise_dynamic(11, 10, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise_dynamic(11, 10, 16, 16, k, d, s, p, 1, 16);

        if (ret != 0)
            return -1;
    }

    return 0;
}

#if NCNN_INT8
static int test_convolutiondepthwise_int8(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int group, bool requant = false, int int8_scale_term = 0, bool input_int8 = false)
{
    ncnn::Mat a = RandomMat(w, h, c);

    if (int8_scale_term == 0)
        int8_scale_term = requant ? 101 : 1;
    const bool use_requant = int8_scale_term > 100;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outch / group * c / group * kernel * kernel * group);
    pd.set(7, group);
    pd.set(8, int8_scale_term); // int8_scale_term

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 5 : 4);
    weights[0] = RandomMat(outch / group * c / group * kernel * kernel * group);
    ncnn::Mat weight_scales;
    if (int8_scale_term == 2 || int8_scale_term == 102)
        weight_scales = scales_mat(weights[0], 1, weights[0].w, weights[0].w);
    else
        weight_scales = scales_mat(weights[0], group, c * kernel * kernel / group, c * kernel * kernel / group);
    ncnn::Mat input_scales = scales_mat(a, 1, w * h * c, a.cstep);
    ncnn::Mat top_scales = use_requant ? scales_mat(a, 1, w * h * c, a.cstep) : ncnn::Mat();

    ncnn::Mat a_int8 = a;
    if (input_int8)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;
        ncnn::quantize_to_int8(a, a_int8, input_scales, opt);
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

    int flag = input_int8 ? TEST_LAYER_DISABLE_AUTO_INPUT_CASTING : 0;
    int ret = 0;
    if (input_int8)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = false;
        opt.use_bf16_storage = false;

        ret = test_layer_opt("ConvolutionDepthWise", pd, weights, opt, a_int8, use_requant ? 1.0f : 0.001f, flag);
    }
    else
    {
        ret = test_layer("ConvolutionDepthWise", pd, weights, a_int8, use_requant ? 1.0f : 0.001f, flag);
    }
    if (ret != 0)
    {
        fprintf(stderr, "test_convolutiondepthwise_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d int8_scale_term=%d input_int8=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, group, int8_scale_term, input_int8, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolutiondepthwise_1()
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
                  || test_convolutiondepthwise_int8(15, 7, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise_int8(15, 7, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise_int8(15, 7, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise_int8(15, 7, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise_int8(15, 7, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise_int8(15, 7, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise_int8(15, 7, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise_int8(15, 7, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise_int8(15, 7, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise_int8(15, 7, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise_int8(15, 7, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise_int8(15, 7, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise_int8(15, 7, 16, 16, k, d, s, p, 1, 16)
                  || test_convolutiondepthwise_int8(15, 7, 8, 8, k, d, s, p, 1, 8, false, 2)
                  || test_convolutiondepthwise_int8(15, 7, 8, 8, k, d, s, p, 1, 2, false, 2);

        if (ret != 0)
            return -1;
    }

    for (int i = 0; i < 16; i++)
    {
        const int k = kdsp[i][0];
        const int d = kdsp[i][1];
        const int s = kdsp[i][2];
        const int p = kdsp[i][3];

        int ret = 0
                  || test_convolutiondepthwise_int8(9, 7, 1, 1, k, d, s, p, 1, 1, true)
                  || test_convolutiondepthwise_int8(9, 7, 2, 2, k, d, s, p, 0, 1, true)
                  || test_convolutiondepthwise_int8(9, 7, 2, 2, k, d, s, p, 1, 2, true)
                  || test_convolutiondepthwise_int8(9, 7, 3, 3, k, d, s, p, 0, 3, true)
                  || test_convolutiondepthwise_int8(9, 7, 4, 2, k, d, s, p, 1, 2, true)
                  || test_convolutiondepthwise_int8(9, 7, 4, 4, k, d, s, p, 0, 4, true)
                  || test_convolutiondepthwise_int8(9, 7, 7, 7, k, d, s, p, 1, 7, true)
                  || test_convolutiondepthwise_int8(9, 7, 8, 8, k, d, s, p, 0, 2, true)
                  || test_convolutiondepthwise_int8(9, 7, 8, 8, k, d, s, p, 1, 8, true)
                  || test_convolutiondepthwise_int8(9, 7, 12, 12, k, d, s, p, 0, 4, true)
                  || test_convolutiondepthwise_int8(9, 7, 15, 15, k, d, s, p, 1, 15, true)
                  || test_convolutiondepthwise_int8(9, 7, 16, 8, k, d, s, p, 0, 2, true)
                  || test_convolutiondepthwise_int8(9, 7, 16, 16, k, d, s, p, 1, 16, true)
                  || test_convolutiondepthwise_int8(9, 7, 8, 8, k, d, s, p, 1, 8, true, 102)
                  || test_convolutiondepthwise_int8(9, 7, 8, 8, k, d, s, p, 1, 2, true, 102);

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_convolutiondepthwise_1_int8_input()
{
    return 0
           || test_convolutiondepthwise_int8(9, 7, 1, 1, 3, 1, 1, 1, 1, 1, false, 1, true)
           || test_convolutiondepthwise_int8(9, 7, 4, 4, 3, 1, 1, 1, 1, 4, false, 1, true)
           || test_convolutiondepthwise_int8(9, 7, 8, 8, 3, 1, 1, 1, 1, 2, false, 1, true)
           || test_convolutiondepthwise_int8(9, 7, 8, 8, 3, 1, 1, 1, 1, 2, false, 2, true)
           || test_convolutiondepthwise_int8(9, 7, 8, 8, 3, 1, 1, 1, 1, 8, true, 101, true)
           || test_convolutiondepthwise_int8(9, 7, 8, 8, 3, 1, 1, 1, 1, 2, true, 102, true);
}
#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return test_convolutiondepthwise_1() || test_convolutiondepthwise_1_int8_input() || test_convolutiondepthwise_2();
#else
    return test_convolutiondepthwise_2();
#endif
}
