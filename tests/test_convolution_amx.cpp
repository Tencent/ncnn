// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_convolution_impl(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int activation_type, const ncnn::Option* opt, int flag)
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

    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? 0.2f : -0.1f; // alpha
    activation_params[1] = 0.3f;                                  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * c * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    float epsilon = 0.001;

    int ret = test_layer_opt("Convolution", pd, weights, *opt, a, epsilon, flag | TEST_LAYER_DISABLE_GPU_TESTING);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type);
        return ret;
    }

    return 0;
}

#if defined __APPLE__ && __aarch64__
// the amx 1x1 kernels need num_input and num_output >= 16 and at least 16 output pixels
static const int amx_channels[][3] = {
    {16, 16, 0},
    {16, 16, 1},
    {17, 19, 1},
    {31, 16, 0},
    {24, 32, 1},
    {64, 64, 1},
};
// exact 16 wide pixel tiles first, then the spatial remainders
// the 32 wide fp16 tiles read the tail backwards from N - 32, so widths just past a
// multiple of 32 (49 and 65) hand an overlapping tile pair to two workers
static const int amx_shapes[][2] = {
    {4, 4},
    {16, 16},
    {9, 7},
    {18, 17},
    {25, 33},
    {7, 7},
    {13, 5},
};

static int test_convolution_apple_amx_fp32()
{
    for (int threading = 0; threading < 2; threading++)
    {
        for (int pack = 0; pack < 2; pack++)
        {
            ncnn::Option opt;
            opt.use_packing_layout = pack;
            // fp16 and bf16 storage have their own dispatch, force the fp32 kernel
            opt.use_fp16_packed = false;
            opt.use_fp16_storage = false;
            opt.use_fp16_arithmetic = false;
            opt.use_bf16_packed = false;
            opt.use_bf16_storage = false;
            opt.use_apple_amx = true;
            opt.use_arm_sme = false;
            opt.use_winograd_convolution = false;

            for (int i = 0; i < 6; i++)
            {
                for (int j = 0; j < 7; j++)
                {
                    if (test_convolution_impl(amx_shapes[j][0], amx_shapes[j][1], amx_channels[i][0], amx_channels[i][1], 1, 1, 1, 1, amx_channels[i][2], 0, &opt, threading ? TEST_LAYER_ENABLE_THREADING : 0) != 0)
                        return -1;
                }
            }
        }
    }

    return 0;
}

static int test_convolution_apple_amx_activation()
{
    // the amx fp32 epilogue writes the bias during the store, the activation runs afterwards
    static const int acts[] = {1, 2, 3, 4, 5, 6};

    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_apple_amx = true;
    opt.use_arm_sme = false;
    opt.use_winograd_convolution = false;

    for (size_t i = 0; i < sizeof(acts) / sizeof(acts[0]); i++)
    {
        if (test_convolution_impl(18, 17, 17, 19, 1, 1, 1, 1, 1, acts[i], &opt, 0) != 0)
            return -1;

        opt.use_packing_layout = false;
        if (test_convolution_impl(18, 17, 17, 19, 1, 1, 1, 1, 1, acts[i], &opt, 0) != 0)
            return -1;

        opt.use_packing_layout = true;
    }

    return 0;
}

static int test_convolution_apple_amx_fp16()
{
    // the fp16 amx kernel is gated on num_output > 16 and num_input >= 64
    static const int channels[][3] = {
        {64, 17, 0},
        {64, 17, 1},
        {80, 33, 1},
        {128, 128, 1},
    };

    for (int threading = 0; threading < 2; threading++)
    {
        for (int arith = 0; arith < 2; arith++)
        {
            ncnn::Option opt;
            opt.use_packing_layout = true;
            opt.use_fp16_packed = true;
            opt.use_fp16_storage = true;
            opt.use_fp16_arithmetic = arith;
            opt.use_apple_amx = true;
            opt.use_arm_sme = false;
            opt.use_winograd_convolution = false;

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 7; j++)
                {
                    if (test_convolution_impl(amx_shapes[j][0], amx_shapes[j][1], channels[i][0], channels[i][1], 1, 1, 1, 1, channels[i][2], 0, &opt, threading ? TEST_LAYER_ENABLE_THREADING : 0) != 0)
                        return -1;
                }
            }
        }
    }

    return 0;
}

#endif // defined __APPLE__ && __aarch64__

int main()
{
    SRAND(7767517);

#if defined __APPLE__ && __aarch64__
    if (!ncnn::cpu_support_apple_amx())
        return 0;

    return 0
           || test_convolution_apple_amx_fp32()
           || test_convolution_apple_amx_activation()
           || test_convolution_apple_amx_fp16();
#else
    return 0;
#endif
}
