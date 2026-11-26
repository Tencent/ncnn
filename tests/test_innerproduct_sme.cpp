// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if defined __aarch64__ && NCNN_ARM_SME

// innerproduct takes the gemm branch when the bottom blob is 2d and its width is
// the number of input features, so the weights are num_output rows of num_input
// the harness packs a 2d blob along h whenever the sample count is a multiple of 4,
// so the shapes below drive both the pack1 and the pack4 input layout of the tile kernel
static int test_innerproduct_impl(int w, int h, int outch, int bias, int activation_type, const ncnn::Option* opt, int flag)
{
    ncnn::Mat a = RandomMat(w, h);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, bias);
    pd.set(2, outch * w);

    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? 0.2f : -0.1f; // alpha
    activation_params[1] = 0.3f;                                  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * w);
    if (bias)
        weights[1] = RandomMat(outch);

    float epsilon = 0.001;

    int ret = test_layer_opt("InnerProduct", pd, weights, *opt, a, epsilon, flag | TEST_LAYER_DISABLE_GPU_TESTING);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct failed w=%d h=%d outch=%d bias=%d act=%d\n", w, h, outch, bias, activation_type);
        return ret;
    }

    return 0;
}

// num_input and num_output >= 16 and at least 16 samples take the tile kernel,
// the rows below 16 on either axis must keep the dot product result
static const int ip_shapes[][3] = {
    {16, 16, 16},
    {16, 16, 17},
    {17, 19, 24},
    {24, 32, 16},
    {31, 16, 31},
    {64, 8, 32},
    {15, 64, 16},
    {20, 16, 4},
    {32, 48, 64},
};

static int test_innerproduct_arm_sme_fp32()
{
    for (int threading = 0; threading < 2; threading++)
    {
        for (int pack = 0; pack < 2; pack++)
        {
            ncnn::Option opt;
            opt.use_packing_layout = pack;
            opt.use_fp16_packed = false;
            opt.use_fp16_storage = false;
            opt.use_fp16_arithmetic = false;
            opt.use_bf16_packed = false;
            opt.use_bf16_storage = false;
            opt.use_arm_sme = true;

            for (int i = 0; i < 9; i++)
            {
                for (int bias = 0; bias < 2; bias++)
                {
                    if (test_innerproduct_impl(ip_shapes[i][0], ip_shapes[i][1], ip_shapes[i][2], bias, 0, &opt, threading ? TEST_LAYER_ENABLE_THREADING : 0) != 0)
                        return -1;
                }
            }
        }
    }

    return 0;
}

static int test_innerproduct_arm_sme_activation()
{
    static const int acts[] = {1, 2, 3, 4, 5, 6};

    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    opt.use_arm_sme = true;

    for (size_t i = 0; i < sizeof(acts) / sizeof(acts[0]); i++)
    {
        for (int bias = 0; bias < 2; bias++)
        {
            if (test_innerproduct_impl(17, 19, 24, bias, acts[i], &opt, 0) != 0)
                return -1;

            opt.use_packing_layout = false;
            if (test_innerproduct_impl(17, 19, 24, bias, acts[i], &opt, 0) != 0)
                return -1;

            opt.use_packing_layout = true;
        }
    }

    return 0;
}

#endif // defined __aarch64__ && NCNN_ARM_SME

int main()
{
    SRAND(7767517);

#if defined __aarch64__ && NCNN_ARM_SME
    if (!ncnn::cpu_support_arm_sme_f32f32())
        return 0;

    return 0
           || test_innerproduct_arm_sme_fp32()
           || test_innerproduct_arm_sme_activation();
#else
    return 0;
#endif
}
