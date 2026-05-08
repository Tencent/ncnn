// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_dequantize(const ncnn::Mat& a, int scale_data_size, int bias_data_size)
{
    ncnn::ParamDict pd;
    pd.set(0, scale_data_size);
    pd.set(1, bias_data_size);

    std::vector<ncnn::Mat> weights(bias_data_size ? 2 : 1);
    weights[0] = RandomMat(scale_data_size);
    if (bias_data_size)
        weights[1] = RandomMat(bias_data_size);

    int flag = TEST_LAYER_DISABLE_AUTO_INPUT_CASTING;
    int ret = test_layer("Dequantize", pd, weights, a, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_dequantize failed a.dims=%d a=(%d %d %d) scale_data_size=%d bias_data_size=%d\n", a.dims, a.w, a.h, a.c, scale_data_size, bias_data_size);
    }

    return ret;
}

static int test_dequantize_pack8(const ncnn::Mat& a, int scale_data_size, int bias_data_size)
{
    ncnn::ParamDict pd;
    pd.set(0, scale_data_size);
    pd.set(1, bias_data_size);

    std::vector<ncnn::Mat> weights(bias_data_size ? 2 : 1);
    weights[0] = RandomMat(scale_data_size);
    if (bias_data_size)
        weights[1] = RandomMat(bias_data_size);

    int flag = TEST_LAYER_DISABLE_AUTO_INPUT_CASTING | TEST_LAYER_ENABLE_FORCE_INPUT_PACK8;
    int ret = test_layer("Dequantize", pd, weights, a, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_dequantize_pack8 failed a.dims=%d a=(%d %d %d) scale_data_size=%d bias_data_size=%d\n", a.dims, a.w, a.h, a.c, scale_data_size, bias_data_size);
    }

    return ret;
}

static int test_dequantize_0()
{
    return 0
           || test_dequantize(RandomIntMat(11, 13, 48), 1, 48)
           || test_dequantize(RandomIntMat(11, 13, 48), 1, 1)
           || test_dequantize(RandomIntMat(11, 13, 48), 1, 0)
           || test_dequantize(RandomIntMat(11, 13, 48), 48, 48)
           || test_dequantize(RandomIntMat(11, 13, 48), 48, 1)
           || test_dequantize(RandomIntMat(11, 13, 48), 48, 0)
           || test_dequantize(RandomIntMat(5, 7, 24), 1, 24)
           || test_dequantize(RandomIntMat(5, 7, 24), 1, 1)
           || test_dequantize(RandomIntMat(5, 7, 24), 1, 0)
           || test_dequantize(RandomIntMat(5, 7, 24), 24, 24)
           || test_dequantize(RandomIntMat(5, 7, 24), 24, 1)
           || test_dequantize(RandomIntMat(5, 7, 24), 24, 0)
           || test_dequantize(RandomIntMat(7, 9, 12), 1, 12)
           || test_dequantize(RandomIntMat(7, 9, 12), 1, 1)
           || test_dequantize(RandomIntMat(7, 9, 12), 1, 0)
           || test_dequantize(RandomIntMat(7, 9, 12), 12, 12)
           || test_dequantize(RandomIntMat(7, 9, 12), 12, 1)
           || test_dequantize(RandomIntMat(7, 9, 12), 12, 0)
           || test_dequantize(RandomIntMat(3, 5, 13), 1, 13)
           || test_dequantize(RandomIntMat(3, 5, 13), 1, 1)
           || test_dequantize(RandomIntMat(3, 5, 13), 1, 0)
           || test_dequantize(RandomIntMat(3, 5, 13), 13, 13)
           || test_dequantize(RandomIntMat(3, 5, 13), 13, 1)
           || test_dequantize(RandomIntMat(3, 5, 13), 13, 0);
}

static int test_dequantize_1()
{
    return 0
           || test_dequantize(RandomIntMat(127, 48), 1, 48)
           || test_dequantize(RandomIntMat(127, 48), 1, 1)
           || test_dequantize(RandomIntMat(127, 48), 1, 0)
           || test_dequantize(RandomIntMat(127, 48), 48, 48)
           || test_dequantize(RandomIntMat(127, 48), 48, 1)
           || test_dequantize(RandomIntMat(127, 48), 48, 0)
           || test_dequantize(RandomIntMat(15, 24), 1, 24)
           || test_dequantize(RandomIntMat(15, 24), 1, 1)
           || test_dequantize(RandomIntMat(15, 24), 1, 0)
           || test_dequantize(RandomIntMat(15, 24), 24, 24)
           || test_dequantize(RandomIntMat(15, 24), 24, 1)
           || test_dequantize(RandomIntMat(15, 24), 24, 0)
           || test_dequantize(RandomIntMat(17, 12), 1, 12)
           || test_dequantize(RandomIntMat(17, 12), 1, 1)
           || test_dequantize(RandomIntMat(17, 12), 1, 0)
           || test_dequantize(RandomIntMat(17, 12), 12, 12)
           || test_dequantize(RandomIntMat(17, 12), 12, 1)
           || test_dequantize(RandomIntMat(17, 12), 12, 0)
           || test_dequantize(RandomIntMat(19, 15), 1, 15)
           || test_dequantize(RandomIntMat(19, 15), 1, 1)
           || test_dequantize(RandomIntMat(19, 15), 1, 0)
           || test_dequantize(RandomIntMat(19, 15), 15, 15)
           || test_dequantize(RandomIntMat(19, 15), 15, 1)
           || test_dequantize(RandomIntMat(19, 15), 15, 0);
}

static int test_dequantize_2()
{
    return 0
           || test_dequantize(RandomIntMat(128), 1, 1)
           || test_dequantize(RandomIntMat(128), 1, 0)
           || test_dequantize(RandomIntMat(120), 1, 1)
           || test_dequantize(RandomIntMat(120), 1, 0)
           || test_dequantize(RandomIntMat(124), 1, 1)
           || test_dequantize(RandomIntMat(124), 1, 0)
           || test_dequantize(RandomIntMat(127), 1, 1)
           || test_dequantize(RandomIntMat(127), 1, 0);
}

static int test_dequantize_3()
{
    return 0
           || test_dequantize_pack8(RandomIntMat(5, 7, 24), 1, 24)
           || test_dequantize_pack8(RandomIntMat(5, 7, 24), 1, 1)
           || test_dequantize_pack8(RandomIntMat(5, 7, 24), 1, 0)
           || test_dequantize_pack8(RandomIntMat(5, 7, 24), 24, 24)
           || test_dequantize_pack8(RandomIntMat(5, 7, 24), 24, 1)
           || test_dequantize_pack8(RandomIntMat(5, 7, 24), 24, 0)
           || test_dequantize_pack8(RandomIntMat(15, 24), 1, 24)
           || test_dequantize_pack8(RandomIntMat(15, 24), 1, 1)
           || test_dequantize_pack8(RandomIntMat(15, 24), 1, 0)
           || test_dequantize_pack8(RandomIntMat(15, 24), 24, 24)
           || test_dequantize_pack8(RandomIntMat(15, 24), 24, 1)
           || test_dequantize_pack8(RandomIntMat(15, 24), 24, 0)
           || test_dequantize_pack8(RandomIntMat(128), 1, 1)
           || test_dequantize_pack8(RandomIntMat(128), 1, 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_dequantize_0()
           || test_dequantize_1()
           || test_dequantize_2()
           || test_dequantize_3();
}
