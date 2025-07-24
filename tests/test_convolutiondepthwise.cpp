// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_convolutiondepthwise(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int group)
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

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(outch / group * c / group * kernel * kernel * group);
    weights[1] = RandomMat(outch);

    int ret = test_layer("ConvolutionDepthWise", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolutiondepthwise failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, group, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolutiondepthwise_0()
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
                  || test_convolutiondepthwise(15, 7, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise(15, 7, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise(15, 7, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise(15, 7, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise(15, 7, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise(15, 7, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise(15, 7, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise(15, 7, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise(15, 7, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise(15, 7, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise(15, 7, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise(15, 7, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise(15, 7, 16, 16, k, d, s, p, 1, 16)
                  || test_convolutiondepthwise(18, 17, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise(18, 17, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise(18, 17, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise(18, 17, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise(18, 17, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise(18, 17, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise(18, 17, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise(18, 17, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise(18, 17, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise(18, 17, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise(18, 17, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise(18, 17, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise(18, 17, 16, 16, k, d, s, p, 1, 16)
                  || test_convolutiondepthwise(25, 33, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise(25, 33, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise(25, 33, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise(25, 33, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise(25, 33, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise(25, 33, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise(25, 33, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise(25, 33, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise(25, 33, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise(25, 33, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise(25, 33, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise(25, 33, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise(25, 33, 16, 16, k, d, s, p, 1, 16);

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_convolutiondepthwise_0();
}
