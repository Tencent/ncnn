// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "layer/convolution.h"
#include "testutil.h"

static int test_convolution(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias)
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
    // larget epsilon for winograd optimization
    if (kernel == 3 && dilation == 1 && stride == 1 && c >= 16 && outch >= 16)
    {
        Randomize(a, -1, 1);
        if (c >= 64 || outch >= 64)
            Randomize(weights[0], -0.3, 0.3);
        else
            Randomize(weights[0], -1, 1);
        epsilon = 0.002;
    }

    int ret = test_layer<ncnn::Convolution>("Convolution", pd, weights, a, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolution_0()
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
                  || test_convolution(9, 7, 1, 1, k, d, s, p, 1)
                  || test_convolution(9, 7, 4, 13, k, d, s, p, 0)
                  || test_convolution(9, 7, 13, 4, k, d, s, p, 1)
                  || test_convolution(9, 7, 12, 12, k, d, s, p, 0)
                  || test_convolution(9, 7, 8, 12, k, d, s, p, 1)
                  || test_convolution(9, 7, 8, 13, k, d, s, p, 0)
                  || test_convolution(9, 7, 13, 8, k, d, s, p, 1)
                  || test_convolution(9, 7, 12, 16, k, d, s, p, 0)
                  || test_convolution(9, 7, 15, 15, k, d, s, p, 0)
                  || test_convolution(9, 7, 16, 16, k, d, s, p, 0)
                  || test_convolution(18, 17, 1, 1, k, d, s, p, 1)
                  || test_convolution(18, 17, 4, 13, k, d, s, p, 0)
                  || test_convolution(18, 17, 13, 4, k, d, s, p, 1)
                  || test_convolution(18, 17, 12, 12, k, d, s, p, 0)
                  || test_convolution(18, 17, 8, 12, k, d, s, p, 1)
                  || test_convolution(18, 17, 8, 13, k, d, s, p, 0)
                  || test_convolution(18, 17, 13, 8, k, d, s, p, 1)
                  || test_convolution(18, 17, 12, 16, k, d, s, p, 0)
                  || test_convolution(18, 17, 15, 15, k, d, s, p, 0)
                  || test_convolution(18, 17, 16, 16, k, d, s, p, 0)
                  || test_convolution(25, 33, 1, 1, k, d, s, p, 1)
                  || test_convolution(25, 33, 4, 13, k, d, s, p, 0)
                  || test_convolution(25, 33, 13, 4, k, d, s, p, 1)
                  || test_convolution(25, 33, 12, 12, k, d, s, p, 0)
                  || test_convolution(25, 33, 8, 12, k, d, s, p, 1)
                  || test_convolution(25, 33, 8, 13, k, d, s, p, 0)
                  || test_convolution(25, 33, 13, 8, k, d, s, p, 1)
                  || test_convolution(25, 33, 12, 16, k, d, s, p, 0)
                  || test_convolution(25, 33, 15, 15, k, d, s, p, 0)
                  || test_convolution(25, 33, 16, 16, k, d, s, p, 0);

        if (ret != 0)
            return -1;
    }

    return 0
           || test_convolution(7, 5, 1, 4, 3, 1, 1, 1, 1)
           || test_convolution(14, 5, 1, 4, 3, 1, 2, 1, 1)
           || test_convolution(11, 5, 2, 12, 2, 2, 2, 1, 1)
           || test_convolution(15, 11, 4, 4, 3, 1, 1, 1, 1)
           || test_convolution(15, 11, 8, 8, 3, 1, 1, 1, 1)
           || test_convolution(11, 11, 8, 16, 3, 1, 1, 1, 1)
           || test_convolution(13, 16, 16, 24, 3, 1, 1, 1, 1)
           || test_convolution(20, 19, 24, 24, 3, 1, 1, 1, 1)
           || test_convolution(8, 8, 16, 24, 3, 1, 1, 1, 0)
           || test_convolution(4, 8, 16, 24, 3, 1, 1, 1, 1)
           || test_convolution(4, 20, 16, 24, 3, 1, 1, 1, 0)
           || test_convolution(6, 7, 64, 64, 3, 1, 2, 0, 1)
           || test_convolution(15, 17, 24, 32, 1, 1, 1, 0, 0)
           || test_convolution(15, 17, 24, 32, 1, 1, 2, 0, 1)
           || test_convolution(15, 17, 24, 32, 3, 1, 2, 0, 1)
           || test_convolution(15, 17, 32, 24, 1, 1, 1, 0, 0)
           || test_convolution(15, 17, 32, 24, 1, 1, 2, 0, 1)
           || test_convolution(15, 17, 32, 24, 3, 1, 2, 0, 1)
           || test_convolution(15, 17, 32, 28, 1, 1, 1, 0, 0)
           || test_convolution(15, 17, 32, 28, 1, 1, 2, 0, 1)
           || test_convolution(15, 17, 32, 28, 3, 1, 2, 0, 1)
           || test_convolution(15, 17, 26, 32, 1, 1, 1, 0, 0)
           || test_convolution(15, 17, 26, 32, 1, 1, 2, 0, 1)
           || test_convolution(15, 17, 26, 32, 3, 1, 2, 0, 1)
           || test_convolution(15, 17, 32, 26, 1, 1, 1, 0, 0)
           || test_convolution(15, 17, 32, 26, 1, 1, 2, 0, 1)
           || test_convolution(15, 17, 32, 26, 3, 1, 2, 0, 1)
           || test_convolution(30, 30, 32, 26, 3, 1, 1, 1, 0)
           || test_convolution(12, 18, 8, 16, 3, 1, 1, 1, 1)
           || test_convolution(42, 18, 32, 160, 3, 1, 1, 1, 1)
           || test_convolution(12, 18, 32, 160, 3, 1, 1, 1, 1)
           || test_convolution(12, 18, 4, 12, 3, 1, 1, 1, 1)
           || test_convolution(42, 18, 28, 140, 3, 1, 1, 1, 1)
           || test_convolution(12, 18, 28, 140, 3, 1, 1, 1, 1);
}

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

    int ret = test_layer<ncnn::Convolution>("Convolution", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution_vec failed w=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
    }

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
           || test_convolution_vec(64, 128, 1, 1, 1, 0, 0);
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

    int ret = test_layer<ncnn::Convolution>("Convolution", pd, weights, as);
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

#if NCNN_INT8
static int test_convolution_int8(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, bool requant = false)
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

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 5 : 4);
    weights[0] = RandomMat(outch * c * kernel * kernel);

    ncnn::Mat weight_scales = scales_mat(weights[0], outch, c * kernel * kernel, c * kernel * kernel);
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
    int ret = test_layer<ncnn::Convolution>("Convolution", pd, weights, a, requant ? 1.0f : 0.001f, 0, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d requant=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, requant, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolution_1()
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
                  || test_convolution_int8(9, 7, 1, 1, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 2, 2, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 3, 3, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 4, 4, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 7, 7, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 8, 8, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 15, 15, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 16, 15, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 15, 16, k, d, s, p, 1)
                  || test_convolution_int8(9, 7, 16, 16, k, d, s, p, 1);

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
                  || test_convolution_int8(9, 7, 1, 1, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 1, 1, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 2, 2, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 3, 3, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 4, 4, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 7, 7, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 8, 8, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 15, 15, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 16, 15, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 15, 16, k, d, s, p, 1, true)
                  || test_convolution_int8(9, 7, 16, 16, k, d, s, p, 1, true);

        if (ret != 0)
            return -1;
    }

    return 0
           || test_convolution_int8(11, 11, 8, 16, 3, 1, 1, 1, 1)
           || test_convolution_int8(13, 16, 16, 24, 3, 1, 1, 1, 1)
           || test_convolution_int8(8, 8, 16, 24, 3, 1, 1, 1, 0)
           || test_convolution_int8(4, 8, 16, 24, 3, 1, 1, 1, 1)
           || test_convolution_int8(4, 20, 16, 24, 3, 1, 1, 1, 0)
           || test_convolution_int8(6, 7, 64, 64, 3, 1, 2, 0, 1)
           || test_convolution_int8(25, 33, 16, 15, 3, 1, 1, 1, 0)
           || test_convolution_int8(7, 7, 15, 12, 3, 1, 1, 1, 0);
}
#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return 0
           || test_convolution_0()
           || test_convolution_1()
           || test_convolution_2()
           || test_convolution_3();
#else
    return 0
           || test_convolution_0()
           || test_convolution_2()
           || test_convolution_3();
#endif
}
