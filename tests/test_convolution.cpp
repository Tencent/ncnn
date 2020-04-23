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

#include "testutil.h"

#include "layer/convolution.h"

static int test_convolution(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);// num_output
    pd.set(1, kernel);// kernel_w
    pd.set(2, dilation);// dilation_w
    pd.set(3, stride);// stride_w
    pd.set(4, pad);// pad_w
    pd.set(5, bias);// bias_term
    pd.set(6, outch*c*kernel*kernel);

    int activation_type = RAND() % 5;// 0 1 2 3 4
    ncnn::Mat activation_params(2);
    activation_params[0] = RandomFloat(-1, 0);// alpha
    activation_params[1] = RandomFloat(0, 1);// beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch*c*kernel*kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::Convolution>("Convolution", pd, weights, opt, a);
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
        {2, 1, 2, 1},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {3, 2, 1, 1},
        {4, 1, 1, 2},
        {4, 1, 2, 2},
        {4, 2, 1, 2},
        {5, 1, 1, 2},
        {5, 1, 2, 2},
        {5, 2, 2, 2},
        {7, 1, 1, 3},
        {7, 1, 2, 3},
        {7, 2, 1, 3},
    };

    for (int i=0; i<16; i++)
    {
        int ret = 0
            || test_convolution(9, 7, 1, 1, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
            || test_convolution(9, 7, 4, 13, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 0)
            || test_convolution(9, 7, 13, 4, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
            || test_convolution(9, 7, 4, 8, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 0)
            || test_convolution(9, 7, 8, 4, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
            || test_convolution(9, 7, 8, 13, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 0)
            || test_convolution(9, 7, 13, 8, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
            || test_convolution(9, 7, 16, 16, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 0)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

void set_param(ncnn::Convolution* layer)
{
    layer->use_int8_requantize = true;
    layer->top_blob_int8_scale = 64.f;
    return;
}

static int test_convolution_int8(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, bool requant = false)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);// num_output
    pd.set(1, kernel);// kernel_w
    pd.set(2, dilation);// dilation_w
    pd.set(3, stride);// stride_w
    pd.set(4, pad);// pad_w
    pd.set(5, bias);// bias_term
    pd.set(6, outch*c*kernel*kernel);
    pd.set(8, 1);// int8_scale_term

    std::vector<ncnn::Mat> weights(bias ? 4 : 3);
    weights[0] = RandomMat(outch*c*kernel*kernel);
    if (bias)
    {
        weights[1] = RandomMat(outch);
        weights[2] = RandomMat(outch);
        weights[3] = RandomMat(1);
    }
    else
    {
        weights[1] = RandomMat(outch);
        weights[2] = RandomMat(1);
    }

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_int8_inference = true;

    int ret = test_layer<ncnn::Convolution>("Convolution", pd, weights, opt, a, 0.001f, requant ? set_param : 0);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d requant=%d\n", w, h, c, outch, kernel, dilation, stride, pad, bias, requant);
    }

    return 0;
}

static int test_convolution_1()
{
    static const int kdsp[24][4] = {
        {1, 1, 1, 0},
        {1, 1, 2, 0},
        {2, 1, 1, 1},
        {2, 1, 2, 1},
        {2, 2, 1, 1},
        {2, 2, 2, 1},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {3, 2, 1, 1},
        {3, 2, 2, 1},
        {4, 1, 1, 2},
        {4, 1, 2, 2},
        {4, 2, 1, 2},
        {4, 2, 2, 2},
        {5, 1, 1, 2},
        {5, 1, 2, 2},
        {5, 2, 1, 2},
        {5, 2, 2, 2},
        {7, 1, 1, 3},
        {7, 1, 2, 3},
        {7, 1, 3, 3},
        {7, 2, 1, 3},
        {7, 2, 2, 3},
        {7, 2, 3, 3},
    };
    for (int i=0; i<24; i++)
    {
        int ret = 0
            || test_convolution_int8(9, 7, 1, 1, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
            || test_convolution_int8(9, 7, 2, 2, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
            || test_convolution_int8(9, 7, 3, 3, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
            || test_convolution_int8(9, 7, 4, 4, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
            || test_convolution_int8(9, 7, 7, 7, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
            || test_convolution_int8(9, 7, 8, 8, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
            || test_convolution_int8(9, 7, 15, 15, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
            || test_convolution_int8(9, 7, 16, 16, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        ;

        if (ret != 0)
            return -1;
    }
    for (int i=0; i<20; i++)
    {
        int ret = 0
            || test_convolution_int8(9, 7, 1, 1, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution_int8(9, 7, 1, 1, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution_int8(9, 7, 2, 2, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution_int8(9, 7, 3, 3, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution_int8(9, 7, 4, 4, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution_int8(9, 7, 7, 7, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution_int8(9, 7, 8, 8, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution_int8(9, 7, 15, 15, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution_int8(9, 7, 16, 16, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
        ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_convolution_0() || test_convolution_1();
}
