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

static int test_convolution(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, bool use_packing_layout)
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

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch*c*kernel*kernel);
    if (bias)
        weights[1] = RandomMat(outch);
    ncnn::ModelBinFromMatArray mb(weights.data());

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_packing_layout = use_packing_layout;

    int ret = test_layer<ncnn::Convolution>("Convolution", pd, mb, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d use_packing_layout=%d\n", w, h, c, outch, kernel, dilation, stride, pad, bias, use_packing_layout);
    }

    return ret;
}

static int test_convolution_0()
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
            || test_convolution(13, 11, 1, 1, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, false)
            || test_convolution(13, 11, 2, 2, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, false)
            || test_convolution(13, 11, 3, 3, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, false)
            || test_convolution(13, 11, 4, 4, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, false)
            || test_convolution(13, 11, 7, 7, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, false)
            || test_convolution(13, 11, 8, 8, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, false)
            || test_convolution(13, 11, 15, 15, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, false)
            || test_convolution(13, 11, 16, 16, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, false)

            || test_convolution(13, 11, 1, 1, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution(13, 11, 2, 2, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution(13, 11, 3, 3, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution(13, 11, 3, 12, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution(13, 11, 4, 4, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution(13, 11, 8, 3, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution(13, 11, 8, 8, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution(13, 11, 16, 4, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            || test_convolution(13, 11, 16, 16, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, true)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_convolution_int8(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias)
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
    ncnn::ModelBinFromMatArray mb(weights.data());

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_int8_inference = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;
    opt.use_packing_layout = false;

    int ret = test_layer<ncnn::Convolution>("Convolution", pd, mb, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d\n", w, h, c, outch, kernel, dilation, stride, pad, bias);
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
        || test_convolution_int8(13, 11, 1, 1, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 2, 2, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 3, 3, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 4, 4, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 7, 7, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 8, 8, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 15, 15, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 16, 16, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)

        || test_convolution_int8(13, 11, 1, 1, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 2, 2, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 3, 3, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 3, 12, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 4, 4, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 8, 3, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 8, 8, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 16, 4, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
        || test_convolution_int8(13, 11, 16, 16, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1)
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
