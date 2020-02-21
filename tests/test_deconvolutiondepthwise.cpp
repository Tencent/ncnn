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

#include "layer/deconvolutiondepthwise.h"

static int test_deconvolutiondepthwise(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int group)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);// num_output
    pd.set(1, kernel);// kernel_w
    pd.set(2, dilation);// dilation_w
    pd.set(3, stride);// stride_w
    pd.set(4, pad);// pad_w
    pd.set(5, bias);// bias_term
    pd.set(6, outch/group*c/group*kernel*kernel*group);
    pd.set(7, group);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(outch/group*c/group*kernel*kernel*group);
    weights[1] = RandomMat(outch);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;

    int ret = test_layer<ncnn::DeconvolutionDepthWise>("DeconvolutionDepthWise", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_deconvolutiondepthwise failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d\n", w, h, c, outch, kernel, dilation, stride, pad, bias, group);
    }

    return ret;
}

static int test_deconvolutiondepthwise_0()
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
            || test_deconvolutiondepthwise(13, 11, 1, 1, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, 1)
            || test_deconvolutiondepthwise(13, 11, 2, 2, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, 1)
            || test_deconvolutiondepthwise(13, 11, 2, 2, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, 2)
            || test_deconvolutiondepthwise(13, 11, 3, 3, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, 3)
            || test_deconvolutiondepthwise(13, 11, 4, 2, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, 2)
            || test_deconvolutiondepthwise(13, 11, 4, 4, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, 4)
            || test_deconvolutiondepthwise(13, 11, 7, 7, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, 7)
            || test_deconvolutiondepthwise(13, 11, 8, 8, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, 2)
            || test_deconvolutiondepthwise(13, 11, 8, 8, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, 8)
            || test_deconvolutiondepthwise(13, 11, 12, 12, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, 4)
            || test_deconvolutiondepthwise(13, 11, 15, 15, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, 15)
            || test_deconvolutiondepthwise(13, 11, 16, 8, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, 2)
            || test_deconvolutiondepthwise(13, 11, 16, 16, kdsp[i][0], kdsp[i][1], kdsp[i][2], kdsp[i][3], 1, 16)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_deconvolutiondepthwise_0();
}
