// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "layer/deconvolutiondepthwise1d.h"
#include "testutil.h"

static int test_deconvolutiondepthwise1d(int w, int h, int outh, int kernel, int dilation, int stride, int pad, int bias, int group, int output_pad_right, int output_w)
{
    ncnn::Mat a = RandomMat(w, h);

    if (output_w > 0 && pad != -233 && pad != -234)
    {
        pad = -233;
    }

    ncnn::ParamDict pd;
    pd.set(0, outh);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outh / group * h / group * kernel * group);
    pd.set(7, group);

    int activation_type = RAND() % 5; // 0 1 2 3 4
    ncnn::Mat activation_params(2);
    activation_params[0] = RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    pd.set(18, output_pad_right);
    pd.set(20, output_w);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(outh / group * h / group * kernel * group);
    weights[1] = RandomMat(outh);

    int ret = test_layer<ncnn::DeconvolutionDepthWise1D>("DeconvolutionDepthWise1D", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_deconvolutiondepthwise1d failed w=%d h=%d outh=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d act=%d actparams=[%f,%f] output_pad_right=%d output_w=%d\n", w, h, outh, kernel, dilation, stride, pad, bias, group, activation_type, activation_params[0], activation_params[1], output_pad_right, output_w);
    }

    return ret;
}

static int test_deconvolutiondepthwise1d_0()
{
    static const int kdsp[16][4] = {
        {1, 1, 1, 0},
        {1, 1, 2, 0},
        {2, 1, 1, 1},
        {2, 1, 2, -233},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {3, 2, 1, 1},
        {4, 1, 1, -233},
        {4, 1, 2, -234},
        {4, 2, 1, -234},
        {5, 1, 1, 2},
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
                  || test_deconvolutiondepthwise1d(15, 1, 1, k, d, s, p, 1, 1, 0, 0)
                  || test_deconvolutiondepthwise1d(15, 2, 2, k, d, s, p, 0, 1, 1, 7)
                  || test_deconvolutiondepthwise1d(15, 2, 2, k, d, s, p, 1, 2, 1, 0)
                  || test_deconvolutiondepthwise1d(15, 3, 3, k, d, s, p, 0, 3, 0, 0)
                  || test_deconvolutiondepthwise1d(15, 4, 2, k, d, s, p, 1, 2, 0, 7)
                  || test_deconvolutiondepthwise1d(15, 4, 4, k, d, s, p, 0, 4, 2, 0)
                  || test_deconvolutiondepthwise1d(15, 7, 7, k, d, s, p, 1, 7, 2, 0)
                  || test_deconvolutiondepthwise1d(15, 8, 8, k, d, s, p, 0, 2, 0, 7)
                  || test_deconvolutiondepthwise1d(15, 8, 8, k, d, s, p, 1, 8, 0, 0)
                  || test_deconvolutiondepthwise1d(15, 12, 12, k, d, s, p, 0, 4, 3, 0)
                  || test_deconvolutiondepthwise1d(15, 15, 15, k, d, s, p, 1, 15, 3, 7)
                  || test_deconvolutiondepthwise1d(15, 16, 8, k, d, s, p, 0, 2, 0, 0)
                  || test_deconvolutiondepthwise1d(15, 16, 16, k, d, s, p, 1, 16, 0, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_deconvolutiondepthwise1d_0();
}
