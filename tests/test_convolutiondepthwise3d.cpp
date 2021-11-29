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

#include "layer/convolutiondepthwise3d.h"
#include "testutil.h"

static int test_convolutiondepthwise3d(int w, int h, int d, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, int group)
{
    ncnn::Mat a = RandomMat(w, h, d, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);    // num_output
    pd.set(1, kernel);   // kernel_w
    pd.set(2, dilation); // dilation_w
    pd.set(3, stride);   // stride_w
    pd.set(4, pad);      // pad_w
    pd.set(5, bias);     // bias_term
    pd.set(6, outch / group * c / group * kernel * kernel * kernel * group);
    pd.set(7, group);

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(outch / group * c / group * kernel * kernel * kernel * group);
    weights[1] = RandomMat(outch);

    int ret = test_layer<ncnn::ConvolutionDepthWise3D>("ConvolutionDepthWise3D", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolutiondepthwise3d failed w=%d h=%d d=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d group=%d act=%d actparams=[%f,%f]\n", w, h, d, c, outch, kernel, dilation, stride, pad, bias, group, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolutiondepthwise3d_0()
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
                  || test_convolutiondepthwise3d(15, 9, 7, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise3d(15, 9, 7, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise3d(15, 9, 7, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise3d(15, 9, 7, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise3d(15, 9, 7, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise3d(15, 9, 7, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise3d(15, 9, 7, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise3d(15, 9, 7, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise3d(15, 9, 7, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise3d(15, 9, 7, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise3d(15, 9, 7, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise3d(15, 9, 7, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise3d(15, 9, 7, 16, 16, k, d, s, p, 1, 16)
                  || test_convolutiondepthwise3d(18, 17, 11, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise3d(18, 17, 11, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise3d(18, 17, 11, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise3d(18, 17, 11, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise3d(18, 17, 11, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise3d(18, 17, 11, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise3d(18, 17, 11, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise3d(18, 17, 11, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise3d(18, 17, 11, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise3d(18, 17, 11, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise3d(18, 17, 11, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise3d(18, 17, 11, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise3d(18, 17, 11, 16, 16, k, d, s, p, 1, 16)
                  || test_convolutiondepthwise3d(25, 23, 13, 1, 1, k, d, s, p, 1, 1)
                  || test_convolutiondepthwise3d(25, 23, 13, 2, 2, k, d, s, p, 0, 1)
                  || test_convolutiondepthwise3d(25, 23, 13, 2, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise3d(25, 23, 13, 3, 3, k, d, s, p, 0, 3)
                  || test_convolutiondepthwise3d(25, 23, 13, 4, 2, k, d, s, p, 1, 2)
                  || test_convolutiondepthwise3d(25, 23, 13, 4, 4, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise3d(25, 23, 13, 7, 7, k, d, s, p, 1, 7)
                  || test_convolutiondepthwise3d(25, 23, 13, 8, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise3d(25, 23, 13, 8, 8, k, d, s, p, 1, 8)
                  || test_convolutiondepthwise3d(25, 23, 13, 12, 12, k, d, s, p, 0, 4)
                  || test_convolutiondepthwise3d(25, 23, 13, 15, 15, k, d, s, p, 1, 15)
                  || test_convolutiondepthwise3d(25, 23, 13, 16, 8, k, d, s, p, 0, 2)
                  || test_convolutiondepthwise3d(25, 23, 13, 16, 16, k, d, s, p, 1, 16);

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_convolutiondepthwise3d_0();
}
