// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

static int test_convolution3d(int w, int h, int d, int c, int outch, int kernel, int dilation, int stride, int pad, int bias)
{
    ncnn::Mat a = RandomMat(w, h, d, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);    // num_output
    pd.set(1, kernel);   // kernel_w
    pd.set(2, dilation); // dilation_w
    pd.set(3, stride);   // stride_w
    pd.set(4, pad);      // pad_w
    pd.set(5, bias);     // bias_term
    pd.set(6, outch * c * kernel * kernel * kernel);

    int activation_type = RAND() % 6; // 0 1 2 3 4 5
    ncnn::Mat activation_params(2);
    activation_params[0] = RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * c * kernel * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer("Convolution3D", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution3d failed w=%d h=%d d=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, d, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_convolution3d_0()
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
                  || test_convolution3d(11, 10, 9, 1, 1, k, d, s, p, 1)
                  || test_convolution3d(11, 10, 9, 4, 13, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 13, 4, k, d, s, p, 1)
                  || test_convolution3d(11, 10, 9, 12, 12, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 8, 12, k, d, s, p, 1)
                  || test_convolution3d(11, 10, 9, 8, 13, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 13, 8, k, d, s, p, 1)
                  || test_convolution3d(11, 10, 9, 12, 16, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 15, 15, k, d, s, p, 0)
                  || test_convolution3d(11, 10, 9, 16, 16, k, d, s, p, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_convolution3d_0();
}
