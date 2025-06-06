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

static int test_deformableconv2d(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias)
{
    const int kernel_extent_w = dilation * (kernel - 1) + 1;
    const int kernel_extent_h = dilation * (kernel - 1) + 1;
    const int out_w = (w + pad + pad - kernel_extent_w) / stride + 1;
    const int out_h = (h + pad + pad - kernel_extent_h) / stride + 1;
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(w, h, c);
    a[1] = RandomMat(out_w, out_h, kernel * kernel * 2);
    a[2] = RandomMat(out_w, out_h, kernel * kernel);

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
    int ret = test_layer("DeformableConv2D", pd, weights, a, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_deformableconv2d failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
    }

    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_storage = false;
        opt.use_shader_pack8 = false;
        opt.use_image_storage = false;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("DeformableConv2D", pd, weights, opt, a, 1, epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_deformableconv2d failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
        }
    }

    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = true;
        opt.use_fp16_storage = true;
        opt.use_fp16_arithmetic = true;
        opt.use_bf16_storage = true;
        opt.use_shader_pack8 = true;
        opt.use_image_storage = true;
        opt.use_sgemm_convolution = false;
        opt.use_winograd_convolution = false;

        ret = test_layer_opt("DeformableConv2D", pd, weights, opt, a, 1, epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_deformableconv2d failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
        }
    }

    return ret;
}

static int test_deformableconv2d_0()
{
    static const int kdsp[10][4] = {
        {1, 1, 1, 0},
        {1, 1, 2, 0},
        {2, 1, 1, 1},
        {2, 1, 2, 0},
        {3, 1, 1, 1},
        {3, 1, 2, 1},
        {3, 2, 1, 1},
        {4, 1, 2, 1},
        {5, 1, 2, 2},
        {5, 2, 2, 2},
    };

    for (int i = 4; i < 6; i++)
    {
        const int k = kdsp[i][0];
        const int d = kdsp[i][1];
        const int s = kdsp[i][2];
        const int p = kdsp[i][3];

        int ret = 0
                  || test_deformableconv2d(9, 7, 1, 1, k, d, s, p, 1)
                  || test_deformableconv2d(9, 7, 4, 13, k, d, s, p, 0)
                  || test_deformableconv2d(9, 7, 13, 4, k, d, s, p, 1)
                  || test_deformableconv2d(9, 7, 4, 8, k, d, s, p, 0)
                  || test_deformableconv2d(9, 7, 8, 4, k, d, s, p, 1)
                  || test_deformableconv2d(9, 7, 8, 13, k, d, s, p, 0)
                  || test_deformableconv2d(9, 7, 13, 8, k, d, s, p, 1)
                  || test_deformableconv2d(9, 7, 16, 16, k, d, s, p, 0)
                  || test_deformableconv2d(16, 16, 1 * 3, 1 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 1 * 3, 4 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 1 * 3, 8 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 1 * 3, 16 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 4 * 3, 1 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 4 * 3, 4 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 4 * 3, 8 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 4 * 3, 16 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 8 * 3, 1 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 8 * 3, 4 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 8 * 3, 8 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 8 * 3, 16 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 16 * 3, 1 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 16 * 3, 4 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 16 * 3, 8 * 3, k, d, s, p, 1)
                  || test_deformableconv2d(16, 16, 16 * 3, 16 * 3, k, d, s, p, 1);

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_deformableconv2d_0();
}
