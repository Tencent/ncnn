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

    int ret = test_layer("Convolution", pd, weights, a, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
        return ret;
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

        ret = test_layer_opt("Convolution", pd, weights, opt, a, epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
            return ret;
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

        ret = test_layer_opt("Convolution", pd, weights, opt, a, epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
            return ret;
        }
    }

#if __aarch64__
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_a53_a55_optimized_kernel = true;

        ret = test_layer_opt("Convolution", pd, weights, opt, a, epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, activation_type, activation_params[0], activation_params[1]);
            return ret;
        }
    }
#endif // __aarch64__

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

    for (int i = 0; i < 12; i++)
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
                  || test_convolution(9, 7, 13, 24, k, d, s, p, 1)
                  || test_convolution(9, 7, 12, 16, k, d, s, p, 0)
                  || test_convolution(9, 7, 15, 15, k, d, s, p, 0)
                  || test_convolution(9, 7, 16, 16, k, d, s, p, 0)
                  || test_convolution(18, 17, 1, 1, k, d, s, p, 1)
                  || test_convolution(18, 17, 4, 13, k, d, s, p, 0)
                  || test_convolution(18, 17, 13, 4, k, d, s, p, 1)
                  || test_convolution(18, 17, 12, 12, k, d, s, p, 0)
                  || test_convolution(18, 17, 8, 12, k, d, s, p, 1)
                  || test_convolution(18, 17, 8, 13, k, d, s, p, 0)
                  || test_convolution(18, 17, 13, 24, k, d, s, p, 1)
                  || test_convolution(18, 17, 12, 16, k, d, s, p, 0)
                  || test_convolution(18, 17, 15, 15, k, d, s, p, 0)
                  || test_convolution(18, 17, 16, 16, k, d, s, p, 0)
                  || test_convolution(25, 33, 1, 1, k, d, s, p, 1)
                  || test_convolution(25, 33, 4, 13, k, d, s, p, 0)
                  || test_convolution(25, 33, 13, 4, k, d, s, p, 1)
                  || test_convolution(25, 33, 12, 12, k, d, s, p, 0)
                  || test_convolution(25, 33, 8, 12, k, d, s, p, 1)
                  || test_convolution(25, 33, 8, 13, k, d, s, p, 0)
                  || test_convolution(25, 33, 13, 24, k, d, s, p, 1)
                  || test_convolution(25, 33, 12, 16, k, d, s, p, 0)
                  || test_convolution(25, 33, 15, 15, k, d, s, p, 0)
                  || test_convolution(25, 33, 16, 16, k, d, s, p, 0);

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_convolution_0();
}
