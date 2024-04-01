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

    Randomize(a, -1, 1);
    Randomize(weights[0], -0.6, 0.6);
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

    return ret;
}

static int test_convolution_0()
{
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
           || test_convolution(12, 18, 28, 140, 3, 1, 1, 1, 1)
           || test_convolution(3, 3, 47, 47, 3, 1, 1, 0, 1)
           || test_convolution(5, 5, 40, 40, 3, 1, 1, 0, 0)
           || test_convolution(13, 13, 53, 47, 3, 1, 1, 0, 1)
           || test_convolution(20, 26, 47, 47, 3, 1, 1, 0, 0)
           || test_convolution(12, 12, 47, 53, 3, 1, 1, 1, 0)
           || test_convolution(23, 23, 53, 53, 3, 1, 1, 1, 0)
           || test_convolution(26, 34, 47, 47, 3, 1, 1, 2, 0)
           || test_convolution(52, 40, 31, 31, 3, 1, 1, 2, 0)
           || test_convolution(6, 7, 7, 17, 2, 2, 2, 1, 1)
           || test_convolution(8, 9, 3, 17, 5, 1, 1, 2, 1)
           || test_convolution(9, 7, 19, 13, 1, 2, 2, 0, 0)
           || test_convolution(15, 12, 19, 3, 4, 1, 2, 2, 1)
           || test_convolution(14, 14, 24, 31, 5, 1, 2, 2, 1)
           || test_convolution(12, 12, 20, 15, 6, 1, 1, 0, 0)
           || test_convolution(11, 10, 12, 7, 4, 2, 1, 2, 1);
}

static int test_convolution_1()
{
    return 0
           || test_convolution(7, 6, 135, 31, 3, 1, 1, 1, 0)
           || test_convolution(8, 7, 31, 135, 3, 1, 1, 1, 0)
           || test_convolution(9, 7, 135, 7, 3, 1, 1, 0, 0)
           || test_convolution(9, 8, 140, 4, 3, 1, 1, 0, 0)
           || test_convolution(8, 9, 160, 6, 3, 1, 1, 0, 0)
           || test_convolution(11, 9, 7, 135, 3, 1, 1, 0, 0)
           || test_convolution(10, 9, 4, 140, 3, 1, 1, 0, 0)
           || test_convolution(9, 10, 6, 160, 3, 1, 1, 0, 0);
}

int main()
{
    SRAND(7767517);

    return test_convolution_0() || test_convolution_1();
}
