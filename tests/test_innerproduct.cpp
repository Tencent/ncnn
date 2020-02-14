// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "layer/innerproduct.h"

static int test_innerproduct(int w, int h, int c, int outch, int bias)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);// num_output
    pd.set(1, bias);// bias_term
    pd.set(2, outch*w*h*c);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch*w*h*c);
    if (bias)
        weights[1] = RandomMat(outch);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;

    int ret = test_layer<ncnn::InnerProduct>("InnerProduct", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct failed w=%d h=%d c=%d outch=%d bias=%d\n", w, h, c, outch, bias);
    }

    return ret;
}

static int test_innerproduct_0()
{
    return 0
        || test_innerproduct(7, 3, 1, 1, 1)
        || test_innerproduct(7, 3, 2, 2, 1)
        || test_innerproduct(7, 3, 3, 3, 1)
        || test_innerproduct(7, 3, 4, 4, 1)
        || test_innerproduct(7, 3, 7, 7, 1)
        || test_innerproduct(7, 3, 8, 8, 1)
        || test_innerproduct(7, 3, 15, 15, 1)
        || test_innerproduct(7, 3, 16, 16, 1)
        ;
}

static int test_innerproduct_int8(int w, int h, int c, int outch, int bias)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);// num_output
    pd.set(1, bias);// bias_term
    pd.set(2, outch*w*h*c);
    pd.set(8, 1);// int8_scale_term

    std::vector<ncnn::Mat> weights(bias ? 4 : 3);
    weights[0] = RandomMat(outch*w*h*c);
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
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;

    int ret = test_layer<ncnn::InnerProduct>("InnerProduct", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_int8 failed w=%d h=%d c=%d outch=%d bias=%d\n", w, h, c, outch, bias);
    }

    return 0;
}

static int test_innerproduct_1()
{
    return 0
        || test_innerproduct_int8(7, 3, 1, 1, 1)
        || test_innerproduct_int8(7, 3, 2, 2, 1)
        || test_innerproduct_int8(7, 3, 3, 3, 1)
        || test_innerproduct_int8(7, 3, 3, 12, 1)
        || test_innerproduct_int8(7, 3, 4, 4, 1)
        || test_innerproduct_int8(7, 3, 7, 7, 1)
        || test_innerproduct_int8(7, 3, 8, 3, 1)
        || test_innerproduct_int8(7, 3, 8, 8, 1)
        || test_innerproduct_int8(7, 3, 15, 15, 1)
        || test_innerproduct_int8(7, 3, 16, 4, 1)
        || test_innerproduct_int8(7, 3, 16, 16, 1)
        ;
}

int main()
{
    SRAND(7767517);

    return test_innerproduct_0() || test_innerproduct_1();
}
