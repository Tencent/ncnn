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

#include "layer/innerproduct.h"
#include "testutil.h"

static int test_innerproduct(const ncnn::Mat& a, int outch, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, outch); // num_output
    pd.set(1, bias);  // bias_term
    pd.set(2, outch * a.w * a.h * a.c);

    int activation_type = RAND() % 6; // 0 1 2 3 4 5
    ncnn::Mat activation_params(2);
    activation_params[0] = RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);  // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * a.w * a.h * a.c);
    if (bias)
        weights[1] = RandomMat(outch);

    int ret = test_layer<ncnn::InnerProduct>("InnerProduct", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct failed a.dims=%d a=(%d %d %d) outch=%d bias=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.c, outch, bias, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_innerproduct_0()
{
    return 0
           || test_innerproduct(RandomMat(1, 3, 1), 1, 1)
           || test_innerproduct(RandomMat(3, 2, 2), 2, 1)
           || test_innerproduct(RandomMat(9, 3, 8), 7, 1)
           || test_innerproduct(RandomMat(2, 2, 8), 8, 1)
           || test_innerproduct(RandomMat(4, 3, 15), 8, 1)
           || test_innerproduct(RandomMat(6, 2, 16), 16, 1)
           || test_innerproduct(RandomMat(6, 2, 16), 7, 1)
           || test_innerproduct(RandomMat(6, 2, 5), 16, 1);
}

static int test_innerproduct_1()
{
    return 0
           || test_innerproduct(RandomMat(1, 1), 1, 1)
           || test_innerproduct(RandomMat(3, 2), 2, 1)
           || test_innerproduct(RandomMat(9, 8), 7, 1)
           || test_innerproduct(RandomMat(2, 8), 8, 1)
           || test_innerproduct(RandomMat(4, 15), 8, 1)
           || test_innerproduct(RandomMat(6, 16), 16, 1)
           || test_innerproduct(RandomMat(6, 16), 7, 1)
           || test_innerproduct(RandomMat(6, 5), 16, 1);
}

static int test_innerproduct_2()
{
    return 0
           || test_innerproduct(RandomMat(1), 1, 1)
           || test_innerproduct(RandomMat(2), 2, 1)
           || test_innerproduct(RandomMat(8), 7, 1)
           || test_innerproduct(RandomMat(8), 8, 1)
           || test_innerproduct(RandomMat(15), 8, 1)
           || test_innerproduct(RandomMat(16), 16, 1)
           || test_innerproduct(RandomMat(16), 7, 1)
           || test_innerproduct(RandomMat(5), 16, 1)
           || test_innerproduct(RandomMat(32), 16, 1)
           || test_innerproduct(RandomMat(12), 16, 1)
           || test_innerproduct(RandomMat(16), 12, 1)
           || test_innerproduct(RandomMat(24), 32, 1);
}

static int test_innerproduct_int8(const ncnn::Mat& a, int outch, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, outch); // num_output
    pd.set(1, bias);  // bias_term
    pd.set(2, outch * a.w * a.h * a.c);
    pd.set(8, 1); // int8_scale_term

    std::vector<ncnn::Mat> weights(bias ? 4 : 3);
    weights[0] = RandomMat(outch * a.w * a.h * a.c);
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

    int ret = test_layer<ncnn::InnerProduct>("InnerProduct", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_innerproduct_int8 failed a.dims=%d a=(%d %d %d) outch=%d bias=%d\n", a.dims, a.w, a.h, a.c, outch, bias);
    }

    return 0;
}

static int test_innerproduct_3()
{
    return 0
           || test_innerproduct_int8(RandomMat(1, 3, 1), 1, 1)
           || test_innerproduct_int8(RandomMat(3, 2, 2), 2, 1)
           || test_innerproduct_int8(RandomMat(5, 3, 3), 3, 1)
           || test_innerproduct_int8(RandomMat(7, 2, 3), 12, 1)
           || test_innerproduct_int8(RandomMat(9, 3, 4), 4, 1)
           || test_innerproduct_int8(RandomMat(2, 2, 7), 7, 1)
           || test_innerproduct_int8(RandomMat(4, 3, 8), 3, 1)
           || test_innerproduct_int8(RandomMat(6, 2, 8), 8, 1)
           || test_innerproduct_int8(RandomMat(8, 3, 15), 15, 1)
           || test_innerproduct_int8(RandomMat(7, 2, 16), 4, 1)
           || test_innerproduct_int8(RandomMat(6, 3, 16), 16, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_innerproduct_0()
           || test_innerproduct_1()
           || test_innerproduct_2()
           || test_innerproduct_3();
}
