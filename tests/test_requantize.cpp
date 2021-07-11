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

#include "layer/requantize.h"
#include "testutil.h"

static int test_requantize(const ncnn::Mat& a, int scale_in_data_size, int scale_out_data_size, int bias_data_size)
{
    ncnn::ParamDict pd;
    pd.set(0, scale_in_data_size);
    pd.set(1, scale_out_data_size);
    pd.set(2, bias_data_size);

    int activation_type = RAND() % 6; // 0 1 2 3 4 5
    ncnn::Mat activation_params(2);
    activation_params[0] = RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);  // beta
    pd.set(3, activation_type);
    pd.set(4, activation_params);

    std::vector<ncnn::Mat> weights(bias_data_size ? 3 : 2);
    weights[0] = RandomMat(scale_in_data_size);
    weights[1] = RandomMat(scale_out_data_size);
    if (bias_data_size)
        weights[2] = RandomMat(bias_data_size);

    int flag = TEST_LAYER_DISABLE_AUTO_INPUT_CASTING;
    int ret = test_layer<ncnn::Requantize>("Requantize", pd, weights, a, 1, 0, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_requantize failed a.dims=%d a=(%d %d %d) scale_in_data_size=%d scale_out_data_size=%d bias_data_size=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.c, scale_in_data_size, scale_out_data_size, bias_data_size, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_requantize_pack8(const ncnn::Mat& a, int scale_in_data_size, int scale_out_data_size, int bias_data_size)
{
    ncnn::ParamDict pd;
    pd.set(0, scale_in_data_size);
    pd.set(1, scale_out_data_size);
    pd.set(2, bias_data_size);

    int activation_type = RAND() % 6; // 0 1 2 3 4 5
    ncnn::Mat activation_params(2);
    activation_params[0] = RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);  // beta
    pd.set(3, activation_type);
    pd.set(4, activation_params);

    std::vector<ncnn::Mat> weights(bias_data_size ? 3 : 2);
    weights[0] = RandomMat(scale_in_data_size);
    weights[1] = RandomMat(scale_out_data_size);
    if (bias_data_size)
        weights[2] = RandomMat(bias_data_size);

    int flag = TEST_LAYER_DISABLE_AUTO_INPUT_CASTING | TEST_LAYER_ENABLE_FORCE_INPUT_PACK8;
    int ret = test_layer<ncnn::Requantize>("Requantize", pd, weights, a, 1, 0, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_requantize_pack8 failed a.dims=%d a=(%d %d %d) scale_in_data_size=%d scale_out_data_size=%d bias_data_size=%d act=%d actparams=[%f,%f]\n", a.dims, a.w, a.h, a.c, scale_in_data_size, scale_out_data_size, bias_data_size, activation_type, activation_params[0], activation_params[1]);
    }

    return ret;
}

static int test_requantize_0()
{
    return 0
           || test_requantize(RandomIntMat(5, 7, 24), 1, 1, 24)
           || test_requantize(RandomIntMat(5, 7, 24), 1, 1, 1)
           || test_requantize(RandomIntMat(5, 7, 24), 1, 1, 0)
           || test_requantize(RandomIntMat(5, 7, 24), 24, 24, 24)
           || test_requantize(RandomIntMat(5, 7, 24), 24, 24, 1)
           || test_requantize(RandomIntMat(5, 7, 24), 24, 24, 0)
           || test_requantize(RandomIntMat(5, 7, 24), 1, 24, 24)
           || test_requantize(RandomIntMat(5, 7, 24), 1, 24, 1)
           || test_requantize(RandomIntMat(5, 7, 24), 1, 24, 0)
           || test_requantize(RandomIntMat(5, 7, 24), 24, 1, 24)
           || test_requantize(RandomIntMat(5, 7, 24), 24, 1, 1)
           || test_requantize(RandomIntMat(5, 7, 24), 24, 1, 0)
           || test_requantize(RandomIntMat(7, 9, 12), 1, 1, 12)
           || test_requantize(RandomIntMat(7, 9, 12), 1, 1, 1)
           || test_requantize(RandomIntMat(7, 9, 12), 1, 1, 0)
           || test_requantize(RandomIntMat(7, 9, 12), 12, 12, 12)
           || test_requantize(RandomIntMat(7, 9, 12), 12, 12, 1)
           || test_requantize(RandomIntMat(7, 9, 12), 12, 12, 0)
           || test_requantize(RandomIntMat(7, 9, 12), 1, 12, 12)
           || test_requantize(RandomIntMat(7, 9, 12), 1, 12, 1)
           || test_requantize(RandomIntMat(7, 9, 12), 1, 12, 0)
           || test_requantize(RandomIntMat(7, 9, 12), 12, 1, 12)
           || test_requantize(RandomIntMat(7, 9, 12), 12, 1, 1)
           || test_requantize(RandomIntMat(7, 9, 12), 12, 1, 0)
           || test_requantize(RandomIntMat(3, 5, 13), 1, 1, 13)
           || test_requantize(RandomIntMat(3, 5, 13), 1, 1, 1)
           || test_requantize(RandomIntMat(3, 5, 13), 1, 1, 0)
           || test_requantize(RandomIntMat(3, 5, 13), 13, 13, 13)
           || test_requantize(RandomIntMat(3, 5, 13), 13, 13, 1)
           || test_requantize(RandomIntMat(3, 5, 13), 13, 13, 0)
           || test_requantize(RandomIntMat(3, 5, 13), 1, 13, 13)
           || test_requantize(RandomIntMat(3, 5, 13), 1, 13, 1)
           || test_requantize(RandomIntMat(3, 5, 13), 1, 13, 0)
           || test_requantize(RandomIntMat(3, 5, 13), 13, 1, 13)
           || test_requantize(RandomIntMat(3, 5, 13), 13, 1, 1)
           || test_requantize(RandomIntMat(3, 5, 13), 13, 1, 0);
}

static int test_requantize_1()
{
    return 0
           || test_requantize(RandomIntMat(15, 24), 1, 1, 24)
           || test_requantize(RandomIntMat(15, 24), 1, 1, 1)
           || test_requantize(RandomIntMat(15, 24), 1, 1, 0)
           || test_requantize(RandomIntMat(15, 24), 24, 24, 24)
           || test_requantize(RandomIntMat(15, 24), 24, 24, 1)
           || test_requantize(RandomIntMat(15, 24), 24, 24, 0)
           || test_requantize(RandomIntMat(15, 24), 1, 24, 24)
           || test_requantize(RandomIntMat(15, 24), 1, 24, 1)
           || test_requantize(RandomIntMat(15, 24), 1, 24, 0)
           || test_requantize(RandomIntMat(15, 24), 24, 1, 24)
           || test_requantize(RandomIntMat(15, 24), 24, 1, 1)
           || test_requantize(RandomIntMat(15, 24), 24, 1, 0)
           || test_requantize(RandomIntMat(17, 12), 1, 1, 12)
           || test_requantize(RandomIntMat(17, 12), 1, 1, 1)
           || test_requantize(RandomIntMat(17, 12), 1, 1, 0)
           || test_requantize(RandomIntMat(17, 12), 12, 12, 12)
           || test_requantize(RandomIntMat(17, 12), 12, 12, 1)
           || test_requantize(RandomIntMat(17, 12), 12, 12, 0)
           || test_requantize(RandomIntMat(17, 12), 1, 12, 12)
           || test_requantize(RandomIntMat(17, 12), 1, 12, 1)
           || test_requantize(RandomIntMat(17, 12), 1, 12, 0)
           || test_requantize(RandomIntMat(17, 12), 12, 1, 12)
           || test_requantize(RandomIntMat(17, 12), 12, 1, 1)
           || test_requantize(RandomIntMat(17, 12), 12, 1, 0)
           || test_requantize(RandomIntMat(19, 15), 1, 1, 15)
           || test_requantize(RandomIntMat(19, 15), 1, 1, 1)
           || test_requantize(RandomIntMat(19, 15), 1, 1, 0)
           || test_requantize(RandomIntMat(19, 15), 15, 15, 15)
           || test_requantize(RandomIntMat(19, 15), 15, 15, 1)
           || test_requantize(RandomIntMat(19, 15), 15, 15, 0)
           || test_requantize(RandomIntMat(19, 15), 1, 15, 15)
           || test_requantize(RandomIntMat(19, 15), 1, 15, 1)
           || test_requantize(RandomIntMat(19, 15), 1, 15, 0)
           || test_requantize(RandomIntMat(19, 15), 15, 1, 15)
           || test_requantize(RandomIntMat(19, 15), 15, 1, 1)
           || test_requantize(RandomIntMat(19, 15), 15, 1, 0);
}

static int test_requantize_2()
{
    return 0
           || test_requantize(RandomIntMat(128), 1, 1, 128)
           || test_requantize(RandomIntMat(128), 1, 1, 1)
           || test_requantize(RandomIntMat(128), 1, 1, 0)
           || test_requantize(RandomIntMat(128), 128, 128, 128)
           || test_requantize(RandomIntMat(128), 128, 128, 1)
           || test_requantize(RandomIntMat(128), 128, 128, 0)
           || test_requantize(RandomIntMat(128), 1, 128, 128)
           || test_requantize(RandomIntMat(128), 1, 128, 1)
           || test_requantize(RandomIntMat(128), 1, 128, 0)
           || test_requantize(RandomIntMat(128), 128, 1, 128)
           || test_requantize(RandomIntMat(128), 128, 1, 1)
           || test_requantize(RandomIntMat(128), 128, 1, 0)
           || test_requantize(RandomIntMat(124), 1, 1, 124)
           || test_requantize(RandomIntMat(124), 1, 1, 1)
           || test_requantize(RandomIntMat(124), 1, 1, 0)
           || test_requantize(RandomIntMat(124), 124, 124, 124)
           || test_requantize(RandomIntMat(124), 124, 124, 1)
           || test_requantize(RandomIntMat(124), 124, 124, 0)
           || test_requantize(RandomIntMat(124), 1, 124, 124)
           || test_requantize(RandomIntMat(124), 1, 124, 1)
           || test_requantize(RandomIntMat(124), 1, 124, 0)
           || test_requantize(RandomIntMat(124), 124, 1, 124)
           || test_requantize(RandomIntMat(124), 124, 1, 1)
           || test_requantize(RandomIntMat(124), 124, 1, 0)
           || test_requantize(RandomIntMat(127), 1, 1, 127)
           || test_requantize(RandomIntMat(127), 1, 1, 1)
           || test_requantize(RandomIntMat(127), 1, 1, 0)
           || test_requantize(RandomIntMat(127), 127, 127, 127)
           || test_requantize(RandomIntMat(127), 127, 127, 1)
           || test_requantize(RandomIntMat(127), 127, 127, 0)
           || test_requantize(RandomIntMat(127), 1, 127, 127)
           || test_requantize(RandomIntMat(127), 1, 127, 1)
           || test_requantize(RandomIntMat(127), 1, 127, 0)
           || test_requantize(RandomIntMat(127), 127, 1, 127)
           || test_requantize(RandomIntMat(127), 127, 1, 1)
           || test_requantize(RandomIntMat(127), 127, 1, 0);
}

static int test_requantize_3()
{
    return 0
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 1, 1, 24)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 1, 1, 1)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 1, 1, 0)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 24, 24, 24)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 24, 24, 1)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 24, 24, 0)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 1, 24, 24)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 1, 24, 1)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 1, 24, 0)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 24, 1, 24)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 24, 1, 1)
           || test_requantize_pack8(RandomIntMat(5, 7, 24), 24, 1, 0)
           || test_requantize_pack8(RandomIntMat(15, 24), 1, 1, 24)
           || test_requantize_pack8(RandomIntMat(15, 24), 1, 1, 1)
           || test_requantize_pack8(RandomIntMat(15, 24), 1, 1, 0)
           || test_requantize_pack8(RandomIntMat(15, 24), 24, 24, 24)
           || test_requantize_pack8(RandomIntMat(15, 24), 24, 24, 1)
           || test_requantize_pack8(RandomIntMat(15, 24), 24, 24, 0)
           || test_requantize_pack8(RandomIntMat(15, 24), 1, 24, 24)
           || test_requantize_pack8(RandomIntMat(15, 24), 1, 24, 1)
           || test_requantize_pack8(RandomIntMat(15, 24), 1, 24, 0)
           || test_requantize_pack8(RandomIntMat(15, 24), 24, 1, 24)
           || test_requantize_pack8(RandomIntMat(15, 24), 24, 1, 1)
           || test_requantize_pack8(RandomIntMat(15, 24), 24, 1, 0)
           || test_requantize_pack8(RandomIntMat(128), 1, 1, 128)
           || test_requantize_pack8(RandomIntMat(128), 1, 1, 1)
           || test_requantize_pack8(RandomIntMat(128), 1, 1, 0)
           || test_requantize_pack8(RandomIntMat(128), 128, 128, 128)
           || test_requantize_pack8(RandomIntMat(128), 128, 128, 1)
           || test_requantize_pack8(RandomIntMat(128), 128, 128, 0)
           || test_requantize_pack8(RandomIntMat(128), 1, 128, 128)
           || test_requantize_pack8(RandomIntMat(128), 1, 128, 1)
           || test_requantize_pack8(RandomIntMat(128), 1, 128, 0)
           || test_requantize_pack8(RandomIntMat(128), 128, 1, 128)
           || test_requantize_pack8(RandomIntMat(128), 128, 1, 1)
           || test_requantize_pack8(RandomIntMat(128), 128, 1, 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_requantize_0()
           || test_requantize_1()
           || test_requantize_2()
           || test_requantize_3();
}
