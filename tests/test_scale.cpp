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

#include "layer/scale.h"
#include "testutil.h"

static int test_scale(const ncnn::Mat& a, int scale_data_size, int bias)
{
    ncnn::ParamDict pd;
    pd.set(0, scale_data_size);
    pd.set(1, bias);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(scale_data_size);
    if (bias)
        weights[1] = RandomMat(scale_data_size);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::Scale>("Scale", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_scale failed a.dims=%d a=(%d %d %d) scale_data_size=%d bias=%d\n", a.dims, a.w, a.h, a.c, scale_data_size, bias);
    }

    return ret;
}

static int test_scale_attention(const ncnn::Mat& a, int scale_data_size)
{
    ncnn::ParamDict pd;
    pd.set(0, -233);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_int8_inference = false;

    std::vector<ncnn::Mat> ab(2);
    ab[0] = a;
    ab[1] = RandomMat(scale_data_size);

    int ret = test_layer<ncnn::Scale>("Scale", pd, weights, opt, ab, 2);
    if (ret != 0)
    {
        fprintf(stderr, "test_scale_attention failed a.dims=%d a=(%d %d %d) scale_data_size=%d\n", a.dims, a.w, a.h, a.c, scale_data_size);
    }

    return ret;
}

static int test_scale_0()
{
    return 0
           || test_scale(RandomMat(6, 7, 16), 16, 0)
           || test_scale(RandomMat(6, 7, 16), 16, 1)
           || test_scale(RandomMat(3, 5, 13), 13, 0)
           || test_scale(RandomMat(3, 5, 13), 13, 1);
}

static int test_scale_1()
{
    return 0
           || test_scale(RandomMat(6, 16), 16, 0)
           || test_scale(RandomMat(6, 16), 16, 1)
           || test_scale(RandomMat(7, 15), 15, 0)
           || test_scale(RandomMat(7, 15), 15, 1);
}

static int test_scale_2()
{
    return 0
           || test_scale(RandomMat(128), 128, 0)
           || test_scale(RandomMat(128), 128, 1)
           || test_scale(RandomMat(127), 127, 0)
           || test_scale(RandomMat(127), 127, 1);
}

static int test_scale_3()
{
    return 0
           || test_scale_attention(RandomMat(6, 7, 16), 16)
           || test_scale_attention(RandomMat(3, 5, 13), 13);
}

static int test_scale_4()
{
    return 0
           || test_scale_attention(RandomMat(6, 16), 16)
           || test_scale_attention(RandomMat(7, 15), 15);
}

static int test_scale_5()
{
    return 0
           || test_scale_attention(RandomMat(128), 128)
           || test_scale_attention(RandomMat(127), 127);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_scale_0()
           || test_scale_1()
           || test_scale_2()
           || test_scale_3()
           || test_scale_4()
           || test_scale_5();
}
