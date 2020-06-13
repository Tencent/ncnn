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

#include "layer/sigmoid.h"
#include "testutil.h"

static int test_sigmoid(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::Sigmoid>("Sigmoid", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_sigmoid failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
    }

    return ret;
}

static int test_sigmoid_0()
{
    return 0
           || test_sigmoid(RandomMat(6, 7, 16))
           || test_sigmoid(RandomMat(3, 5, 13));
}

static int test_sigmoid_1()
{
    return 0
           || test_sigmoid(RandomMat(6, 16))
           || test_sigmoid(RandomMat(7, 15));
}

static int test_sigmoid_2()
{
    return 0
           || test_sigmoid(RandomMat(128))
           || test_sigmoid(RandomMat(127));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_sigmoid_0()
           || test_sigmoid_1()
           || test_sigmoid_2();
}
