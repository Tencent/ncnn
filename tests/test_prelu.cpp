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

#include "layer/prelu.h"
#include "testutil.h"

static int test_prelu(const ncnn::Mat& a, int num_slope)
{
    ncnn::ParamDict pd;
    pd.set(0, num_slope);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(num_slope);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::PReLU>("PReLU", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_prelu failed a.dims=%d a=(%d %d %d) num_slope=%d\n", a.dims, a.w, a.h, a.c, num_slope);
    }

    return ret;
}

static int test_prelu_0()
{
    return 0
           || test_prelu(RandomMat(6, 7, 16), 16)
           || test_prelu(RandomMat(6, 7, 16), 1)
           || test_prelu(RandomMat(3, 5, 13), 13)
           || test_prelu(RandomMat(3, 5, 13), 1);
}

static int test_prelu_1()
{
    return 0
           || test_prelu(RandomMat(6, 16), 16)
           || test_prelu(RandomMat(6, 16), 1)
           || test_prelu(RandomMat(7, 15), 15)
           || test_prelu(RandomMat(7, 15), 1);
}

static int test_prelu_2()
{
    return 0
           || test_prelu(RandomMat(128), 128)
           || test_prelu(RandomMat(128), 1)
           || test_prelu(RandomMat(127), 127)
           || test_prelu(RandomMat(127), 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_prelu_0()
           || test_prelu_1()
           || test_prelu_2();
}
