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

#include "layer/relu.h"
#include "testutil.h"

static int test_relu(const ncnn::Mat& a, float slope)
{
    ncnn::ParamDict pd;
    pd.set(0, slope); //slope

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::ReLU>("ReLU", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_relu failed a.dims=%d a=(%d %d %d) slope=%f\n", a.dims, a.w, a.h, a.c, slope);
    }

    return ret;
}

static int test_relu_0()
{
    return 0
           || test_relu(RandomMat(6, 7, 16), 0.f)
           || test_relu(RandomMat(6, 7, 16), 0.1f)
           || test_relu(RandomMat(3, 5, 13), 0.f)
           || test_relu(RandomMat(3, 5, 13), 0.1f);
}

static int test_relu_1()
{
    return 0
           || test_relu(RandomMat(6, 16), 0.f)
           || test_relu(RandomMat(6, 16), 0.1f)
           || test_relu(RandomMat(7, 15), 0.f)
           || test_relu(RandomMat(7, 15), 0.1f);
}

static int test_relu_2()
{
    return 0
           || test_relu(RandomMat(128), 0.f)
           || test_relu(RandomMat(128), 0.1f)
           || test_relu(RandomMat(127), 0.f)
           || test_relu(RandomMat(127), 0.1f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_relu_0()
           || test_relu_1()
           || test_relu_2();
}
