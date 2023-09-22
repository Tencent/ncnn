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

#include "layer/range.h"
#include "testutil.h"

static int test_range(const ncnn::Mat& a, const ncnn::Mat& b, const ncnn::Mat& c)
{

    // the values should be greater than 0
    RandomizeInt(a, 0, 100000);
    RandomizeInt(b, a[0], a[0]* 10);
    RandomizeInt(c, 1, 5);

    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);
    std::vector<ncnn::Mat> inputs;
    inputs.push_back(a);
    inputs.push_back(b);
    if(!c.empty())
        inputs.push_back(c);

    int ret = test_layer<ncnn::Range>("Range", pd, weights, inputs);
    if (ret != 0)
    {
        fprintf(stderr, "test_range failed a.dims=%d a=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
    }

    return ret;
}

static int test_range_0()
{
    return 0
           || test_range(RandomIntMat(1), RandomIntMat(1))
           || test_range(RandomIntMat(1), RandomIntMat(1))
           || test_range(RandomIntMat(1), RandomIntMat(1));
}

static int test_bias_1()
{
    return 0
           || test_range(RandomIntMat(1), RandomIntMat(1), RandomIntMat(1))
           || test_range(RandomIntMat(1), RandomIntMat(1), RandomIntMat(1))
           || test_range(RandomIntMat(1), RandomIntMat(1), RandomIntMat(1));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_range_0()
           || test_range_1();
}
