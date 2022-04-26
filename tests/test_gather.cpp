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

#include "layer/gather.h"
#include "testutil.h"

static int test_gather(const std::vector<ncnn::Mat>& a, const int axis = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);

    std::vector<ncnn::Mat> weights(0);
    int ret = test_layer<ncnn::Gather>("Gather", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_gather failed a[0].dims=%d a[0]=(%d,%d,%d) a[1]=(%d)  axis=%d\n", a[0].dims, a[0].w, a[0].h, a[0].c, a[1].w, axis);
    }

    return ret;
}

static int test_gather_0()
{
    std::vector<ncnn::Mat> inp(2);
    inp[0] = RandomMat(5, 7, 24);
    inp[1] = RandomMat(60, 0.f, 4.f);
    return 0
           || test_gather(inp, 0)
           || test_gather(inp, 1)
           || test_gather(inp, 2);
}

static int test_gather_1()
{
    std::vector<ncnn::Mat> inp(2);
    inp[0] = RandomMat(8, 7);
    inp[1] = RandomMat(24, 0.f, 6.f);
    return 0
           || test_gather(inp, 0)
           || test_gather(inp, 1);
}

static int test_gather_2()
{
    std::vector<ncnn::Mat> inp(2);
    inp[0] = RandomMat(80);
    inp[1] = RandomMat(24, 0.f, 79.f);
    return 0
           || test_gather(inp, 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_gather_0()
           || test_gather_1()
           || test_gather_2();
}
