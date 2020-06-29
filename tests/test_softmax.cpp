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

#include "layer/softmax.h"
#include "testutil.h"

static int test_softmax(const ncnn::Mat& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); // axis
    pd.set(1, 1);    // fixbug0

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::Softmax>("Softmax", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_softmax failed a.dims=%d a=(%d %d %d) axis=%d\n", a.dims, a.w, a.h, a.c, axis);
    }

    return ret;
}

static int test_softmax_0()
{
    ncnn::Mat a = RandomMat(6, 7, 16);

    return 0
           || test_softmax(a, 0)
           || test_softmax(a, 1)
           || test_softmax(a, 2);
}

static int test_softmax_1()
{
    ncnn::Mat a = RandomMat(3, 5, 13);

    return 0
           || test_softmax(a, 0)
           || test_softmax(a, 1)
           || test_softmax(a, 2);
}

static int test_softmax_2()
{
    ncnn::Mat a = RandomMat(6, 16);

    return 0
           || test_softmax(a, 0)
           || test_softmax(a, 1);
}

static int test_softmax_3()
{
    ncnn::Mat a = RandomMat(7, 15);

    return 0
           || test_softmax(a, 0)
           || test_softmax(a, 1);
}

static int test_softmax_4()
{
    ncnn::Mat a = RandomMat(128);

    return test_softmax(a, 0);
}

static int test_softmax_5()
{
    ncnn::Mat a = RandomMat(127);

    return test_softmax(a, 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_softmax_0()
           || test_softmax_1()
           || test_softmax_2()
           || test_softmax_3()
           || test_softmax_4()
           || test_softmax_5();
}
