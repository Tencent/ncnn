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

static int test_range(const ncnn::Mat& _a, const ncnn::Mat& _b, const ncnn::Mat& _c)
{
    ncnn::Mat a = _a;
    ncnn::Mat b = _b;
    ncnn::Mat c = _c;

    // the values should be greater than 0
    a = a.clone();
    Randomize(a, 0.0f, 100.0f);
    float* a_ptr = a;
    b = b.clone();
    Randomize(b, a_ptr[0] + 10.0f, a_ptr[0] + 100.0f);
    if (!c.empty())
    {
        c = c.clone();
        Randomize(c, 1.0f, 5.0f);
    }
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);
    std::vector<ncnn::Mat> inputs(!c.empty() ? 3 : 2);
    inputs[0] = a;
    inputs[1] = b;
    if (!c.empty())
        inputs[2] = c;

    int ret = test_layer<ncnn::Range>("Range", pd, weights, inputs);
    if (ret != 0)
    {
        fprintf(stderr, "test_range failed a.dims=%d a=(%d %d %d %d), b.dims=%d, b=(%d, %d, %d, %d), c.dims=%d, c=(%d, %d, %d, %d)\n", a.dims, a.w, a.h, a.d, a.c, b.dims, b.w, b.h, b.d, b.c, b.dims, b.w, b.h, b.d, b.c);
    }

    return ret;
}

static int test_range_0()
{
    return 0
           || test_range(RandomMat(1), RandomMat(1), ncnn::Mat())
           || test_range(RandomMat(1), RandomMat(1), ncnn::Mat())
           || test_range(RandomMat(1), RandomMat(1), ncnn::Mat());
}

static int test_range_1()
{
    return 0
           || test_range(RandomMat(1), RandomMat(1), RandomMat(1))
           || test_range(RandomMat(1), RandomMat(1), RandomMat(1))
           || test_range(RandomMat(1), RandomMat(1), RandomMat(1));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_range_0()
           || test_range_1();
}
