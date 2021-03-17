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

#include "layer/quantize.h"
#include "testutil.h"

static int test_quantize(const ncnn::Mat& a, float scale_low, float scale_high)
{
    ncnn::Mat scale_data;
    if (scale_low == scale_high)
    {
        scale_data.create(1);
        scale_data[0] = scale_low;
    }
    else
    {
        if (a.dims == 1) scale_data.create(a.w);
        if (a.dims == 2) scale_data.create(a.h);
        if (a.dims == 3) scale_data.create(a.c);
        Randomize(scale_data, scale_low, scale_high);
    }

    ncnn::ParamDict pd;
    pd.set(0, scale_data.w);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = scale_data;

    int ret = test_layer<ncnn::Quantize>("Quantize", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_quantize failed a.dims=%d a=(%d %d %d) scale_low=%f scale_high=%f\n", a.dims, a.w, a.h, a.c, scale_low, scale_high);
    }

    return ret;
}

static int test_quantize_0()
{
    return 0
           || test_quantize(RandomMat(5, 7, 24), 100.f, 100.f)
           || test_quantize(RandomMat(5, 7, 24), 120.f, 140.f)
           || test_quantize(RandomMat(7, 9, 12), 100.f, 100.f)
           || test_quantize(RandomMat(7, 9, 12), 120.f, 140.f)
           || test_quantize(RandomMat(3, 5, 13), 100.f, 100.f)
           || test_quantize(RandomMat(3, 5, 13), 120.f, 140.f);
}

static int test_quantize_1()
{
    return 0
           || test_quantize(RandomMat(15, 24), 100.f, 100.f)
           || test_quantize(RandomMat(15, 24), 120.f, 140.f)
           || test_quantize(RandomMat(17, 12), 100.f, 100.f)
           || test_quantize(RandomMat(17, 12), 120.f, 140.f)
           || test_quantize(RandomMat(19, 15), 100.f, 100.f)
           || test_quantize(RandomMat(19, 15), 120.f, 140.f);
}

static int test_quantize_2()
{
    return 0
           || test_quantize(RandomMat(128), 100.f, 100.f)
           || test_quantize(RandomMat(128), 120.f, 140.f)
           || test_quantize(RandomMat(124), 100.f, 100.f)
           || test_quantize(RandomMat(124), 120.f, 140.f)
           || test_quantize(RandomMat(127), 100.f, 100.f)
           || test_quantize(RandomMat(127), 120.f, 140.f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_quantize_0()
           || test_quantize_1()
           || test_quantize_2();
}
