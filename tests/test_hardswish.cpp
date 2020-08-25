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

#include "layer/hardswish.h"
#include "testutil.h"

static int test_hardswish(const ncnn::Mat& a, float alpha, float beta)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(0, beta);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::HardSwish>("HardSwish", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_hardswish failed a.dims=%d a=(%d %d %d) alpha=%f beta=%f\n", a.dims, a.w, a.h, a.c, alpha, beta);
    }

    return ret;
}

static int test_hardswish_0()
{
    return 0
           || test_hardswish(RandomMat(5, 7, 24), 0.2f, 0.5f)
           || test_hardswish(RandomMat(7, 9, 12), 0.2f, 0.5f)
           || test_hardswish(RandomMat(3, 5, 13), 0.2f, 0.5f);
}

static int test_hardswish_1()
{
    return 0
           || test_hardswish(RandomMat(15, 24), 0.2f, 0.5f)
           || test_hardswish(RandomMat(17, 12), 0.2f, 0.5f)
           || test_hardswish(RandomMat(19, 15), 0.2f, 0.5f);
}

static int test_hardswish_2()
{
    return 0
           || test_hardswish(RandomMat(128), 0.2f, 0.5f)
           || test_hardswish(RandomMat(124), 0.2f, 0.5f)
           || test_hardswish(RandomMat(127), 0.2f, 0.5f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_hardswish_0()
           || test_hardswish_1()
           || test_hardswish_2();
}
