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

#include "testutil.h"

static int test_shrink(const ncnn::Mat& a, float lambd, float bias)
{
    ncnn::ParamDict pd;
    pd.set(0, bias);
    pd.set(1, lambd);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Shrink", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_shrink failed a.dims=%d a=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
    }

    return ret;
}

static int test_shrink_0()
{
    return 0
           || test_shrink(RandomMat(12, 24, 8, 3), 0.5f, 5.0f)
           || test_shrink(RandomMat(9, 7, 16, 4), 1.0f, 0.3f)
           || test_shrink(RandomMat(6, 9, 4, 3), 4.5, 6.1);
}

static int test_shrink_1()
{
    return 0
           || test_shrink(RandomMat(12, 6, 24), 0.5f, 5.0f)
           || test_shrink(RandomMat(7, 8, 24), 1.0f, 0.3f)
           || test_shrink(RandomMat(3, 4, 5), 4.5, 6.1);
}

static int test_shrink_2()
{
    return 0
           || test_shrink(RandomMat(5, 7), 3.4f, 0.3f)
           || test_shrink(RandomMat(7, 9), 3.1f, 4.0f)
           || test_shrink(RandomMat(3, 5), 2.0f, 4.0f);
}

static int test_shrink_3()
{
    return 0
           || test_shrink(RandomMat(25), 3.4f, 0.3f)
           || test_shrink(RandomMat(63), 3.1f, 4.0f)
           || test_shrink(RandomMat(1024), 2.0f, 4.0f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_shrink_0()
           || test_shrink_1()
           || test_shrink_2()
           || test_shrink_3();
}
