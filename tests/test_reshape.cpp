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

#include "layer/reshape.h"
#include "testutil.h"

static int test_reshape(const ncnn::Mat& a, int outw, int outh, int outc)
{
    ncnn::ParamDict pd;
    pd.set(0, outw); // w
    pd.set(1, outh); // h
    pd.set(2, outc); // c

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Reshape>("Reshape", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_reshape failed a.dims=%d a=(%d %d %d) outw=%d outh=%d outc=%d\n", a.dims, a.w, a.h, a.c, outw, outh, outc);
    }

    return ret;
}

static int test_reshape_permute(const ncnn::Mat& a, int outw, int outh, int outc)
{
    ncnn::ParamDict pd;
    pd.set(0, outw); // w
    pd.set(1, outh); // h
    pd.set(2, outc); // c
    pd.set(3, 1);    // permute

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Reshape>("Reshape", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_reshape_permute failed a.dims=%d a=(%d %d %d) outw=%d outh=%d outc=%d\n", a.dims, a.w, a.h, a.c, outw, outh, outc);
    }

    return ret;
}

static int test_reshape_0()
{
    ncnn::Mat a = RandomMat(3, 7, 16);

    return 0
           || test_reshape(a, 7, 3, 16)
           || test_reshape(a, 3, 16, 7)
           || test_reshape(a, 16, 7, 3)
           || test_reshape(a, 2, 3, -1)
           || test_reshape(a, -1, 8, 2)
           || test_reshape(a, -1, 4, -233)
           || test_reshape(a, 8, -1, -233)
           || test_reshape(a, 16, 21, -233)
           || test_reshape(a, -1, -233, -233);
}

static int test_reshape_1()
{
    ncnn::Mat a = RandomMat(4, 14, 13);

    return 0
           || test_reshape(a, 14, 4, 13)
           || test_reshape(a, 4, 13, 14)
           || test_reshape(a, 13, 14, 4)
           || test_reshape(a, 2, 7, -1)
           || test_reshape(a, -1, 13, 2)
           || test_reshape(a, -1, 4, -233)
           || test_reshape(a, 8, -1, -233)
           || test_reshape(a, 8, 91, -233)
           || test_reshape(a, -1, -233, -233);
}

static int test_reshape_2()
{
    ncnn::Mat a = RandomMat(14, 16);

    return 0
           || test_reshape(a, 7, 2, 16)
           || test_reshape(a, 2, 16, 7)
           || test_reshape(a, 16, 7, 2)
           || test_reshape(a, 2, 4, -1)
           || test_reshape(a, -1, 8, 2)
           || test_reshape(a, 28, 8, -233)
           || test_reshape(a, -1, 7, -233)
           || test_reshape(a, 16, -1, -233)
           || test_reshape(a, -1, -233, -233);
}

static int test_reshape_3()
{
    ncnn::Mat a = RandomMat(12, 14);

    return 0
           || test_reshape(a, 7, 2, 12)
           || test_reshape(a, 2, 12, 7)
           || test_reshape(a, 12, 7, 2)
           || test_reshape(a, 2, 4, -1)
           || test_reshape(a, -1, 4, 2)
           || test_reshape(a, 21, 8, -233)
           || test_reshape(a, -1, 7, -233)
           || test_reshape(a, 3, -1, -233)
           || test_reshape(a, -1, -233, -233);
}

static int test_reshape_4()
{
    ncnn::Mat a = RandomMat(120);

    return 0
           || test_reshape(a, 3, 5, 8)
           || test_reshape(a, 3, 8, 5)
           || test_reshape(a, 8, 5, 3)
           || test_reshape(a, 2, 5, -1)
           || test_reshape(a, -1, 5, 2)
           || test_reshape(a, 4, 30, -233)
           || test_reshape(a, -1, 2, -233)
           || test_reshape(a, 24, -1, -233)
           || test_reshape(a, -1, -233, -233);
}

static int test_reshape_5()
{
    ncnn::Mat a = RandomMat(210);

    return 0
           || test_reshape(a, 3, 5, 14)
           || test_reshape(a, 3, 14, 5)
           || test_reshape(a, 14, 5, 3)
           || test_reshape(a, 2, 5, -1)
           || test_reshape(a, -1, 5, 2)
           || test_reshape(a, 6, 35, -233)
           || test_reshape(a, -1, 7, -233)
           || test_reshape(a, 21, -1, -233)
           || test_reshape(a, -1, -233, -233);
}

static int test_reshape_6()
{
    ncnn::Mat a = RandomMat(3, 7, 16);

    return 0
           || test_reshape_permute(a, 7, 3, 16)
           || test_reshape_permute(a, 3, 16, 7)
           || test_reshape_permute(a, 16, 7, 3)
           || test_reshape_permute(a, 2, 3, -1)
           || test_reshape_permute(a, -1, 8, 2)
           || test_reshape_permute(a, -1, 4, -233)
           || test_reshape_permute(a, 8, -1, -233)
           || test_reshape_permute(a, 16, 21, -233)
           || test_reshape_permute(a, -1, -233, -233);
}

static int test_reshape_7()
{
    ncnn::Mat a = RandomMat(4, 14, 13);

    return 0
           || test_reshape_permute(a, 14, 4, 13)
           || test_reshape_permute(a, 4, 13, 14)
           || test_reshape_permute(a, 13, 14, 4)
           || test_reshape_permute(a, 2, 7, -1)
           || test_reshape_permute(a, -1, 13, 2)
           || test_reshape_permute(a, -1, 4, -233)
           || test_reshape_permute(a, 8, -1, -233)
           || test_reshape_permute(a, 8, 91, -233)
           || test_reshape_permute(a, -1, -233, -233);
}

static int test_reshape_8()
{
    ncnn::Mat a = RandomMat(14, 16);

    return 0
           || test_reshape_permute(a, 7, 2, 16)
           || test_reshape_permute(a, 2, 16, 7)
           || test_reshape_permute(a, 16, 7, 2)
           || test_reshape_permute(a, 2, 4, -1)
           || test_reshape_permute(a, -1, 8, 2)
           || test_reshape_permute(a, 28, 8, -233)
           || test_reshape_permute(a, -1, 7, -233)
           || test_reshape_permute(a, 16, -1, -233)
           || test_reshape_permute(a, -1, -233, -233);
}

static int test_reshape_9()
{
    ncnn::Mat a = RandomMat(12, 14);

    return 0
           || test_reshape_permute(a, 7, 2, 12)
           || test_reshape_permute(a, 2, 12, 7)
           || test_reshape_permute(a, 12, 7, 2)
           || test_reshape_permute(a, 2, 4, -1)
           || test_reshape_permute(a, -1, 4, 2)
           || test_reshape_permute(a, 21, 8, -233)
           || test_reshape_permute(a, -1, 7, -233)
           || test_reshape_permute(a, 3, -1, -233)
           || test_reshape_permute(a, -1, -233, -233);
}

static int test_reshape_10()
{
    ncnn::Mat a = RandomMat(120);

    return 0
           || test_reshape_permute(a, 3, 5, 8)
           || test_reshape_permute(a, 3, 8, 5)
           || test_reshape_permute(a, 8, 5, 3)
           || test_reshape_permute(a, 2, 5, -1)
           || test_reshape_permute(a, -1, 5, 2)
           || test_reshape_permute(a, 4, 30, -233)
           || test_reshape_permute(a, -1, 2, -233)
           || test_reshape_permute(a, 24, -1, -233)
           || test_reshape_permute(a, -1, -233, -233);
}

static int test_reshape_11()
{
    ncnn::Mat a = RandomMat(210);

    return 0
           || test_reshape_permute(a, 3, 5, 14)
           || test_reshape_permute(a, 3, 14, 5)
           || test_reshape_permute(a, 14, 5, 3)
           || test_reshape_permute(a, 2, 5, -1)
           || test_reshape_permute(a, -1, 5, 2)
           || test_reshape_permute(a, 6, 35, -233)
           || test_reshape_permute(a, -1, 7, -233)
           || test_reshape_permute(a, 21, -1, -233)
           || test_reshape_permute(a, -1, -233, -233);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_reshape_0()
           || test_reshape_1()
           || test_reshape_2()
           || test_reshape_3()
           || test_reshape_4()
           || test_reshape_5()
           || test_reshape_6()
           || test_reshape_7()
           || test_reshape_8()
           || test_reshape_9()
           || test_reshape_10()
           || test_reshape_11();
}
