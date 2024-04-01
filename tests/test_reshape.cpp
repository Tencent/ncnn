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

#include "testutil.h"

static int test_reshape(const ncnn::Mat& a, int outw, int outh, int outd, int outc)
{
    ncnn::ParamDict pd;
    pd.set(0, outw);  // w
    pd.set(1, outh);  // h
    pd.set(11, outd); // d
    pd.set(2, outc);  // c

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Reshape", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_reshape failed a.dims=%d a=(%d %d %d %d) outw=%d outh=%d outd=%d outc=%d\n", a.dims, a.w, a.h, a.d, a.c, outw, outh, outd, outc);
    }

    return ret;
}

static int test_reshape_0()
{
    ncnn::Mat a = RandomMat(3, 8, 5, 32);

    return 0
           || test_reshape(a, 5, 8, 3, 32)
           || test_reshape(a, 3, 8, 32, 5)
           || test_reshape(a, 32, 5, 8, 3)
           || test_reshape(a, 4, 3, 16, -1)
           || test_reshape(a, 6, 16, -1, 2)
           || test_reshape(a, 4, -1, 8, 8)
           || test_reshape(a, -1, 16, 6, 2)
           || test_reshape(a, 16, 6, -233, -1)
           || test_reshape(a, 8, -1, -233, 8)
           || test_reshape(a, -1, 6, -233, 16)
           || test_reshape(a, 8, -1, -233, -233)
           || test_reshape(a, -1, 6, -233, -233)
           || test_reshape(a, -1, -233, -233, -233);
}

static int test_reshape_1()
{
    ncnn::Mat a = RandomMat(4, 5, 12, 26);

    return 0
           || test_reshape(a, 12, 5, 4, 26)
           || test_reshape(a, 4, 26, 12, 5)
           || test_reshape(a, 13, 5, 12, 8)
           || test_reshape(a, 2, 10, 8, -1)
           || test_reshape(a, 13, 4, -1, 10)
           || test_reshape(a, 13, -1, 12, 4)
           || test_reshape(a, -1, 26, 3, 16)
           || test_reshape(a, 13, 4, -233, -1)
           || test_reshape(a, 26, -1, -233, 12)
           || test_reshape(a, -1, 13, -233, 16)
           || test_reshape(a, 12, -1, -233, -233)
           || test_reshape(a, -1, 24, -233, -233)
           || test_reshape(a, -1, -233, -233, -233);
}

static int test_reshape_2()
{
    ncnn::Mat a = RandomMat(6, 7, 32);

    return 0
           || test_reshape(a, 6, 8, 4, 7)
           || test_reshape(a, 2, 6, 7, 16)
           || test_reshape(a, 7, 6, -233, 32)
           || test_reshape(a, 6, 32, -233, 7)
           || test_reshape(a, 32, 7, -233, 6)
           || test_reshape(a, 2, 6, -233, -1)
           || test_reshape(a, -1, 8, -233, 2)
           || test_reshape(a, -1, 4, -233, -233)
           || test_reshape(a, 8, -1, -233, -233)
           || test_reshape(a, 32, 42, -233, -233)
           || test_reshape(a, -1, -233, -233, -233);
}

static int test_reshape_3()
{
    ncnn::Mat a = RandomMat(4, 14, 26);

    return 0
           || test_reshape(a, 13, 4, 2, 14)
           || test_reshape(a, 1, 13, 7, 16)
           || test_reshape(a, 14, 8, -233, 13)
           || test_reshape(a, 4, 13, -233, 28)
           || test_reshape(a, 13, 14, -233, 8)
           || test_reshape(a, 2, 7, -233, -1)
           || test_reshape(a, -1, 13, -233, 2)
           || test_reshape(a, -1, 4, -233, -233)
           || test_reshape(a, 8, -1, -233, -233)
           || test_reshape(a, 16, 91, -233, -233)
           || test_reshape(a, -1, -233, -233, -233);
}

static int test_reshape_4()
{
    ncnn::Mat a = RandomMat(14, 32);

    return 0
           || test_reshape(a, 2, 7, 2, 16)
           || test_reshape(a, 16, 1, 7, 4)
           || test_reshape(a, 7, 2, -233, 32)
           || test_reshape(a, 4, 16, -233, 7)
           || test_reshape(a, 16, 14, -233, 2)
           || test_reshape(a, 2, 4, -233, -1)
           || test_reshape(a, -1, 8, -233, 2)
           || test_reshape(a, 28, 16, -233, -233)
           || test_reshape(a, -1, 14, -233, -233)
           || test_reshape(a, 16, -1, -233, -233)
           || test_reshape(a, -1, -233, -233, -233);
}

static int test_reshape_5()
{
    ncnn::Mat a = RandomMat(12, 28);

    return 0
           || test_reshape(a, 4, 3, 2, 14)
           || test_reshape(a, 1, 3, 14, 8)
           || test_reshape(a, 7, 2, -233, 24)
           || test_reshape(a, 2, 24, -233, 7)
           || test_reshape(a, 12, 14, -233, 2)
           || test_reshape(a, 2, 4, -233, -1)
           || test_reshape(a, -1, 4, -233, 2)
           || test_reshape(a, 21, 16, -233, -233)
           || test_reshape(a, -1, 14, -233, -233)
           || test_reshape(a, 3, -1, -233, -233)
           || test_reshape(a, -1, -233, -233, -233);
}

static int test_reshape_6()
{
    ncnn::Mat a = RandomMat(240);

    return 0
           || test_reshape(a, 1, 1, 1, 240)
           || test_reshape(a, 10, 1, 1, 24)
           || test_reshape(a, 3, 5, -233, 16)
           || test_reshape(a, 3, 8, -233, 10)
           || test_reshape(a, 8, 5, -233, 6)
           || test_reshape(a, 2, 5, -233, -1)
           || test_reshape(a, -1, 5, -233, 4)
           || test_reshape(a, 8, 30, -233, -233)
           || test_reshape(a, -1, 2, -233, -233)
           || test_reshape(a, 24, -1, -233, -233)
           || test_reshape(a, -1, -233, -233, -233);
}

static int test_reshape_7()
{
    ncnn::Mat a = RandomMat(210);

    return 0
           || test_reshape(a, 1, 1, 210, 1)
           || test_reshape(a, 5, 2, 7, 3)
           || test_reshape(a, 3, 5, -233, 14)
           || test_reshape(a, 3, 14, -233, 5)
           || test_reshape(a, 14, 5, -233, 3)
           || test_reshape(a, 2, 5, -233, -1)
           || test_reshape(a, -1, 5, -233, 2)
           || test_reshape(a, 6, 35, -233, -233)
           || test_reshape(a, -1, 7, -233, -233)
           || test_reshape(a, 21, -1, -233, -233)
           || test_reshape(a, -1, -233, -233, -233);
}

static int test_reshape_8()
{
    ncnn::Mat a = RandomMat(1, 19, 15, 18);

    return test_reshape(a, 19, 15, -233, 18);
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
           || test_reshape_8();
}
