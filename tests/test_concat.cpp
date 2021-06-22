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

#include "layer/concat.h"
#include "testutil.h"

static int test_concat(const std::vector<ncnn::Mat>& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); //axis

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Concat>("Concat", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_concat failed a[0].dims=%d a[0]=(%d %d %d) axis=%d\n", a[0].dims, a[0].w, a[0].h, a[0].c, axis);
    }

    return ret;
}

static int test_concat_0()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(16, 12, 64);
    a[1] = RandomMat(16, 12, 64);
    a[2] = RandomMat(16, 12, 64);

    return 0
           || test_concat(a, 0)
           || test_concat(a, 1)
           || test_concat(a, 2)
           || test_concat(a, -1)
           || test_concat(a, -2)
           || test_concat(a, -3);
}

static int test_concat_1()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(7, 3, 6);
    a[1] = RandomMat(7, 3, 16);
    a[2] = RandomMat(7, 3, 10);

    std::vector<ncnn::Mat> b(3);
    b[0] = RandomMat(9, 5, 16);
    b[1] = RandomMat(9, 5, 8);
    b[2] = RandomMat(9, 5, 24);

    return 0
           || test_concat(a, 0)
           || test_concat(a, -3)

           || test_concat(b, 0)
           || test_concat(b, -3);
}

static int test_concat_2()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(7, 3, 10);
    a[1] = RandomMat(7, 8, 10);
    a[2] = RandomMat(7, 5, 10);

    std::vector<ncnn::Mat> b(3);
    b[0] = RandomMat(9, 8, 24);
    b[1] = RandomMat(9, 3, 24);
    b[2] = RandomMat(9, 5, 24);

    return 0
           || test_concat(a, 1)
           || test_concat(a, -2)

           || test_concat(b, 1)
           || test_concat(b, -2);
}

static int test_concat_3()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(8, 9, 6);
    a[1] = RandomMat(3, 9, 6);
    a[2] = RandomMat(5, 9, 6);

    std::vector<ncnn::Mat> b(3);
    b[0] = RandomMat(1, 7, 32);
    b[1] = RandomMat(8, 7, 32);
    b[2] = RandomMat(7, 7, 32);

    return 0
           || test_concat(a, 2)
           || test_concat(a, -1)

           || test_concat(b, 2)
           || test_concat(b, -1);
}

static int test_concat_4()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(11, 6);
    a[1] = RandomMat(11, 16);
    a[2] = RandomMat(11, 10);

    std::vector<ncnn::Mat> b(3);
    b[0] = RandomMat(15, 24);
    b[1] = RandomMat(15, 16);
    b[2] = RandomMat(15, 8);

    return 0
           || test_concat(a, 0)
           || test_concat(a, -2)

           || test_concat(b, 0)
           || test_concat(b, -2);
}

static int test_concat_5()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(9, 14);
    a[1] = RandomMat(8, 14);
    a[2] = RandomMat(11, 14);

    std::vector<ncnn::Mat> b(3);
    b[0] = RandomMat(13, 48);
    b[1] = RandomMat(18, 48);
    b[2] = RandomMat(15, 48);

    return 0
           || test_concat(a, 1)
           || test_concat(a, -1)

           || test_concat(b, 1)
           || test_concat(b, -1);
}

static int test_concat_6()
{
    std::vector<ncnn::Mat> a(3);
    a[0] = RandomMat(6);
    a[1] = RandomMat(16);
    a[2] = RandomMat(10);

    std::vector<ncnn::Mat> b(3);
    b[0] = RandomMat(8);
    b[1] = RandomMat(16);
    b[2] = RandomMat(24);

    return 0
           || test_concat(a, 0)
           || test_concat(a, -1)

           || test_concat(b, 0)
           || test_concat(b, -1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_concat_0()
           || test_concat_1()
           || test_concat_2()
           || test_concat_3()
           || test_concat_4()
           || test_concat_5()
           || test_concat_6();
}
