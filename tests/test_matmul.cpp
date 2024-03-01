// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static int test_matmul(const ncnn::Mat& a, const ncnn::Mat& b)
{
    ncnn::ParamDict pd;
    pd.set(0, 0); // transB

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = b;

    int ret = test_layer("MatMul", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_matmul failed a.dims=%d a=(%d %d %d %d) b.dims=%d b=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c, b.dims, b.w, b.h, b.d, b.c);
    }

    return ret;
}

static int test_matmul_transb(const ncnn::Mat& a, const ncnn::Mat& b)
{
    ncnn::ParamDict pd;
    pd.set(0, 1); // transB

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = b;

    int ret = test_layer("MatMul", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_matmul_transb failed a.dims=%d a=(%d %d %d %d) b.dims=%d b=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c, b.dims, b.w, b.h, b.d, b.c);
    }

    return ret;
}

static int test_matmul_0()
{
    return 0
           || test_matmul(RandomMat(124), RandomMat(124))
           || test_matmul(RandomMat(127), RandomMat(127))
           || test_matmul(RandomMat(128), RandomMat(128));
}

static int test_matmul_1()
{
    return 0
           || test_matmul(RandomMat(5), RandomMat(6, 5))
           || test_matmul(RandomMat(16), RandomMat(12, 16))
           || test_matmul(RandomMat(11), RandomMat(16, 11))

           || test_matmul_transb(RandomMat(5), RandomMat(5, 6))
           || test_matmul_transb(RandomMat(16), RandomMat(16, 12))
           || test_matmul_transb(RandomMat(11), RandomMat(11, 16));
}

static int test_matmul_2()
{
    return 0
           || test_matmul(RandomMat(13), RandomMat(7, 13, 12))
           || test_matmul(RandomMat(24), RandomMat(6, 24, 16))
           || test_matmul(RandomMat(20), RandomMat(8, 20, 19))

           || test_matmul_transb(RandomMat(13), RandomMat(13, 7, 12))
           || test_matmul_transb(RandomMat(24), RandomMat(24, 6, 16))
           || test_matmul_transb(RandomMat(20), RandomMat(20, 8, 19));
}

static int test_matmul_3()
{
    return 0
           || test_matmul(RandomMat(13), RandomMat(7, 13, 5, 12))
           || test_matmul(RandomMat(24), RandomMat(6, 24, 4, 16))
           || test_matmul(RandomMat(20), RandomMat(8, 20, 3, 19))

           || test_matmul_transb(RandomMat(13), RandomMat(13, 7, 5, 12))
           || test_matmul_transb(RandomMat(24), RandomMat(24, 6, 4, 16))
           || test_matmul_transb(RandomMat(20), RandomMat(20, 8, 3, 19));
}

static int test_matmul_4()
{
    return 0
           || test_matmul(RandomMat(5, 6), RandomMat(5))
           || test_matmul(RandomMat(16, 12), RandomMat(16))
           || test_matmul(RandomMat(11, 16), RandomMat(11));
}

static int test_matmul_5()
{
    return 0
           || test_matmul(RandomMat(32, 3, 10), RandomMat(32))
           || test_matmul(RandomMat(31, 4, 16), RandomMat(31))
           || test_matmul(RandomMat(28, 5, 28), RandomMat(28));
}

static int test_matmul_6()
{
    return 0
           || test_matmul(RandomMat(32, 3, 4, 10), RandomMat(32))
           || test_matmul(RandomMat(31, 4, 5, 16), RandomMat(31))
           || test_matmul(RandomMat(18, 5, 6, 28), RandomMat(18));
}

static int test_matmul_7()
{
    return 0
           || test_matmul(RandomMat(14, 10), RandomMat(5, 14))
           || test_matmul(RandomMat(16, 16), RandomMat(10, 16))
           || test_matmul(RandomMat(14, 28), RandomMat(9, 14))

           || test_matmul_transb(RandomMat(14, 10), RandomMat(14, 5))
           || test_matmul_transb(RandomMat(16, 16), RandomMat(16, 10))
           || test_matmul_transb(RandomMat(14, 28), RandomMat(14, 9));
}

static int test_matmul_8()
{
    return 0
           || test_matmul(RandomMat(5, 4), RandomMat(4, 5, 12))
           || test_matmul(RandomMat(5, 14), RandomMat(5, 5, 16))
           || test_matmul(RandomMat(5, 24), RandomMat(6, 5, 19))

           || test_matmul_transb(RandomMat(5, 4), RandomMat(5, 4, 12))
           || test_matmul_transb(RandomMat(5, 14), RandomMat(5, 5, 16))
           || test_matmul_transb(RandomMat(5, 24), RandomMat(5, 6, 19));
}

static int test_matmul_9()
{
    return 0
           || test_matmul(RandomMat(5, 4), RandomMat(4, 5, 2, 12))
           || test_matmul(RandomMat(5, 14), RandomMat(5, 5, 3, 16))
           || test_matmul(RandomMat(5, 24), RandomMat(6, 5, 4, 19))

           || test_matmul_transb(RandomMat(5, 4), RandomMat(5, 4, 2, 12))
           || test_matmul_transb(RandomMat(5, 14), RandomMat(5, 5, 3, 16))
           || test_matmul_transb(RandomMat(5, 24), RandomMat(5, 6, 4, 19));
}

static int test_matmul_10()
{
    return 0
           || test_matmul(RandomMat(14, 23, 10), RandomMat(5, 14))
           || test_matmul(RandomMat(16, 22, 16), RandomMat(10, 16))
           || test_matmul(RandomMat(14, 20, 28), RandomMat(9, 14))

           || test_matmul_transb(RandomMat(14, 23, 10), RandomMat(14, 5))
           || test_matmul_transb(RandomMat(16, 22, 16), RandomMat(16, 10))
           || test_matmul_transb(RandomMat(14, 20, 28), RandomMat(14, 9));
}

static int test_matmul_11()
{
    return 0
           || test_matmul(RandomMat(14, 13, 2, 10), RandomMat(5, 14))
           || test_matmul(RandomMat(16, 12, 3, 16), RandomMat(10, 16))
           || test_matmul(RandomMat(14, 10, 4, 28), RandomMat(9, 14))

           || test_matmul_transb(RandomMat(14, 13, 2, 10), RandomMat(14, 5))
           || test_matmul_transb(RandomMat(16, 12, 3, 16), RandomMat(16, 10))
           || test_matmul_transb(RandomMat(14, 10, 4, 28), RandomMat(14, 9));
}

static int test_matmul_12()
{
    return 0
           || test_matmul(RandomMat(14, 23, 10), RandomMat(5, 14, 10))
           || test_matmul(RandomMat(16, 22, 16), RandomMat(10, 16, 16))
           || test_matmul(RandomMat(14, 20, 28), RandomMat(9, 14, 28))

           || test_matmul_transb(RandomMat(14, 23, 10), RandomMat(14, 5, 10))
           || test_matmul_transb(RandomMat(16, 22, 16), RandomMat(16, 10, 16))
           || test_matmul_transb(RandomMat(14, 20, 28), RandomMat(14, 9, 28));
}

static int test_matmul_13()
{
    return 0
           || test_matmul(RandomMat(14, 23, 10), RandomMat(5, 14, 1, 16))
           || test_matmul(RandomMat(16, 22, 9), RandomMat(10, 16, 1, 17))
           || test_matmul(RandomMat(14, 20, 8), RandomMat(9, 14, 1, 18))

           || test_matmul_transb(RandomMat(14, 23, 10), RandomMat(14, 5, 1, 16))
           || test_matmul_transb(RandomMat(16, 22, 9), RandomMat(16, 10, 1, 17))
           || test_matmul_transb(RandomMat(14, 20, 8), RandomMat(14, 9, 1, 18));
}

static int test_matmul_14()
{
    return 0
           || test_matmul(RandomMat(14, 23, 10, 1), RandomMat(5, 14, 1, 16))
           || test_matmul(RandomMat(16, 22, 9, 1), RandomMat(10, 16, 1, 17))
           || test_matmul(RandomMat(14, 20, 8, 1), RandomMat(9, 14, 1, 18))

           || test_matmul_transb(RandomMat(14, 23, 10, 1), RandomMat(14, 5, 1, 16))
           || test_matmul_transb(RandomMat(16, 22, 9, 1), RandomMat(16, 10, 1, 17))
           || test_matmul_transb(RandomMat(14, 20, 8, 1), RandomMat(14, 9, 1, 18));
}

static int test_matmul_15()
{
    return 0
           || test_matmul(RandomMat(14, 23, 10, 16), RandomMat(5, 14, 10, 16))
           || test_matmul(RandomMat(16, 22, 9, 17), RandomMat(10, 16, 9, 17))
           || test_matmul(RandomMat(14, 20, 8, 18), RandomMat(9, 14, 8, 18))

           || test_matmul_transb(RandomMat(14, 23, 10, 16), RandomMat(14, 5, 10, 16))
           || test_matmul_transb(RandomMat(16, 22, 9, 17), RandomMat(16, 10, 9, 17))
           || test_matmul_transb(RandomMat(14, 20, 8, 18), RandomMat(14, 9, 8, 18));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_matmul_0()
           || test_matmul_1()
           || test_matmul_2()
           || test_matmul_3()
           || test_matmul_4()
           || test_matmul_5()
           || test_matmul_6()
           || test_matmul_7()
           || test_matmul_8()
           || test_matmul_9()
           || test_matmul_10()
           || test_matmul_11()
           || test_matmul_12()
           || test_matmul_13()
           || test_matmul_14()
           || test_matmul_15();
}
