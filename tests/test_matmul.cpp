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

#include "layer/matmul.h"
#include "testutil.h"

static int test_matmul(const ncnn::Mat& a, const ncnn::Mat& b)
{
    ncnn::ParamDict pd;
    pd.set(0, 0); // transB

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = b;

    int ret = test_layer<ncnn::MatMul>("MatMul", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_matmul failed a.dims=%d a=(%d %d %d %d) b.dims=%d b=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c, b.dims, b.w, b.h, b.d, b.c);
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
           || test_matmul(RandomMat(5, 6), RandomMat(5))
           || test_matmul(RandomMat(16, 12), RandomMat(16))
           || test_matmul(RandomMat(11, 16), RandomMat(11));
}

static int test_matmul_2()
{
    return 0
           || test_matmul(RandomMat(32, 3, 10), RandomMat(32))
           || test_matmul(RandomMat(31, 4, 16), RandomMat(31))
           || test_matmul(RandomMat(28, 5, 28), RandomMat(28));
}

static int test_matmul_3()
{
    return 0
           || test_matmul(RandomMat(14, 23, 10), RandomMat(5, 14, 10))
           || test_matmul(RandomMat(16, 22, 16), RandomMat(10, 16, 16))
           || test_matmul(RandomMat(14, 20, 28), RandomMat(9, 14, 28));
}

static int test_matmul_4()
{
    return 0
           || test_matmul(RandomMat(14, 23, 10), RandomMat(5, 14))
           || test_matmul(RandomMat(16, 22, 16), RandomMat(10, 16))
           || test_matmul(RandomMat(14, 20, 28), RandomMat(9, 14));
}

static int test_matmul_5()
{
    return 0
           || test_matmul(RandomMat(14, 23, 10, 16), RandomMat(5, 14, 10, 16))
           || test_matmul(RandomMat(16, 22, 9, 17), RandomMat(10, 16, 9, 17))
           || test_matmul(RandomMat(14, 20, 8, 18), RandomMat(9, 14, 8, 18));
}

static int test_matmul_6()
{
    return 0
           || test_matmul(RandomMat(13), RandomMat(7, 13, 12))
           || test_matmul(RandomMat(24), RandomMat(6, 24, 16))
           || test_matmul(RandomMat(20), RandomMat(8, 20, 19));
}

static int test_matmul_7()
{
    return 0
           || test_matmul(RandomMat(5, 4), RandomMat(4, 5, 12))
           || test_matmul(RandomMat(5, 14), RandomMat(5, 5, 16))
           || test_matmul(RandomMat(5, 24), RandomMat(6, 5, 19));
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
           || test_matmul_7();
}
