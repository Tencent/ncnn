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

#include "layer/reduction.h"
#include "testutil.h"

#define OP_TYPE_MAX 11

static int op_type = 0;

static ncnn::Mat IntArrayMat(int a0)
{
    ncnn::Mat m(1);
    int* p = m;
    p[0] = a0;
    return m;
}

static ncnn::Mat IntArrayMat(int a0, int a1)
{
    ncnn::Mat m(2);
    int* p = m;
    p[0] = a0;
    p[1] = a1;
    return m;
}

static ncnn::Mat IntArrayMat(int a0, int a1, int a2)
{
    ncnn::Mat m(3);
    int* p = m;
    p[0] = a0;
    p[1] = a1;
    p[2] = a2;
    return m;
}

static ncnn::Mat IntArrayMat(int a0, int a1, int a2, int a3)
{
    ncnn::Mat m(4);
    int* p = m;
    p[0] = a0;
    p[1] = a1;
    p[2] = a2;
    p[3] = a3;
    return m;
}

static void print_int_array(const ncnn::Mat& a)
{
    const int* pa = a;

    fprintf(stderr, "[");
    for (int i = 0; i < a.w; i++)
    {
        fprintf(stderr, " %d", pa[i]);
    }
    fprintf(stderr, " ]");
}

static int test_reduction(const ncnn::Mat& _a, float coeff, int keepdims)
{
    ncnn::Mat a = _a;
    if (op_type == 9 || op_type == 10)
    {
        // value must be positive for logsum and logsumexp
        Randomize(a, 0.001f, 2.f);
    }

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 1); // reduce_all
    pd.set(2, coeff);
    pd.set(4, keepdims);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Reduction>("Reduction", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_reduction failed a.dims=%d a=(%d %d %d %d) op_type=%d coeff=%f keepdims=%d reduce_all=1\n", a.dims, a.w, a.h, a.d, a.c, op_type, coeff, keepdims);
    }

    return ret;
}

static int test_reduction(const ncnn::Mat& _a, float coeff, int keepdims, const ncnn::Mat& axes)
{
    ncnn::Mat a = _a;
    if (op_type == 9 || op_type == 10)
    {
        // value must be positive for logsum and logsumexp
        Randomize(a, 0.001f, 2.f);
    }

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0); // reduce_all
    pd.set(2, coeff);
    pd.set(3, axes);
    pd.set(4, keepdims);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Reduction>("Reduction", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_reduction failed a.dims=%d a=(%d %d %d %d) op_type=%d coeff=%f keepdims=%d", a.dims, a.w, a.h, a.d, a.c, op_type, coeff, keepdims);
        fprintf(stderr, " axes=");
        print_int_array(axes);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_reduction_0()
{
    return 0
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 0)
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 0)
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 0)
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 0)
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 0)
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 0)

           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 1)
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 1)
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 1)
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 1)
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 1)
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 1)

           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 0, IntArrayMat(1))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 0, IntArrayMat(2))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 0, IntArrayMat(3))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 0, IntArrayMat(0, 1))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 0, IntArrayMat(0, 2))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 0, IntArrayMat(0, 3))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 0, IntArrayMat(1, 2))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 0, IntArrayMat(1, 3))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 0, IntArrayMat(2, 3))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 0, IntArrayMat(0, 1, 2))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 0, IntArrayMat(0, 1, 3))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 0, IntArrayMat(0, 2, 3))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 0, IntArrayMat(1, 2, 3))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 0, IntArrayMat(0, 1, 2, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 0, IntArrayMat(1))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 0, IntArrayMat(2))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 0, IntArrayMat(3))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 0, IntArrayMat(0, 1))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 0, IntArrayMat(0, 2))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 0, IntArrayMat(0, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 0, IntArrayMat(1, 2))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 0, IntArrayMat(1, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 0, IntArrayMat(2, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 0, IntArrayMat(0, 1, 2))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 0, IntArrayMat(0, 1, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 0, IntArrayMat(0, 2, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 0, IntArrayMat(1, 2, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 0, IntArrayMat(0, 1, 2, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 0, IntArrayMat(1))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 0, IntArrayMat(2))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 0, IntArrayMat(3))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 0, IntArrayMat(0, 1))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 0, IntArrayMat(0, 2))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 0, IntArrayMat(0, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 0, IntArrayMat(1, 2))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 0, IntArrayMat(1, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 0, IntArrayMat(2, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 0, IntArrayMat(0, 1, 2))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 0, IntArrayMat(0, 1, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 0, IntArrayMat(0, 2, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 0, IntArrayMat(1, 2, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 0, IntArrayMat(0, 1, 2, 3))

           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 1, IntArrayMat(1))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 1, IntArrayMat(2))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 1, IntArrayMat(3))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 1, IntArrayMat(0, 1))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 1, IntArrayMat(0, 2))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 1, IntArrayMat(0, 3))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 1, IntArrayMat(1, 2))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 1, IntArrayMat(1, 3))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 1, IntArrayMat(2, 3))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 1, IntArrayMat(0, 1, 2))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 1, IntArrayMat(0, 1, 3))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 1, IntArrayMat(0, 2, 3))
           || test_reduction(RandomMat(5, 6, 7, 24), 2.f, 1, IntArrayMat(1, 2, 3))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 1, IntArrayMat(0, 1, 2, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 1, IntArrayMat(1))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 1, IntArrayMat(2))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 1, IntArrayMat(3))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 1, IntArrayMat(0, 1))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 1, IntArrayMat(0, 2))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 1, IntArrayMat(0, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 1, IntArrayMat(1, 2))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 1, IntArrayMat(1, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 1, IntArrayMat(2, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 1, IntArrayMat(0, 1, 2))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 1, IntArrayMat(0, 1, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 1, IntArrayMat(0, 2, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 2.f, 1, IntArrayMat(1, 2, 3))
           || test_reduction(RandomMat(7, 8, 9, 12), 1.f, 1, IntArrayMat(0, 1, 2, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 1, IntArrayMat(1))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 1, IntArrayMat(2))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 1, IntArrayMat(3))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 1, IntArrayMat(0, 1))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 1, IntArrayMat(0, 2))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 1, IntArrayMat(0, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 1, IntArrayMat(1, 2))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 1, IntArrayMat(1, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 1, IntArrayMat(2, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 1, IntArrayMat(0, 1, 2))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 1, IntArrayMat(0, 1, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 1, IntArrayMat(0, 2, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 2.f, 1, IntArrayMat(1, 2, 3))
           || test_reduction(RandomMat(3, 4, 5, 13), 1.f, 1, IntArrayMat(0, 1, 2, 3));
}

static int test_reduction_1()
{
    return 0
           || test_reduction(RandomMat(5, 7, 24), 1.f, 0)
           || test_reduction(RandomMat(5, 7, 24), 2.f, 0)
           || test_reduction(RandomMat(7, 9, 12), 1.f, 0)
           || test_reduction(RandomMat(7, 9, 12), 2.f, 0)
           || test_reduction(RandomMat(3, 5, 13), 1.f, 0)
           || test_reduction(RandomMat(3, 5, 13), 2.f, 0)

           || test_reduction(RandomMat(5, 7, 24), 1.f, 1)
           || test_reduction(RandomMat(5, 7, 24), 2.f, 1)
           || test_reduction(RandomMat(7, 9, 12), 1.f, 1)
           || test_reduction(RandomMat(7, 9, 12), 2.f, 1)
           || test_reduction(RandomMat(3, 5, 13), 1.f, 1)
           || test_reduction(RandomMat(3, 5, 13), 2.f, 1)

           || test_reduction(RandomMat(5, 7, 24), 1.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(5, 7, 24), 2.f, 0, IntArrayMat(1))
           || test_reduction(RandomMat(5, 7, 24), 1.f, 0, IntArrayMat(0, 1))
           || test_reduction(RandomMat(5, 7, 24), 2.f, 0, IntArrayMat(0, 2))
           || test_reduction(RandomMat(5, 7, 24), 1.f, 0, IntArrayMat(1, 2))
           || test_reduction(RandomMat(5, 7, 24), 2.f, 0, IntArrayMat(0, 1, 2))
           || test_reduction(RandomMat(7, 9, 12), 1.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(7, 9, 12), 2.f, 0, IntArrayMat(1))
           || test_reduction(RandomMat(7, 9, 12), 1.f, 0, IntArrayMat(0, 1))
           || test_reduction(RandomMat(7, 9, 12), 2.f, 0, IntArrayMat(0, 2))
           || test_reduction(RandomMat(7, 9, 12), 1.f, 0, IntArrayMat(1, 2))
           || test_reduction(RandomMat(7, 9, 12), 2.f, 0, IntArrayMat(0, 1, 2))
           || test_reduction(RandomMat(3, 5, 13), 1.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(3, 5, 13), 2.f, 0, IntArrayMat(1))
           || test_reduction(RandomMat(3, 5, 13), 1.f, 0, IntArrayMat(0, 1))
           || test_reduction(RandomMat(3, 5, 13), 2.f, 0, IntArrayMat(0, 2))
           || test_reduction(RandomMat(3, 5, 13), 1.f, 0, IntArrayMat(1, 2))
           || test_reduction(RandomMat(3, 5, 13), 2.f, 0, IntArrayMat(0, 1, 2))

           || test_reduction(RandomMat(5, 7, 24), 1.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(5, 7, 24), 2.f, 1, IntArrayMat(1))
           || test_reduction(RandomMat(5, 7, 24), 1.f, 1, IntArrayMat(0, 1))
           || test_reduction(RandomMat(5, 7, 24), 2.f, 1, IntArrayMat(0, 2))
           || test_reduction(RandomMat(5, 7, 24), 1.f, 1, IntArrayMat(1, 2))
           || test_reduction(RandomMat(5, 7, 24), 2.f, 1, IntArrayMat(0, 1, 2))
           || test_reduction(RandomMat(7, 9, 12), 1.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(7, 9, 12), 2.f, 1, IntArrayMat(1))
           || test_reduction(RandomMat(7, 9, 12), 1.f, 1, IntArrayMat(0, 1))
           || test_reduction(RandomMat(7, 9, 12), 2.f, 1, IntArrayMat(0, 2))
           || test_reduction(RandomMat(7, 9, 12), 1.f, 1, IntArrayMat(1, 2))
           || test_reduction(RandomMat(7, 9, 12), 2.f, 1, IntArrayMat(0, 1, 2))
           || test_reduction(RandomMat(3, 5, 13), 1.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(3, 5, 13), 2.f, 1, IntArrayMat(1))
           || test_reduction(RandomMat(3, 5, 13), 1.f, 1, IntArrayMat(0, 1))
           || test_reduction(RandomMat(3, 5, 13), 2.f, 1, IntArrayMat(0, 2))
           || test_reduction(RandomMat(3, 5, 13), 1.f, 1, IntArrayMat(1, 2))
           || test_reduction(RandomMat(3, 5, 13), 2.f, 1, IntArrayMat(0, 1, 2));
}

static int test_reduction_2()
{
    return 0
           || test_reduction(RandomMat(15, 24), 1.f, 0)
           || test_reduction(RandomMat(15, 24), 2.f, 0)
           || test_reduction(RandomMat(17, 12), 1.f, 0)
           || test_reduction(RandomMat(17, 12), 2.f, 0)
           || test_reduction(RandomMat(19, 15), 1.f, 0)
           || test_reduction(RandomMat(19, 15), 2.f, 0)

           || test_reduction(RandomMat(15, 24), 1.f, 1)
           || test_reduction(RandomMat(15, 24), 2.f, 1)
           || test_reduction(RandomMat(17, 12), 1.f, 1)
           || test_reduction(RandomMat(17, 12), 2.f, 1)
           || test_reduction(RandomMat(19, 15), 1.f, 1)
           || test_reduction(RandomMat(19, 15), 2.f, 1)

           || test_reduction(RandomMat(15, 24), 1.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(15, 24), 2.f, 0, IntArrayMat(1))
           || test_reduction(RandomMat(15, 24), 1.f, 0, IntArrayMat(0, 1))
           || test_reduction(RandomMat(17, 12), 2.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(17, 12), 1.f, 0, IntArrayMat(1))
           || test_reduction(RandomMat(17, 12), 2.f, 0, IntArrayMat(0, 1))
           || test_reduction(RandomMat(19, 15), 1.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(19, 15), 2.f, 0, IntArrayMat(1))
           || test_reduction(RandomMat(19, 15), 1.f, 0, IntArrayMat(0, 1))

           || test_reduction(RandomMat(15, 24), 2.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(15, 24), 1.f, 1, IntArrayMat(1))
           || test_reduction(RandomMat(15, 24), 2.f, 1, IntArrayMat(0, 1))
           || test_reduction(RandomMat(17, 12), 1.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(17, 12), 2.f, 1, IntArrayMat(1))
           || test_reduction(RandomMat(17, 12), 1.f, 1, IntArrayMat(0, 1))
           || test_reduction(RandomMat(19, 15), 2.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(19, 15), 1.f, 1, IntArrayMat(1))
           || test_reduction(RandomMat(19, 15), 2.f, 1, IntArrayMat(0, 1));
}

static int test_reduction_3()
{
    return 0
           || test_reduction(RandomMat(128), 1.f, 0)
           || test_reduction(RandomMat(128), 2.f, 0)
           || test_reduction(RandomMat(124), 1.f, 0)
           || test_reduction(RandomMat(124), 2.f, 0)
           || test_reduction(RandomMat(127), 1.f, 0)
           || test_reduction(RandomMat(127), 2.f, 0)

           || test_reduction(RandomMat(128), 1.f, 1)
           || test_reduction(RandomMat(128), 2.f, 1)
           || test_reduction(RandomMat(124), 1.f, 1)
           || test_reduction(RandomMat(124), 2.f, 1)
           || test_reduction(RandomMat(127), 1.f, 1)
           || test_reduction(RandomMat(127), 2.f, 1)

           || test_reduction(RandomMat(128), 1.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(128), 2.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(124), 1.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(124), 2.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(127), 1.f, 0, IntArrayMat(0))
           || test_reduction(RandomMat(127), 2.f, 0, IntArrayMat(0))

           || test_reduction(RandomMat(128), 1.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(128), 2.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(124), 1.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(124), 2.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(127), 1.f, 1, IntArrayMat(0))
           || test_reduction(RandomMat(127), 1.f, 1, IntArrayMat(0));
}

int main()
{
    SRAND(7767517);

    for (op_type = 0; op_type < OP_TYPE_MAX; op_type++)
    {
        int ret = 0
                  || test_reduction_0()
                  || test_reduction_1()
                  || test_reduction_2()
                  || test_reduction_3();

        if (ret != 0)
            return ret;
    }

    return 0;
}
