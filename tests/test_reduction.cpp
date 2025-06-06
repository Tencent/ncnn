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

#define OP_TYPE_MAX 11

static int op_type = 0;

static std::vector<int> IntArray(int a0)
{
    std::vector<int> m(1);
    m[0] = a0;
    return m;
}

static std::vector<int> IntArray(int a0, int a1)
{
    std::vector<int> m(2);
    m[0] = a0;
    m[1] = a1;
    return m;
}

static std::vector<int> IntArray(int a0, int a1, int a2)
{
    std::vector<int> m(3);
    m[0] = a0;
    m[1] = a1;
    m[2] = a2;
    return m;
}

static std::vector<int> IntArray(int a0, int a1, int a2, int a3)
{
    std::vector<int> m(4);
    m[0] = a0;
    m[1] = a1;
    m[2] = a2;
    m[3] = a3;
    return m;
}

static void print_int_array(const std::vector<int>& a)
{
    fprintf(stderr, "[");
    for (size_t i = 0; i < a.size(); i++)
    {
        fprintf(stderr, " %d", a[i]);
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

    int ret = test_layer("Reduction", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_reduction failed a.dims=%d a=(%d %d %d %d) op_type=%d coeff=%f keepdims=%d reduce_all=1\n", a.dims, a.w, a.h, a.d, a.c, op_type, coeff, keepdims);
    }

    return ret;
}

static int test_reduction(const ncnn::Mat& _a, float coeff, int keepdims, const std::vector<int>& axes_array)
{
    ncnn::Mat a = _a;
    if (op_type == 9 || op_type == 10)
    {
        // value must be positive for logsum and logsumexp
        Randomize(a, 0.001f, 2.f);
    }

    ncnn::Mat axes(axes_array.size());
    {
        int* p = axes;
        for (size_t i = 0; i < axes_array.size(); i++)
        {
            p[i] = axes_array[i];
        }
    }

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0); // reduce_all
    pd.set(2, coeff);
    pd.set(3, axes);
    pd.set(4, keepdims);
    pd.set(5, 1); // fixbug0

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Reduction", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_reduction failed a.dims=%d a=(%d %d %d %d) op_type=%d coeff=%f keepdims=%d", a.dims, a.w, a.h, a.d, a.c, op_type, coeff, keepdims);
        fprintf(stderr, " axes=");
        print_int_array(axes_array);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_reduction_nd(const ncnn::Mat& a)
{
    int ret1 = 0
               || test_reduction(a, 1.f, 0)
               || test_reduction(a, 2.f, 0)
               || test_reduction(a, 1.f, 1)
               || test_reduction(a, 2.f, 1)
               || test_reduction(a, 1.f, 0, IntArray(0))
               || test_reduction(a, 1.f, 1, IntArray(0));

    if (a.dims == 1 || ret1 != 0)
        return ret1;

    int ret2 = 0
               || test_reduction(a, 2.f, 0, IntArray(1))
               || test_reduction(a, 2.f, 1, IntArray(1))
               || test_reduction(a, 1.f, 0, IntArray(0, 1))
               || test_reduction(a, 1.f, 1, IntArray(0, 1));

    if (a.dims == 2 || ret2 != 0)
        return ret2;

    int ret3 = 0
               || test_reduction(a, 1.f, 0, IntArray(2))
               || test_reduction(a, 1.f, 1, IntArray(2))
               || test_reduction(a, 2.f, 0, IntArray(0, 2))
               || test_reduction(a, 2.f, 0, IntArray(1, 2))
               || test_reduction(a, 2.f, 1, IntArray(0, 2))
               || test_reduction(a, 2.f, 1, IntArray(1, 2))
               || test_reduction(a, 1.f, 0, IntArray(0, 1, 2))
               || test_reduction(a, 1.f, 1, IntArray(0, 1, 2));

    if (a.dims == 3 || ret3 != 0)
        return ret3;

    int ret4 = 0
               || test_reduction(a, 2.f, 0, IntArray(3))
               || test_reduction(a, 2.f, 1, IntArray(3))
               || test_reduction(a, 1.f, 0, IntArray(0, 3))
               || test_reduction(a, 1.f, 0, IntArray(1, 3))
               || test_reduction(a, 2.f, 0, IntArray(2, 3))
               || test_reduction(a, 1.f, 1, IntArray(0, 3))
               || test_reduction(a, 1.f, 1, IntArray(1, 3))
               || test_reduction(a, 2.f, 1, IntArray(2, 3))
               || test_reduction(a, 2.f, 0, IntArray(0, 1, 3))
               || test_reduction(a, 1.f, 0, IntArray(0, 2, 3))
               || test_reduction(a, 2.f, 0, IntArray(1, 2, 3))
               || test_reduction(a, 2.f, 1, IntArray(0, 1, 3))
               || test_reduction(a, 1.f, 1, IntArray(0, 2, 3))
               || test_reduction(a, 2.f, 1, IntArray(1, 2, 3))
               || test_reduction(a, 1.f, 0, IntArray(0, 1, 2, 3))
               || test_reduction(a, 1.f, 1, IntArray(0, 1, 2, 3));

    return ret4;
}

static int test_reduction_0()
{
    ncnn::Mat a = RandomMat(5, 6, 7, 24);
    ncnn::Mat b = RandomMat(7, 8, 9, 12);
    ncnn::Mat c = RandomMat(3, 4, 5, 13);

    return 0
           || test_reduction_nd(a)
           || test_reduction_nd(b)
           || test_reduction_nd(c);
}

static int test_reduction_1()
{
    ncnn::Mat a = RandomMat(5, 7, 24);
    ncnn::Mat b = RandomMat(7, 9, 12);
    ncnn::Mat c = RandomMat(3, 5, 13);

    return 0
           || test_reduction_nd(a)
           || test_reduction_nd(b)
           || test_reduction_nd(c);
}

static int test_reduction_2()
{
    ncnn::Mat a = RandomMat(15, 24);
    ncnn::Mat b = RandomMat(17, 12);
    ncnn::Mat c = RandomMat(19, 15);

    return 0
           || test_reduction_nd(a)
           || test_reduction_nd(b)
           || test_reduction_nd(c);
}

static int test_reduction_3()
{
    ncnn::Mat a = RandomMat(128);
    ncnn::Mat b = RandomMat(124);
    ncnn::Mat c = RandomMat(127);

    return 0
           || test_reduction_nd(a)
           || test_reduction_nd(b)
           || test_reduction_nd(c);
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
