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

static int test_expanddims(const ncnn::Mat& a, int expand_w, int expand_h, int expand_d, int expand_c)
{
    ncnn::ParamDict pd;
    pd.set(0, expand_w);
    pd.set(1, expand_h);
    pd.set(11, expand_d);
    pd.set(2, expand_c);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("ExpandDims", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_expanddims failed a.dims=%d a=(%d %d %d %d) expand_w=%d expand_h=%d expand_d=%d expand_c=%d\n", a.dims, a.w, a.h, a.d, a.c, expand_w, expand_h, expand_d, expand_c);
    }

    return ret;
}

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

static int test_expanddims_axes(const ncnn::Mat& a, const std::vector<int>& axes_array)
{
    ncnn::Mat axes(axes_array.size());
    {
        int* p = axes;
        for (size_t i = 0; i < axes_array.size(); i++)
        {
            p[i] = axes_array[i];
        }
    }

    ncnn::ParamDict pd;
    pd.set(3, axes);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("ExpandDims", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_expanddims_axes failed a.dims=%d a=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
        fprintf(stderr, " axes=");
        print_int_array(axes_array);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_expanddims_all_params(const ncnn::Mat& a)
{
    return 0
           || test_expanddims(a, 0, 0, 0, 0)
           || test_expanddims(a, 0, 0, 0, 1)
           || test_expanddims(a, 0, 0, 1, 0)
           || test_expanddims(a, 0, 0, 1, 1)
           || test_expanddims(a, 0, 1, 0, 0)
           || test_expanddims(a, 0, 1, 0, 1)
           || test_expanddims(a, 0, 1, 1, 0)
           || test_expanddims(a, 0, 1, 1, 1)
           || test_expanddims(a, 1, 0, 0, 0)
           || test_expanddims(a, 1, 0, 0, 1)
           || test_expanddims(a, 1, 0, 1, 0)
           || test_expanddims(a, 1, 0, 1, 1)
           || test_expanddims(a, 1, 1, 0, 0)
           || test_expanddims(a, 1, 1, 0, 1)
           || test_expanddims(a, 1, 1, 1, 0)
           || test_expanddims(a, 1, 1, 1, 1)

           || test_expanddims_axes(a, IntArray(0))
           || test_expanddims_axes(a, IntArray(1))
           || test_expanddims_axes(a, IntArray(2))
           || test_expanddims_axes(a, IntArray(3))
           || test_expanddims_axes(a, IntArray(0, 1))
           || test_expanddims_axes(a, IntArray(0, 2))
           || test_expanddims_axes(a, IntArray(0, 3))
           || test_expanddims_axes(a, IntArray(1, 2))
           || test_expanddims_axes(a, IntArray(1, 3))
           || test_expanddims_axes(a, IntArray(2, 3))
           || test_expanddims_axes(a, IntArray(0, 1, 2))
           || test_expanddims_axes(a, IntArray(0, 1, 3))
           || test_expanddims_axes(a, IntArray(0, 2, 3))
           || test_expanddims_axes(a, IntArray(1, 2, 3))
           || test_expanddims_axes(a, IntArray(0, 1, 2, 3));
}

static int test_expanddims_0()
{
    return 0
           || test_expanddims_all_params(RandomMat(3, 12, 16))
           || test_expanddims_all_params(RandomMat(3, 1, 16))
           || test_expanddims_all_params(RandomMat(1, 33, 15))
           || test_expanddims_all_params(RandomMat(1, 14, 1))
           || test_expanddims_all_params(RandomMat(12, 13, 1))
           || test_expanddims_all_params(RandomMat(1, 1, 1));
}

static int test_expanddims_1()
{
    return 0
           || test_expanddims_all_params(RandomMat(14, 16))
           || test_expanddims_all_params(RandomMat(1, 14))
           || test_expanddims_all_params(RandomMat(11, 1))
           || test_expanddims_all_params(RandomMat(1, 1));
}

static int test_expanddims_2()
{
    return 0
           || test_expanddims_all_params(RandomMat(120))
           || test_expanddims_all_params(RandomMat(1));
}

int main()
{
    SRAND(7767517);

    return test_expanddims_0() || test_expanddims_1() || test_expanddims_2();
}
