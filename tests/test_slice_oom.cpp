// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

static int test_slice_oom(const ncnn::Mat& a, const ncnn::Mat& slices, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, slices);
    pd.set(1, axis);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a0(1);
    a0[0] = a;

    int ret = test_layer_oom("Slice", pd, weights, a0, slices.w);
    if (ret != 0)
    {
        fprintf(stderr, "test_slice_oom failed a.dims=%d a=(%d %d %d %d)", a.dims, a.w, a.h, a.d, a.c);
        fprintf(stderr, " slices=");
        print_int_array(slices);
        fprintf(stderr, " axis=%d\n", axis);
    }

    return ret;
}

static int test_slice_oom_indices(const ncnn::Mat& a, const ncnn::Mat& indices, int axis)
{
    ncnn::ParamDict pd;
    pd.set(1, axis);
    pd.set(2, indices);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a0(1);
    a0[0] = a;

    int ret = test_layer_oom("Slice", pd, weights, a0, indices.w);
    if (ret != 0)
    {
        fprintf(stderr, "test_slice_oom_indices failed a.dims=%d a=(%d %d %d %d)", a.dims, a.w, a.h, a.d, a.c);
        fprintf(stderr, " indices=");
        print_int_array(indices);
        fprintf(stderr, " axis=%d\n", axis);
    }

    return ret;
}

static int test_slice_0()
{
    ncnn::Mat a = RandomMat(48, 48, 48, 48);

    return 0
           || test_slice_oom(a, IntArrayMat(3, 12, 16, -233), 0)
           || test_slice_oom(a, IntArrayMat(3, 12, 16, -233), 1)
           || test_slice_oom(a, IntArrayMat(3, 12, 16, -233), 2)
           || test_slice_oom(a, IntArrayMat(3, 12, 16, -233), 3)
           || test_slice_oom_indices(a, IntArrayMat(2, -24, -8), 0);
}

static int test_slice_1()
{
    ncnn::Mat a = RandomMat(48, 48, 48);

    return 0
           || test_slice_oom(a, IntArrayMat(3, 12, 16, -233), 0)
           || test_slice_oom(a, IntArrayMat(3, 12, 16, -233), 1)
           || test_slice_oom(a, IntArrayMat(3, 12, 16, -233), 2)
           || test_slice_oom_indices(a, IntArrayMat(2, -24, -8), 0);
}

static int test_slice_2()
{
    ncnn::Mat a = RandomMat(48, 48);

    return 0
           || test_slice_oom(a, IntArrayMat(3, 12, 16, -233), 0)
           || test_slice_oom(a, IntArrayMat(3, 12, 16, -233), 1)
           || test_slice_oom_indices(a, IntArrayMat(2, -24, -8), 0);
}

static int test_slice_3()
{
    ncnn::Mat a = RandomMat(48);

    return 0
           || test_slice_oom(a, IntArrayMat(3, 12, 16, -233), 0)
           || test_slice_oom_indices(a, IntArrayMat(2, -24, -8), 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_slice_0()
           || test_slice_1()
           || test_slice_2()
           || test_slice_3();
}
