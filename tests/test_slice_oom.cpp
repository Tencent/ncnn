// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

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

static int test_slice_oom(const ncnn::Mat& a, const std::vector<int>& slices_array, int axis)
{
    ncnn::Mat slices(slices_array.size());
    {
        int* p = slices;
        for (size_t i = 0; i < slices_array.size(); i++)
        {
            p[i] = slices_array[i];
        }
    }

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
        print_int_array(slices_array);
        fprintf(stderr, " axis=%d\n", axis);
    }

    return ret;
}

static int test_slice_oom_indices(const ncnn::Mat& a, const std::vector<int>& indices_array, int axis)
{
    ncnn::Mat indices(indices_array.size());
    {
        int* p = indices;
        for (size_t i = 0; i < indices_array.size(); i++)
        {
            p[i] = indices_array[i];
        }
    }

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
        print_int_array(indices_array);
        fprintf(stderr, " axis=%d\n", axis);
    }

    return ret;
}

static int test_slice_0()
{
    ncnn::Mat a = RandomMat(48, 48, 48, 48);

    return 0
           || test_slice_oom(a, IntArray(3, 12, 16, -233), 0)
           || test_slice_oom(a, IntArray(3, 12, 16, -233), 1)
           || test_slice_oom(a, IntArray(3, 12, 16, -233), 2)
           || test_slice_oom(a, IntArray(3, 12, 16, -233), 3)
           || test_slice_oom_indices(a, IntArray(2, -24, -8), 0);
}

static int test_slice_1()
{
    ncnn::Mat a = RandomMat(48, 48, 48);

    return 0
           || test_slice_oom(a, IntArray(3, 12, 16, -233), 0)
           || test_slice_oom(a, IntArray(3, 12, 16, -233), 1)
           || test_slice_oom(a, IntArray(3, 12, 16, -233), 2)
           || test_slice_oom_indices(a, IntArray(2, -24, -8), 0);
}

static int test_slice_2()
{
    ncnn::Mat a = RandomMat(48, 48);

    return 0
           || test_slice_oom(a, IntArray(3, 12, 16, -233), 0)
           || test_slice_oom(a, IntArray(3, 12, 16, -233), 1)
           || test_slice_oom_indices(a, IntArray(2, -24, -8), 0);
}

static int test_slice_3()
{
    ncnn::Mat a = RandomMat(48);

    return 0
           || test_slice_oom(a, IntArray(3, 12, 16, -233), 0)
           || test_slice_oom_indices(a, IntArray(2, -24, -8), 0);
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
