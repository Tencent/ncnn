// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

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

static int test_slice(const ncnn::Mat& a, const std::vector<int>& slices_array, int axis, int flag = 0)
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

    int ret = test_layer("Slice", pd, weights, a0, slices.w, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_slice failed a.dims=%d a=(%d %d %d %d)", a.dims, a.w, a.h, a.d, a.c);
        fprintf(stderr, " slices=");
        print_int_array(slices_array);
        fprintf(stderr, " axis=%d\n", axis);
    }

    return ret;
}

static int test_slice_indices(const ncnn::Mat& a, const std::vector<int>& indices_array, int axis, int flag = 0)
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

    int ret = test_layer("Slice", pd, weights, a0, indices.w, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_slice_indices failed a.dims=%d a=(%d %d %d %d)", a.dims, a.w, a.h, a.d, a.c);
        fprintf(stderr, " indices=");
        print_int_array(indices_array);
        fprintf(stderr, " axis=%d\n", axis);
    }

    return ret;
}

static int test_slice_0(const ncnn::Mat& a, int flag = 0)
{
    return 0
           || test_slice(a, IntArray(-233, -233, -233), 0, flag)
           || test_slice(a, IntArray(-233, -233, -233), 1, flag)
           || test_slice(a, IntArray(-233, -233, -233), -2, flag)
           || test_slice(a, IntArray(-233, -233, -233), 3, flag)
           || test_slice(a, IntArray(3, 12, 16, -233), 0, flag)
           || test_slice(a, IntArray(12, 16, -233), 0, flag)
           || test_slice(a, IntArray(32, 8, -233), 0, flag)
           || test_slice(a, IntArray(2, 12, 16, -233), 1, flag)
           || test_slice(a, IntArray(16, 4, 5, -233), -2, flag)
           || test_slice(a, IntArray(8, 2, 16, -233), 3, flag)
           || test_slice_indices(a, IntArray(2, -24, -8), 0, flag)
           || test_slice_indices(a, IntArray(4, 20, 4), 1, flag)
           || test_slice_indices(a, IntArray(16, -16), -2, flag)
           || test_slice_indices(a, IntArray(1, -12), 3, flag);
}

static int test_slice_0()
{
    ncnn::Mat a[] = {
        RandomMat(30, 32, 36, 48),
        RandomMat(36, 30, 32, 51),
        RandomMat(30, 32, 36, 60)
    };

    // the 60-element case retains the same Vulkan input/output packing
    return 0
           || test_slice_0(a[0], TEST_LAYER_DISABLE_GPU_TESTING)
           || test_slice_0(a[1])
           || test_slice_0(a[2]);
}

static int test_slice_1(const ncnn::Mat& a, int flag = 0)
{
    return 0
           || test_slice(a, IntArray(-233, -233, -233), 0, flag)
           || test_slice(a, IntArray(-233, -233, -233), 1, flag)
           || test_slice(a, IntArray(-233, -233, -233), -1, flag)
           || test_slice(a, IntArray(3, 12, 16, -233), 0, flag)
           || test_slice(a, IntArray(12, 16, -233), 0, flag)
           || test_slice(a, IntArray(32, 8, -233), 0, flag)
           || test_slice(a, IntArray(2, 12, 16, -233), 1, flag)
           || test_slice(a, IntArray(16, 4, 5, -233), -1, flag)
           || test_slice_indices(a, IntArray(2, -24, -8), 0, flag)
           || test_slice_indices(a, IntArray(4, 20, 4), 1, flag)
           || test_slice_indices(a, IntArray(1, -12), 2, flag);
}

static int test_slice_1()
{
    ncnn::Mat a[] = {
        RandomMat(51, 36, 48),
        RandomMat(48, 48, 51),
        RandomMat(51, 36, 60)
    };

    // the 60-element case retains the same Vulkan input/output packing
    return 0
           || test_slice_1(a[0], TEST_LAYER_DISABLE_GPU_TESTING)
           || test_slice_1(a[1])
           || test_slice_1(a[2]);
}

static int test_slice_2(const ncnn::Mat& a, int flag = 0)
{
    return 0
           || test_slice(a, IntArray(-233, -233, -233), 0, flag)
           || test_slice(a, IntArray(-233, -233, -233), -1, flag)
           || test_slice(a, IntArray(3, 12, 16, -233), 0, flag)
           || test_slice(a, IntArray(12, 16, -233), 0, flag)
           || test_slice(a, IntArray(32, 8, -233), -2, flag)
           || test_slice(a, IntArray(2, 12, 16, -233), -1, flag)
           || test_slice_indices(a, IntArray(2, -24, -8), 0, flag)
           || test_slice_indices(a, IntArray(1, -12), 1, flag);
}

static int test_slice_2()
{
    ncnn::Mat a[] = {
        RandomMat(36, 48),
        RandomMat(48, 51),
        RandomMat(36, 60)
    };

    // the 60-element case retains the same Vulkan input/output packing
    return 0
           || test_slice_2(a[0], TEST_LAYER_DISABLE_GPU_TESTING)
           || test_slice_2(a[1])
           || test_slice_2(a[2]);
}

static int test_slice_3(const ncnn::Mat& a, int flag = 0)
{
    return 0
           || test_slice(a, IntArray(-233, -233, -233), 0, flag)
           || test_slice(a, IntArray(3, 12, 16, -233), 0, flag)
           || test_slice(a, IntArray(12, 16, -233), 0, flag)
           || test_slice(a, IntArray(32, 8, -233), -1, flag)
           || test_slice_indices(a, IntArray(2, -24, -8), 0, flag);
}

static int test_slice_3()
{
    ncnn::Mat a[] = {
        RandomMat(48),
        RandomMat(51),
        RandomMat(60)
    };

    // the 60-element case retains the same Vulkan input/output packing
    return 0
           || test_slice_3(a[0], TEST_LAYER_DISABLE_GPU_TESTING)
           || test_slice_3(a[1])
           || test_slice_3(a[2]);
}

static int test_slice_4()
{
    ncnn::Mat a = RandomMat(31, 28);
    ncnn::Mat b = RandomMat(1, 21, 36);

    return test_slice(a, IntArray(4, -233), 0) || test_slice(b, IntArray(12, -233), 0);
}

static ncnn::Mat param_int_array(int size, int value)
{
    ncnn::Mat m(size);
    int* p = m;
    for (int i = 0; i < size; i++)
        p[i] = value;

    return m;
}

#if NCNN_VALIDATION
static int test_slice_load_param()
{
    ncnn::ParamDict base;
    base.set(0, param_int_array(1, 1));
    if (test_layer_param(ncnn::LayerType::Slice, base, 0) != 0)
        return -1;

    {
        ncnn::ParamDict pd = base;
        pd.set(0, ncnn::Mat(0));
        pd.set(2, ncnn::Mat(0));
        if (test_layer_param(ncnn::LayerType::Slice, pd, 0) != 0)
            return -1;
    }

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    return 0
           || test_layer_param(ncnn::LayerType::Slice, base, 0, missing_data, -1)
           || test_layer_param(ncnn::LayerType::Slice, base, 2, missing_data, -1)
           || test_layer_param(ncnn::LayerType::Slice, base, 0, ncnn::Mat(1, (size_t)1u), -1)
           || test_layer_param(ncnn::LayerType::Slice, base, 0, ncnn::Mat(1, 2), -1)
           || test_layer_param(ncnn::LayerType::Slice, base, 0, 1.f, -1)
           || test_layer_param(ncnn::LayerType::Slice, base, 0, param_int_array(1, INT_MIN), -1);
}

static int test_slice_load_param_axis()
{
    ncnn::ParamDict base;
    for (int i = -4; i <= 3; i++)
    {
        if (test_layer_param(ncnn::LayerType::Slice, base, 1, i, 0) != 0)
            return -1;
    }

    const int invalid[] = {-5, 4, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        if (test_layer_param(ncnn::LayerType::Slice, base, 1, invalid[i], -1) != 0)
            return -1;
    }

    return 0;
}
#endif // NCNN_VALIDATION

int main()
{
    SRAND(7767517);

    return 0
           || test_slice_0()
           || test_slice_1()
           || test_slice_2()
           || test_slice_3()
           || test_slice_4()
#if NCNN_VALIDATION
           || test_slice_load_param()
           || test_slice_load_param_axis()
#endif // NCNN_VALIDATION
           ;
}
