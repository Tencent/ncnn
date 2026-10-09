// Copyright 2021 Tencent
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

static int test_expanddims_3()
{
    ncnn::Mat a = RandomMat(3, 4, 5);
    ncnn::Mat b = RandomMat(3, 4);

    return 0
           || test_expanddims_axes(a, IntArray(-4))
           || test_expanddims_axes(a, IntArray(-3))
           || test_expanddims_axes(a, IntArray(-2))
           || test_expanddims_axes(a, IntArray(-1))
           || test_expanddims_axes(b, IntArray(0, 1))
           || test_expanddims_axes(b, IntArray(0, 3))
           || test_expanddims_axes(b, IntArray(2, 3));
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
static int test_expanddims_load_param()
{
    ncnn::ParamDict base;
    base.set(3, param_int_array(1, 0));
    if (test_layer_param(ncnn::LayerType::ExpandDims, base, 0)
            || test_layer_param(ncnn::LayerType::ExpandDims, base, 3, ncnn::Mat(0), 0))
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    return 0
           || test_layer_param(ncnn::LayerType::ExpandDims, base, 3, missing_data, -1)
           || test_layer_param(ncnn::LayerType::ExpandDims, base, 3, ncnn::Mat(1, (size_t)1u), -1)
           || test_layer_param(ncnn::LayerType::ExpandDims, base, 3, ncnn::Mat(1, 2), -1)
           || test_layer_param(ncnn::LayerType::ExpandDims, base, 3, 1.f, -1)
           || test_layer_param(ncnn::LayerType::ExpandDims, base, 3, param_int_array(5, 1), -1)
           || test_layer_param(ncnn::LayerType::ExpandDims, base, 3, param_int_array(1, -4), 0)
           || test_layer_param(ncnn::LayerType::ExpandDims, base, 3, param_int_array(1, 3), 0)
           || test_layer_param(ncnn::LayerType::ExpandDims, base, 3, param_int_array(1, -5), -1)
           || test_layer_param(ncnn::LayerType::ExpandDims, base, 3, param_int_array(1, 4), -1)
           || test_layer_param(ncnn::LayerType::ExpandDims, base, 3, param_int_array(1, INT_MIN), -1);
}
#endif // NCNN_VALIDATION

int main()
{
    SRAND(7767517);

    return 0
           || test_expanddims_0()
           || test_expanddims_1()
           || test_expanddims_2()
           || test_expanddims_3()
#if NCNN_VALIDATION
           || test_expanddims_load_param()
#endif // NCNN_VALIDATION
           ;
}
