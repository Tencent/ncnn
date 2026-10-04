// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

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

    int ret = test_layer("Reduction", pd, weights, a, 0.001);
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

    int ret = test_layer("Reduction", pd, weights, a, 0.001);
    if (ret != 0)
    {
        fprintf(stderr, "test_reduction failed a.dims=%d a=(%d %d %d %d) op_type=%d coeff=%f keepdims=%d", a.dims, a.w, a.h, a.d, a.c, op_type, coeff, keepdims);
        fprintf(stderr, " axes=");
        print_int_array(axes_array);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_reduction_axes(const ncnn::Mat& a)
{
    // axis geometry and output rank are independent of coefficient scaling
    int ret = test_reduction(a, 1.f, 0) || test_reduction(a, 2.f, 1);
    if (ret != 0)
        return ret;
    const int axes1[][4] = {{0, -1, -1, -1}};
    const int axes2[][4] = {{0, -1, -1, -1}, {1, -1, -1, -1}, {0, 1, -1, -1}};
    const int axes3[][4] = {
        {0, -1, -1, -1}, {1, -1, -1, -1}, {2, -1, -1, -1},
        {0, 1, -1, -1}, {0, 2, -1, -1}, {1, 2, -1, -1}, {0, 1, 2, -1}
    };
    const int axes4[][4] = {
        {0, -1, -1, -1}, {1, -1, -1, -1}, {2, -1, -1, -1}, {3, -1, -1, -1},
        {0, 1, -1, -1}, {0, 2, -1, -1}, {0, 3, -1, -1},
        {1, 2, -1, -1}, {1, 3, -1, -1}, {2, 3, -1, -1},
        {0, 1, 2, -1}, {0, 1, 3, -1}, {0, 2, 3, -1}, {1, 2, 3, -1}, {0, 1, 2, 3}
    };
    const int (*axis_sets)[4] = a.dims == 4 ? axes4 : a.dims == 3 ? axes3 : a.dims == 2 ? axes2 : axes1;
    const int counts[] = {1, 3, 7, 15};
    for (int i = 0; i < counts[a.dims - 1]; i++)
    {
        std::vector<int> axes;
        for (int j = 0; j < 4 && axis_sets[i][j] >= 0; j++)
            axes.push_back(axis_sets[i][j]);
        ret = test_reduction(a, 1.f, 0, axes) || test_reduction(a, 2.f, 1, axes);
        if (ret != 0)
            return ret;
    }
    return 0;
}

static int test_reduction_arithmetic()
{
    // each operation exercises every axis kernel and both output layouts
    return test_reduction_axes(RandomMat(127))
           || test_reduction_axes(RandomMat(19, 15))
           || test_reduction_axes(RandomMat(7, 9, 12))
           || test_reduction_axes(RandomMat(7, 8, 9, 12))
           || test_reduction(RandomMat(3, 1, 5, 1), 2.f, 0, IntArray(1, 3));
}

static int test_reduction_vector_boundaries()
{
    // lengths below, at and above SIMD widths retain vector and scalar loops
    const int sizes[] = {1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 128};
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
    {
        ncnn::Mat a = RandomMat(sizes[i]);
        ncnn::Mat b = RandomMat(sizes[i], 3);
        int ret = test_reduction(a, 1.f, 0)
                  || test_reduction(a, 2.f, 1)
                  || test_reduction(b, 1.f, 0)
                  || test_reduction(b, 2.f, 1, IntArray(1));
        if (ret != 0)
            return ret;
    }
    return 0;
}

static int test_reduction_geometry()
{
    // singleton ranks, packed transfers and a large workgroup reduction
    return test_reduction_axes(RandomMat(1, 3, 1, 5))
           || test_reduction_axes(RandomMat(3, 1, 5, 1))
           || test_reduction_axes(RandomMat(1, 1, 1, 1))
           || test_reduction_axes(RandomMat(1, 3, 5))
           || test_reduction_axes(RandomMat(3, 1, 5))
           || test_reduction_axes(RandomMat(1, 5))
           || test_reduction_axes(RandomMat(5, 1))
           || test_reduction_axes(RandomMat(3, 4, 5, 13))
           || test_reduction(RandomMat(5, 6, 7, 24), 1.f, 0)
           || test_reduction(RandomMat(17, 12), 1.f, 0)
           || test_reduction(RandomMat(5, 7, 24), 2.f, 1, IntArray(2));
}

static int test_reduction_numeric_boundaries()
{
    // zero L2 results exercise the subnormal square-root guard
    op_type = 8;
    ncnn::Mat a(33);
    ncnn::Mat b(3, 2, 4);
    a.fill(0.f);
    b.fill(0.f);
    int ret = test_reduction(a, 1.f, 0)
              || test_reduction(b, 2.f, 1, IntArray(2));
    if (ret != 0)
        return ret;

    // negative axes preserve each operation and both output layouts
    for (op_type = 0; op_type < OP_TYPE_MAX; op_type++)
    {
        ret = test_reduction(RandomMat(3, 2, 4, 5), 1.f, 0, IntArray(-1))
              || test_reduction(RandomMat(3, 2, 4, 5), 2.f, 1, IntArray(-4, -2));
        if (ret != 0)
            return ret;
    }
    return 0;
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
static int test_reduction_load_param()
{
    ncnn::ParamDict base;
    base.set(5, 1); // fixbug0
    base.set(3, param_int_array(1, 0));
    if (test_layer_param(ncnn::LayerType::Reduction, base, 0)
            || test_layer_param(ncnn::LayerType::Reduction, base, 3, ncnn::Mat(0), 0))
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    return 0
           || test_layer_param(ncnn::LayerType::Reduction, base, 3, missing_data, -1)
           || test_layer_param(ncnn::LayerType::Reduction, base, 3, ncnn::Mat(1, (size_t)1u), -1)
           || test_layer_param(ncnn::LayerType::Reduction, base, 3, ncnn::Mat(1, 2), -1)
           || test_layer_param(ncnn::LayerType::Reduction, base, 3, 1.f, -1)
           || test_layer_param(ncnn::LayerType::Reduction, base, 3, param_int_array(5, 1), -1)
           || test_layer_param(ncnn::LayerType::Reduction, base, 3, param_int_array(1, INT_MIN), -1);
}

static int test_reduction_load_param_type()
{
    ncnn::ParamDict base;
    if (test_layer_param(ncnn::LayerType::Reduction, base, 0) != 0)
        return -1;

    for (int i = 0; i <= 10; i++)
    {
        if (test_layer_param(ncnn::LayerType::Reduction, base, 0, i, 0) != 0)
            return -1;
    }

    const int invalid[] = {-1, 11, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        if (test_layer_param(ncnn::LayerType::Reduction, base, 0, invalid[i], -1) != 0)
            return -1;
    }

    return 0;
}

static int test_reduction_load_param_text()
{
#if NCNN_STRING
    const char* params[] = {"5=1 -23303=1,0", "5=1 -23303=1,0.0"};
    for (int i = 0; i < 2; i++)
    {
        TestParamDict pd;
        if (pd.load_param(params[i]) != 0 || pd.type(3) != 5 + i)
            return -1;

        if (test_layer_param(ncnn::LayerType::Reduction, pd, i == 0 ? 0 : -1) != 0)
            return -1;
    }
#endif
    return 0;
}
#endif // NCNN_VALIDATION

static int test_reduction_depth_padding_boundaries()
{
    // reducing depth isolates input channel padding from output depth indexing
    ncnn::Mat a(1, 1, 2, 1);
    a[0] = 1.f;
    a[1] = 2.f;

    // reducing height preserves two output depths with independent channel tail padding
    ncnn::Mat b(1, 4, 2, 1);
    for (int i = 0; i < 4; i++)
    {
        b[i] = 1.f;
        b[4 + i] = 2.f;
    }

    op_type = 0;
    return test_reduction(a, 1.f, 1, IntArray(1))
           || test_reduction(a, 1.f, 1, IntArray(-3))
           || test_reduction(b, 1.f, 1, IntArray(0, 2))
           || test_reduction(b, 1.f, 1, IntArray(-4, -2));
}

int main()
{
    SRAND(7767517);

    for (op_type = 0; op_type < OP_TYPE_MAX; op_type++)
    {
        int ret = 0
                  || test_reduction_arithmetic()
                  || test_reduction_vector_boundaries();

        if (ret != 0)
            return ret;
    }

    op_type = 0;
    return test_reduction_geometry()
           || test_reduction_numeric_boundaries()
#if NCNN_VALIDATION
           || test_reduction_load_param()
           || test_reduction_load_param_type()
           || test_reduction_load_param_text()
#endif // NCNN_VALIDATION
           || test_reduction_depth_padding_boundaries()
           ;
}
