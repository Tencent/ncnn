// Copyright 2025 Tencent
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

static int test_flip(const ncnn::Mat& a, const std::vector<int>& axes_array)
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
    pd.set(0, axes);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Flip", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_flip failed a.dims=%d a=(%d %d %d %d)", a.dims, a.w, a.h, a.d, a.c);
        fprintf(stderr, " axes=");
        print_int_array(axes_array);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_flip_nd(const ncnn::Mat& a)
{
    int ret1 = test_flip(a, IntArray(0));

    if (a.dims == 1 || ret1 != 0)
        return ret1;

    int ret2 = 0
               || test_flip(a, IntArray(0))
               || test_flip(a, IntArray(1))
               || test_flip(a, IntArray(0, 1));

    if (a.dims == 2 || ret2 != 0)
        return ret2;

    int ret3 = 0
               || test_flip(a, IntArray(0))
               || test_flip(a, IntArray(1))
               || test_flip(a, IntArray(2))
               || test_flip(a, IntArray(0, 1))
               || test_flip(a, IntArray(0, 2))
               || test_flip(a, IntArray(1, 2))
               || test_flip(a, IntArray(0, 1, 2));

    if (a.dims == 3 || ret3 != 0)
        return ret3;

    int ret4 = 0
               || test_flip(a, IntArray(0))
               || test_flip(a, IntArray(1))
               || test_flip(a, IntArray(2))
               || test_flip(a, IntArray(3))
               || test_flip(a, IntArray(0, 1))
               || test_flip(a, IntArray(0, 2))
               || test_flip(a, IntArray(0, 3))
               || test_flip(a, IntArray(1, 2))
               || test_flip(a, IntArray(1, 3))
               || test_flip(a, IntArray(2, 3))
               || test_flip(a, IntArray(0, 1, 2))
               || test_flip(a, IntArray(0, 1, 3))
               || test_flip(a, IntArray(0, 2, 3))
               || test_flip(a, IntArray(1, 2, 3))
               || test_flip(a, IntArray(0, 1, 2, 3));

    return ret4;
}

static int test_flip_0()
{
    ncnn::Mat a = RandomMat(5, 6, 7, 24);
    ncnn::Mat b = RandomMat(7, 8, 9, 12);
    ncnn::Mat c = RandomMat(3, 4, 5, 13);

    return 0
           || test_flip_nd(a)
           || test_flip_nd(b)
           || test_flip_nd(c);
}

static int test_flip_1()
{
    ncnn::Mat a = RandomMat(5, 7, 24);
    ncnn::Mat b = RandomMat(7, 9, 12);
    ncnn::Mat c = RandomMat(3, 5, 13);

    return 0
           || test_flip_nd(a)
           || test_flip_nd(b)
           || test_flip_nd(c);
}

static int test_flip_2()
{
    ncnn::Mat a = RandomMat(15, 24);
    ncnn::Mat b = RandomMat(17, 12);
    ncnn::Mat c = RandomMat(19, 15);

    return 0
           || test_flip_nd(a)
           || test_flip_nd(b)
           || test_flip_nd(c);
}

static int test_flip_3()
{
    ncnn::Mat a = RandomMat(128);
    ncnn::Mat b = RandomMat(124);
    ncnn::Mat c = RandomMat(127);

    return 0
           || test_flip_nd(a)
           || test_flip_nd(b)
           || test_flip_nd(c);
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
static int test_flip_load_param()
{
    ncnn::ParamDict base;
    base.set(0, param_int_array(1, 0));
    if (test_layer_param(ncnn::LayerType::Flip, base, 0)
            || test_layer_param(ncnn::LayerType::Flip, base, 0, ncnn::Mat(0), 0))
        return -1;

    ncnn::Mat missing_data(0);
    missing_data.w = 1;

    if (test_layer_param(ncnn::LayerType::Flip, base, 0, missing_data, -1) != 0)
        return -1;

    ncnn::Mat negative_length(0);
    negative_length.w = -1;

    return 0
           || test_layer_param(ncnn::LayerType::Flip, base, 0, negative_length, -1)
           || test_layer_param(ncnn::LayerType::Flip, base, 0, ncnn::Mat(0, (size_t)1u), -1)
           || test_layer_param(ncnn::LayerType::Flip, base, 0, ncnn::Mat(1, (size_t)1u), -1)
           || test_layer_param(ncnn::LayerType::Flip, base, 0, ncnn::Mat(1, 2), -1)
           || test_layer_param(ncnn::LayerType::Flip, base, 0, 1.f, -1)
           || test_layer_param(ncnn::LayerType::Flip, base, 0, param_int_array(5, 1), -1)
           || test_layer_param(ncnn::LayerType::Flip, base, 0, param_int_array(1, INT_MIN), -1);
}

static int test_flip_load_param_serialized()
{
    TestParamDict typed;
#if NCNN_STRING
    if (typed.load_param("-23300=1,0.0") != 0)
        return -1;

    if (test_layer_param(ncnn::LayerType::Flip, typed, -1) != 0)
        return -1;

    if (typed.load_param("-23300=0") != 0)
        return -1;

    if (test_layer_param(ncnn::LayerType::Flip, typed, 0) != 0)
        return -1;
#endif
    // binary parameters use little-endian byte order
    const unsigned char binary[] = {
        0xfc, 0xa4, 0xff, 0xff,
        0x01, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x17, 0xff, 0xff, 0xff
    };
    if (typed.load_param_bin(binary) != 0)
        return -1;

    if (test_layer_param(ncnn::LayerType::Flip, typed, 0) != 0)
        return -1;

    const unsigned char empty_binary[] = {
        0xfc, 0xa4, 0xff, 0xff,
        0x00, 0x00, 0x00, 0x00,
        0x17, 0xff, 0xff, 0xff
    };
    if (typed.load_param_bin(empty_binary) != 0)
        return -1;

    return test_layer_param(ncnn::LayerType::Flip, typed, 0);
}
#endif // NCNN_VALIDATION

int main()
{
    SRAND(7767517);

    return 0
           || test_flip_0()
           || test_flip_1()
           || test_flip_2()
           || test_flip_3()
#if NCNN_VALIDATION
           || test_flip_load_param()
           || test_flip_load_param_serialized()
#endif // NCNN_VALIDATION
           ;
}
