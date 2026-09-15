// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"
#include "datareader.h"

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

static int test_flip_load_param_case(const ncnn::ParamDict& pd, bool valid)
{
    ncnn::Layer* layer = ncnn::create_layer_naive(ncnn::LayerType::Flip);
    if (!layer)
        return -1;

    int ret = layer->load_param(pd);
    delete layer;

    if ((ret == 0) != valid)
    {
        fprintf(stderr, "Flip load_param returned %d, expected %s\n", ret, valid ? "success" : "failure");
        return -1;
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

static int test_flip_load_param()
{
    ncnn::ParamDict base;
    base.set(0, param_int_array(1, 0));
    if (test_flip_load_param_case(base, true) != 0)
        return -1;

    ncnn::ParamDict pd = base;
    pd.set(0, ncnn::Mat(1, (size_t)1u));
    if (test_flip_load_param_case(pd, false) != 0)
        return -1;

    pd.set(0, ncnn::Mat(1, 2));
    if (test_flip_load_param_case(pd, false) != 0)
        return -1;

    pd.set(0, 1.f);
    if (test_flip_load_param_case(pd, false) != 0)
        return -1;

    pd.set(0, param_int_array(5, 1));
    if (test_flip_load_param_case(pd, false) != 0)
        return -1;

    pd.set(0, param_int_array(1, INT_MIN));
    if (test_flip_load_param_case(pd, false) != 0)
        return -1;
    class FlipParamDict : public ncnn::ParamDict
    {
    public:
        using ncnn::ParamDict::load_param;
        using ncnn::ParamDict::load_param_bin;
    };

    FlipParamDict typed;
#if NCNN_STRING
    const unsigned char* text = (const unsigned char*)"-23300=1,0.0";
    ncnn::DataReaderFromMemory text_reader(text);
    if (typed.load_param(text_reader) != 0)
        return -1;
    if (test_flip_load_param_case(typed, false) != 0)
        return -1;
#endif
    const int binary[] = {-23300, 1, 0, -233};
    const unsigned char* data = (const unsigned char*)binary;
    ncnn::DataReaderFromMemory binary_reader(data);
    if (typed.load_param_bin(binary_reader) != 0)
        return -1;
    if (test_flip_load_param_case(typed, true) != 0)
        return -1;

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_flip_0()
           || test_flip_1()
           || test_flip_2()
           || test_flip_3()
           || test_flip_load_param();
}
