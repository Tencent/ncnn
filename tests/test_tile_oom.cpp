// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_tile_oom(const ncnn::Mat& a, int axis, int tiles)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);
    pd.set(1, tiles);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer_oom("Tile", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_tile_oom failed a.dims=%d a=(%d %d %d %d) axis=%d tiles=%d\n", a.dims, a.w, a.h, a.d, a.c, axis, tiles);
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

static int test_tile_oom(const ncnn::Mat& a, const std::vector<int>& repeats_array)
{
    ncnn::Mat repeats(repeats_array.size());
    {
        int* p = repeats;
        for (size_t i = 0; i < repeats_array.size(); i++)
        {
            p[i] = repeats_array[i];
        }
    }

    ncnn::ParamDict pd;
    pd.set(2, repeats);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer_oom("Tile", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_tile_oom failed a.dims=%d a=(%d %d %d %d) repeats=", a.dims, a.w, a.h, a.d, a.c);
        print_int_array(repeats_array);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_tile_0()
{
    ncnn::Mat a = RandomMat(5, 6, 7, 24);
    ncnn::Mat b = RandomMat(5, 7, 24);
    ncnn::Mat c = RandomMat(15, 24);
    ncnn::Mat d = RandomMat(128);

    return 0
           || test_tile_oom(a, 0, 3)
           || test_tile_oom(a, 1, 4)
           || test_tile_oom(a, 2, 5)
           || test_tile_oom(a, 3, 2)
           || test_tile_oom(b, 0, 5)
           || test_tile_oom(b, 1, 4)
           || test_tile_oom(b, 2, 4)
           || test_tile_oom(c, 0, 2)
           || test_tile_oom(c, 1, 1)
           || test_tile_oom(d, 0, 2)

           || test_tile_oom(a, IntArray(3))
           || test_tile_oom(a, IntArray(2, 4))
           || test_tile_oom(a, IntArray(2, 2, 5))
           || test_tile_oom(a, IntArray(3, 1, 3, 2))
           || test_tile_oom(b, IntArray(5))
           || test_tile_oom(b, IntArray(1, 4))
           || test_tile_oom(b, IntArray(2, 1, 4))
           || test_tile_oom(b, IntArray(2, 4, 4, 1))
           || test_tile_oom(c, IntArray(5))
           || test_tile_oom(c, IntArray(6, 1))
           || test_tile_oom(c, IntArray(6, 1, 6))
           || test_tile_oom(c, IntArray(3, 2, 1, 1))
           || test_tile_oom(d, IntArray(10))
           || test_tile_oom(d, IntArray(10, 1))
           || test_tile_oom(d, IntArray(5, 2, 1))
           || test_tile_oom(d, IntArray(2, 2, 2, 3));
}

int main()
{
    SRAND(7767517);

    return test_tile_0();
}
