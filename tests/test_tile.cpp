// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

static int test_tile(const ncnn::Mat& a, int axis, int tiles)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);
    pd.set(1, tiles);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Tile", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_tile failed a.dims=%d a=(%d %d %d %d) axis=%d tiles=%d\n", a.dims, a.w, a.h, a.d, a.c, axis, tiles);
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

static int test_tile(const ncnn::Mat& a, const std::vector<int>& repeats_array)
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

    int ret = test_layer("Tile", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_tile failed a.dims=%d a=(%d %d %d %d) repeats=", a.dims, a.w, a.h, a.d, a.c);
        print_int_array(repeats_array);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_tile_0()
{
    ncnn::Mat a = RandomMat(5, 6, 7, 24);
    ncnn::Mat b = RandomMat(7, 8, 9, 12);
    ncnn::Mat c = RandomMat(3, 4, 5, 13);

    return 0
           || test_tile(a, 0, 3)
           || test_tile(a, 1, 4)
           || test_tile(a, 2, 5)
           || test_tile(a, 3, 2)
           || test_tile(b, 0, 3)
           || test_tile(b, 1, 4)
           || test_tile(b, 2, 1)
           || test_tile(b, 3, 2)
           || test_tile(c, 0, 3)
           || test_tile(c, 1, 4)
           || test_tile(c, 2, 5)
           || test_tile(c, 3, 2)

           || test_tile(a, IntArray(3))
           || test_tile(a, IntArray(2, 4))
           || test_tile(a, IntArray(2, 2, 5))
           || test_tile(a, IntArray(3, 1, 3, 2))
           || test_tile(b, IntArray(3, 1))
           || test_tile(b, IntArray(4, 1, 4))
           || test_tile(b, IntArray(2, 2, 2, 1))
           || test_tile(b, IntArray(3, 2, 1))
           || test_tile(c, IntArray(3))
           || test_tile(c, IntArray(1, 1, 4))
           || test_tile(c, IntArray(2, 2, 5))
           || test_tile(c, IntArray(3, 2, 1, 9));
}

static int test_tile_1()
{
    ncnn::Mat a = RandomMat(5, 7, 24);
    ncnn::Mat b = RandomMat(7, 9, 12);
    ncnn::Mat c = RandomMat(3, 5, 13);

    return 0
           || test_tile(a, 0, 5)
           || test_tile(a, 1, 4)
           || test_tile(a, 2, 4)
           || test_tile(b, 0, 3)
           || test_tile(b, 1, 3)
           || test_tile(b, 2, 3)
           || test_tile(c, 0, 1)
           || test_tile(c, 1, 2)
           || test_tile(c, 2, 2)

           || test_tile(a, IntArray(5))
           || test_tile(a, IntArray(1, 4))
           || test_tile(a, IntArray(2, 1, 4))
           || test_tile(a, IntArray(1, 2, 1, 4))
           || test_tile(b, IntArray(3))
           || test_tile(b, IntArray(1, 3, 3))
           || test_tile(b, IntArray(2, 3))
           || test_tile(b, IntArray(2, 3, 3, 3))
           || test_tile(c, IntArray(1))
           || test_tile(c, IntArray(2, 1))
           || test_tile(c, IntArray(2, 2, 2))
           || test_tile(c, IntArray(2, 1, 2, 1));
}

static int test_tile_2()
{
    ncnn::Mat a = RandomMat(15, 24);
    ncnn::Mat b = RandomMat(17, 12);
    ncnn::Mat c = RandomMat(19, 13);

    return 0
           || test_tile(a, 0, 2)
           || test_tile(a, 1, 1)
           || test_tile(b, 0, 3)
           || test_tile(b, 1, 4)
           || test_tile(c, 0, 5)
           || test_tile(c, 1, 6)

           || test_tile(a, IntArray(2))
           || test_tile(a, IntArray(1, 1))
           || test_tile(a, IntArray(4, 1, 1))
           || test_tile(a, IntArray(2, 4, 4, 1))
           || test_tile(b, IntArray(3))
           || test_tile(b, IntArray(2, 4))
           || test_tile(b, IntArray(2, 4, 3, 1))
           || test_tile(b, IntArray(1, 2, 1, 4))
           || test_tile(c, IntArray(5))
           || test_tile(c, IntArray(6, 1))
           || test_tile(c, IntArray(6, 1, 6))
           || test_tile(c, IntArray(3, 2, 1, 1));
}

static int test_tile_3()
{
    ncnn::Mat a = RandomMat(128);
    ncnn::Mat b = RandomMat(124);
    ncnn::Mat c = RandomMat(127);

    return 0
           || test_tile(a, 0, 1)
           || test_tile(a, 0, 2)
           || test_tile(b, 0, 3)
           || test_tile(c, 0, 4)

           || test_tile(a, IntArray(10))
           || test_tile(a, IntArray(10, 1))
           || test_tile(a, IntArray(5, 2, 1))
           || test_tile(a, IntArray(2, 2, 2, 3))
           || test_tile(b, IntArray(2))
           || test_tile(b, IntArray(2, 2))
           || test_tile(b, IntArray(2, 2, 1))
           || test_tile(b, IntArray(4, 1, 2, 2))
           || test_tile(c, IntArray(3))
           || test_tile(c, IntArray(4, 3))
           || test_tile(c, IntArray(1))
           || test_tile(c, IntArray(1, 1))
           || test_tile(c, IntArray(1, 1, 1))
           || test_tile(c, IntArray(1, 3, 2, 2));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_tile_0()
           || test_tile_1()
           || test_tile_2()
           || test_tile_3();
}
