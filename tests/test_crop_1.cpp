// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

static int test_crop(const ncnn::Mat& a, const std::vector<int>& starts_array, const std::vector<int>& ends_array, const std::vector<int>& axes_array)
{
    ncnn::Mat starts(starts_array.size());
    {
        int* p = starts;
        for (size_t i = 0; i < starts_array.size(); i++)
        {
            p[i] = starts_array[i];
        }
    }

    ncnn::Mat ends(ends_array.size());
    {
        int* p = ends;
        for (size_t i = 0; i < ends_array.size(); i++)
        {
            p[i] = ends_array[i];
        }
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
    pd.set(9, starts); // starts
    pd.set(10, ends);  // ends
    pd.set(11, axes);  // axes

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Crop", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_crop failed a.dims=%d a=(%d %d %d %d)", a.dims, a.w, a.h, a.d, a.c);
        fprintf(stderr, " starts=");
        print_int_array(starts_array);
        fprintf(stderr, " ends=");
        print_int_array(ends_array);
        fprintf(stderr, " axes=");
        print_int_array(axes_array);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_crop_1d(const ncnn::Mat& a)
{
    std::vector<int> params[][3] = {
        {IntArray(12), IntArray(-233), IntArray(0)},
        {IntArray(16), IntArray(-233), IntArray(0)},
        {IntArray(11), IntArray(11 + 16), IntArray(0)},
        {IntArray(12), IntArray(12 + 7), IntArray(-1)},
        {IntArray(16), IntArray(16 + 12), std::vector<int>()},
        {IntArray(11), IntArray(-7 + 1), IntArray(0)},
        {IntArray(12), IntArray(-12 + 1), IntArray(-1)},
        {IntArray(16), IntArray(-16 + 1), std::vector<int>()}
    };

    for (int i = 0; i < sizeof(params) / sizeof(params[0]); i++)
    {
        int ret = test_crop(a, params[i][0], params[i][1], params[i][2]);
        if (ret)
            return ret;
    }

    return 0;
}

static int test_crop_2d(const ncnn::Mat& a)
{
    std::vector<int> params[][3] = {
        {IntArray(12), IntArray(-233), IntArray(0)},
        {IntArray(8), IntArray(-233), IntArray(0)},
        {IntArray(4), IntArray(-233), IntArray(1)},
        {IntArray(5, 11), IntArray(-233, -233), IntArray(0, 1)},
        {IntArray(11), IntArray(11 + 16), IntArray(0)},
        {IntArray(12), IntArray(12 + 7), IntArray(0)},
        {IntArray(8), IntArray(8 + 12), IntArray(-2)},
        {IntArray(5), IntArray(8), IntArray(1)},
        {IntArray(6), IntArray(9), IntArray(1)},
        {IntArray(4), IntArray(12), IntArray(-1)},
        {IntArray(11, 5), IntArray(11 + 7, 11), IntArray(0, 1)},
        {IntArray(12, 6), IntArray(12 + 12, 12), IntArray(0, 1)},
        {IntArray(8, 4), IntArray(8 + 16, 10), IntArray(0, -1)},
        {IntArray(11), IntArray(-16 + 1), IntArray(0)},
        {IntArray(12), IntArray(-7 + 1), IntArray(0)},
        {IntArray(8), IntArray(-12 + 1), IntArray(-2)},
        {IntArray(5), IntArray(-5 + 1), IntArray(1)},
        {IntArray(6), IntArray(-6 + 1), IntArray(1)},
        {IntArray(4), IntArray(-4 + 1), IntArray(-1)},
        {IntArray(11, 5), IntArray(-12 + 1, -6 + 1), IntArray(0, 1)},
        {IntArray(12, 6), IntArray(-16 + 1, -5 + 1), IntArray(0, 1)},
        {IntArray(8, 4), IntArray(-7 + 1, -4 + 1), IntArray(-2, -1)}
    };

    for (int i = 0; i < sizeof(params) / sizeof(params[0]); i++)
    {
        int ret = test_crop(a, params[i][0], params[i][1], params[i][2]);
        if (ret)
            return ret;
    }

    return 0;
}

static int test_crop_3d(const ncnn::Mat& a)
{
    std::vector<int> params[][3] = {
        {IntArray(11), IntArray(-233), IntArray(0)},
        {IntArray(8), IntArray(-233), IntArray(0)},
        {IntArray(5), IntArray(-233), IntArray(1)},
        {IntArray(6), IntArray(-233), IntArray(2)},
        {IntArray(4), IntArray(-233), IntArray(-1)},
        {IntArray(12, 6), IntArray(-233, -233), IntArray(0, 1)},
        {IntArray(11, 5), IntArray(-233, -233), IntArray(0, -1)},
        {IntArray(8, 4), IntArray(-233, -233), IntArray(0, 2)},
        {IntArray(6, 6), IntArray(-233, -233), IntArray(1, -1)},
        {IntArray(11, 5, 5), IntArray(-233, -233, -233), IntArray(0, 1, 2)},
        {IntArray(8, 4, 4), IntArray(-233, -233, -233), IntArray(0, 1, -1)},
        {IntArray(11), IntArray(11 + 7), IntArray(0)},
        {IntArray(12), IntArray(12 + 12), IntArray(0)},
        {IntArray(8), IntArray(8 + 16), IntArray(0)},
        {IntArray(5), IntArray(13), IntArray(1)},
        {IntArray(6), IntArray(12), IntArray(1)},
        {IntArray(4), IntArray(11), IntArray(-2)},
        {IntArray(5), IntArray(12), IntArray(2)},
        {IntArray(6), IntArray(11), IntArray(2)},
        {IntArray(4), IntArray(13), IntArray(-1)},
        {IntArray(11, 5), IntArray(11 + 7, 11), IntArray(0, 1)},
        {IntArray(12, 6), IntArray(12 + 16, 12), IntArray(0, 1)},
        {IntArray(8, 4), IntArray(8 + 12, 13), IntArray(0, -2)},
        {IntArray(11, 5), IntArray(11 + 16, 13), IntArray(0, 2)},
        {IntArray(12, 6), IntArray(12 + 12, 11), IntArray(0, 2)},
        {IntArray(8, 4), IntArray(8 + 7, 12), IntArray(0, -1)},
        {IntArray(5, 4), IntArray(12, 12), IntArray(1, 2)},
        {IntArray(6, 3), IntArray(13, 13), IntArray(1, 2)},
        {IntArray(4, 2), IntArray(11, 11), IntArray(-2, -1)},
        {IntArray(11, 5, 2), IntArray(11 + 7, 11, 11), IntArray(0, 1, 2)},
        {IntArray(12, 6, 4), IntArray(12 + 16, 12, 12), IntArray(0, 1, 2)},
        {IntArray(8, 4, 3), IntArray(8 + 12, 13, 13), IntArray(-3, -2, -1)},
        {IntArray(11), IntArray(-7 + 1), IntArray(0)},
        {IntArray(12), IntArray(-12 + 1), IntArray(0)},
        {IntArray(8), IntArray(-16 + 1), IntArray(-3)},
        {IntArray(5), IntArray(-6 + 1), IntArray(1)},
        {IntArray(6), IntArray(-5 + 1), IntArray(1)},
        {IntArray(4), IntArray(-4 + 1), IntArray(-2)},
        {IntArray(5), IntArray(-5 + 1), IntArray(2)},
        {IntArray(6), IntArray(-4 + 1), IntArray(2)},
        {IntArray(4), IntArray(-6 + 1), IntArray(-1)},
        {IntArray(11, 5), IntArray(-7 + 1, -4 + 1), IntArray(0, 1)},
        {IntArray(12, 6), IntArray(-12 + 1, -6 + 1), IntArray(0, 1)},
        {IntArray(8, 4), IntArray(-16 + 1, -5 + 1), IntArray(-3, -2)},
        {IntArray(11, 5), IntArray(-12 + 1, -6 + 1), IntArray(0, 2)},
        {IntArray(12, 6), IntArray(-16 + 1, -5 + 1), IntArray(0, 2)},
        {IntArray(8, 4), IntArray(-7 + 1, -4 + 1), IntArray(-3, -1)},
        {IntArray(5, 2), IntArray(-5 + 1, -5 + 1), IntArray(1, 2)},
        {IntArray(6, 4), IntArray(-4 + 1, -4 + 1), IntArray(1, 2)},
        {IntArray(4, 3), IntArray(-6 + 1, -6 + 1), IntArray(-2, -1)},
        {IntArray(11, 5, 4), IntArray(-7 + 1, -5 + 1, -5 + 1), IntArray(0, 1, 2)},
        {IntArray(12, 6, 3), IntArray(-12 + 1, -6 + 1, -6 + 1), IntArray(0, 1, 2)},
        {IntArray(8, 4, 2), IntArray(-16 + 1, -4 + 1, -4 + 1), IntArray(-3, -2, -1)}
    };

    for (int i = 0; i < sizeof(params) / sizeof(params[0]); i++)
    {
        int ret = test_crop(a, params[i][0], params[i][1], params[i][2]);
        if (ret)
            return ret;
    }

    return 0;
}

static int test_crop_4d(const ncnn::Mat& a)
{
    std::vector<int> params[][3] = {
        {IntArray(11), IntArray(-233), IntArray(0)},
        {IntArray(8), IntArray(-233), IntArray(0)},
        {IntArray(6), IntArray(-233), IntArray(1)},
        {IntArray(5), IntArray(-233), IntArray(2)},
        {IntArray(4), IntArray(-233), IntArray(-2)},
        {IntArray(6), IntArray(-233), IntArray(3)},
        {IntArray(5), IntArray(-233), IntArray(-1)},
        {IntArray(8, 4), IntArray(-233, -233), IntArray(0, 1)},
        {IntArray(12, 6), IntArray(-233, -233), IntArray(0, 2)},
        {IntArray(11, 5), IntArray(-233, -233), IntArray(-4, -2)},
        {IntArray(4, 4), IntArray(-233, -233), IntArray(1, 2)},
        {IntArray(12, 6), IntArray(-233, -233), IntArray(0, 3)},
        {IntArray(5, 5), IntArray(-233, -233), IntArray(1, 3)},
        {IntArray(4, 4), IntArray(-233, -233), IntArray(2, 3)},
        {IntArray(12, 6, 6), IntArray(-233, -233, -233), IntArray(0, 1, 2)},
        {IntArray(11, 5, 5), IntArray(-233, -233, -233), IntArray(0, 1, 2)},
        {IntArray(8, 4, 4), IntArray(-233, -233, -233), IntArray(0, 1, 3)},
        {IntArray(12, 6, 6), IntArray(-233, -233, -233), IntArray(0, 2, 3)},
        {IntArray(11, 5, 5), IntArray(-233, -233, -233), IntArray(0, 2, 3)},
        {IntArray(4, 4, 4), IntArray(-233, -233, -233), IntArray(1, 2, 3)},
        {IntArray(6, 6, 6), IntArray(-233, -233, -233), IntArray(1, 2, 3)},
        {IntArray(11, 5, 5, 5), IntArray(-233, -233, -233, -233), IntArray(0, 1, 2, 3)},
        {IntArray(8, 4, 4, 4), IntArray(-233, -233, -233, -233), IntArray(0, 1, 2, 3)},
        {IntArray(12, 6, 6, 6), IntArray(-233, -233, -233, -233), IntArray(-4, -3, -2, -1)},
        {IntArray(11), IntArray(11 + 16), IntArray(0)},
        {IntArray(12), IntArray(12 + 7), IntArray(0)},
        {IntArray(8), IntArray(8 + 12), IntArray(-4)},
        {IntArray(5), IntArray(11), IntArray(1)},
        {IntArray(6), IntArray(13), IntArray(1)},
        {IntArray(4), IntArray(12), IntArray(-3)},
        {IntArray(3), IntArray(12), IntArray(2)},
        {IntArray(4), IntArray(13), IntArray(2)},
        {IntArray(5), IntArray(11), IntArray(-2)},
        {IntArray(1), IntArray(8), IntArray(3)},
        {IntArray(2), IntArray(7), IntArray(3)},
        {IntArray(3), IntArray(6), IntArray(-1)},
        {IntArray(11, 5), IntArray(11 + 7, 11), IntArray(0, 1)},
        {IntArray(12, 6), IntArray(12 + 12, 12), IntArray(0, 1)},
        {IntArray(8, 4), IntArray(8 + 16, 13), IntArray(-4, -3)},
        {IntArray(11, 4), IntArray(11 + 12, 13), IntArray(0, 2)},
        {IntArray(12, 3), IntArray(12 + 16, 11), IntArray(0, 2)},
        {IntArray(8, 2), IntArray(8 + 7, 12), IntArray(-4, -2)},
        {IntArray(11, 1), IntArray(11 + 16, 5), IntArray(0, 3)},
        {IntArray(12, 2), IntArray(12 + 7, 6), IntArray(0, 3)},
        {IntArray(8, 3), IntArray(8 + 12, 7), IntArray(-4, -1)},
        {IntArray(3, 3), IntArray(13, 4), IntArray(1, 2)},
        {IntArray(4, 2), IntArray(12, 3), IntArray(1, 2)},
        {IntArray(5, 1), IntArray(11, 2), IntArray(-3, -2)},
        {IntArray(5, 5), IntArray(11, 8), IntArray(1, 3)},
        {IntArray(4, 6), IntArray(12, 9), IntArray(1, 3)},
        {IntArray(3, 4), IntArray(13, 7), IntArray(-3, -1)},
        {IntArray(2, 3), IntArray(12, 9), IntArray(2, 3)},
        {IntArray(3, 2), IntArray(11, 7), IntArray(2, 3)},
        {IntArray(4, 1), IntArray(10, 8), IntArray(-2, -1)},
        {IntArray(11, 2, 2), IntArray(11 + 6, 9, 9), IntArray(0, 1, 2)},
        {IntArray(12, 3, 3), IntArray(12 + 1, 10, 10), IntArray(0, 1, 2)},
        {IntArray(8, 4, 4), IntArray(8 + 3, 11, 11), IntArray(-4, -3, -2)},
        {IntArray(11, 4, 4), IntArray(11 + 12, 12, 12), IntArray(0, 1, 3)},
        {IntArray(12, 5, 5), IntArray(12 + 8, 11, 11), IntArray(0, 1, 3)},
        {IntArray(8, 6, 6), IntArray(8 + 4, 13, 13), IntArray(-4, -3, -1)},
        {IntArray(11, 1, 4), IntArray(11 + 5, 12, 12), IntArray(0, 2, 3)},
        {IntArray(12, 3, 3), IntArray(12 + 3, 11, 11), IntArray(0, 2, 3)},
        {IntArray(8, 2, 5), IntArray(8 + 2, 10, 10), IntArray(-4, -2, -1)},
        {IntArray(1, 1, 1), IntArray(7, 7, 7), IntArray(1, 2, 3)},
        {IntArray(2, 2, 2), IntArray(8, 9, 10), IntArray(1, 2, 3)},
        {IntArray(3, 3, 3), IntArray(11, 12, 13), IntArray(-3, -2, -1)},
        {IntArray(11, 2, 3, 6), IntArray(11 + 11, 10, 12, 11), IntArray(0, 1, 2, 3)},
        {IntArray(12, 3, 4, 5), IntArray(12 + 12, 9, 11, 13), IntArray(0, 1, 2, 3)},
        {IntArray(8, 4, 5, 4), IntArray(8 + 8, 8, 10, 12), IntArray(-4, -3, -2, -1)},
        {IntArray(11), IntArray(-7 + 1), IntArray(0)},
        {IntArray(12), IntArray(-12 + 1), IntArray(0)},
        {IntArray(8), IntArray(-16 + 1), IntArray(-4)},
        {IntArray(5), IntArray(-6 + 1), IntArray(1)},
        {IntArray(6), IntArray(-5 + 1), IntArray(1)},
        {IntArray(4), IntArray(-4 + 1), IntArray(-3)},
        {IntArray(4), IntArray(-4 + 1), IntArray(2)},
        {IntArray(5), IntArray(-5 + 1), IntArray(2)},
        {IntArray(6), IntArray(-6 + 1), IntArray(-2)},
        {IntArray(1), IntArray(-5 + 1), IntArray(3)},
        {IntArray(2), IntArray(-4 + 1), IntArray(3)},
        {IntArray(3), IntArray(-3 + 1), IntArray(-1)},
        {IntArray(11, 3), IntArray(-7 + 1, -3 + 1), IntArray(0, 1)},
        {IntArray(12, 4), IntArray(-12 + 1, -4 + 1), IntArray(0, 1)},
        {IntArray(8, 5), IntArray(-16 + 1, -5 + 1), IntArray(-4, -3)},
        {IntArray(11, 1), IntArray(-12 + 1, -5 + 1), IntArray(0, 2)},
        {IntArray(12, 2), IntArray(-16 + 1, -4 + 1), IntArray(0, 2)},
        {IntArray(8, 3), IntArray(-7 + 1, -6 + 1), IntArray(-4, -2)},
        {IntArray(11, 3), IntArray(-12 + 1, -2 + 1), IntArray(0, 3)},
        {IntArray(12, 4), IntArray(-16 + 1, -3 + 1), IntArray(0, 3)},
        {IntArray(8, 5), IntArray(-7 + 1, -4 + 1), IntArray(-4, -1)},
        {IntArray(2, 3), IntArray(-4 + 1, -2 + 1), IntArray(1, 2)},
        {IntArray(3, 4), IntArray(-2 + 1, -3 + 1), IntArray(1, 2)},
        {IntArray(4, 5), IntArray(-3 + 1, -4 + 1), IntArray(-3, -2)},
        {IntArray(3, 2), IntArray(-2 + 1, -4 + 1), IntArray(1, 3)},
        {IntArray(4, 3), IntArray(-3 + 1, -2 + 1), IntArray(1, 3)},
        {IntArray(5, 4), IntArray(-4 + 1, -3 + 1), IntArray(-3, -1)},
        {IntArray(2, 3), IntArray(-4 + 1, -6 + 1), IntArray(2, 3)},
        {IntArray(1, 2), IntArray(-5 + 1, -5 + 1), IntArray(2, 3)},
        {IntArray(3, 1), IntArray(-6 + 1, -4 + 1), IntArray(-2, -1)},
        {IntArray(11, 3, 3), IntArray(-7 + 1, -3 + 1, -4 + 1), IntArray(0, 1, 2)},
        {IntArray(12, 4, 4), IntArray(-12 + 1, -4 + 1, -3 + 1), IntArray(0, 1, 2)},
        {IntArray(8, 5, 5), IntArray(-16 + 1, -5 + 1, -5 + 1), IntArray(-4, -3, -2)},
        {IntArray(11, 2, 2), IntArray(-7 + 1, -5 + 1, -4 + 1), IntArray(0, 1, 3)},
        {IntArray(12, 1, 1), IntArray(-12 + 1, -6 + 1, -5 + 1), IntArray(0, 1, 3)},
        {IntArray(8, 3, 3), IntArray(-16 + 1, -4 + 1, -6 + 1), IntArray(-4, -3, -1)},
        {IntArray(11, 2, 5), IntArray(-7 + 1, -2 + 1, -5 + 1), IntArray(0, 2, 3)},
        {IntArray(12, 3, 3), IntArray(-12 + 1, -3 + 1, -4 + 1), IntArray(0, 2, 3)},
        {IntArray(8, 4, 4), IntArray(-16 + 1, -4 + 1, -3 + 1), IntArray(-4, -2, -1)},
        {IntArray(1, 3, 3), IntArray(-3 + 1, -6 + 1, -4 + 1), IntArray(1, 2, 3)},
        {IntArray(2, 2, 2), IntArray(-4 + 1, -4 + 1, -5 + 1), IntArray(1, 2, 3)},
        {IntArray(3, 1, 1), IntArray(-5 + 1, -5 + 1, -6 + 1), IntArray(-3, -2, -1)},
        {IntArray(11, 3, 4, 4), IntArray(-7 + 1, -3 + 1, -2 + 1, -4 + 1), IntArray(0, 1, 2, 3)},
        {IntArray(12, 4, 5, 3), IntArray(-12 + 1, -4 + 1, -3 + 1, -5 + 1), IntArray(0, 1, 2, 3)},
        {IntArray(8, 5, 6, 2), IntArray(-16 + 1, -5 + 1, -4 + 1, -3 + 1), IntArray(-4, -3, -2, -1)}
    };

    for (int i = 0; i < sizeof(params) / sizeof(params[0]); i++)
    {
        int ret = test_crop(a, params[i][0], params[i][1], params[i][2]);
        if (ret)
            return ret;
    }

    return 0;
}

int main()
{
    SRAND(776757);

    return 0
           || test_crop_1d(RandomMat(112))
           || test_crop_1d(RandomMat(126))
           || test_crop_1d(RandomMat(127))
           || test_crop_2d(RandomMat(20, 48))
           || test_crop_2d(RandomMat(15, 36))
           || test_crop_2d(RandomMat(16, 33))
           || test_crop_3d(RandomMat(20, 20, 48))
           || test_crop_3d(RandomMat(15, 15, 36))
           || test_crop_3d(RandomMat(16, 16, 33))
           || test_crop_4d(RandomMat(20, 20, 20, 48))
           || test_crop_4d(RandomMat(15, 15, 15, 36))
           || test_crop_4d(RandomMat(16, 16, 16, 33));
}
