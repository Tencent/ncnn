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

#include "layer/crop.h"
#include "testutil.h"

static int test_crop(const ncnn::Mat& a, int woffset, int hoffset, int coffset, int outw, int outh, int outc, int woffset2, int hoffset2, int coffset2)
{
    ncnn::ParamDict pd;
    pd.set(0, woffset);  // woffset
    pd.set(1, hoffset);  // hoffset
    pd.set(2, coffset);  // coffset
    pd.set(3, outw);     // outw
    pd.set(4, outh);     // outh
    pd.set(5, outc);     // outc
    pd.set(6, woffset2); // woffset2
    pd.set(7, hoffset2); // hoffset2
    pd.set(8, coffset2); // coffset2

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Crop>("Crop", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_crop failed a.dims=%d a=(%d %d %d) woffset=%d hoffset=%d coffset=%d outw=%d outh=%d outc=%d woffset2=%d hoffset2=%d coffset2=%d\n", a.dims, a.w, a.h, a.c, woffset, hoffset, coffset, outw, outh, outc, woffset2, hoffset2, coffset2);
    }

    return ret;
}

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

static int test_crop(const ncnn::Mat& a, const ncnn::Mat& starts, const ncnn::Mat& ends, const ncnn::Mat& axes)
{
    ncnn::ParamDict pd;
    pd.set(9, starts); // starts
    pd.set(10, ends);  // ends
    pd.set(11, axes);  // axes

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Crop>("Crop", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_crop failed a.dims=%d a=(%d %d %d)", a.dims, a.w, a.h, a.c);
        fprintf(stderr, " starts=");
        print_int_array(starts);
        fprintf(stderr, " ends=");
        print_int_array(ends);
        fprintf(stderr, " axes=");
        print_int_array(axes);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_crop_0()
{
    ncnn::Mat a = RandomMat(13, 11, 16);

    return 0
           || test_crop(a, 0, 0, 0, -233, -233, -233, 0, 0, 0)
           || test_crop(a, 0, 3, 0, 10, -233, -233, 0, 0, 0)
           || test_crop(a, 5, 0, 0, -233, 6, -233, 0, 0, 0)
           || test_crop(a, 5, 3, 0, -233, -233, 12, 0, 0, 0)
           || test_crop(a, 0, 3, 4, 7, -233, 8, 0, 0, 0)
           || test_crop(a, 5, 0, 5, 4, -233, 4, 0, 0, 0)
           || test_crop(a, 5, 3, 6, 2, 7, 9, 0, 0, 0)
           || test_crop(a, 0, 3, 4, -233, 4, 5, 0, 0, 0)
           || test_crop(a, 5, 0, 5, 6, 6, 8, 0, 0, 0)
           || test_crop(a, 5, 3, 6, 4, 4, 4, 0, 0, 0);
}

static int test_crop_1()
{
    ncnn::Mat a = RandomMat(13, 11, 16);

    return 0
           || test_crop(a, 0, 0, 0, -233, -233, -233, 3, 4, 6)
           || test_crop(a, 0, 3, 0, 10, -233, -233, 0, 3, 4)
           || test_crop(a, 5, 0, 0, -233, 6, -233, 5, 0, 2)
           || test_crop(a, 5, 3, 0, -233, -233, 12, 2, 1, 1)
           || test_crop(a, 0, 3, 4, 7, -233, 8, 3, 4, 4)
           || test_crop(a, 5, 0, 5, 4, -233, 4, 0, 3, 0)
           || test_crop(a, 5, 3, 6, 2, 7, 9, 1, 2, 3)
           || test_crop(a, 0, 3, 4, -233, 4, 5, 3, 2, 1);
}

static int test_crop_2()
{
    ncnn::Mat a = RandomMat(13, 11, 17);

    return 0
           || test_crop(a, 0, 0, 0, -233, -233, -233, 3, 4, 6)
           || test_crop(a, 0, 3, 0, 10, -233, -233, 0, 3, 4)
           || test_crop(a, 5, 0, 0, -233, 6, -233, 5, 0, 2)
           || test_crop(a, 5, 3, 0, -233, -233, 12, 2, 1, 1)
           || test_crop(a, 0, 3, 4, 7, -233, 8, 3, 4, 4)
           || test_crop(a, 5, 0, 5, 4, -233, 4, 0, 3, 0)
           || test_crop(a, 5, 3, 6, 2, 7, 9, 1, 2, 3)
           || test_crop(a, 0, 3, 4, -233, 4, 5, 3, 2, 1);
}

static int test_crop_3()
{
    ncnn::Mat a = RandomMat(13, 11, 16);

    return 0
           || test_crop(a, IntArrayMat(0, 0, 0), IntArrayMat(100, 100, 100), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(4), IntArrayMat(8), IntArrayMat(0))
           || test_crop(a, IntArrayMat(2), IntArrayMat(7), IntArrayMat(1))
           || test_crop(a, IntArrayMat(3), IntArrayMat(5), IntArrayMat(2))
           || test_crop(a, IntArrayMat(2, 1), IntArrayMat(4, -2), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(1, 4), IntArrayMat(-5, 7), IntArrayMat(1, -3))
           || test_crop(a, IntArrayMat(2, 1), IntArrayMat(4, -2), IntArrayMat(-1, -2))
           || test_crop(a, IntArrayMat(1, 2, 3), IntArrayMat(-3, -2, -1), IntArrayMat(-3, -2, -1));
}

static int test_crop_4()
{
    ncnn::Mat a = RandomMat(13, 11, 17);

    return 0
           || test_crop(a, IntArrayMat(0, 0, 0), IntArrayMat(100, 100, 100), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(4), IntArrayMat(8), IntArrayMat(0))
           || test_crop(a, IntArrayMat(2), IntArrayMat(7), IntArrayMat(1))
           || test_crop(a, IntArrayMat(3), IntArrayMat(5), IntArrayMat(2))
           || test_crop(a, IntArrayMat(2, 1), IntArrayMat(4, -2), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(1, 4), IntArrayMat(-5, 7), IntArrayMat(1, -3))
           || test_crop(a, IntArrayMat(2, 1), IntArrayMat(4, -2), IntArrayMat(-1, -2))
           || test_crop(a, IntArrayMat(1, 2, 3), IntArrayMat(-3, -2, -1), IntArrayMat(-3, -2, -1));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_crop_0()
           || test_crop_1()
           || test_crop_2()
           || test_crop_3()
           || test_crop_4();
}
