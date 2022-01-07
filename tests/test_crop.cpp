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

static int test_crop(const ncnn::Mat& a, int woffset, int hoffset, int doffset, int coffset, int outw, int outh, int outd, int outc, int woffset2, int hoffset2, int doffset2, int coffset2)
{
    ncnn::ParamDict pd;
    pd.set(0, woffset);   // woffset
    pd.set(1, hoffset);   // hoffset
    pd.set(13, doffset);  // doffset
    pd.set(2, coffset);   // coffset
    pd.set(3, outw);      // outw
    pd.set(4, outh);      // outh
    pd.set(14, outd);     // outd
    pd.set(5, outc);      // outc
    pd.set(6, woffset2);  // woffset2
    pd.set(7, hoffset2);  // hoffset2
    pd.set(15, doffset2); // doffset2
    pd.set(8, coffset2);  // coffset2

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Crop>("Crop", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_crop failed a.dims=%d a=(%d %d %d %d) woffset=%d hoffset=%d doffset=%d coffset=%d outw=%d outh=%d outd=%d outc=%d woffset2=%d hoffset2=%d doffset2=%d coffset2=%d\n", a.dims, a.w, a.h, a.d, a.c, woffset, hoffset, doffset, coffset, outw, outh, outd, outc, woffset2, hoffset2, doffset2, coffset2);
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

static ncnn::Mat IntArrayMat(int a0, int a1, int a2, int a3)
{
    ncnn::Mat m(4);
    int* p = m;
    p[0] = a0;
    p[1] = a1;
    p[2] = a2;
    p[3] = a3;
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
        fprintf(stderr, "test_crop failed a.dims=%d a=(%d %d %d %d)", a.dims, a.w, a.h, a.d, a.c);
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

static int test_crop(const ncnn::Mat& a, int woffset, int hoffset, int doffset, int coffset, const ncnn::Mat& ref)
{
    ncnn::ParamDict pd;
    pd.set(0, woffset);
    pd.set(1, hoffset);
    pd.set(13, doffset);
    pd.set(2, coffset);
    pd.set(3, 0);  // outw
    pd.set(4, 0);  // outh
    pd.set(14, 0); // outd
    pd.set(5, 0);  // outc
    pd.set(6, 0);  // woffset2
    pd.set(7, 0);  // hoffset2
    pd.set(15, 0); // doffset2
    pd.set(8, 0);  // coffset2

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> ab(2);
    ab[0] = a;
    ab[1] = ref;

    int ret = test_layer<ncnn::Crop>("Crop", pd, weights, ab);
    if (ret != 0)
    {
        fprintf(stderr, "test_crop failed a.dims=%d a=(%d %d %d %d) woffset=%d hoffset=%d doffset=%d coffset=%d ref.dims=%d ref=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c, woffset, hoffset, doffset, coffset, ref.dims, ref.w, ref.h, ref.d, ref.c);
    }

    return ret;
}

static int test_crop_0(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, -233, 0, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 11, 0, 0, 0, -233, 0, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 11, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 11, 0, 0, 0, -233, 0, 0, 0, 12, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 0, -233, 0, 0, 0, 16, 0, 0, 0)
           || test_crop(a, 16, 0, 0, 0, -233, 0, 0, 0, 7, 0, 0, 0);
}

static int test_crop_1(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, IntArrayMat(12), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(16), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(11), IntArrayMat(11 + 16), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(12 + 7), IntArrayMat(-1))
           || test_crop(a, IntArrayMat(16), IntArrayMat(16 + 12), ncnn::Mat())
           || test_crop(a, IntArrayMat(11), IntArrayMat(-7 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-12 + 1), IntArrayMat(-1))
           || test_crop(a, IntArrayMat(16), IntArrayMat(-16 + 1), ncnn::Mat());
}

static int test_crop_2(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, a)
           || test_crop(a, 0, 0, 0, 0, ncnn::Mat(27))

           || test_crop(a, 11, 0, 0, 0, ncnn::Mat(7))
           || test_crop(a, 12, 0, 0, 0, ncnn::Mat(12))
           || test_crop(a, 16, 0, 0, 0, ncnn::Mat(16));
}

static int test_crop_3(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 5, 0, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 0, 11, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 6, 12, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 4, 8, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 0, 5, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, 6, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 0, 4, -233, 0, 0, 0, 0, 0, 0)

           || test_crop(a, 0, 11, 0, 0, -233, 12, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 0, 0, -233, 16, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 0, 8, 0, 0, -233, 7, 0, 0, 0, 0, 0, 0)

           || test_crop(a, 5, 11, 0, 0, 4, 16, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 6, 12, 0, 0, 5, 7, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 4, 8, 0, 0, 6, 12, 0, 0, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 0, -233, -233, 0, 0, 5, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, -233, -233, 0, 0, 6, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 0, -233, -233, 0, 0, 4, 0, 0, 0)

           || test_crop(a, 0, 11, 0, 0, -233, -233, 0, 0, 0, 12, 0, 0)
           || test_crop(a, 0, 12, 0, 0, -233, -233, 0, 0, 0, 16, 0, 0)
           || test_crop(a, 0, 8, 0, 0, -233, -233, 0, 0, 0, 7, 0, 0)

           || test_crop(a, 5, 11, 0, 0, -233, -233, 0, 0, 4, 16, 0, 0)
           || test_crop(a, 6, 12, 0, 0, -233, -233, 0, 0, 5, 7, 0, 0)
           || test_crop(a, 4, 8, 0, 0, -233, -233, 0, 0, 6, 12, 0, 0);
}

static int test_crop_4(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, IntArrayMat(12), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(4), IntArrayMat(-233), IntArrayMat(1))
           || test_crop(a, IntArrayMat(5, 11), IntArrayMat(-233, -233), IntArrayMat(0, 1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(11 + 16), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(12 + 7), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(8 + 12), IntArrayMat(-2))

           || test_crop(a, IntArrayMat(5), IntArrayMat(8), IntArrayMat(1))
           || test_crop(a, IntArrayMat(6), IntArrayMat(9), IntArrayMat(1))
           || test_crop(a, IntArrayMat(4), IntArrayMat(12), IntArrayMat(-1))

           || test_crop(a, IntArrayMat(11, 5), IntArrayMat(11 + 7, 11), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 6), IntArrayMat(12 + 12, 12), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(8, 4), IntArrayMat(8 + 16, 10), IntArrayMat(0, -1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-16 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-7 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-12 + 1), IntArrayMat(-2))

           || test_crop(a, IntArrayMat(5), IntArrayMat(-5 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(6), IntArrayMat(-6 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(4), IntArrayMat(-4 + 1), IntArrayMat(-1))

           || test_crop(a, IntArrayMat(11, 5), IntArrayMat(-12 + 1, -6 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 6), IntArrayMat(-16 + 1, -5 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(8, 4), IntArrayMat(-7 + 1, -4 + 1), IntArrayMat(-2, -1));
}

static int test_crop_5(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, a)

           || test_crop(a, 0, 12, 0, 0, ncnn::Mat(8, 7))
           || test_crop(a, 5, 0, 0, 0, ncnn::Mat(7, 27))

           || test_crop(a, 5, 11, 0, 0, ncnn::Mat(5, 12))
           || test_crop(a, 6, 12, 0, 0, ncnn::Mat(4, 16))
           || test_crop(a, 4, 8, 0, 0, ncnn::Mat(6, 7));
}

static int test_crop_6(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 5, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 4, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 12, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 5, 5, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 4, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 6, 0, 12, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 5, 0, 0, 11, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 8, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 6, 0, 12, -233, -233, 0, -233, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 0, 6, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, 4, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 0, 5, -233, 0, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 5, 0, 0, -233, 4, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 6, 0, 0, -233, 5, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 4, 0, 0, -233, 6, 0, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 0, 0, 11, -233, -233, 0, 7, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 12, -233, -233, 0, 12, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 8, -233, -233, 0, 16, 0, 0, 0, 0)

           || test_crop(a, 5, 5, 0, 0, 4, 4, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 6, 0, 0, 6, 6, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 4, 0, 0, 5, 5, 0, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 5, 0, 11, -233, 6, 0, 12, 0, 0, 0, 0)
           || test_crop(a, 0, 6, 0, 12, -233, 4, 0, 16, 0, 0, 0, 0)
           || test_crop(a, 0, 4, 0, 8, -233, 5, 0, 7, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 11, 4, -233, 0, 16, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 12, 5, -233, 0, 7, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 8, 4, -233, 0, 12, 0, 0, 0, 0)

           || test_crop(a, 5, 3, 0, 11, 6, 5, 0, 12, 0, 0, 0, 0)
           || test_crop(a, 6, 4, 0, 12, 4, 4, 0, 16, 0, 0, 0, 0)
           || test_crop(a, 4, 5, 0, 8, 5, 3, 0, 7, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 0, -233, -233, 0, -233, 4, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, -233, -233, 0, -233, 5, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 0, -233, -233, 0, -233, 6, 0, 0, 0)

           || test_crop(a, 0, 5, 0, 0, -233, -233, 0, -233, 0, 5, 0, 0)
           || test_crop(a, 0, 6, 0, 0, -233, -233, 0, -233, 0, 6, 0, 0)
           || test_crop(a, 0, 4, 0, 0, -233, -233, 0, -233, 0, 4, 0, 0)

           || test_crop(a, 0, 0, 0, 11, -233, -233, 0, -233, 0, 0, 0, 12)
           || test_crop(a, 0, 0, 0, 12, -233, -233, 0, -233, 0, 0, 0, 16)
           || test_crop(a, 0, 0, 0, 8, -233, -233, 0, -233, 0, 0, 0, 7)

           || test_crop(a, 5, 4, 0, 0, -233, -233, 0, -233, 4, 2, 0, 0)
           || test_crop(a, 6, 3, 0, 0, -233, -233, 0, -233, 5, 3, 0, 0)
           || test_crop(a, 4, 2, 0, 0, -233, -233, 0, -233, 6, 4, 0, 0)

           || test_crop(a, 0, 5, 0, 11, -233, -233, 0, -233, 0, 5, 0, 7)
           || test_crop(a, 0, 6, 0, 12, -233, -233, 0, -233, 0, 6, 0, 12)
           || test_crop(a, 0, 4, 0, 8, -233, -233, 0, -233, 0, 4, 0, 16)

           || test_crop(a, 5, 0, 0, 11, -233, -233, 0, -233, 6, 0, 0, 12)
           || test_crop(a, 6, 0, 0, 12, -233, -233, 0, -233, 4, 0, 0, 16)
           || test_crop(a, 4, 0, 0, 8, -233, -233, 0, -233, 5, 0, 0, 7)

           || test_crop(a, 5, 2, 0, 11, -233, -233, 0, -233, 4, 3, 0, 16)
           || test_crop(a, 6, 3, 0, 12, -233, -233, 0, -233, 5, 4, 0, 7)
           || test_crop(a, 4, 4, 0, 8, -233, -233, 0, -233, 6, 2, 0, 12);
}

static int test_crop_7(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, IntArrayMat(11), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(5), IntArrayMat(-233), IntArrayMat(1))
           || test_crop(a, IntArrayMat(6), IntArrayMat(-233), IntArrayMat(2))
           || test_crop(a, IntArrayMat(4), IntArrayMat(-233), IntArrayMat(-1))
           || test_crop(a, IntArrayMat(12, 6), IntArrayMat(-233, -233), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(11, 5), IntArrayMat(-233, -233), IntArrayMat(0, -1))
           || test_crop(a, IntArrayMat(8, 4), IntArrayMat(-233, -233), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(6, 6), IntArrayMat(-233, -233), IntArrayMat(1, -1))
           || test_crop(a, IntArrayMat(11, 5, 5), IntArrayMat(-233, -233, -233), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(8, 4, 4), IntArrayMat(-233, -233, -233), IntArrayMat(0, 1, -1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(11 + 7), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(12 + 12), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(8 + 16), IntArrayMat(0))

           || test_crop(a, IntArrayMat(5), IntArrayMat(13), IntArrayMat(1))
           || test_crop(a, IntArrayMat(6), IntArrayMat(12), IntArrayMat(1))
           || test_crop(a, IntArrayMat(4), IntArrayMat(11), IntArrayMat(-2))

           || test_crop(a, IntArrayMat(5), IntArrayMat(12), IntArrayMat(2))
           || test_crop(a, IntArrayMat(6), IntArrayMat(11), IntArrayMat(2))
           || test_crop(a, IntArrayMat(4), IntArrayMat(13), IntArrayMat(-1))

           || test_crop(a, IntArrayMat(11, 5), IntArrayMat(11 + 7, 11), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 6), IntArrayMat(12 + 16, 12), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(8, 4), IntArrayMat(8 + 12, 13), IntArrayMat(0, -2))

           || test_crop(a, IntArrayMat(11, 5), IntArrayMat(11 + 16, 13), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(12, 6), IntArrayMat(12 + 12, 11), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(8, 4), IntArrayMat(8 + 7, 12), IntArrayMat(0, -1))

           || test_crop(a, IntArrayMat(5, 4), IntArrayMat(12, 12), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(6, 3), IntArrayMat(13, 13), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(4, 2), IntArrayMat(11, 11), IntArrayMat(-2, -1))

           || test_crop(a, IntArrayMat(11, 5, 2), IntArrayMat(11 + 7, 11, 11), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(12, 6, 4), IntArrayMat(12 + 16, 12, 12), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(8, 4, 3), IntArrayMat(8 + 12, 13, 13), IntArrayMat(-3, -2, -1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-7 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-12 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-16 + 1), IntArrayMat(-3))

           || test_crop(a, IntArrayMat(5), IntArrayMat(-6 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(6), IntArrayMat(-5 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(4), IntArrayMat(-4 + 1), IntArrayMat(-2))

           || test_crop(a, IntArrayMat(5), IntArrayMat(-5 + 1), IntArrayMat(2))
           || test_crop(a, IntArrayMat(6), IntArrayMat(-4 + 1), IntArrayMat(2))
           || test_crop(a, IntArrayMat(4), IntArrayMat(-6 + 1), IntArrayMat(-1))

           || test_crop(a, IntArrayMat(11, 5), IntArrayMat(-7 + 1, -4 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 6), IntArrayMat(-12 + 1, -6 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(8, 4), IntArrayMat(-16 + 1, -5 + 1), IntArrayMat(-3, -2))

           || test_crop(a, IntArrayMat(11, 5), IntArrayMat(-12 + 1, -6 + 1), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(12, 6), IntArrayMat(-16 + 1, -5 + 1), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(8, 4), IntArrayMat(-7 + 1, -4 + 1), IntArrayMat(-3, -1))

           || test_crop(a, IntArrayMat(5, 2), IntArrayMat(-5 + 1, -5 + 1), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(6, 4), IntArrayMat(-4 + 1, -4 + 1), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(4, 3), IntArrayMat(-6 + 1, -6 + 1), IntArrayMat(-2, -1))

           || test_crop(a, IntArrayMat(11, 5, 4), IntArrayMat(-7 + 1, -5 + 1, -5 + 1), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(12, 6, 3), IntArrayMat(-12 + 1, -6 + 1, -6 + 1), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(8, 4, 2), IntArrayMat(-16 + 1, -4 + 1, -4 + 1), IntArrayMat(-3, -2, -1));
}

static int test_crop_8(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, a)

           || test_crop(a, 0, 5, 0, 0, ncnn::Mat(6, 6))
           || test_crop(a, 6, 0, 0, 0, ncnn::Mat(8, 8))
           || test_crop(a, 5, 2, 0, 0, ncnn::Mat(6, 3))
           || test_crop(a, 6, 3, 0, 0, ncnn::Mat(8, 4))
           || test_crop(a, 4, 4, 0, 0, ncnn::Mat(7, 5))

           || test_crop(a, 5, 3, 0, 11, ncnn::Mat(7, 3, 7))
           || test_crop(a, 6, 4, 0, 12, ncnn::Mat(6, 4, 12))
           || test_crop(a, 4, 2, 0, 8, ncnn::Mat(5, 5, 16));
}

static int test_crop_9(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 5, 0, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 4, 0, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 5, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 4, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 12, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 5, 5, 5, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 4, 4, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 6, 6, 12, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 5, 0, 5, 11, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 4, 8, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 6, 6, 12, -233, -233, -233, -233, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 0, 6, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, 5, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 0, 4, -233, -233, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 5, 0, 0, -233, 4, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 6, 0, 0, -233, 5, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 4, 0, 0, -233, 6, -233, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 0, 5, 0, -233, -233, 6, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 6, 0, -233, -233, 4, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 4, 0, -233, -233, 5, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 0, 0, 11, -233, -233, -233, 7, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 12, -233, -233, -233, 12, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 8, -233, -233, -233, 16, 0, 0, 0, 0)

           || test_crop(a, 5, 2, 0, 0, 6, 5, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 3, 0, 0, 5, 6, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 4, 0, 0, 4, 4, -233, -233, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 3, 0, 2, -233, 5, -233, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 2, 0, 3, -233, 3, -233, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 4, 0, 4, -233, 4, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 5, 0, 11, -233, 1, -233, 12, 0, 0, 0, 0)
           || test_crop(a, 0, 6, 0, 12, -233, 2, -233, 16, 0, 0, 0, 0)
           || test_crop(a, 0, 4, 0, 8, -233, 3, -233, 7, 0, 0, 0, 0)

           || test_crop(a, 0, 0, 4, 11, -233, -233, 3, 16, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 5, 12, -233, -233, 2, 7, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 6, 8, -233, -233, 1, 16, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 5, 11, 1, -233, 2, 16, 0, 0, 0, 0)
           || test_crop(a, 6, 0, 6, 12, 2, -233, 3, 7, 0, 0, 0, 0)
           || test_crop(a, 4, 0, 4, 8, 3, -233, 1, 16, 0, 0, 0, 0)

           || test_crop(a, 4, 6, 3, 11, 2, 3, 4, 12, 0, 0, 0, 0)
           || test_crop(a, 5, 5, 4, 12, 3, 4, 5, 16, 0, 0, 0, 0)
           || test_crop(a, 6, 4, 2, 8, 4, 5, 6, 7, 0, 0, 0, 0)

           || test_crop(a, 5, 0, 0, 0, -233, -233, -233, -233, 2, 0, 0, 0)
           || test_crop(a, 6, 0, 0, 0, -233, -233, -233, -233, 3, 0, 0, 0)
           || test_crop(a, 4, 0, 0, 0, -233, -233, -233, -233, 4, 0, 0, 0)

           || test_crop(a, 0, 4, 0, 0, -233, -233, -233, -233, 0, 4, 0, 0)
           || test_crop(a, 0, 3, 0, 0, -233, -233, -233, -233, 0, 3, 0, 0)
           || test_crop(a, 0, 2, 0, 0, -233, -233, -233, -233, 0, 2, 0, 0)

           || test_crop(a, 0, 0, 4, 0, -233, -233, -233, -233, 0, 0, 4, 0)
           || test_crop(a, 0, 0, 5, 0, -233, -233, -233, -233, 0, 0, 2, 0)
           || test_crop(a, 0, 0, 6, 0, -233, -233, -233, -233, 0, 0, 3, 0)

           || test_crop(a, 0, 0, 0, 11, -233, -233, -233, -233, 0, 0, 0, 12)
           || test_crop(a, 0, 0, 0, 12, -233, -233, -233, -233, 0, 0, 0, 16)
           || test_crop(a, 0, 0, 0, 8, -233, -233, -233, -233, 0, 0, 0, 7)

           || test_crop(a, 5, 3, 0, 0, -233, -233, -233, -233, 5, 2, 0, 0)
           || test_crop(a, 6, 4, 0, 0, -233, -233, -233, -233, 3, 3, 0, 0)
           || test_crop(a, 4, 4, 0, 0, -233, -233, -233, -233, 2, 5, 0, 0)

           || test_crop(a, 0, 4, 0, 11, -233, -233, -233, -233, 0, 3, 0, 7)
           || test_crop(a, 0, 3, 0, 12, -233, -233, -233, -233, 0, 4, 0, 12)
           || test_crop(a, 0, 2, 0, 8, -233, -233, -233, -233, 0, 5, 0, 16)

           || test_crop(a, 0, 4, 4, 0, -233, -233, -233, -233, 0, 4, 1, 0)
           || test_crop(a, 0, 5, 5, 0, -233, -233, -233, -233, 0, 2, 2, 0)
           || test_crop(a, 0, 2, 6, 0, -233, -233, -233, -233, 0, 1, 3, 0)

           || test_crop(a, 3, 0, 0, 11, -233, -233, -233, -233, 3, 0, 0, 12)
           || test_crop(a, 4, 0, 0, 12, -233, -233, -233, -233, 4, 0, 0, 16)
           || test_crop(a, 5, 0, 0, 8, -233, -233, -233, -233, 2, 0, 0, 7)

           || test_crop(a, 0, 4, 4, 11, -233, -233, -233, -233, 0, 4, 4, 12)
           || test_crop(a, 0, 5, 5, 12, -233, -233, -233, -233, 0, 4, 4, 16)
           || test_crop(a, 0, 6, 6, 8, -233, -233, -233, -233, 0, 3, 3, 7)

           || test_crop(a, 1, 1, 1, 11, -233, -233, -233, -233, 1, 1, 1, 16)
           || test_crop(a, 2, 2, 2, 12, -233, -233, -233, -233, 2, 2, 2, 7)
           || test_crop(a, 3, 3, 3, 8, -233, -233, -233, -233, 3, 3, 3, 12);
}

static int test_crop_10(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, IntArrayMat(11), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(6), IntArrayMat(-233), IntArrayMat(1))
           || test_crop(a, IntArrayMat(5), IntArrayMat(-233), IntArrayMat(2))
           || test_crop(a, IntArrayMat(4), IntArrayMat(-233), IntArrayMat(-2))
           || test_crop(a, IntArrayMat(6), IntArrayMat(-233), IntArrayMat(3))
           || test_crop(a, IntArrayMat(5), IntArrayMat(-233), IntArrayMat(-1))
           || test_crop(a, IntArrayMat(8, 4), IntArrayMat(-233, -233), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 6), IntArrayMat(-233, -233), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(11, 5), IntArrayMat(-233, -233), IntArrayMat(-4, -2))
           || test_crop(a, IntArrayMat(4, 4), IntArrayMat(-233, -233), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(12, 6), IntArrayMat(-233, -233), IntArrayMat(0, 3))
           || test_crop(a, IntArrayMat(5, 5), IntArrayMat(-233, -233), IntArrayMat(1, 3))
           || test_crop(a, IntArrayMat(4, 4), IntArrayMat(-233, -233), IntArrayMat(2, 3))
           || test_crop(a, IntArrayMat(12, 6, 6), IntArrayMat(-233, -233, -233), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(11, 5, 5), IntArrayMat(-233, -233, -233), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(8, 4, 4), IntArrayMat(-233, -233, -233), IntArrayMat(0, 1, 3))
           || test_crop(a, IntArrayMat(12, 6, 6), IntArrayMat(-233, -233, -233), IntArrayMat(0, 2, 3))
           || test_crop(a, IntArrayMat(11, 5, 5), IntArrayMat(-233, -233, -233), IntArrayMat(0, 2, 3))
           || test_crop(a, IntArrayMat(4, 4, 4), IntArrayMat(-233, -233, -233), IntArrayMat(1, 2, 3))
           || test_crop(a, IntArrayMat(6, 6, 6), IntArrayMat(-233, -233, -233), IntArrayMat(1, 2, 3))
           || test_crop(a, IntArrayMat(11, 5, 5, 5), IntArrayMat(-233, -233, -233, -233), IntArrayMat(0, 1, 2, 3))
           || test_crop(a, IntArrayMat(8, 4, 4, 4), IntArrayMat(-233, -233, -233, -233), IntArrayMat(0, 1, 2, 3))
           || test_crop(a, IntArrayMat(12, 6, 6, 6), IntArrayMat(-233, -233, -233, -233), IntArrayMat(-4, -3, -2, -1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(11 + 16), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(12 + 7), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(8 + 12), IntArrayMat(-4))

           || test_crop(a, IntArrayMat(5), IntArrayMat(11), IntArrayMat(1))
           || test_crop(a, IntArrayMat(6), IntArrayMat(13), IntArrayMat(1))
           || test_crop(a, IntArrayMat(4), IntArrayMat(12), IntArrayMat(-3))

           || test_crop(a, IntArrayMat(3), IntArrayMat(12), IntArrayMat(2))
           || test_crop(a, IntArrayMat(4), IntArrayMat(13), IntArrayMat(2))
           || test_crop(a, IntArrayMat(5), IntArrayMat(11), IntArrayMat(-2))

           || test_crop(a, IntArrayMat(1), IntArrayMat(8), IntArrayMat(3))
           || test_crop(a, IntArrayMat(2), IntArrayMat(7), IntArrayMat(3))
           || test_crop(a, IntArrayMat(3), IntArrayMat(6), IntArrayMat(-1))

           || test_crop(a, IntArrayMat(11, 5), IntArrayMat(11 + 7, 11), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 6), IntArrayMat(12 + 12, 12), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(8, 4), IntArrayMat(8 + 16, 13), IntArrayMat(-4, -3))

           || test_crop(a, IntArrayMat(11, 4), IntArrayMat(11 + 12, 13), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(12, 3), IntArrayMat(12 + 16, 11), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(8, 2), IntArrayMat(8 + 7, 12), IntArrayMat(-4, -2))

           || test_crop(a, IntArrayMat(11, 1), IntArrayMat(11 + 16, 5), IntArrayMat(0, 3))
           || test_crop(a, IntArrayMat(12, 2), IntArrayMat(12 + 7, 6), IntArrayMat(0, 3))
           || test_crop(a, IntArrayMat(8, 3), IntArrayMat(8 + 12, 7), IntArrayMat(-4, -1))

           || test_crop(a, IntArrayMat(3, 3), IntArrayMat(13, 4), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(4, 2), IntArrayMat(12, 3), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(5, 1), IntArrayMat(11, 2), IntArrayMat(-3, -2))

           || test_crop(a, IntArrayMat(5, 5), IntArrayMat(11, 8), IntArrayMat(1, 3))
           || test_crop(a, IntArrayMat(4, 6), IntArrayMat(12, 9), IntArrayMat(1, 3))
           || test_crop(a, IntArrayMat(3, 4), IntArrayMat(13, 7), IntArrayMat(-3, -1))

           || test_crop(a, IntArrayMat(2, 3), IntArrayMat(12, 9), IntArrayMat(2, 3))
           || test_crop(a, IntArrayMat(3, 2), IntArrayMat(11, 7), IntArrayMat(2, 3))
           || test_crop(a, IntArrayMat(4, 1), IntArrayMat(10, 8), IntArrayMat(-2, -1))

           || test_crop(a, IntArrayMat(11, 2, 2), IntArrayMat(11 + 6, 9, 9), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(12, 3, 3), IntArrayMat(12 + 1, 10, 10), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(8, 4, 4), IntArrayMat(8 + 3, 11, 11), IntArrayMat(-4, -3, -2))

           || test_crop(a, IntArrayMat(11, 4, 4), IntArrayMat(11 + 12, 12, 12), IntArrayMat(0, 1, 3))
           || test_crop(a, IntArrayMat(12, 5, 5), IntArrayMat(12 + 8, 11, 11), IntArrayMat(0, 1, 3))
           || test_crop(a, IntArrayMat(8, 6, 6), IntArrayMat(8 + 4, 13, 13), IntArrayMat(-4, -3, -1))

           || test_crop(a, IntArrayMat(11, 1, 4), IntArrayMat(11 + 5, 12, 12), IntArrayMat(0, 2, 3))
           || test_crop(a, IntArrayMat(12, 3, 3), IntArrayMat(12 + 3, 11, 11), IntArrayMat(0, 2, 3))
           || test_crop(a, IntArrayMat(8, 2, 5), IntArrayMat(8 + 2, 10, 10), IntArrayMat(-4, -2, -1))

           || test_crop(a, IntArrayMat(1, 1, 1), IntArrayMat(7, 7, 7), IntArrayMat(1, 2, 3))
           || test_crop(a, IntArrayMat(2, 2, 2), IntArrayMat(8, 9, 10), IntArrayMat(1, 2, 3))
           || test_crop(a, IntArrayMat(3, 3, 3), IntArrayMat(11, 12, 13), IntArrayMat(-3, -2, -1))

           || test_crop(a, IntArrayMat(11, 2, 3, 6), IntArrayMat(11 + 11, 10, 12, 11), IntArrayMat(0, 1, 2, 3))
           || test_crop(a, IntArrayMat(12, 3, 4, 5), IntArrayMat(12 + 12, 9, 11, 13), IntArrayMat(0, 1, 2, 3))
           || test_crop(a, IntArrayMat(8, 4, 5, 4), IntArrayMat(8 + 8, 8, 10, 12), IntArrayMat(-4, -3, -2, -1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-7 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-12 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-16 + 1), IntArrayMat(-4))

           || test_crop(a, IntArrayMat(5), IntArrayMat(-6 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(6), IntArrayMat(-5 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(4), IntArrayMat(-4 + 1), IntArrayMat(-3))

           || test_crop(a, IntArrayMat(4), IntArrayMat(-4 + 1), IntArrayMat(2))
           || test_crop(a, IntArrayMat(5), IntArrayMat(-5 + 1), IntArrayMat(2))
           || test_crop(a, IntArrayMat(6), IntArrayMat(-6 + 1), IntArrayMat(-2))

           || test_crop(a, IntArrayMat(1), IntArrayMat(-5 + 1), IntArrayMat(3))
           || test_crop(a, IntArrayMat(2), IntArrayMat(-4 + 1), IntArrayMat(3))
           || test_crop(a, IntArrayMat(3), IntArrayMat(-3 + 1), IntArrayMat(-1))

           || test_crop(a, IntArrayMat(11, 3), IntArrayMat(-7 + 1, -3 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 4), IntArrayMat(-12 + 1, -4 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(8, 5), IntArrayMat(-16 + 1, -5 + 1), IntArrayMat(-4, -3))

           || test_crop(a, IntArrayMat(11, 1), IntArrayMat(-12 + 1, -5 + 1), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(12, 2), IntArrayMat(-16 + 1, -4 + 1), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(8, 3), IntArrayMat(-7 + 1, -6 + 1), IntArrayMat(-4, -2))

           || test_crop(a, IntArrayMat(11, 3), IntArrayMat(-12 + 1, -2 + 1), IntArrayMat(0, 3))
           || test_crop(a, IntArrayMat(12, 4), IntArrayMat(-16 + 1, -3 + 1), IntArrayMat(0, 3))
           || test_crop(a, IntArrayMat(8, 5), IntArrayMat(-7 + 1, -4 + 1), IntArrayMat(-4, -1))

           || test_crop(a, IntArrayMat(2, 3), IntArrayMat(-4 + 1, -2 + 1), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(3, 4), IntArrayMat(-2 + 1, -3 + 1), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(4, 5), IntArrayMat(-3 + 1, -4 + 1), IntArrayMat(-3, -2))

           || test_crop(a, IntArrayMat(3, 2), IntArrayMat(-2 + 1, -4 + 1), IntArrayMat(1, 3))
           || test_crop(a, IntArrayMat(4, 3), IntArrayMat(-3 + 1, -2 + 1), IntArrayMat(1, 3))
           || test_crop(a, IntArrayMat(5, 4), IntArrayMat(-4 + 1, -3 + 1), IntArrayMat(-3, -1))

           || test_crop(a, IntArrayMat(2, 3), IntArrayMat(-4 + 1, -6 + 1), IntArrayMat(2, 3))
           || test_crop(a, IntArrayMat(1, 2), IntArrayMat(-5 + 1, -5 + 1), IntArrayMat(2, 3))
           || test_crop(a, IntArrayMat(3, 1), IntArrayMat(-6 + 1, -4 + 1), IntArrayMat(-2, -1))

           || test_crop(a, IntArrayMat(11, 3, 3), IntArrayMat(-7 + 1, -3 + 1, -4 + 1), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(12, 4, 4), IntArrayMat(-12 + 1, -4 + 1, -3 + 1), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(8, 5, 5), IntArrayMat(-16 + 1, -5 + 1, -5 + 1), IntArrayMat(-4, -3, -2))

           || test_crop(a, IntArrayMat(11, 2, 2), IntArrayMat(-7 + 1, -5 + 1, -4 + 1), IntArrayMat(0, 1, 3))
           || test_crop(a, IntArrayMat(12, 1, 1), IntArrayMat(-12 + 1, -6 + 1, -5 + 1), IntArrayMat(0, 1, 3))
           || test_crop(a, IntArrayMat(8, 3, 3), IntArrayMat(-16 + 1, -4 + 1, -6 + 1), IntArrayMat(-4, -3, -1))

           || test_crop(a, IntArrayMat(11, 2, 5), IntArrayMat(-7 + 1, -2 + 1, -5 + 1), IntArrayMat(0, 2, 3))
           || test_crop(a, IntArrayMat(12, 3, 3), IntArrayMat(-12 + 1, -3 + 1, -4 + 1), IntArrayMat(0, 2, 3))
           || test_crop(a, IntArrayMat(8, 4, 4), IntArrayMat(-16 + 1, -4 + 1, -3 + 1), IntArrayMat(-4, -2, -1))

           || test_crop(a, IntArrayMat(1, 3, 3), IntArrayMat(-3 + 1, -6 + 1, -4 + 1), IntArrayMat(1, 2, 3))
           || test_crop(a, IntArrayMat(2, 2, 2), IntArrayMat(-4 + 1, -4 + 1, -5 + 1), IntArrayMat(1, 2, 3))
           || test_crop(a, IntArrayMat(3, 1, 1), IntArrayMat(-5 + 1, -5 + 1, -6 + 1), IntArrayMat(-3, -2, -1))

           || test_crop(a, IntArrayMat(11, 3, 4, 4), IntArrayMat(-7 + 1, -3 + 1, -2 + 1, -4 + 1), IntArrayMat(0, 1, 2, 3))
           || test_crop(a, IntArrayMat(12, 4, 5, 3), IntArrayMat(-12 + 1, -4 + 1, -3 + 1, -5 + 1), IntArrayMat(0, 1, 2, 3))
           || test_crop(a, IntArrayMat(8, 5, 6, 2), IntArrayMat(-16 + 1, -5 + 1, -4 + 1, -3 + 1), IntArrayMat(-4, -3, -2, -1));
}

static int test_crop_11(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, a)

           || test_crop(a, 0, 5, 0, 0, ncnn::Mat(6, 6, 6))
           || test_crop(a, 6, 0, 0, 0, ncnn::Mat(8, 8, 8))
           || test_crop(a, 5, 5, 5, 0, ncnn::Mat(6, 6, 6))
           || test_crop(a, 6, 6, 6, 0, ncnn::Mat(8, 8, 8))
           || test_crop(a, 4, 4, 4, 0, ncnn::Mat(5, 5, 5))

           || test_crop(a, 3, 3, 3, 11, ncnn::Mat(3, 3, 3, 7))
           || test_crop(a, 4, 4, 4, 12, ncnn::Mat(6, 6, 6, 12))
           || test_crop(a, 5, 5, 5, 8, ncnn::Mat(8, 8, 8, 16));
}

int main()
{
    SRAND(776757);

    return 0
           || test_crop_0(RandomMat(112))
           || test_crop_0(RandomMat(126))
           || test_crop_0(RandomMat(127))
           || test_crop_1(RandomMat(112))
           || test_crop_1(RandomMat(126))
           || test_crop_1(RandomMat(127))
           || test_crop_2(RandomMat(112))
           || test_crop_2(RandomMat(126))
           || test_crop_2(RandomMat(127))
           || test_crop_3(RandomMat(20, 40))
           || test_crop_3(RandomMat(15, 36))
           || test_crop_3(RandomMat(16, 33))
           || test_crop_4(RandomMat(20, 40))
           || test_crop_4(RandomMat(15, 36))
           || test_crop_4(RandomMat(16, 33))
           || test_crop_5(RandomMat(20, 40))
           || test_crop_5(RandomMat(15, 36))
           || test_crop_5(RandomMat(16, 33))
           || test_crop_6(RandomMat(20, 20, 40))
           || test_crop_6(RandomMat(15, 15, 36))
           || test_crop_6(RandomMat(16, 16, 33))
           || test_crop_7(RandomMat(20, 20, 40))
           || test_crop_7(RandomMat(15, 15, 36))
           || test_crop_7(RandomMat(16, 16, 33))
           || test_crop_8(RandomMat(20, 20, 40))
           || test_crop_8(RandomMat(15, 15, 36))
           || test_crop_8(RandomMat(16, 16, 33))
           || test_crop_9(RandomMat(20, 20, 20, 40))
           || test_crop_9(RandomMat(15, 15, 15, 36))
           || test_crop_9(RandomMat(16, 16, 16, 33))
           || test_crop_10(RandomMat(20, 20, 20, 40))
           || test_crop_10(RandomMat(15, 15, 15, 36))
           || test_crop_10(RandomMat(16, 16, 16, 33))
           || test_crop_11(RandomMat(20, 20, 20, 40))
           || test_crop_11(RandomMat(15, 15, 15, 36))
           || test_crop_11(RandomMat(16, 16, 16, 33));
}
