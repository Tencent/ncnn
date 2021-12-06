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
        fprintf(stderr, "test_crop failed a.dims=%d a=(%d %d %d %d) woffset=%d hoffset=%d coffset=%d ref.dims=%d ref=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c, woffset, hoffset, doffset, coffset, ref.dims, ref.w, ref.h, ref.d, ref.c);
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
           || test_crop(a, IntArrayMat(12), IntArrayMat(12 + 7), IntArrayMat(0))
           || test_crop(a, IntArrayMat(16), IntArrayMat(16 + 12), IntArrayMat(0))
           || test_crop(a, IntArrayMat(11), IntArrayMat(-7 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-12 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(16), IntArrayMat(-16 + 1), IntArrayMat(0));
}

static int test_crop_2(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, ncnn::Mat(27))

           || test_crop(a, 11, 0, 0, 0, ncnn::Mat(7))
           || test_crop(a, 12, 0, 0, 0, ncnn::Mat(12))
           || test_crop(a, 16, 0, 0, 0, ncnn::Mat(16));
}

static int test_crop_3(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 11, 0, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 0, 11, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 12, 12, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 8, 8, 0, 0, -233, -233, 0, 0, 0, 0, 0, 0)

           || test_crop(a, 11, 0, 0, 0, 7, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 0, 12, -233, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 8, 0, 0, 0, 16, -233, 0, 0, 0, 0, 0, 0)

           || test_crop(a, 0, 11, 0, 0, -233, 12, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 0, 0, -233, 16, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 0, 8, 0, 0, -233, 7, 0, 0, 0, 0, 0, 0)

           || test_crop(a, 11, 11, 0, 0, 16, 16, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 12, 12, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0)
           || test_crop(a, 8, 8, 0, 0, 12, 12, 0, 0, 0, 0, 0, 0)

           || test_crop(a, 11, 0, 0, 0, -233, -233, 0, 0, 7, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 0, -233, -233, 0, 0, 12, 0, 0, 0)
           || test_crop(a, 8, 0, 0, 0, -233, -233, 0, 0, 16, 0, 0, 0)

           || test_crop(a, 0, 11, 0, 0, -233, -233, 0, 0, 0, 12, 0, 0)
           || test_crop(a, 0, 12, 0, 0, -233, -233, 0, 0, 0, 16, 0, 0)
           || test_crop(a, 0, 8, 0, 0, -233, -233, 0, 0, 0, 7, 0, 0)

           || test_crop(a, 11, 11, 0, 0, -233, -233, 0, 0, 16, 16, 0, 0)
           || test_crop(a, 12, 12, 0, 0, -233, -233, 0, 0, 7, 7, 0, 0)
           || test_crop(a, 8, 8, 0, 0, -233, -233, 0, 0, 12, 12, 0, 0);
}

static int test_crop_4(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, IntArrayMat(12), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-233), IntArrayMat(1))
           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-233, -233), IntArrayMat(0, 1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(24), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(16), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(17), IntArrayMat(0))

           || test_crop(a, IntArrayMat(11), IntArrayMat(16), IntArrayMat(1))
           || test_crop(a, IntArrayMat(12), IntArrayMat(17), IntArrayMat(1))
           || test_crop(a, IntArrayMat(8), IntArrayMat(24), IntArrayMat(1))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(17, 17), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(24, 24), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(16, 16), IntArrayMat(0, 1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-16 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-7 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-12 + 1), IntArrayMat(0))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-7 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-12 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-16 + 1), IntArrayMat(1))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-12 + 1, -12 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-16 + 1, -16 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-7 + 1, -7 + 1), IntArrayMat(0, 1));
}

static int test_crop_5(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 12, 0, 0, ncnn::Mat(16, 7))
           || test_crop(a, 11, 0, 0, 0, ncnn::Mat(7, 27))

           || test_crop(a, 11, 11, 0, 0, ncnn::Mat(12, 12))
           || test_crop(a, 12, 12, 0, 0, ncnn::Mat(16, 16))
           || test_crop(a, 8, 8, 0, 0, ncnn::Mat(7, 7));
}

static int test_crop_6(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 11, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 8, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 12, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 11, 11, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 8, 8, 0, 0, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 0, 12, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 11, 0, 0, 11, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 8, 0, 0, 8, -233, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 12, 12, 0, 12, -233, -233, 0, -233, 0, 0, 0, 0)

           || test_crop(a, 11, 0, 0, 0, 12, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 0, 16, -233, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 8, 0, 0, 0, 7, -233, 0, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 11, 0, 0, -233, 16, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 0, 0, -233, 7, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 8, 0, 0, -233, 12, 0, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 0, 0, 11, -233, -233, 0, 7, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 12, -233, -233, 0, 12, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 8, -233, -233, 0, 16, 0, 0, 0, 0)

           || test_crop(a, 11, 11, 0, 0, 7, 7, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 12, 12, 0, 0, 12, 12, 0, -233, 0, 0, 0, 0)
           || test_crop(a, 8, 8, 0, 0, 16, 16, 0, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 11, 0, 11, -233, 12, 0, 12, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 0, 12, -233, 16, 0, 16, 0, 0, 0, 0)
           || test_crop(a, 0, 8, 0, 8, -233, 7, 0, 7, 0, 0, 0, 0)

           || test_crop(a, 11, 0, 0, 11, 16, -233, 0, 16, 0, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 12, 7, -233, 0, 7, 0, 0, 0, 0)
           || test_crop(a, 8, 0, 0, 8, 16, -233, 0, 16, 0, 0, 0, 0)

           || test_crop(a, 11, 11, 0, 11, 12, 12, 0, 12, 0, 0, 0, 0)
           || test_crop(a, 12, 12, 0, 12, 16, 16, 0, 16, 0, 0, 0, 0)
           || test_crop(a, 8, 8, 0, 8, 7, 7, 0, 7, 0, 0, 0, 0)

           || test_crop(a, 11, 0, 0, 0, -233, -233, 0, -233, 16, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 0, -233, -233, 0, -233, 7, 0, 0, 0)
           || test_crop(a, 8, 0, 0, 0, -233, -233, 0, -233, 12, 0, 0, 0)

           || test_crop(a, 0, 11, 0, 0, -233, -233, 0, -233, 0, 7, 0, 0)
           || test_crop(a, 0, 12, 0, 0, -233, -233, 0, -233, 0, 12, 0, 0)
           || test_crop(a, 0, 8, 0, 0, -233, -233, 0, -233, 0, 16, 0, 0)

           || test_crop(a, 0, 0, 0, 11, -233, -233, 0, -233, 0, 0, 0, 12)
           || test_crop(a, 0, 0, 0, 12, -233, -233, 0, -233, 0, 0, 0, 16)
           || test_crop(a, 0, 0, 0, 8, -233, -233, 0, -233, 0, 0, 0, 7)

           || test_crop(a, 11, 11, 0, 0, -233, -233, 0, -233, 16, 16, 0, 0)
           || test_crop(a, 12, 12, 0, 0, -233, -233, 0, -233, 7, 7, 0, 0)
           || test_crop(a, 8, 8, 0, 0, -233, -233, 0, -233, 12, 12, 0, 0)

           || test_crop(a, 0, 11, 0, 11, -233, -233, 0, -233, 0, 7, 0, 7)
           || test_crop(a, 0, 12, 0, 12, -233, -233, 0, -233, 0, 12, 0, 12)
           || test_crop(a, 0, 8, 0, 8, -233, -233, 0, -233, 0, 16, 0, 16)

           || test_crop(a, 11, 0, 0, 11, -233, -233, 0, -233, 12, 0, 0, 12)
           || test_crop(a, 12, 0, 0, 12, -233, -233, 0, -233, 16, 0, 0, 16)
           || test_crop(a, 8, 0, 0, 8, -233, -233, 0, -233, 7, 0, 0, 7)

           || test_crop(a, 11, 11, 0, 11, -233, -233, 0, -233, 16, 16, 0, 16)
           || test_crop(a, 12, 12, 0, 12, -233, -233, 0, -233, 7, 7, 0, 7)
           || test_crop(a, 8, 8, 0, 8, -233, -233, 0, -233, 12, 12, 0, 12);
}

static int test_crop_7(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, IntArrayMat(11), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-233), IntArrayMat(1))
           || test_crop(a, IntArrayMat(11), IntArrayMat(-233), IntArrayMat(2))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-233), IntArrayMat(2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-233, -233), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-233, -233), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-233, -233), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-233, -233), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(-233, -233, -233), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(8, 8, 8), IntArrayMat(-233, -233, -233), IntArrayMat(0, 1, 2))

           || test_crop(a, IntArrayMat(11), IntArrayMat(16), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(17), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(24), IntArrayMat(0))

           || test_crop(a, IntArrayMat(11), IntArrayMat(17), IntArrayMat(1))
           || test_crop(a, IntArrayMat(12), IntArrayMat(24), IntArrayMat(1))
           || test_crop(a, IntArrayMat(8), IntArrayMat(16), IntArrayMat(1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(24), IntArrayMat(2))
           || test_crop(a, IntArrayMat(12), IntArrayMat(16), IntArrayMat(2))
           || test_crop(a, IntArrayMat(8), IntArrayMat(17), IntArrayMat(2))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(16, 16), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(17, 17), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(24, 24), IntArrayMat(0, 1))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(17, 17), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(24, 24), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(16, 16), IntArrayMat(0, 2))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(24, 24), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(16, 16), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(17, 17), IntArrayMat(1, 2))

           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(16, 16, 16), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(17, 17, 17), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(8, 8, 8), IntArrayMat(24, 24, 24), IntArrayMat(0, 1, 2))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-7 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-12 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-16 + 1), IntArrayMat(0))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-12 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-16 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-7 + 1), IntArrayMat(1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-16 + 1), IntArrayMat(2))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-7 + 1), IntArrayMat(2))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-12 + 1), IntArrayMat(2))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-7 + 1, -7 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-12 + 1, -12 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-16 + 1, -16 + 1), IntArrayMat(0, 1))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-12 + 1, -12 + 1), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-16 + 1, -16 + 1), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-7 + 1, -7 + 1), IntArrayMat(0, 2))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-16 + 1, -16 + 1), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-7 + 1, -7 + 1), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-12 + 1, -12 + 1), IntArrayMat(1, 2))

           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(-7 + 1, -7 + 1, -7 + 1), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(-12 + 1, -12 + 1, -12 + 1), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(8, 8, 8), IntArrayMat(-16 + 1, -16 + 1, -16 + 1), IntArrayMat(0, 1, 2));
}

static int test_crop_8(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 11, 0, 0, ncnn::Mat(12, 12))
           || test_crop(a, 12, 0, 0, 0, ncnn::Mat(16, 16))
           || test_crop(a, 11, 11, 0, 0, ncnn::Mat(12, 12))
           || test_crop(a, 12, 12, 0, 0, ncnn::Mat(16, 16))
           || test_crop(a, 8, 8, 0, 0, ncnn::Mat(7, 7))

           || test_crop(a, 11, 11, 0, 11, ncnn::Mat(7, 7, 7))
           || test_crop(a, 12, 12, 0, 12, ncnn::Mat(12, 12, 12))
           || test_crop(a, 8, 8, 0, 8, ncnn::Mat(16, 16, 16));
}

static int test_crop_9(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 11, 0, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 8, 0, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 11, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 8, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 12, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 11, 11, 11, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 8, 8, 8, 0, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 12, 12, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 11, 0, 11, 11, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 8, 0, 8, 8, -233, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 12, 12, 12, 12, -233, -233, -233, -233, 0, 0, 0, 0)

           || test_crop(a, 11, 0, 0, 0, 12, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 0, 16, -233, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 8, 0, 0, 0, 7, -233, -233, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 11, 0, 0, -233, 16, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 0, 0, -233, 7, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 8, 0, 0, -233, 12, -233, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 0, 11, 0, -233, -233, 16, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 12, 0, -233, -233, 7, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 8, 0, -233, -233, 12, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 0, 0, 11, -233, -233, -233, 7, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 12, -233, -233, -233, 12, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 0, 8, -233, -233, -233, 16, 0, 0, 0, 0)

           || test_crop(a, 11, 11, 0, 0, 7, 7, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 12, 12, 0, 0, 12, 12, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 8, 8, 0, 0, 16, 16, -233, -233, 0, 0, 0, 0)

           || test_crop(a, 11, 0, 11, 0, 7, -233, 7, -233, 0, 0, 0, 0)
           || test_crop(a, 12, 0, 12, 0, 12, -233, 12, -233, 0, 0, 0, 0)
           || test_crop(a, 8, 0, 8, 0, 16, -233, 16, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 11, 0, 11, -233, 12, -233, 12, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 0, 12, -233, 16, -233, 16, 0, 0, 0, 0)
           || test_crop(a, 0, 8, 0, 8, -233, 7, -233, 7, 0, 0, 0, 0)

           || test_crop(a, 0, 0, 11, 11, -233, -233, 16, 16, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 12, 12, -233, -233, 7, 7, 0, 0, 0, 0)
           || test_crop(a, 0, 0, 8, 8, -233, -233, 16, 16, 0, 0, 0, 0)

           || test_crop(a, 11, 0, 11, 11, 16, -233, 16, 16, 0, 0, 0, 0)
           || test_crop(a, 12, 0, 12, 12, 7, -233, 7, 7, 0, 0, 0, 0)
           || test_crop(a, 8, 0, 8, 8, 16, -233, 16, 16, 0, 0, 0, 0)

           || test_crop(a, 11, 11, 11, 11, 12, 12, 12, 12, 0, 0, 0, 0)
           || test_crop(a, 12, 12, 12, 12, 16, 16, 16, 16, 0, 0, 0, 0)
           || test_crop(a, 8, 8, 8, 8, 7, 7, 7, 7, 0, 0, 0, 0)

           || test_crop(a, 11, 0, 0, 0, -233, -233, -233, -233, 16, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 0, -233, -233, -233, -233, 7, 0, 0, 0)
           || test_crop(a, 8, 0, 0, 0, -233, -233, -233, -233, 12, 0, 0, 0)

           || test_crop(a, 0, 11, 0, 0, -233, -233, -233, -233, 0, 7, 0, 0)
           || test_crop(a, 0, 12, 0, 0, -233, -233, -233, -233, 0, 12, 0, 0)
           || test_crop(a, 0, 8, 0, 0, -233, -233, -233, -233, 0, 16, 0, 0)

           || test_crop(a, 0, 0, 11, 0, -233, -233, -233, -233, 0, 0, 7, 0)
           || test_crop(a, 0, 0, 12, 0, -233, -233, -233, -233, 0, 0, 12, 0)
           || test_crop(a, 0, 0, 8, 0, -233, -233, -233, -233, 0, 0, 16, 0)

           || test_crop(a, 0, 0, 0, 11, -233, -233, -233, -233, 0, 0, 0, 12)
           || test_crop(a, 0, 0, 0, 12, -233, -233, -233, -233, 0, 0, 0, 16)
           || test_crop(a, 0, 0, 0, 8, -233, -233, -233, -233, 0, 0, 0, 7)

           || test_crop(a, 11, 11, 0, 0, -233, -233, -233, -233, 16, 16, 0, 0)
           || test_crop(a, 12, 12, 0, 0, -233, -233, -233, -233, 7, 7, 0, 0)
           || test_crop(a, 8, 8, 0, 0, -233, -233, -233, -233, 12, 12, 0, 0)

           || test_crop(a, 0, 11, 0, 11, -233, -233, -233, -233, 0, 7, 0, 7)
           || test_crop(a, 0, 12, 0, 12, -233, -233, -233, -233, 0, 12, 0, 12)
           || test_crop(a, 0, 8, 0, 8, -233, -233, -233, -233, 0, 16, 0, 16)

           || test_crop(a, 0, 11, 11, 0, -233, -233, -233, -233, 0, 7, 7, 0)
           || test_crop(a, 0, 12, 12, 0, -233, -233, -233, -233, 0, 12, 12, 0)
           || test_crop(a, 0, 8, 8, 0, -233, -233, -233, -233, 0, 16, 16, 0)

           || test_crop(a, 11, 0, 0, 11, -233, -233, -233, -233, 12, 0, 0, 12)
           || test_crop(a, 12, 0, 0, 12, -233, -233, -233, -233, 16, 0, 0, 16)
           || test_crop(a, 8, 0, 0, 8, -233, -233, -233, -233, 7, 0, 0, 7)

           || test_crop(a, 0, 11, 11, 11, -233, -233, -233, -233, 0, 12, 12, 12)
           || test_crop(a, 0, 12, 12, 12, -233, -233, -233, -233, 0, 16, 16, 16)
           || test_crop(a, 0, 8, 8, 8, -233, -233, -233, -233, 0, 7, 7, 7)

           || test_crop(a, 11, 11, 11, 11, -233, -233, -233, -233, 16, 16, 16, 16)
           || test_crop(a, 12, 12, 12, 12, -233, -233, -233, -233, 7, 7, 7, 7)
           || test_crop(a, 8, 8, 8, 8, -233, -233, -233, -233, 12, 12, 12, 12);
}

static int test_crop_10(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, IntArrayMat(11), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-233), IntArrayMat(1))
           || test_crop(a, IntArrayMat(11), IntArrayMat(-233), IntArrayMat(2))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-233), IntArrayMat(2))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-233), IntArrayMat(3))
           || test_crop(a, IntArrayMat(11), IntArrayMat(-233), IntArrayMat(3))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-233, -233), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-233, -233), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-233, -233), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-233, -233), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-233, -233), IntArrayMat(0, 3))
           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-233, -233), IntArrayMat(1, 3))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-233, -233), IntArrayMat(2, 3))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(-233, -233, -233), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(-233, -233, -233), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(8, 8, 8), IntArrayMat(-233, -233, -233), IntArrayMat(0, 1, 3))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(-233, -233, -233), IntArrayMat(0, 2, 3))
           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(-233, -233, -233), IntArrayMat(0, 2, 3))
           || test_crop(a, IntArrayMat(8, 8, 8), IntArrayMat(-233, -233, -233), IntArrayMat(1, 2, 3))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(-233, -233, -233), IntArrayMat(1, 2, 3))
           || test_crop(a, IntArrayMat(11, 11, 11, 11), IntArrayMat(-233, -233, -233, -233), IntArrayMat(0, 1, 2, 3))
           || test_crop(a, IntArrayMat(8, 8, 8, 8), IntArrayMat(-233, -233, -233, -233), IntArrayMat(0, 1, 2, 3))
           || test_crop(a, IntArrayMat(12, 12, 12, 12), IntArrayMat(-233, -233, -233, -233), IntArrayMat(0, 1, 2, 3))

           || test_crop(a, IntArrayMat(11), IntArrayMat(16), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(17), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(24), IntArrayMat(0))

           || test_crop(a, IntArrayMat(11), IntArrayMat(17), IntArrayMat(1))
           || test_crop(a, IntArrayMat(12), IntArrayMat(24), IntArrayMat(1))
           || test_crop(a, IntArrayMat(8), IntArrayMat(16), IntArrayMat(1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(24), IntArrayMat(2))
           || test_crop(a, IntArrayMat(12), IntArrayMat(16), IntArrayMat(2))
           || test_crop(a, IntArrayMat(8), IntArrayMat(17), IntArrayMat(2))

           || test_crop(a, IntArrayMat(11), IntArrayMat(24), IntArrayMat(3))
           || test_crop(a, IntArrayMat(12), IntArrayMat(16), IntArrayMat(3))
           || test_crop(a, IntArrayMat(8), IntArrayMat(17), IntArrayMat(3))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(16, 16), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(17, 17), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(24, 24), IntArrayMat(0, 1))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(17, 17), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(24, 24), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(16, 16), IntArrayMat(0, 2))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(17, 17), IntArrayMat(0, 3))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(24, 24), IntArrayMat(0, 3))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(16, 16), IntArrayMat(0, 3))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(24, 24), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(16, 16), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(17, 17), IntArrayMat(1, 2))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(24, 24), IntArrayMat(1, 3))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(16, 16), IntArrayMat(1, 3))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(17, 17), IntArrayMat(1, 3))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(24, 24), IntArrayMat(2, 3))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(16, 16), IntArrayMat(2, 3))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(17, 17), IntArrayMat(2, 3))

           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(16, 16, 16), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(17, 17, 17), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(8, 8, 8), IntArrayMat(24, 24, 24), IntArrayMat(0, 1, 2))

           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(16, 16, 16), IntArrayMat(0, 1, 3))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(17, 17, 17), IntArrayMat(0, 1, 3))
           || test_crop(a, IntArrayMat(8, 8, 8), IntArrayMat(24, 24, 24), IntArrayMat(0, 1, 3))

           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(16, 16, 16), IntArrayMat(0, 2, 3))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(17, 17, 17), IntArrayMat(0, 2, 3))
           || test_crop(a, IntArrayMat(8, 8, 8), IntArrayMat(24, 24, 24), IntArrayMat(0, 2, 3))

           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(16, 16, 16), IntArrayMat(1, 2, 3))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(17, 17, 17), IntArrayMat(1, 2, 3))
           || test_crop(a, IntArrayMat(8, 8, 8), IntArrayMat(24, 24, 24), IntArrayMat(1, 2, 3))

           || test_crop(a, IntArrayMat(11, 11, 11, 11), IntArrayMat(16, 16, 16, 16), IntArrayMat(0, 1, 2, 3))
           || test_crop(a, IntArrayMat(12, 12, 12, 12), IntArrayMat(17, 17, 17, 17), IntArrayMat(0, 1, 2, 3))
           || test_crop(a, IntArrayMat(8, 8, 8, 8), IntArrayMat(24, 24, 24, 24), IntArrayMat(0, 1, 2, 3))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-7 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-12 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-16 + 1), IntArrayMat(0))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-12 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-16 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-7 + 1), IntArrayMat(1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-16 + 1), IntArrayMat(2))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-7 + 1), IntArrayMat(2))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-12 + 1), IntArrayMat(2))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-16 + 1), IntArrayMat(3))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-7 + 1), IntArrayMat(3))
           || test_crop(a, IntArrayMat(8), IntArrayMat(-12 + 1), IntArrayMat(3))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-7 + 1, -7 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-12 + 1, -12 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-16 + 1, -16 + 1), IntArrayMat(0, 1))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-12 + 1, -12 + 1), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-16 + 1, -16 + 1), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-7 + 1, -7 + 1), IntArrayMat(0, 2))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-12 + 1, -12 + 1), IntArrayMat(0, 3))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-16 + 1, -16 + 1), IntArrayMat(0, 3))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-7 + 1, -7 + 1), IntArrayMat(0, 3))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-16 + 1, -16 + 1), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-7 + 1, -7 + 1), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-12 + 1, -12 + 1), IntArrayMat(1, 2))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-12 + 1, -12 + 1), IntArrayMat(1, 3))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-16 + 1, -16 + 1), IntArrayMat(1, 3))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-7 + 1, -7 + 1), IntArrayMat(1, 3))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-12 + 1, -12 + 1), IntArrayMat(2, 3))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-16 + 1, -16 + 1), IntArrayMat(2, 3))
           || test_crop(a, IntArrayMat(8, 8), IntArrayMat(-7 + 1, -7 + 1), IntArrayMat(2, 3))

           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(-7 + 1, -7 + 1, -7 + 1), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(-12 + 1, -12 + 1, -12 + 1), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(8, 8, 8), IntArrayMat(-16 + 1, -16 + 1, -16 + 1), IntArrayMat(0, 1, 2))

           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(-7 + 1, -7 + 1, -7 + 1), IntArrayMat(0, 1, 3))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(-12 + 1, -12 + 1, -12 + 1), IntArrayMat(0, 1, 3))
           || test_crop(a, IntArrayMat(8, 8, 8), IntArrayMat(-16 + 1, -16 + 1, -16 + 1), IntArrayMat(0, 1, 3))

           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(-7 + 1, -7 + 1, -7 + 1), IntArrayMat(0, 2, 3))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(-12 + 1, -12 + 1, -12 + 1), IntArrayMat(0, 2, 3))
           || test_crop(a, IntArrayMat(8, 8, 8), IntArrayMat(-16 + 1, -16 + 1, -16 + 1), IntArrayMat(0, 2, 3))

           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(-7 + 1, -7 + 1, -7 + 1), IntArrayMat(1, 2, 3))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(-12 + 1, -12 + 1, -12 + 1), IntArrayMat(1, 2, 3))
           || test_crop(a, IntArrayMat(8, 8, 8), IntArrayMat(-16 + 1, -16 + 1, -16 + 1), IntArrayMat(1, 2, 3))

           || test_crop(a, IntArrayMat(11, 11, 11, 11), IntArrayMat(-7 + 1, -7 + 1, -7 + 1, -7 + 1), IntArrayMat(0, 1, 2, 3))
           || test_crop(a, IntArrayMat(12, 12, 12, 12), IntArrayMat(-12 + 1, -12 + 1, -12 + 1, -12 + 1), IntArrayMat(0, 1, 2, 3))
           || test_crop(a, IntArrayMat(8, 8, 8, 8), IntArrayMat(-16 + 1, -16 + 1, -16 + 1, -16 + 1), IntArrayMat(0, 1, 2, 3));
}

static int test_crop_11(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 11, 0, 0, ncnn::Mat(12, 12, 12))
           || test_crop(a, 12, 0, 0, 0, ncnn::Mat(16, 16, 16))
           || test_crop(a, 11, 11, 11, 0, ncnn::Mat(12, 12, 12))
           || test_crop(a, 12, 12, 12, 0, ncnn::Mat(16, 16, 16))
           || test_crop(a, 8, 8, 8, 0, ncnn::Mat(7, 7, 7))

           || test_crop(a, 11, 11, 11, 11, ncnn::Mat(7, 7, 7, 7))
           || test_crop(a, 12, 12, 12, 12, ncnn::Mat(12, 12, 12, 12))
           || test_crop(a, 8, 8, 8, 8, ncnn::Mat(16, 16, 16, 16));
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
           || test_crop_3(RandomMat(32, 32))
           || test_crop_3(RandomMat(30, 36))
           || test_crop_3(RandomMat(31, 31))
           || test_crop_4(RandomMat(32, 32))
           || test_crop_4(RandomMat(30, 36))
           || test_crop_4(RandomMat(31, 31))
           || test_crop_5(RandomMat(32, 32))
           || test_crop_5(RandomMat(30, 36))
           || test_crop_5(RandomMat(31, 31))
           || test_crop_6(RandomMat(32, 32, 32))
           || test_crop_6(RandomMat(30, 30, 36))
           || test_crop_6(RandomMat(31, 31, 31))
           || test_crop_7(RandomMat(32, 32, 32))
           || test_crop_7(RandomMat(30, 30, 36))
           || test_crop_7(RandomMat(31, 31, 31))
           || test_crop_8(RandomMat(32, 32, 32))
           || test_crop_8(RandomMat(30, 30, 36))
           || test_crop_8(RandomMat(31, 31, 31))
           || test_crop_9(RandomMat(32, 32, 32, 32))
           || test_crop_9(RandomMat(30, 30, 30, 36))
           || test_crop_9(RandomMat(31, 31, 31, 31))
           || test_crop_10(RandomMat(32, 32, 32, 32))
           || test_crop_10(RandomMat(30, 30, 30, 36))
           || test_crop_10(RandomMat(31, 31, 31, 31))
           || test_crop_11(RandomMat(32, 32, 32, 32))
           || test_crop_11(RandomMat(30, 30, 30, 36))
           || test_crop_11(RandomMat(31, 31, 31, 31));
}
