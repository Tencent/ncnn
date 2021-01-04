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

static int test_crop(const ncnn::Mat& a, int woffset, int hoffset, int coffset, const ncnn::Mat& ref)
{
    ncnn::ParamDict pd;
    pd.set(0, woffset);
    pd.set(1, hoffset);
    pd.set(2, coffset);
    pd.set(3, 0); // outw
    pd.set(4, 0); // outh
    pd.set(5, 0); // outc
    pd.set(6, 0); // woffset2
    pd.set(7, 0); // hoffset2
    pd.set(8, 0); // coffset2

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> ab(2);
    ab[0] = a;
    ab[1] = ref;

    int ret = test_layer<ncnn::Crop>("Crop", pd, weights, ab);
    if (ret != 0)
    {
        fprintf(stderr, "test_crop failed a.dims=%d a=(%d %d %d) woffset=%d hoffset=%d coffset=%d ref.dims=%d ref=(%d %d %d)\n", a.dims, a.w, a.h, a.c, woffset, hoffset, coffset, ref.dims, ref.w, ref.h, ref.c);
    }

    return ret;
}

static int test_crop_0(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, -233, 0, 0, 0, 0, 0)
           || test_crop(a, 11, 0, 0, -233, 0, 0, 0, 0, 0)
           || test_crop(a, 11, 0, 0, 17, 0, 0, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 28, 0, 0, 0, 0, 0)
           || test_crop(a, 24, 0, 0, 24, 0, 0, 0, 0, 0)
           || test_crop(a, 11, 0, 0, -233, 0, 0, 28, 0, 0)
           || test_crop(a, 12, 0, 0, -233, 0, 0, 24, 0, 0)
           || test_crop(a, 24, 0, 0, -233, 0, 0, 17, 0, 0);
}

static int test_crop_1(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, IntArrayMat(12), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(24), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(11), IntArrayMat(11 + 24), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(12 + 17), IntArrayMat(0))
           || test_crop(a, IntArrayMat(24), IntArrayMat(24 + 28), IntArrayMat(0))
           || test_crop(a, IntArrayMat(11), IntArrayMat(-17 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-28 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(24), IntArrayMat(-24 + 1), IntArrayMat(0));
}

static int test_crop_2(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, ncnn::Mat(27))

           || test_crop(a, 11, 0, 0, ncnn::Mat(17))
           || test_crop(a, 12, 0, 0, ncnn::Mat(28))
           || test_crop(a, 24, 0, 0, ncnn::Mat(24));
}

static int test_crop_3(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 11, 0, 0, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 11, 0, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 0, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 12, 12, 0, -233, -233, 0, 0, 0, 0)
           || test_crop(a, 16, 16, 0, -233, -233, 0, 0, 0, 0)

           || test_crop(a, 11, 0, 0, 17, -233, 0, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 28, -233, 0, 0, 0, 0)
           || test_crop(a, 16, 0, 0, 24, -233, 0, 0, 0, 0)

           || test_crop(a, 0, 11, 0, -233, 28, 0, 0, 0, 0)
           || test_crop(a, 0, 12, 0, -233, 24, 0, 0, 0, 0)
           || test_crop(a, 0, 16, 0, -233, 17, 0, 0, 0, 0)

           || test_crop(a, 11, 11, 0, 24, 24, 0, 0, 0, 0)
           || test_crop(a, 12, 12, 0, 17, 17, 0, 0, 0, 0)
           || test_crop(a, 16, 16, 0, 28, 28, 0, 0, 0, 0)

           || test_crop(a, 11, 0, 0, -233, -233, 0, 17, 0, 0)
           || test_crop(a, 12, 0, 0, -233, -233, 0, 28, 0, 0)
           || test_crop(a, 16, 0, 0, -233, -233, 0, 24, 0, 0)

           || test_crop(a, 0, 11, 0, -233, -233, 0, 0, 28, 0)
           || test_crop(a, 0, 12, 0, -233, -233, 0, 0, 24, 0)
           || test_crop(a, 0, 16, 0, -233, -233, 0, 0, 17, 0)

           || test_crop(a, 11, 11, 0, -233, -233, 0, 24, 24, 0)
           || test_crop(a, 12, 12, 0, -233, -233, 0, 17, 17, 0)
           || test_crop(a, 16, 16, 0, -233, -233, 0, 28, 28, 0);
}

static int test_crop_4(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, IntArrayMat(12), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(16), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(16), IntArrayMat(-233), IntArrayMat(1))
           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-233, -233), IntArrayMat(0, 1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(28), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(24), IntArrayMat(0))
           || test_crop(a, IntArrayMat(16), IntArrayMat(17), IntArrayMat(0))

           || test_crop(a, IntArrayMat(11), IntArrayMat(24), IntArrayMat(1))
           || test_crop(a, IntArrayMat(12), IntArrayMat(17), IntArrayMat(1))
           || test_crop(a, IntArrayMat(16), IntArrayMat(28), IntArrayMat(1))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(17, 17), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(28, 28), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(16, 16), IntArrayMat(24, 24), IntArrayMat(0, 1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-24 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-17 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(16), IntArrayMat(-28 + 1), IntArrayMat(0))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-17 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-28 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(16), IntArrayMat(-24 + 1), IntArrayMat(1))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-28 + 1, -28 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-24 + 1, -24 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(16, 16), IntArrayMat(-17 + 1, -17 + 1), IntArrayMat(0, 1));
}

static int test_crop_5(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 12, 0, ncnn::Mat(24, 17))
           || test_crop(a, 11, 0, 0, ncnn::Mat(17, 27))

           || test_crop(a, 11, 11, 0, ncnn::Mat(28, 28))
           || test_crop(a, 12, 12, 0, ncnn::Mat(24, 24))
           || test_crop(a, 16, 16, 0, ncnn::Mat(17, 17));
}

static int test_crop_6(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 0, 0, -233, -233, -233, 0, 0, 0)
           || test_crop(a, 12, 0, 0, -233, -233, -233, 0, 0, 0)
           || test_crop(a, 0, 11, 0, -233, -233, -233, 0, 0, 0)
           || test_crop(a, 0, 16, 0, -233, -233, -233, 0, 0, 0)
           || test_crop(a, 0, 0, 12, -233, -233, -233, 0, 0, 0)
           || test_crop(a, 11, 11, 0, -233, -233, -233, 0, 0, 0)
           || test_crop(a, 16, 16, 0, -233, -233, -233, 0, 0, 0)
           || test_crop(a, 0, 12, 12, -233, -233, -233, 0, 0, 0)
           || test_crop(a, 11, 0, 11, -233, -233, -233, 0, 0, 0)
           || test_crop(a, 16, 0, 16, -233, -233, -233, 0, 0, 0)
           || test_crop(a, 12, 12, 12, -233, -233, -233, 0, 0, 0)

           || test_crop(a, 11, 0, 0, 28, -233, -233, 0, 0, 0)
           || test_crop(a, 12, 0, 0, 24, -233, -233, 0, 0, 0)
           || test_crop(a, 16, 0, 0, 17, -233, -233, 0, 0, 0)

           || test_crop(a, 0, 11, 0, -233, 24, -233, 0, 0, 0)
           || test_crop(a, 0, 12, 0, -233, 17, -233, 0, 0, 0)
           || test_crop(a, 0, 16, 0, -233, 28, -233, 0, 0, 0)

           || test_crop(a, 0, 0, 11, -233, -233, 17, 0, 0, 0)
           || test_crop(a, 0, 0, 12, -233, -233, 28, 0, 0, 0)
           || test_crop(a, 0, 0, 16, -233, -233, 24, 0, 0, 0)

           || test_crop(a, 11, 11, 0, 17, 17, -233, 0, 0, 0)
           || test_crop(a, 12, 12, 0, 28, 28, -233, 0, 0, 0)
           || test_crop(a, 16, 16, 0, 24, 24, -233, 0, 0, 0)

           || test_crop(a, 0, 11, 11, -233, 28, 28, 0, 0, 0)
           || test_crop(a, 0, 12, 12, -233, 24, 24, 0, 0, 0)
           || test_crop(a, 0, 16, 16, -233, 17, 17, 0, 0, 0)

           || test_crop(a, 11, 0, 11, 24, -233, 24, 0, 0, 0)
           || test_crop(a, 12, 0, 12, 17, -233, 17, 0, 0, 0)
           || test_crop(a, 16, 0, 16, 24, -233, 24, 0, 0, 0)

           || test_crop(a, 11, 11, 11, 28, 28, 28, 0, 0, 0)
           || test_crop(a, 12, 12, 12, 24, 24, 24, 0, 0, 0)
           || test_crop(a, 16, 16, 16, 17, 17, 17, 0, 0, 0)

           || test_crop(a, 11, 0, 0, -233, -233, -233, 24, 0, 0)
           || test_crop(a, 12, 0, 0, -233, -233, -233, 17, 0, 0)
           || test_crop(a, 16, 0, 0, -233, -233, -233, 28, 0, 0)

           || test_crop(a, 0, 11, 0, -233, -233, -233, 0, 17, 0)
           || test_crop(a, 0, 12, 0, -233, -233, -233, 0, 28, 0)
           || test_crop(a, 0, 16, 0, -233, -233, -233, 0, 24, 0)

           || test_crop(a, 0, 0, 11, -233, -233, -233, 0, 0, 28)
           || test_crop(a, 0, 0, 12, -233, -233, -233, 0, 0, 24)
           || test_crop(a, 0, 0, 16, -233, -233, -233, 0, 0, 17)

           || test_crop(a, 11, 11, 0, -233, -233, -233, 24, 24, 0)
           || test_crop(a, 12, 12, 0, -233, -233, -233, 17, 17, 0)
           || test_crop(a, 16, 16, 0, -233, -233, -233, 28, 28, 0)

           || test_crop(a, 0, 11, 11, -233, -233, -233, 0, 17, 17)
           || test_crop(a, 0, 12, 12, -233, -233, -233, 0, 28, 28)
           || test_crop(a, 0, 16, 16, -233, -233, -233, 0, 24, 24)

           || test_crop(a, 11, 0, 11, -233, -233, -233, 28, 0, 28)
           || test_crop(a, 12, 0, 12, -233, -233, -233, 24, 0, 24)
           || test_crop(a, 16, 0, 16, -233, -233, -233, 17, 0, 17)

           || test_crop(a, 11, 11, 11, -233, -233, -233, 24, 24, 24)
           || test_crop(a, 12, 12, 12, -233, -233, -233, 17, 17, 17)
           || test_crop(a, 16, 16, 16, -233, -233, -233, 28, 28, 28);
}

static int test_crop_7(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, IntArrayMat(11), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(16), IntArrayMat(-233), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-233), IntArrayMat(1))
           || test_crop(a, IntArrayMat(11), IntArrayMat(-233), IntArrayMat(2))
           || test_crop(a, IntArrayMat(16), IntArrayMat(-233), IntArrayMat(2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-233, -233), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-233, -233), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(16, 16), IntArrayMat(-233, -233), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-233, -233), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(-233, -233, -233), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(16, 16, 16), IntArrayMat(-233, -233, -233), IntArrayMat(0, 1, 2))

           || test_crop(a, IntArrayMat(11), IntArrayMat(24), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(17), IntArrayMat(0))
           || test_crop(a, IntArrayMat(16), IntArrayMat(28), IntArrayMat(0))

           || test_crop(a, IntArrayMat(11), IntArrayMat(17), IntArrayMat(1))
           || test_crop(a, IntArrayMat(12), IntArrayMat(28), IntArrayMat(1))
           || test_crop(a, IntArrayMat(16), IntArrayMat(24), IntArrayMat(1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(28), IntArrayMat(2))
           || test_crop(a, IntArrayMat(12), IntArrayMat(24), IntArrayMat(2))
           || test_crop(a, IntArrayMat(16), IntArrayMat(17), IntArrayMat(2))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(24, 24), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(17, 17), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(16, 16), IntArrayMat(28, 28), IntArrayMat(0, 1))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(17, 17), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(28, 28), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(16, 16), IntArrayMat(24, 24), IntArrayMat(0, 2))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(28, 28), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(24, 24), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(16, 16), IntArrayMat(17, 17), IntArrayMat(1, 2))

           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(24, 24, 24), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(17, 17, 17), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(16, 16, 16), IntArrayMat(28, 28, 28), IntArrayMat(0, 1, 2))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-17 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-28 + 1), IntArrayMat(0))
           || test_crop(a, IntArrayMat(16), IntArrayMat(-24 + 1), IntArrayMat(0))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-28 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-24 + 1), IntArrayMat(1))
           || test_crop(a, IntArrayMat(16), IntArrayMat(-17 + 1), IntArrayMat(1))

           || test_crop(a, IntArrayMat(11), IntArrayMat(-24 + 1), IntArrayMat(2))
           || test_crop(a, IntArrayMat(12), IntArrayMat(-17 + 1), IntArrayMat(2))
           || test_crop(a, IntArrayMat(16), IntArrayMat(-28 + 1), IntArrayMat(2))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-17 + 1, -17 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-28 + 1, -28 + 1), IntArrayMat(0, 1))
           || test_crop(a, IntArrayMat(16, 16), IntArrayMat(-24 + 1, -24 + 1), IntArrayMat(0, 1))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-28 + 1, -28 + 1), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-24 + 1, -24 + 1), IntArrayMat(0, 2))
           || test_crop(a, IntArrayMat(16, 16), IntArrayMat(-17 + 1, -17 + 1), IntArrayMat(0, 2))

           || test_crop(a, IntArrayMat(11, 11), IntArrayMat(-24 + 1, -24 + 1), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(12, 12), IntArrayMat(-17 + 1, -17 + 1), IntArrayMat(1, 2))
           || test_crop(a, IntArrayMat(16, 16), IntArrayMat(-28 + 1, -28 + 1), IntArrayMat(1, 2))

           || test_crop(a, IntArrayMat(11, 11, 11), IntArrayMat(-17 + 1, -17 + 1, -17 + 1), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(12, 12, 12), IntArrayMat(-28 + 1, -28 + 1, -28 + 1), IntArrayMat(0, 1, 2))
           || test_crop(a, IntArrayMat(16, 16, 16), IntArrayMat(-24 + 1, -24 + 1, -24 + 1), IntArrayMat(0, 1, 2));
}

static int test_crop_8(const ncnn::Mat& a)
{
    return 0
           || test_crop(a, 0, 11, 0, ncnn::Mat(28, 28))
           || test_crop(a, 12, 0, 0, ncnn::Mat(24, 24))
           || test_crop(a, 11, 11, 0, ncnn::Mat(28, 28))
           || test_crop(a, 12, 12, 0, ncnn::Mat(24, 24))
           || test_crop(a, 16, 16, 0, ncnn::Mat(17, 17))

           || test_crop(a, 11, 11, 11, ncnn::Mat(17, 17, 17))
           || test_crop(a, 12, 12, 12, ncnn::Mat(28, 28, 28))
           || test_crop(a, 16, 16, 16, ncnn::Mat(24, 24, 24));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_crop_0(RandomMat(128))
           || test_crop_0(RandomMat(124))
           || test_crop_0(RandomMat(127))
           || test_crop_1(RandomMat(128))
           || test_crop_1(RandomMat(124))
           || test_crop_1(RandomMat(127))
           || test_crop_2(RandomMat(128))
           || test_crop_2(RandomMat(124))
           || test_crop_2(RandomMat(127))
           || test_crop_3(RandomMat(64, 64))
           || test_crop_3(RandomMat(60, 60))
           || test_crop_3(RandomMat(63, 63))
           || test_crop_4(RandomMat(64, 64))
           || test_crop_4(RandomMat(60, 60))
           || test_crop_4(RandomMat(63, 63))
           || test_crop_5(RandomMat(64, 64))
           || test_crop_5(RandomMat(60, 60))
           || test_crop_5(RandomMat(63, 63))
           || test_crop_6(RandomMat(64, 64, 64))
           || test_crop_6(RandomMat(60, 60, 60))
           || test_crop_6(RandomMat(63, 63, 63))
           || test_crop_7(RandomMat(64, 64, 64))
           || test_crop_7(RandomMat(60, 60, 60))
           || test_crop_7(RandomMat(63, 63, 63))
           || test_crop_8(RandomMat(64, 64, 64))
           || test_crop_8(RandomMat(60, 60, 60))
           || test_crop_8(RandomMat(63, 63, 63));
}
