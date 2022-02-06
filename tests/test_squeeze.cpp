// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "layer/squeeze.h"
#include "testutil.h"

static int test_squeeze(const ncnn::Mat& a, int squeeze_w, int squeeze_h, int squeeze_c)
{
    ncnn::ParamDict pd;
    pd.set(0, squeeze_w);
    pd.set(1, squeeze_h);
    pd.set(2, squeeze_c);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Squeeze>("Squeeze", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_squeeze failed a.dims=%d a=(%d %d %d) squeeze_w=%d squeeze_h=%d squeeze_c=%d\n", a.dims, a.w, a.h, a.c, squeeze_w, squeeze_h, squeeze_c);
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

static int test_squeeze_axes(const ncnn::Mat& a, const ncnn::Mat& axes)
{
    ncnn::ParamDict pd;
    pd.set(3, axes);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Squeeze>("Squeeze", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_squeeze_axes failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
        fprintf(stderr, " axes=");
        print_int_array(axes);
        fprintf(stderr, "\n");
    }

    return ret;
}

static int test_squeeze_0()
{
    ncnn::Mat as[12];
    as[0] = RandomMat(3, 12, 16);
    as[1] = RandomMat(3, 1, 16);
    as[2] = RandomMat(1, 33, 15);
    as[3] = RandomMat(1, 14, 1);
    as[4] = RandomMat(12, 13, 1);
    as[5] = RandomMat(1, 1, 1);
    as[6] = RandomMat(14, 16);
    as[7] = RandomMat(1, 14);
    as[8] = RandomMat(11, 1);
    as[9] = RandomMat(1, 1);
    as[10] = RandomMat(120);
    as[11] = RandomMat(1);

    for (int i = 0; i < 12; i++)
    {
        const ncnn::Mat& a = as[i];
        int ret = 0
                  || test_squeeze(a, 0, 0, 0)
                  || test_squeeze(a, 0, 0, 1)
                  || test_squeeze(a, 0, 1, 0)
                  || test_squeeze(a, 0, 1, 1)
                  || test_squeeze(a, 1, 0, 0)
                  || test_squeeze(a, 1, 0, 1)
                  || test_squeeze(a, 1, 1, 0)
                  || test_squeeze(a, 1, 1, 1)

                  || test_squeeze_axes(a, IntArrayMat(0))
                  || test_squeeze_axes(a, IntArrayMat(1))
                  || test_squeeze_axes(a, IntArrayMat(2))
                  || test_squeeze_axes(a, IntArrayMat(0, 1))
                  || test_squeeze_axes(a, IntArrayMat(0, 2))
                  || test_squeeze_axes(a, IntArrayMat(1, 2))
                  || test_squeeze_axes(a, IntArrayMat(0, 1, 2));

        if (ret != 0)
            return ret;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_squeeze_0();
}
