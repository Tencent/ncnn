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

#include "layer/binaryop.h"
#include "testutil.h"

#define OP_TYPE_MAX 12

static int op_type = 0;

static int test_binaryop(const ncnn::Mat& _a, const ncnn::Mat& _b)
{
    ncnn::Mat a = _a;
    ncnn::Mat b = _b;
    if (op_type == 6 || op_type == 9)
    {
        // value must be positive for pow/rpow
        a = a.clone();
        b = b.clone();
        Randomize(a, 0.001f, 2.f);
        Randomize(b, 0.001f, 2.f);
    }
    if (op_type == 3 || op_type == 8)
    {
        // value must be positive for div/rdiv
        a = a.clone();
        b = b.clone();
        Randomize(a, 0.1f, 10.f);
        Randomize(b, 0.1f, 10.f);
    }

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0);   // with_scalar
    pd.set(2, 0.f); // b

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> ab(2);
    ab[0] = a;
    ab[1] = b;

    int ret = test_layer<ncnn::BinaryOp>("BinaryOp", pd, weights, ab);
    if (ret != 0)
    {
        fprintf(stderr, "test_binaryop failed a.dims=%d a=(%d %d %d %d) b.dims=%d b=(%d %d %d %d) op_type=%d\n", a.dims, a.w, a.h, a.d, a.c, b.dims, b.w, b.h, b.d, b.c, op_type);
    }

    return ret;
}

static int test_binaryop(const ncnn::Mat& _a, float b)
{
    ncnn::Mat a = _a;
    if (op_type == 6 || op_type == 9)
    {
        // value must be positive for pow
        Randomize(a, 0.001f, 2.f);
        b = RandomFloat(0.001f, 2.f);
    }
    if (op_type == 3 || op_type == 8)
    {
        // value must be positive for div/rdiv
        a = a.clone();
        Randomize(a, 0.1f, 10.f);
    }

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 1); // with_scalar
    pd.set(2, b); // b

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::BinaryOp>("BinaryOp", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_binaryop failed a.dims=%d a=(%d %d %d %d) b=%f op_type=%d\n", a.dims, a.w, a.h, a.d, a.c, b, op_type);
    }

    return ret;
}

static int test_binaryop_1()
{
    ncnn::Mat a[] = {
        RandomMat(31),
        RandomMat(28),
        RandomMat(24),
        RandomMat(32),
        RandomMat(13, 31),
        RandomMat(14, 28),
        RandomMat(15, 24),
        RandomMat(16, 32),
        RandomMat(7, 3, 31),
        RandomMat(6, 4, 28),
        RandomMat(5, 5, 24),
        RandomMat(4, 6, 32),
        RandomMat(2, 7, 3, 31),
        RandomMat(3, 6, 4, 28),
        RandomMat(4, 5, 5, 24),
        RandomMat(5, 4, 6, 32)
    };

    ncnn::Mat b[] = {
        RandomMat(1),
        RandomMat(1, 1),
        RandomMat(1, 1, 1),
        RandomMat(1, 1, 1, 1)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        for (int j = 0; j < sizeof(b) / sizeof(b[0]); j++)
        {
            int ret = test_binaryop(a[i], b[j]) || test_binaryop(b[j], a[i]);
            if (ret != 0)
                return ret;
        }

        int ret = test_binaryop(a[i], 0.2f);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_2()
{
    ncnn::Mat a[] = {
        RandomMat(31),
        RandomMat(28),
        RandomMat(24),
        RandomMat(32),
        RandomMat(13, 31),
        RandomMat(14, 28),
        RandomMat(15, 24),
        RandomMat(16, 32),
        RandomMat(7, 3, 31),
        RandomMat(6, 4, 28),
        RandomMat(5, 5, 24),
        RandomMat(4, 6, 32),
        RandomMat(2, 7, 3, 31),
        RandomMat(3, 6, 4, 28),
        RandomMat(4, 5, 5, 24),
        RandomMat(5, 4, 6, 32)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        ncnn::Mat b;
        b.create_like(a[i]);
        Randomize(b);

        int ret = test_binaryop(a[i], b) || test_binaryop(b, a[i]);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_3()
{
    ncnn::Mat a[] = {
        RandomMat(13, 31),
        RandomMat(14, 28),
        RandomMat(15, 24),
        RandomMat(16, 32)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        ncnn::Mat b0(a[i].h);
        ncnn::Mat b1(1, a[i].h);
        Randomize(b0);
        Randomize(b1);

        int ret = test_binaryop(a[i], b0) || test_binaryop(b0, a[i])
                  || test_binaryop(a[i], b1) || test_binaryop(b1, a[i]);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_4()
{
    ncnn::Mat a[] = {
        RandomMat(7, 3, 31),
        RandomMat(6, 4, 28),
        RandomMat(5, 5, 24),
        RandomMat(4, 6, 32)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        ncnn::Mat b0(a[i].c);
        ncnn::Mat b1(1, 1, a[i].c);
        ncnn::Mat b2(a[i].h, a[i].c);
        ncnn::Mat b3(1, a[i].h, a[i].c);
        Randomize(b0);
        Randomize(b1);
        Randomize(b2);
        Randomize(b3);

        int ret = test_binaryop(a[i], b0) || test_binaryop(b0, a[i])
                  || test_binaryop(a[i], b1) || test_binaryop(b1, a[i])
                  || test_binaryop(a[i], b2) || test_binaryop(b2, a[i])
                  || test_binaryop(a[i], b3) || test_binaryop(b3, a[i]);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_5()
{
    ncnn::Mat a[] = {
        RandomMat(2, 7, 3, 31),
        RandomMat(3, 6, 4, 28),
        RandomMat(4, 5, 5, 24),
        RandomMat(5, 4, 6, 32)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        ncnn::Mat b0(a[i].c);
        ncnn::Mat b1(1, 1, 1, a[i].c);
        ncnn::Mat b2(a[i].d, a[i].c);
        ncnn::Mat b3(1, 1, a[i].d, a[i].c);
        ncnn::Mat b4(a[i].h, a[i].d, a[i].c);
        ncnn::Mat b5(1, a[i].h, a[i].d, a[i].c);
        Randomize(b0);
        Randomize(b1);
        Randomize(b2);
        Randomize(b3);
        Randomize(b4);
        Randomize(b5);

        int ret = test_binaryop(a[i], b0) || test_binaryop(b0, a[i])
                  || test_binaryop(a[i], b1) || test_binaryop(b1, a[i])
                  || test_binaryop(a[i], b2) || test_binaryop(b2, a[i])
                  || test_binaryop(a[i], b3) || test_binaryop(b3, a[i])
                  || test_binaryop(a[i], b4) || test_binaryop(b4, a[i])
                  || test_binaryop(a[i], b5) || test_binaryop(b5, a[i]);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_6()
{
    ncnn::Mat a[] = {
        RandomMat(13, 31),
        RandomMat(14, 28),
        RandomMat(15, 24),
        RandomMat(16, 32)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        ncnn::Mat b0(a[i].w, 1);
        Randomize(b0);

        int ret = test_binaryop(a[i], b0) || test_binaryop(b0, a[i]);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_7()
{
    ncnn::Mat a[] = {
        RandomMat(7, 3, 31),
        RandomMat(6, 4, 28),
        RandomMat(5, 5, 24),
        RandomMat(4, 6, 32)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        ncnn::Mat b0(a[i].w, 1, 1);
        ncnn::Mat b1(a[i].w, a[i].h, 1);
        Randomize(b0);
        Randomize(b1);

        int ret = test_binaryop(a[i], b0) || test_binaryop(b0, a[i])
                  || test_binaryop(a[i], b1) || test_binaryop(b1, a[i]);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_8()
{
    ncnn::Mat a[] = {
        RandomMat(2, 7, 3, 31),
        RandomMat(3, 6, 4, 28),
        RandomMat(4, 5, 5, 24),
        RandomMat(5, 4, 6, 32)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        ncnn::Mat b0(a[i].w, 1, 1, 1);
        ncnn::Mat b1(a[i].w, a[i].h, 1, 1);
        ncnn::Mat b2(a[i].w, a[i].h, a[i].d, 1);
        Randomize(b0);
        Randomize(b1);
        Randomize(b2);

        int ret = test_binaryop(a[i], b0) || test_binaryop(b0, a[i])
                  || test_binaryop(a[i], b1) || test_binaryop(b1, a[i])
                  || test_binaryop(a[i], b2) || test_binaryop(b2, a[i]);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_9()
{
    ncnn::Mat a[] = {
        RandomMat(7, 3, 31),
        RandomMat(6, 4, 28),
        RandomMat(5, 5, 24),
        RandomMat(4, 6, 32)
    };

    for (int i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        ncnn::Mat b0(a[i].w, 1, a[i].c);
        Randomize(b0);

        int ret = test_binaryop(a[i], b0) || test_binaryop(b0, a[i]);
        if (ret != 0)
            return ret;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    for (op_type = 0; op_type < 3; op_type++)
    {
        int ret = 0
                  || test_binaryop_1()
                  || test_binaryop_2()
                  || test_binaryop_3()
                  || test_binaryop_4()
                  || test_binaryop_5()
                  || test_binaryop_6()
                  || test_binaryop_7()
                  || test_binaryop_8()
                  || test_binaryop_9();

        if (ret != 0)
            return ret;
    }

    return 0;
}
