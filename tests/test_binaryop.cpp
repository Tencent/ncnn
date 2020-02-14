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

#include "layer/binaryop.h"

#define OP_TYPE_MAX 9

static int test_binaryop(const ncnn::Mat& _a, const ncnn::Mat& _b, int op_type)
{
    ncnn::Mat a = _a;
    ncnn::Mat b = _b;
    if (op_type == 6)
    {
        // value must be positive for pow
        Randomize(a, 0.001f, 2.f);
        Randomize(b, 0.001f, 2.f);
    }

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0);// with_scalar
    pd.set(2, 0.f);// b

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;

    std::vector<ncnn::Mat> ab(2);
    ab[0] = a;
    ab[1] = b;

    int ret = test_layer<ncnn::BinaryOp>("BinaryOp", pd, weights, opt, ab);
    if (ret != 0)
    {
        fprintf(stderr, "test_binaryop failed a.dims=%d a=(%d %d %d) b.dims=%d b=(%d %d %d) op_type=%d\n", a.dims, a.w, a.h, a.c, b.dims, b.w, b.h, b.c, op_type);
    }

    return ret;
}

static int test_binaryop(const ncnn::Mat& _a, float b, int op_type)
{
    ncnn::Mat a = _a;
    if (op_type == 6)
    {
        // value must be positive for pow
        Randomize(a, 0.001f, 2.f);
        b = RandomFloat(0.001f, 2.f);
    }

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 1);// with_scalar
    pd.set(2, b);// b

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = false;
    opt.use_int8_arithmetic = false;

    int ret = test_layer<ncnn::BinaryOp>("BinaryOp", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_binaryop failed a.dims=%d a=(%d %d %d) b=%f op_type=%d\n", a.dims, a.w, a.h, a.c, b, op_type);
    }

    return ret;
}

// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting

static int test_binaryop_1()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(1), 1.f, op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_2()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(1), RandomMat(1), op_type)
            || test_binaryop(RandomMat(1), RandomMat(4), op_type)
            || test_binaryop(RandomMat(1), RandomMat(8), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_3()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(1), RandomMat(2, 3), op_type)
            || test_binaryop(RandomMat(1), RandomMat(2, 4), op_type)
            || test_binaryop(RandomMat(1), RandomMat(2, 8), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_4()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(1), RandomMat(2, 3, 2), op_type)
            || test_binaryop(RandomMat(1), RandomMat(2, 3, 4), op_type)
            || test_binaryop(RandomMat(1), RandomMat(2, 3, 8), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_5()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2), 1.f, op_type)
            || test_binaryop(RandomMat(4), 1.f, op_type)
            || test_binaryop(RandomMat(8), 1.f, op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_6()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2), RandomMat(1), op_type)
            || test_binaryop(RandomMat(4), RandomMat(1), op_type)
            || test_binaryop(RandomMat(8), RandomMat(1), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_7()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2), RandomMat(2), op_type)
            || test_binaryop(RandomMat(4), RandomMat(4), op_type)
            || test_binaryop(RandomMat(8), RandomMat(8), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_8()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(3), RandomMat(2, 3), op_type)
            || test_binaryop(RandomMat(4), RandomMat(2, 4), op_type)
            || test_binaryop(RandomMat(8), RandomMat(2, 8), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_9()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2), RandomMat(2, 3, 2), op_type)
            || test_binaryop(RandomMat(4), RandomMat(2, 3, 4), op_type)
            || test_binaryop(RandomMat(8), RandomMat(2, 3, 8), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_10()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2, 3), 1.f, op_type)
            || test_binaryop(RandomMat(2, 4), 1.f, op_type)
            || test_binaryop(RandomMat(2, 8), 1.f, op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_11()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2, 3), RandomMat(1), op_type)
            || test_binaryop(RandomMat(2, 4), RandomMat(1), op_type)
            || test_binaryop(RandomMat(2, 8), RandomMat(1), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_12()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2, 3), RandomMat(3), op_type)
            || test_binaryop(RandomMat(2, 4), RandomMat(4), op_type)
            || test_binaryop(RandomMat(2, 8), RandomMat(8), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_13()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2, 3), RandomMat(2, 3), op_type)
            || test_binaryop(RandomMat(2, 4), RandomMat(2, 4), op_type)
            || test_binaryop(RandomMat(2, 8), RandomMat(2, 8), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_14()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(3, 2), RandomMat(2, 3, 2), op_type)
            || test_binaryop(RandomMat(3, 4), RandomMat(2, 3, 4), op_type)
            || test_binaryop(RandomMat(3, 8), RandomMat(2, 3, 8), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_15()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2, 3, 2), 1.f, op_type)
            || test_binaryop(RandomMat(2, 3, 4), 1.f, op_type)
            || test_binaryop(RandomMat(2, 3, 8), 1.f, op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_16()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2, 3, 2), RandomMat(1), op_type)
            || test_binaryop(RandomMat(2, 3, 4), RandomMat(1), op_type)
            || test_binaryop(RandomMat(2, 3, 8), RandomMat(1), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_17()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2, 3, 2), RandomMat(2), op_type)
            || test_binaryop(RandomMat(2, 3, 4), RandomMat(4), op_type)
            || test_binaryop(RandomMat(2, 3, 8), RandomMat(8), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_18()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2, 3, 2), RandomMat(3, 2), op_type)
            || test_binaryop(RandomMat(2, 3, 4), RandomMat(3, 4), op_type)
            || test_binaryop(RandomMat(2, 3, 8), RandomMat(3, 8), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_19()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2, 3, 2), RandomMat(2, 3, 2), op_type)
            || test_binaryop(RandomMat(2, 3, 4), RandomMat(2, 3, 4), op_type)
            || test_binaryop(RandomMat(2, 3, 8), RandomMat(2, 3, 8), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_s1()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2, 3, 2), RandomMat(1, 1, 2), op_type)
            || test_binaryop(RandomMat(2, 3, 4), RandomMat(1, 1, 4), op_type)
            || test_binaryop(RandomMat(2, 3, 8), RandomMat(1, 1, 8), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

static int test_binaryop_s2()
{
    for (int op_type=0; op_type<OP_TYPE_MAX; op_type++)
    {
        int ret = 0
            || test_binaryop(RandomMat(2, 3, 2), RandomMat(2, 3, 1), op_type)
            || test_binaryop(RandomMat(2, 3, 4), RandomMat(2, 3, 1), op_type)
            || test_binaryop(RandomMat(2, 3, 8), RandomMat(2, 3, 1), op_type)
            ;

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
        || test_binaryop_1()
        || test_binaryop_2()
        || test_binaryop_3()
        || test_binaryop_4()
        || test_binaryop_5()
        || test_binaryop_6()
        || test_binaryop_7()
        || test_binaryop_8()
        || test_binaryop_9()
        || test_binaryop_10()
        || test_binaryop_11()
        || test_binaryop_12()
        || test_binaryop_13()
        || test_binaryop_14()
        || test_binaryop_15()
        || test_binaryop_16()
        || test_binaryop_17()
        || test_binaryop_18()
        || test_binaryop_19()
        || test_binaryop_s1()
        || test_binaryop_s2()
        ;
}
