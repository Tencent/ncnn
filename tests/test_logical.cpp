// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int op_type = 0;

static int test_logical(const ncnn::Mat& _a, const ncnn::Mat& _b, int flag)
{
    ncnn::Mat a = _a;
    ncnn::Mat b = _b;

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0);
    pd.set(2, 0);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> ab(2);
    ab[0] = a;
    ab[1] = b;

    int ret = test_layer("Logical", pd, weights, ab, 1, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_logical failed a.dims=%d a=(%d %d %d %d) b.dims=%d b=(%d %d %d %d) op_type=%d\n", a.dims, a.w, a.h, a.d, a.c, b.dims, b.w, b.h, b.d, b.c, op_type);
    }

    return ret;
}

static int test_logical(const ncnn::Mat& _a, int b, int flag)
{
    ncnn::Mat a = _a;

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 1);
    pd.set(2, b);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Logical", pd, weights, a, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_logical failed a.dims=%d a=(%d %d %d %d) b=%d op_type=%d\n", a.dims, a.w, a.h, a.d, a.c, b, op_type);
    }

    return ret;
}

static int test_logical_0()
{
    return 0
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 3, 2), 0)
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 3, 2), 0)
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 3, 2), 0);
}

static int test_logical_1()
{
    return 0
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 3, 2), 0)
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 3, 2), 0)
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 3, 2), 0);
}

static int test_logical_2()
{
    return 0
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 3, 2), 0)
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 3, 2), 0)
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 3, 2), 0);
}

static int test_logical_3()
{
    return 0
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 3, 2), 0)
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 3, 2), 0)
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 3, 2), 0);
}

static int test_logical_scalar()
{
    return 0
           || test_logical(RandomBoolMat(4, 3, 2), 0, 0)
           || test_logical(RandomBoolMat(4, 3, 2), 1, 0)
           || test_logical(RandomBoolMat(4, 3, 2), 0, 0)
           || test_logical(RandomBoolMat(4, 3, 2), 1, 0);
}

static int test_logical_broadcast()
{
    return 0
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 1, 1), 0)
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(1, 3, 1), 0)
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(1, 1, 2), 0)
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(1, 1, 1), 0)
           || test_logical(RandomBoolMat(4, 3, 2), RandomBoolMat(4, 3, 2), 0);
}

int main()
{
    SRAND(7767517);

    op_type = 0;
    printf("test_logical NOT\n");
    if (test_logical_scalar() != 0)
        return -1;

    op_type = 1;
    printf("test_logical AND\n");
    if (test_logical_1() != 0)
        return -1;
    if (test_logical_scalar() != 0)
        return -1;
    if (test_logical_broadcast() != 0)
        return -1;

    op_type = 2;
    printf("test_logical OR\n");
    if (test_logical_2() != 0)
        return -1;
    if (test_logical_scalar() != 0)
        return -1;
    if (test_logical_broadcast() != 0)
        return -1;

    op_type = 3;
    printf("test_logical XOR\n");
    if (test_logical_3() != 0)
        return -1;
    if (test_logical_scalar() != 0)
        return -1;
    if (test_logical_broadcast() != 0)
        return -1;

    return 0;
}
