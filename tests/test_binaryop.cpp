// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

#define OP_TYPE_MAX 14

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
    if (op_type == 10 || op_type == 11)
    {
        // value must be non-zero for atan2/ratan2
        a = a.clone();
        b = b.clone();
        for (int i = 0; i < a.total(); i++)
        {
            if (a[i] == 0.f)
                a[i] = 0.001f;
        }
        for (int i = 0; i < b.total(); i++)
        {
            if (b[i] == 0.f)
                b[i] = 0.001f;
        }
    }

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0);   // with_scalar
    pd.set(2, 0.f); // b

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> ab(2);
    ab[0] = a;
    ab[1] = b;

    int ret = test_layer("BinaryOp", pd, weights, ab, 1, 0.001);
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
    if (op_type == 10 || op_type == 11)
    {
        // value must be non-zero for atan2/ratan2
        a = a.clone();
        for (int i = 0; i < a.total(); i++)
        {
            if (a[i] == 0.f)
                a[i] = 0.001f;
        }
    }

    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 1); // with_scalar
    pd.set(2, b); // b

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("BinaryOp", pd, weights, a, 0.001);
    if (ret != 0)
    {
        fprintf(stderr, "test_binaryop failed a.dims=%d a=(%d %d %d %d) b=%f op_type=%d\n", a.dims, a.w, a.h, a.d, a.c, b, op_type);
    }

    return ret;
}

static int test_binaryop_1(int w)
{
    // same-shape arithmetic, singleton axes and complementary output growth
    ncnn::Mat inputs[] = {
        RandomMat(w),
        RandomMat(1)
    };
    const int broadcast_pairs[][2] = {
        {0, 1}
    };
    int ret = test_binaryop(inputs[0], inputs[0])
              || test_binaryop(inputs[1], inputs[1]);
    if (ret != 0)
        return ret;
    for (size_t i = 0; i < sizeof(broadcast_pairs) / sizeof(broadcast_pairs[0]); i++)
    {
        const ncnn::Mat& a = inputs[broadcast_pairs[i][0]];
        const ncnn::Mat& b = inputs[broadcast_pairs[i][1]];
        int ret = test_binaryop(a, b) || test_binaryop(b, a);
        if (ret != 0)
            return ret;
    }
    // scalar parameters exercise the dense vector and single-element kernels
    return test_binaryop(inputs[0], 0.2f)
           || test_binaryop(inputs[1], 0.2f);
}

static int test_binaryop_1()
{
    // pack1, pack4, pack8 and pack16 include both vector loops and remainders
    return 0
           || test_binaryop_1(31)
           || test_binaryop_1(28)
           || test_binaryop_1(24)
           || test_binaryop_1(32);
}

static int test_binaryop_2(int w, int h)
{
    // same-shape arithmetic, singleton axes and complementary output growth
    ncnn::Mat inputs[] = {
        RandomMat(w, h),
        RandomMat(1, h),
        RandomMat(w, 1),
        RandomMat(1, 1)
    };
    const int broadcast_pairs[][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}
    };
    int ret = test_binaryop(inputs[0], inputs[0])
              || test_binaryop(inputs[1], inputs[1]);
    if (ret != 0)
        return ret;
    for (size_t i = 0; i < sizeof(broadcast_pairs) / sizeof(broadcast_pairs[0]); i++)
    {
        const ncnn::Mat& a = inputs[broadcast_pairs[i][0]];
        const ncnn::Mat& b = inputs[broadcast_pairs[i][1]];
        int ret = test_binaryop(a, b) || test_binaryop(b, a);
        if (ret != 0)
            return ret;
    }
    // scalar parameters exercise the dense vector and single-element kernels
    return test_binaryop(inputs[0], 0.2f)
           || test_binaryop(inputs[3], 0.2f);
}

static int test_binaryop_2()
{
    // singleton width includes a short row count before the packing boundaries
    ncnn::Mat rows = RandomMat(1, 3);
    return test_binaryop(rows, rows)
           || test_binaryop_2(13, 31)
           || test_binaryop_2(14, 28)
           || test_binaryop_2(15, 24)
           || test_binaryop_2(16, 32);
}

static int test_binaryop_3(int w, int h, int c)
{
    // same-shape arithmetic, singleton axes and complementary output growth
    ncnn::Mat inputs[] = {
        RandomMat(w, h, c),
        RandomMat(1, h, c),
        RandomMat(w, 1, c),
        RandomMat(1, 1, c),
        RandomMat(w, h, 1),
        RandomMat(1, h, 1),
        RandomMat(w, 1, 1),
        RandomMat(1, 1, 1)
    };
    const int broadcast_pairs[][2] = {
        {1, 2}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {1, 6}, {2, 5}, {3, 4}
    };
    int ret = test_binaryop(inputs[0], inputs[0])
              || test_binaryop(inputs[1], inputs[1])
              || test_binaryop(inputs[3], inputs[3]);
    if (ret != 0)
        return ret;
    for (size_t i = 0; i < sizeof(broadcast_pairs) / sizeof(broadcast_pairs[0]); i++)
    {
        const ncnn::Mat& a = inputs[broadcast_pairs[i][0]];
        const ncnn::Mat& b = inputs[broadcast_pairs[i][1]];
        int ret = test_binaryop(a, b) || test_binaryop(b, a);
        if (ret != 0)
            return ret;
    }
    // scalar parameters exercise the dense vector and single-element kernels
    return test_binaryop(inputs[0], 0.2f)
           || test_binaryop(inputs[7], 0.2f);
}

static int test_binaryop_channel_broadcast(int c)
{
    // equal packing broadcasts the channel vector over width in both operand directions
    ncnn::Mat a = RandomMat(1, 9, c);
    ncnn::Mat b = RandomMat(7, 1, c);
    return test_binaryop(a, b) || test_binaryop(b, a);
}

static int test_binaryop_mixed_channel_broadcast()
{
    // scalar row values broadcast over packed channels and width
    ncnn::Mat a = RandomMat(7, 3, 32);
    ncnn::Mat b = RandomMat(1, 3, 1);
    return test_binaryop(a, b) || test_binaryop(b, a);
}

static int test_binaryop_3()
{
    // spatial broadcasting uses scalar and packed channels independently of vector widths
    return 0
           || test_binaryop_3(7, 3, 31)
           || test_binaryop_3(7, 9, 28)
           || test_binaryop_channel_broadcast(24)
           || test_binaryop_channel_broadcast(32)
           || test_binaryop_mixed_channel_broadcast();
}

static int test_binaryop_4(int w, int h, int d, int c)
{
    // same-shape arithmetic, singleton axes and complementary output growth
    ncnn::Mat inputs[] = {
        RandomMat(w, h, d, c),
        RandomMat(1, h, d, c),
        RandomMat(w, 1, d, c),
        RandomMat(1, 1, d, c),
        RandomMat(w, h, 1, c),
        RandomMat(1, h, 1, c),
        RandomMat(w, 1, 1, c),
        RandomMat(1, 1, 1, c),
        RandomMat(w, h, d, 1),
        RandomMat(1, h, d, 1),
        RandomMat(w, 1, d, 1),
        RandomMat(1, 1, d, 1),
        RandomMat(w, h, 1, 1),
        RandomMat(1, h, 1, 1),
        RandomMat(w, 1, 1, 1),
        RandomMat(1, 1, 1, 1)
    };
    const int broadcast_pairs[][2] = {
        {0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 3}, {0, 7}, {0, 15}, {1, 14}, {2, 13}, {3, 12}, {4, 11}, {5, 10}, {6, 9}, {7, 8}
    };
    int ret = test_binaryop(inputs[0], inputs[0])
              || test_binaryop(inputs[15], inputs[15]);
    if (ret != 0)
        return ret;
    for (size_t i = 0; i < sizeof(broadcast_pairs) / sizeof(broadcast_pairs[0]); i++)
    {
        const ncnn::Mat& a = inputs[broadcast_pairs[i][0]];
        const ncnn::Mat& b = inputs[broadcast_pairs[i][1]];
        int ret = test_binaryop(a, b) || test_binaryop(b, a);
        if (ret != 0)
            return ret;
    }
    // scalar parameters exercise the dense vector and single-element kernels
    return test_binaryop(inputs[0], 0.2f)
           || test_binaryop(inputs[15], 0.2f);
}

static int test_binaryop_4()
{
    // depth broadcasting uses scalar and packed channels independently of vector widths
    return 0
           || test_binaryop_4(2, 7, 3, 31)
           || test_binaryop_4(3, 6, 4, 28);
}

static int test_binaryop_5(int w, int h, int d, int c)
{
    ncnn::Mat a[4] = {
        RandomMat(c),
        RandomMat(d, c),
        RandomMat(h, d, c),
        RandomMat(w, h, d, c),
    };

    // every implicit-rank expansion is checked in both operand directions
    const int rank_pairs[][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
    };
    for (size_t i = 0; i < sizeof(rank_pairs) / sizeof(rank_pairs[0]); i++)
    {
        const int j = rank_pairs[i][0];
        const int k = rank_pairs[i][1];
        int ret = test_binaryop(a[j], a[k]) || test_binaryop(a[k], a[j]);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_5()
{
    // implicit-rank broadcasts retain each packing boundary
    return 0
           || test_binaryop_5(2, 7, 3, 31)
           || test_binaryop_5(3, 6, 4, 28)
           || test_binaryop_5(4, 5, 5, 24)
           || test_binaryop_5(5, 4, 6, 32);
}

static int test_binaryop_6(int w, int h, int d, int c)
{
    ncnn::Mat a[3] = {
        RandomMat(d, c),
        RandomMat(h, d, c),
        RandomMat(w, h, d, c),
    };

    for (int j = 0; j < 3; j++)
    {
        ncnn::Mat b = RandomMat(a[j].w);

        int ret = test_binaryop(a[j], b) || test_binaryop(b, a[j]);
        if (ret != 0)
            return ret;
    }

    ncnn::Mat aa[3] = {
        RandomMat(c, c),
        RandomMat(c, d, c),
        RandomMat(c, h, d, c),
    };

    for (int j = 0; j < 3; j++)
    {
        ncnn::Mat b = RandomMat(aa[j].w);

        int ret = test_binaryop(aa[j], b) || test_binaryop(b, aa[j]);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_6()
{
    // keep the mixed packing conversions for one-dimensional broadcasts
    return 0
           || test_binaryop_6(16, 15, 12, 31)
           || test_binaryop_6(12, 16, 14, 28)
           || test_binaryop_6(16, 15, 12, 24)
           || test_binaryop_6(15, 12, 16, 32);
}

#if NCNN_VALIDATION
static int test_binaryop_load_param()
{
    ncnn::ParamDict base;
    if (test_layer_param(ncnn::LayerType::BinaryOp, base, 0) != 0)
        return -1;

    for (int i = 0; i <= 18; i++)
    {
        if (test_layer_param(ncnn::LayerType::BinaryOp, base, 0, i, 0) != 0)
            return -1;
    }

    const int invalid[] = {-1, 19, INT_MIN, INT_MAX};
    for (int i = 0; i < 4; i++)
    {
        if (test_layer_param(ncnn::LayerType::BinaryOp, base, 0, invalid[i], -1) != 0)
            return -1;
    }

    return 0;
}
#endif // NCNN_VALIDATION

static int test_binaryop_packing_tails()
{
    // small 4d planes retain scalar remainders when unpacking each vector width
    const int channels[] = {28, 24, 32};
    for (int i = 0; i < 3; i++)
    {
        ncnn::Mat a = RandomMat(3, 1, 1, channels[i]);
        ncnn::Mat b = RandomMat(3, 1, 1, channels[i]);
        int ret = test_binaryop(a, b);
        if (ret != 0)
            return ret;
    }
    return 0;
}

int main()
{
    SRAND(7767517);

    int ret = test_binaryop_packing_tails();
    if (ret != 0)
        return ret;

    for (op_type = 0; op_type < 3; op_type++)
    {
        int ret = 0
                  || test_binaryop_1()
                  || test_binaryop_2()
                  || test_binaryop_3()
                  || test_binaryop_4()
                  || test_binaryop_5()
                  || test_binaryop_6();

        if (ret != 0)
            return ret;
    }

    return 0
#if NCNN_VALIDATION
           || test_binaryop_load_param()
#endif // NCNN_VALIDATION
           ;
}
