// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int op_type = 0;

static int test_binaryop(const ncnn::Mat& _a, const ncnn::Mat& _b)
{
    ncnn::Mat a = _a;
    ncnn::Mat b = _b;
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0);   // with_scalar
    pd.set(2, 0.f); // b

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> ab(2);
    ab[0] = a;
    ab[1] = b;

    int ret = test_layer("BinaryOp", pd, weights, ab, 1, 0.0001);
    if (ret != 0)
    {
        fprintf(stderr, "test_binaryop failed a.dims=%d a=(%d %d %d %d) b.dims=%d b=(%d %d %d %d) op_type=%d\n", a.dims, a.w, a.h, a.d, a.c, b.dims, b.w, b.h, b.d, b.c, op_type);
    }

    return ret;
}

static int test_binaryop(const ncnn::Mat& _a, float b)
{
    ncnn::Mat a = _a;
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 1); // with_scalar
    pd.set(2, b); // b

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("BinaryOp", pd, weights, a, 0.0001);
    if (ret != 0)
    {
        fprintf(stderr, "test_binaryop failed a.dims=%d a=(%d %d %d %d) b=%f op_type=%d\n", a.dims, a.w, a.h, a.d, a.c, b, op_type);
    }

    return ret;
}

static int test_binaryop_1(int w)
{
    // dense widths and scalar broadcasts retain each packing boundary
    ncnn::Mat a = RandomMat(w, 1.0f, 1.1f);
    ncnn::Mat b = RandomMat(w, 0.8f, 0.9f);
    ncnn::Mat scalar = RandomMat(1, 0.8f, 0.9f);
    return test_binaryop(a, b)
           || test_binaryop(a, scalar)
           || test_binaryop(scalar, a)
           || test_binaryop(a, 0.7f);
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
        RandomMat(w, h, 1.0f, 1.1f),
        RandomMat(1, h, 1.0f, 1.1f)
    };
    ncnn::Mat second_inputs[] = {
        RandomMat(w, h, 0.8f, 0.9f),
        RandomMat(1, h, 0.8f, 0.9f),
        RandomMat(w, 1, 0.8f, 0.9f),
        RandomMat(1, 1, 0.8f, 0.9f)
    };
    const int broadcast_pairs[][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}
    };
    int ret = test_binaryop(inputs[0], second_inputs[0])
              || test_binaryop(inputs[1], second_inputs[1]);
    if (ret != 0)
        return ret;
    for (size_t i = 0; i < sizeof(broadcast_pairs) / sizeof(broadcast_pairs[0]); i++)
    {
        const ncnn::Mat& a = inputs[broadcast_pairs[i][0]];
        const ncnn::Mat& b = second_inputs[broadcast_pairs[i][1]];
        int ret = test_binaryop(a, b) || test_binaryop(b, a);
        if (ret != 0)
            return ret;
    }
    // scalar parameters exercise the dense vector kernels
    return test_binaryop(inputs[0], 0.7f);
}

static int test_binaryop_2()
{
    // singleton width includes a short row count before the packing boundaries
    ncnn::Mat rows = RandomMat(1, 3, 1.0f, 1.1f);
    ncnn::Mat second_rows = RandomMat(1, 3, 0.8f, 0.9f);
    return test_binaryop(rows, second_rows)
           || test_binaryop_2(13, 31)
           || test_binaryop_2(14, 28)
           || test_binaryop_2(15, 24)
           || test_binaryop_2(16, 32);
}

static int test_binaryop_3(int w, int h, int c)
{
    // same-shape arithmetic, singleton axes and complementary output growth
    ncnn::Mat inputs[] = {
        RandomMat(w, h, c, 1.0f, 1.1f),
        RandomMat(1, h, c, 1.0f, 1.1f),
        RandomMat(w, 1, c, 1.0f, 1.1f),
        RandomMat(1, 1, c, 1.0f, 1.1f)
    };
    ncnn::Mat second_inputs[] = {
        RandomMat(w, h, c, 0.8f, 0.9f),
        RandomMat(1, h, c, 0.8f, 0.9f),
        RandomMat(w, 1, c, 0.8f, 0.9f),
        RandomMat(1, 1, c, 0.8f, 0.9f),
        RandomMat(w, h, 1, 0.8f, 0.9f),
        RandomMat(1, h, 1, 0.8f, 0.9f),
        RandomMat(w, 1, 1, 0.8f, 0.9f),
        RandomMat(1, 1, 1, 0.8f, 0.9f)
    };
    const int broadcast_pairs[][2] = {
        {1, 2}, {0, 1}, {0, 2}, {0, 3},
        {0, 4}, {0, 5}, {0, 6}, {0, 7},
        {1, 6}, {2, 5}, {3, 4}
    };
    int ret = test_binaryop(inputs[0], second_inputs[0])
              || test_binaryop(inputs[1], second_inputs[1])
              || test_binaryop(inputs[3], second_inputs[3]);
    if (ret != 0)
        return ret;
    for (size_t i = 0; i < sizeof(broadcast_pairs) / sizeof(broadcast_pairs[0]); i++)
    {
        const ncnn::Mat& a = inputs[broadcast_pairs[i][0]];
        const ncnn::Mat& b = second_inputs[broadcast_pairs[i][1]];
        int ret = test_binaryop(a, b) || test_binaryop(b, a);
        if (ret != 0)
            return ret;
    }
    // scalar parameters exercise the dense vector kernels
    return test_binaryop(inputs[0], 0.7f);
}

static int test_binaryop_channel_broadcast(int c)
{
    // equal packing broadcasts the channel vector over width in both operand directions
    ncnn::Mat a = RandomMat(1, 9, c, 1.0f, 1.1f);
    ncnn::Mat b = RandomMat(7, 1, c, 0.8f, 0.9f);
    return test_binaryop(a, b) || test_binaryop(b, a);
}

static int test_binaryop_mixed_channel_broadcast()
{
    // scalar row values broadcast over packed channels and width
    ncnn::Mat a = RandomMat(7, 3, 32, 1.0f, 1.1f);
    ncnn::Mat b = RandomMat(1, 3, 1, 0.8f, 0.9f);
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

static int test_binaryop_4_dense(int w, int h, int d, int c)
{
    // dense and scalar arithmetic retain both channel packing representatives
    ncnn::Mat a = RandomMat(w, h, d, c, 1.0f, 1.1f);
    ncnn::Mat b = RandomMat(w, h, d, c, 0.8f, 0.9f);
    return test_binaryop(a, b) || test_binaryop(a, 0.7f);
}

static int test_binaryop_4_dense()
{
    return test_binaryop_4_dense(2, 7, 3, 31)
           || test_binaryop_4_dense(3, 6, 4, 28);
}

static int test_binaryop_4_broadcast(int w, int h, int d, int c)
{
    // inverse operation pairs retain both canonical arithmetic directions
    ncnn::Mat inputs[] = {
        RandomMat(w, h, d, c, 1.0f, 1.1f),
        RandomMat(1, h, d, c, 1.0f, 1.1f),
        RandomMat(w, 1, d, c, 1.0f, 1.1f),
        RandomMat(1, 1, d, c, 1.0f, 1.1f),
        RandomMat(w, h, 1, c, 1.0f, 1.1f),
        RandomMat(1, h, 1, c, 1.0f, 1.1f),
        RandomMat(w, 1, 1, c, 1.0f, 1.1f),
        RandomMat(1, 1, 1, c, 1.0f, 1.1f)
    };
    ncnn::Mat second_inputs[] = {
        RandomMat(1, h, d, c, 0.8f, 0.9f),
        RandomMat(w, 1, d, c, 0.8f, 0.9f),
        RandomMat(1, 1, d, c, 0.8f, 0.9f),
        RandomMat(w, h, 1, c, 0.8f, 0.9f),
        RandomMat(1, h, 1, c, 0.8f, 0.9f),
        RandomMat(w, 1, 1, c, 0.8f, 0.9f),
        RandomMat(1, 1, 1, c, 0.8f, 0.9f),
        RandomMat(w, h, d, 1, 0.8f, 0.9f),
        RandomMat(1, h, d, 1, 0.8f, 0.9f),
        RandomMat(w, 1, d, 1, 0.8f, 0.9f),
        RandomMat(1, 1, d, 1, 0.8f, 0.9f),
        RandomMat(w, h, 1, 1, 0.8f, 0.9f),
        RandomMat(1, h, 1, 1, 0.8f, 0.9f),
        RandomMat(w, 1, 1, 1, 0.8f, 0.9f),
        RandomMat(1, 1, 1, 1, 0.8f, 0.9f)
    };
    const int broadcast_pairs[][2] = {
        {0, 0}, {0, 1}, {0, 3}, {0, 7},
        {0, 2}, {0, 6}, {0, 14},
        {1, 13}, {2, 12}, {3, 11}, {4, 10},
        {5, 9}, {6, 8}, {7, 7}
    };
    const int ordered_operations[][3] = {
        {15, 0, 1}, {16, 0, 1},
        {17, 0, 1}, {18, 0, 1}
    };
    for (size_t j = 0; j < sizeof(ordered_operations) / sizeof(ordered_operations[0]); j++)
    {
        op_type = ordered_operations[j][0];
        for (size_t i = 0; i < sizeof(broadcast_pairs) / sizeof(broadcast_pairs[0]); i++)
        {
            const ncnn::Mat operands[] = {
                inputs[broadcast_pairs[i][0]],
                second_inputs[broadcast_pairs[i][1]]
            };
            int ret = test_binaryop(operands[ordered_operations[j][1]], operands[ordered_operations[j][2]]);
            if (ret != 0)
                return ret;
        }
    }
    return 0;
}

static int test_binaryop_4_broadcast()
{
    // depth, spatial and channel expansion retain scalar and packed outputs
    return test_binaryop_4_broadcast(2, 7, 3, 31)
           || test_binaryop_4_broadcast(3, 6, 4, 28);
}

static int test_binaryop_5(int w, int h, int d, int c)
{
    ncnn::Mat a[4] = {
        RandomMat(c, 1.0f, 1.1f),
        RandomMat(d, c, 1.0f, 1.1f),
        RandomMat(h, d, c, 1.0f, 1.1f),
        RandomMat(w, h, d, c, 1.0f, 1.1f),
    };

    ncnn::Mat b[4] = {
        RandomMat(c, 0.8f, 0.9f),
        RandomMat(d, c, 0.8f, 0.9f),
        RandomMat(h, d, c, 0.8f, 0.9f),
        RandomMat(w, h, d, c, 0.8f, 0.9f),
    };

    // every implicit-rank expansion is checked in both operand directions
    const int rank_pairs[][2] = {
        {0, 1}, {0, 2}, {0, 3},
        {1, 2}, {1, 3}, {2, 3}
    };
    for (size_t i = 0; i < sizeof(rank_pairs) / sizeof(rank_pairs[0]); i++)
    {
        const int j = rank_pairs[i][0];
        const int k = rank_pairs[i][1];
        int ret = test_binaryop(a[j], b[k]) || test_binaryop(a[k], b[j]);
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
        RandomMat(d, c, 1.0f, 1.1f),
        RandomMat(h, d, c, 1.0f, 1.1f),
        RandomMat(w, h, d, c, 1.0f, 1.1f),
    };

    for (int j = 0; j < 3; j++)
    {
        ncnn::Mat b = RandomMat(a[j].w, 0.8f, 0.9f);

        int ret = test_binaryop(a[j], b) || test_binaryop(b, a[j]);
        if (ret != 0)
            return ret;
    }

    ncnn::Mat aa[3] = {
        RandomMat(c, c, 1.0f, 1.1f),
        RandomMat(c, 3, c, 1.0f, 1.1f),
        RandomMat(c, 3, 2, c, 1.0f, 1.1f),
    };

    for (int j = 0; j < 3; j++)
    {
        ncnn::Mat b = RandomMat(aa[j].w, 0.8f, 0.9f);

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

static int test_binaryop_singletons()
{
    // rank-specific single-element layouts are independent of packing boundaries
    ncnn::Mat a1 = RandomMat(1, 1.0f, 1.1f);
    ncnn::Mat b1 = RandomMat(1, 0.8f, 0.9f);
    ncnn::Mat a2 = RandomMat(1, 1, 1.0f, 1.1f);
    ncnn::Mat a3 = RandomMat(1, 1, 1, 1.0f, 1.1f);
    ncnn::Mat a4 = RandomMat(1, 1, 1, 1, 1.0f, 1.1f);
    ncnn::Mat b4 = RandomMat(1, 1, 1, 1, 0.8f, 0.9f);
    return test_binaryop(a1, b1)
           || test_binaryop(a1, 0.7f)
           || test_binaryop(a2, 0.7f)
           || test_binaryop(a3, 0.7f)
           || test_binaryop(a4, b4)
           || test_binaryop(a4, 0.7f);
}

static int test_binaryop_signed_remainders()
{
    // binary fractions avoid rounding across integer quotient boundaries
    ncnn::Mat a(33);
    ncnn::Mat b(33);
    const float values[] = {-2.25f, -1.125f, -0.375f, 0.375f, 1.125f, 2.25f};
    for (int i = 0; i < 33; i++)
    {
        a[i] = values[i % 6];
        b[i] = i % 2 == 0 ? 0.875f : -0.875f;
    }
    return test_binaryop(a, b)
           || test_binaryop(b, a)
           || test_binaryop(a, 0.875f)
           || test_binaryop(a, -0.875f);
}

int main()
{
    SRAND(7767517);

    for (op_type = 15; op_type < 19; op_type++)
    {
        int ret = 0
                  || test_binaryop_1()
                  || test_binaryop_2()
                  || test_binaryop_3()
                  || test_binaryop_4_dense()
                  || test_binaryop_5()
                  || test_binaryop_6()
                  || test_binaryop_singletons()
                  || test_binaryop_signed_remainders();

        if (ret != 0)
            return ret;
    }

    return test_binaryop_4_broadcast();
}
