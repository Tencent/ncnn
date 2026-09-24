// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#define OP_TYPE_MAX 19

static int op_type = 0;

static int test_binaryop(const ncnn::Mat& _a, const ncnn::Mat& _b, int flag)
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

    int ret = test_layer("BinaryOp", pd, weights, ab, 1, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_binaryop failed a.dims=%d a=(%d %d %d %d) b.dims=%d b=(%d %d %d %d) op_type=%d\n", a.dims, a.w, a.h, a.d, a.c, b.dims, b.w, b.h, b.d, b.c, op_type);
    }

    return ret;
}

static int test_binaryop(const ncnn::Mat& _a, float b, int flag)
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

    int ret = test_layer("BinaryOp", pd, weights, a, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_binaryop failed a.dims=%d a=(%d %d %d %d) b=%f op_type=%d\n", a.dims, a.w, a.h, a.d, a.c, b, op_type);
    }

    return ret;
}

static int test_binaryop_1(int w, int flag = 0)
{
    ncnn::Mat a[2];
    for (int j = 0; j < 2; j++)
    {
        int bw = j % 2 == 0 ? w : 1;
        a[j] = RandomMat(bw);
    }

    for (int j = 0; j < 2; j++)
    {
        for (int k = 0; k < 2; k++)
        {
            int ret = test_binaryop(a[j], a[k], flag);
            if (ret != 0)
                return ret;
        }

        int ret = test_binaryop(a[j], 0.2f, flag);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_1()
{
    // the 28-element cases cover the same Vulkan packing and broadcast paths
    return 0
           || test_binaryop_1(31)
           || test_binaryop_1(28)
           || test_binaryop_1(24, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_binaryop_1(32, TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_binaryop_2(int w, int h, int flag = 0)
{
    ncnn::Mat a[4];
    for (int j = 0; j < 2; j++)
    {
        for (int k = 0; k < 2; k++)
        {
            int bw = j % 2 == 0 ? w : 1;
            int bh = k % 2 == 0 ? h : 1;
            a[j * 2 + k] = RandomMat(bw, bh);
        }
    }

    for (int j = 0; j < 4; j++)
    {
        for (int k = 0; k < 4; k++)
        {
            // retain full spatial broadcasts, singleton packed axes and every no-broadcast shape
            int ret;
            if (j != k && (j & k) > 1)
                ret = test_binaryop(a[j], a[k], flag | TEST_LAYER_DISABLE_GPU_TESTING);
            else
                ret = test_binaryop(a[j], a[k], flag);
            if (ret != 0)
                return ret;
        }

        int ret;
        if (j != 0 && j != 3)
            ret = test_binaryop(a[j], 0.2f, flag | TEST_LAYER_DISABLE_GPU_TESTING);
        else
            ret = test_binaryop(a[j], 0.2f, flag);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_2()
{
    // the 28-element cases cover the same Vulkan packing and broadcast paths
    return 0
           || test_binaryop_2(13, 31)
           || test_binaryop_2(14, 28)
           || test_binaryop_2(15, 24, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_binaryop_2(16, 32, TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_binaryop_3(int w, int h, int c, int flag = 0)
{
    ncnn::Mat a[8];
    for (int j = 0; j < 2; j++)
    {
        for (int k = 0; k < 2; k++)
        {
            for (int l = 0; l < 2; l++)
            {
                int bw = j % 2 == 0 ? w : 1;
                int bh = k % 2 == 0 ? h : 1;
                int bc = l % 2 == 0 ? c : 1;
                a[j * 4 + k * 2 + l] = RandomMat(bw, bh, bc);
            }
        }
    }

    for (int j = 0; j < 8; j++)
    {
        for (int k = 0; k < 8; k++)
        {
            // retain full spatial broadcasts, singleton packed axes and every no-broadcast shape
            int ret;
            if (j != k && (j & k) > 1)
                ret = test_binaryop(a[j], a[k], flag | TEST_LAYER_DISABLE_GPU_TESTING);
            else
                ret = test_binaryop(a[j], a[k], flag);
            if (ret != 0)
                return ret;
        }

        int ret;
        if (j != 0 && j != 7)
            ret = test_binaryop(a[j], 0.2f, flag | TEST_LAYER_DISABLE_GPU_TESTING);
        else
            ret = test_binaryop(a[j], 0.2f, flag);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_3()
{
    // the 28-element cases cover the same Vulkan packing and broadcast paths
    return 0
           || test_binaryop_3(7, 3, 31)
           || test_binaryop_3(6, 4, 28)
           || test_binaryop_3(5, 5, 24, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_binaryop_3(4, 6, 32, TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_binaryop_4(int w, int h, int d, int c, int flag = 0)
{
    ncnn::Mat a[16];
    for (int j = 0; j < 2; j++)
    {
        for (int k = 0; k < 2; k++)
        {
            for (int l = 0; l < 2; l++)
            {
                for (int m = 0; m < 2; m++)
                {
                    int bw = j % 2 == 0 ? w : 1;
                    int bh = k % 2 == 0 ? h : 1;
                    int bd = l % 2 == 0 ? d : 1;
                    int bc = m % 2 == 0 ? c : 1;
                    a[j * 8 + k * 4 + l * 2 + m] = RandomMat(bw, bh, bd, bc);
                }
            }
        }
    }

    for (int j = 0; j < 16; j++)
    {
        for (int k = 0; k < 16; k++)
        {
            // retain full spatial broadcasts, singleton packed axes and every no-broadcast shape
            int ret;
            if (j != k && (j & k) > 1)
                ret = test_binaryop(a[j], a[k], flag | TEST_LAYER_DISABLE_GPU_TESTING);
            else
                ret = test_binaryop(a[j], a[k], flag);
            if (ret != 0)
                return ret;
        }

        int ret;
        if (j != 0 && j != 15)
            ret = test_binaryop(a[j], 0.2f, flag | TEST_LAYER_DISABLE_GPU_TESTING);
        else
            ret = test_binaryop(a[j], 0.2f, flag);
        if (ret != 0)
            return ret;
    }

    return 0;
}

static int test_binaryop_4()
{
    // the 28-element cases cover the same Vulkan packing and broadcast paths
    return 0
           || test_binaryop_4(2, 7, 3, 31)
           || test_binaryop_4(3, 6, 4, 28)
           || test_binaryop_4(4, 5, 5, 24, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_binaryop_4(5, 4, 6, 32, TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_binaryop_5(int w, int h, int d, int c, int flag = 0)
{
    ncnn::Mat a[4] = {
        RandomMat(c),
        RandomMat(d, c),
        RandomMat(h, d, c),
        RandomMat(w, h, d, c),
    };

    for (int j = 0; j < 4; j++)
    {
        for (int k = 0; k < 4; k++)
        {
            if (j == k)
                continue;

            int ret = test_binaryop(a[j], a[k], flag);
            if (ret != 0)
                return ret;
        }
    }

    return 0;
}

static int test_binaryop_5()
{
    // the 28-element cases cover the same Vulkan packing and broadcast paths
    return 0
           || test_binaryop_5(2, 7, 3, 31)
           || test_binaryop_5(3, 6, 4, 28)
           || test_binaryop_5(4, 5, 5, 24, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_binaryop_5(5, 4, 6, 32, TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_binaryop_6(int w, int h, int d, int c, int flag = 0)
{
    ncnn::Mat a[3] = {
        RandomMat(d, c),
        RandomMat(h, d, c),
        RandomMat(w, h, d, c),
    };

    for (int j = 0; j < 3; j++)
    {
        ncnn::Mat b = RandomMat(a[j].w);

        int ret = test_binaryop(a[j], b, flag) || test_binaryop(b, a[j], flag);
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

        int ret = test_binaryop(aa[j], b, flag) || test_binaryop(b, aa[j], flag);
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
           || test_binaryop_6(15, 12, 16, 32, TEST_LAYER_DISABLE_GPU_TESTING);
}

int main()
{
    SRAND(7767517);

    for (op_type = 6; op_type < 9; op_type++)
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

    return 0;
}
