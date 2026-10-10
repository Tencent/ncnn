// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_concat(const std::vector<ncnn::Mat>& a, int axis, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); //axis

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Concat", pd, weights, a, 1, 0.001, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_concat failed a[0].dims=%d a[0]=(%d %d %d %d) axis=%d\n", a[0].dims, a[0].w, a[0].h, a[0].d, a[0].c, axis);
    }

    return ret;
}

static int test_concat(const ncnn::Mat& a, int axis, int flag = 0)
{
    std::vector<ncnn::Mat> as(3);
    as[0] = a;
    as[1] = a;
    as[2] = a;

    return test_concat(as, axis, flag) || test_concat(as, axis - a.dims, flag);
}

static int test_concat_0()
{
    ncnn::Mat a[] = {
        RandomMat(15, 5, 6, 13),
        RandomMat(15, 5, 6, 20),
        RandomMat(15, 5, 6, 24),
        RandomMat(15, 5, 6, 48)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                std::vector<ncnn::Mat> as(4);
                as[0] = a[i];
                as[1] = a[j];
                as[2] = a[k];
                as[3] = a[k];

                // indices 0/1 retain all Vulkan pack1/pack4 combinations
                int ret;
                if (i >= 2 || j >= 2 || k >= 2)
                    ret = test_concat(as, 0, TEST_LAYER_DISABLE_GPU_TESTING) || test_concat(as, -4, TEST_LAYER_DISABLE_GPU_TESTING);
                else
                    ret = test_concat(as, 0) || test_concat(as, -4);
                if (ret != 0)
                    return ret;
            }
        }
    }

    return 0;
}

static int test_concat_1()
{
    ncnn::Mat a[] = {
        RandomMat(15, 3, 15, 13),
        RandomMat(15, 3, 16, 20),
        RandomMat(15, 3, 17, 24),
        RandomMat(15, 3, 18, 48)
    };

    // indices 0/1 retain all Vulkan pack1/pack4 combinations
    return 0
           || test_concat(a[0], 1)
           || test_concat(a[1], 1)
           || test_concat(a[2], 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_concat(a[3], 1, TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_concat_2()
{
    ncnn::Mat a[] = {
        RandomMat(15, 15, 6, 13),
        RandomMat(15, 16, 6, 20),
        RandomMat(15, 17, 6, 24),
        RandomMat(15, 18, 6, 48)
    };

    // indices 0/1 retain all Vulkan pack1/pack4 combinations
    return 0
           || test_concat(a[0], 2)
           || test_concat(a[1], 2)
           || test_concat(a[2], 2, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_concat(a[3], 2, TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_concat_3()
{
    ncnn::Mat a[] = {
        RandomMat(15, 5, 7, 13),
        RandomMat(16, 5, 7, 20),
        RandomMat(17, 5, 7, 24),
        RandomMat(18, 5, 7, 48)
    };

    // indices 0/1 retain all Vulkan pack1/pack4 combinations
    return 0
           || test_concat(a[0], 3)
           || test_concat(a[1], 3)
           || test_concat(a[2], 3, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_concat(a[3], 3, TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_concat_4()
{
    ncnn::Mat a[] = {
        RandomMat(15, 13, 13),
        RandomMat(15, 13, 20),
        RandomMat(15, 13, 24),
        RandomMat(15, 13, 48)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                std::vector<ncnn::Mat> as(4);
                as[0] = a[i];
                as[1] = a[j];
                as[2] = a[k];
                as[3] = a[k];

                // indices 0/1 retain all Vulkan pack1/pack4 combinations
                int ret;
                if (i >= 2 || j >= 2 || k >= 2)
                    ret = test_concat(as, 0, TEST_LAYER_DISABLE_GPU_TESTING) || test_concat(as, -3, TEST_LAYER_DISABLE_GPU_TESTING);
                else
                    ret = test_concat(as, 0) || test_concat(as, -3);
                if (ret != 0)
                    return ret;
            }
        }
    }

    return 0;
}

static int test_concat_5()
{
    ncnn::Mat a[] = {
        RandomMat(15, 15, 13),
        RandomMat(15, 16, 20),
        RandomMat(15, 17, 24),
        RandomMat(15, 18, 48)
    };

    // indices 0/1 retain all Vulkan pack1/pack4 combinations
    return 0
           || test_concat(a[0], 1)
           || test_concat(a[1], 1)
           || test_concat(a[2], 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_concat(a[3], 1, TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_concat_6()
{
    ncnn::Mat a[] = {
        RandomMat(15, 13, 13),
        RandomMat(16, 13, 20),
        RandomMat(17, 13, 24),
        RandomMat(18, 13, 48)
    };

    // indices 0/1 retain all Vulkan pack1/pack4 combinations
    return 0
           || test_concat(a[0], 2)
           || test_concat(a[1], 2)
           || test_concat(a[2], 2, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_concat(a[3], 2, TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_concat_7()
{
    ncnn::Mat a[] = {
        RandomMat(19, 29),
        RandomMat(19, 44),
        RandomMat(19, 56),
        RandomMat(19, 80)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                std::vector<ncnn::Mat> as(4);
                as[0] = a[i];
                as[1] = a[j];
                as[2] = a[k];
                as[3] = a[k];

                // indices 0/1 retain all Vulkan pack1/pack4 combinations
                int ret;
                if (i >= 2 || j >= 2 || k >= 2)
                    ret = test_concat(as, 0, TEST_LAYER_DISABLE_GPU_TESTING) || test_concat(as, -2, TEST_LAYER_DISABLE_GPU_TESTING);
                else
                    ret = test_concat(as, 0) || test_concat(as, -2);
                if (ret != 0)
                    return ret;
            }
        }
    }

    return 0;
}

static int test_concat_8()
{
    ncnn::Mat a[] = {
        RandomMat(19, 29),
        RandomMat(16, 44),
        RandomMat(17, 56),
        RandomMat(18, 80)
    };

    // indices 0/1 retain all Vulkan pack1/pack4 combinations
    return 0
           || test_concat(a[0], 1)
           || test_concat(a[1], 1)
           || test_concat(a[2], 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_concat(a[3], 1, TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_concat_9()
{
    ncnn::Mat a[] = {
        RandomMat(29),
        RandomMat(44),
        RandomMat(56),
        RandomMat(80)
    };

    const int n = sizeof(a) / sizeof(a[0]);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                std::vector<ncnn::Mat> as(4);
                as[0] = a[i];
                as[1] = a[j];
                as[2] = a[k];
                as[3] = a[k];

                // indices 0/1 retain all Vulkan pack1/pack4 combinations
                int ret;
                if (i >= 2 || j >= 2 || k >= 2)
                    ret = test_concat(as, 0, TEST_LAYER_DISABLE_GPU_TESTING) || test_concat(as, -1, TEST_LAYER_DISABLE_GPU_TESTING);
                else
                    ret = test_concat(as, 0) || test_concat(as, -1);
                if (ret != 0)
                    return ret;
            }
        }
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_concat_0()
           || test_concat_1()
           || test_concat_2()
           || test_concat_3()
           || test_concat_4()
           || test_concat_5()
           || test_concat_6()
           || test_concat_7()
           || test_concat_8()
           || test_concat_9();
}
