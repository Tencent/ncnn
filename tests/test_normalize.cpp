// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_normalize(const ncnn::Mat& a, int across_spatial, int across_channel, int channel_shared, float eps, int eps_mode, int flag = 0)
{
    int scale_data_size = channel_shared ? 1 : a.c;

    ncnn::ParamDict pd;
    pd.set(0, across_spatial);
    pd.set(4, across_channel);
    pd.set(1, channel_shared);
    pd.set(2, eps);
    pd.set(3, scale_data_size);
    pd.set(9, eps_mode);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(scale_data_size);

    int ret = test_layer("Normalize", pd, weights, a, 0.001f, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_normalize failed a.dims=%d a=(%d %d %d %d) across_spatial=%d across_channel=%d channel_shared=%d eps=%f eps_mode=%d\n", a.dims, a.w, a.h, a.d, a.c, across_spatial,
                across_channel, channel_shared, eps, eps_mode);
    }

    return ret;
}

static int test_normalize_0()
{
    ncnn::Mat a = RandomMat(6, 4, 2);
    ncnn::Mat b = RandomMat(5, 7, 8);

    return 0
           || test_normalize(a, 1, 0, 0, 0.01f, 0)
           || test_normalize(a, 1, 0, 0, 0.001f, 1)
           || test_normalize(a, 1, 0, 0, 0.002f, 2)
           || test_normalize(a, 1, 0, 1, 0.01f, 0)
           || test_normalize(a, 1, 0, 1, 0.001f, 1)
           || test_normalize(a, 1, 0, 1, 0.002f, 2)
           || test_normalize(b, 1, 0, 0, 0.01f, 0)
           || test_normalize(b, 1, 0, 0, 0.001f, 1)
           || test_normalize(b, 1, 0, 0, 0.002f, 2)
           || test_normalize(b, 1, 0, 1, 0.01f, 0)
           || test_normalize(b, 1, 0, 1, 0.001f, 1)
           || test_normalize(b, 1, 0, 1, 0.002f, 2);
}

static int test_normalize_1()
{
    ncnn::Mat a = RandomMat(5, 6, 3);
    ncnn::Mat b = RandomMat(3, 4, 8);

    return 0
           || test_normalize(a, 0, 1, 0, 0.01f, 0)
           || test_normalize(a, 0, 1, 0, 0.001f, 1)
           || test_normalize(a, 0, 1, 0, 0.002f, 2)
           || test_normalize(a, 0, 1, 1, 0.01f, 0)
           || test_normalize(a, 0, 1, 1, 0.001f, 1)
           || test_normalize(a, 0, 1, 1, 0.002f, 2)
           || test_normalize(b, 0, 1, 0, 0.01f, 0)
           || test_normalize(b, 0, 1, 0, 0.001f, 1)
           || test_normalize(b, 0, 1, 0, 0.002f, 2)
           || test_normalize(b, 0, 1, 1, 0.01f, 0)
           || test_normalize(b, 0, 1, 1, 0.001f, 1)
           || test_normalize(b, 0, 1, 1, 0.002f, 2);
}

static int test_normalize_2()
{
    ncnn::Mat a = RandomMat(2, 3, 5);
    ncnn::Mat b = RandomMat(4, 6, 8);

    return 0
           || test_normalize(a, 1, 1, 0, 0.01f, 0)
           || test_normalize(a, 1, 1, 0, 0.001f, 1)
           || test_normalize(a, 1, 1, 0, 0.002f, 2)
           || test_normalize(a, 1, 1, 1, 0.01f, 0)
           || test_normalize(a, 1, 1, 1, 0.001f, 1)
           || test_normalize(a, 1, 1, 1, 0.002f, 2)
           || test_normalize(b, 1, 1, 0, 0.01f, 0)
           || test_normalize(b, 1, 1, 0, 0.001f, 1)
           || test_normalize(b, 1, 1, 0, 0.002f, 2)
           || test_normalize(b, 1, 1, 1, 0.01f, 0)
           || test_normalize(b, 1, 1, 1, 0.001f, 1)
           || test_normalize(b, 1, 1, 1, 0.002f, 2);
}

static int test_normalize_3()
{
    ncnn::Mat a = RandomMat(5, 4, 3, 3);
    ncnn::Mat b = RandomMat(3, 3, 2, 8);

    return 0
           || test_normalize(a, 1, 0, 0, 0.01f, 0)
           || test_normalize(a, 1, 0, 1, 0.001f, 1)
           || test_normalize(b, 0, 1, 0, 0.002f, 2)
           || test_normalize(b, 0, 1, 1, 0.01f, 0)
           || test_normalize(a, 1, 1, 0, 0.001f, 1)
           || test_normalize(b, 1, 1, 1, 0.002f, 2);
}

static int test_normalize_4()
{
    ncnn::Mat a = RandomMat(17, 13, 24);
    ncnn::Mat b = RandomMat(9, 7, 3, 16);

    return 0
           || test_normalize(a, 1, 0, 0, 0.01f, 0, TEST_LAYER_ENABLE_THREADING | TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(a, 1, 0, 1, 0.001f, 1, TEST_LAYER_ENABLE_THREADING | TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(a, 0, 1, 0, 0.002f, 2, TEST_LAYER_ENABLE_THREADING | TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(a, 1, 1, 1, 0.01f, 0, TEST_LAYER_ENABLE_THREADING | TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(b, 1, 0, 0, 0.001f, 1, TEST_LAYER_ENABLE_THREADING | TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(b, 0, 1, 1, 0.002f, 2, TEST_LAYER_ENABLE_THREADING | TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(b, 1, 1, 0, 0.01f, 0, TEST_LAYER_ENABLE_THREADING | TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_normalize_5()
{
    ncnn::Mat a = RandomMat(7, 5, 16, -0.00001f, 0.00001f);
    ncnn::Mat b = RandomMat(9, 7, 24, -0.00001f, 0.00001f);

    return 0
           || test_normalize(a, 1, 0, 0, 0.0001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(a, 0, 1, 1, 0.0001f, 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(a, 1, 1, 0, 0.0001f, 2, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(b, 1, 0, 0, 0.0001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(b, 0, 1, 1, 0.0001f, 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(b, 1, 1, 0, 0.0001f, 2, TEST_LAYER_DISABLE_GPU_TESTING);
}

static int test_normalize_6()
{
    const ncnn::Mat inputs[] = {
        RandomMat(1),
        RandomMat(7),
        RandomMat(12),
        RandomMat(24),
        RandomMat(32),
        RandomMat(7, 3),
        RandomMat(7, 12),
        RandomMat(7, 24),
        RandomMat(7, 32),
        RandomMat(1, 1, 12),
        RandomMat(2, 1, 12),
        RandomMat(3, 1, 12),
        RandomMat(4, 1, 12),
        RandomMat(1, 1, 24),
        RandomMat(1, 1, 16),
        RandomMat(7, 5, 3),
        RandomMat(7, 5, 12),
        RandomMat(7, 5, 24),
        RandomMat(7, 5, 32),
        RandomMat(3, 5, 3, 12),
        RandomMat(3, 5, 3, 24),
        RandomMat(3, 5, 3, 32),
        RandomMat(12, -0.00001f, 0.00001f),
        RandomMat(7, 24, -0.00001f, 0.00001f),
        RandomMat(3, 5, 12, -0.00001f, 0.00001f),
        RandomMat(3, 5, 3, 32, -0.00001f, 0.00001f),
        RandomMat(3, 5, 16, 0.f, 0.f)
    };

    for (int i = 0; i < (int)(sizeof(inputs) / sizeof(inputs[0])); i++)
    {
        for (int across_spatial = 0; across_spatial <= 1; across_spatial++)
        {
            for (int across_channel = 0; across_channel <= 1; across_channel++)
            {
                for (int channel_shared = 0; channel_shared <= 1; channel_shared++)
                {
                    for (int eps_mode = 0; eps_mode < 3; eps_mode++)
                    {
                        int ret = test_normalize(inputs[i], across_spatial, across_channel, channel_shared, 0.0001f, eps_mode, TEST_LAYER_DISABLE_GPU_TESTING);
                        if (ret != 0)
                            return ret;
                    }
                }
            }
        }
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_normalize_0()
           || test_normalize_1()
           || test_normalize_2()
           || test_normalize_3()
           || test_normalize_4()
           || test_normalize_5()
           || test_normalize_6();
}
