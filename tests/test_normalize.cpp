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
    return 0
           // dims=1/2, packing is along spatial dimensions
           || test_normalize(RandomMat(7), 1, 0, 0, 0.0001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(12), 1, 0, 1, 0.0001f, 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(24), 1, 1, 0, 0.0001f, 2, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(32), 1, 1, 1, 0.0001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(7, 3), 1, 0, 0, 0.0001f, 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(7, 12), 1, 1, 1, 0.0001f, 2, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(7, 24), 1, 0, 1, 0.0001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(7, 32), 1, 1, 0, 0.0001f, 1, TEST_LAYER_DISABLE_GPU_TESTING)

           // dims=3, packing is along channels
           // pack1 / pack4 / pack8 / pack16
           || test_normalize(RandomMat(7, 5, 3), 1, 0, 0, 0.0001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(7, 5, 12), 1, 0, 1, 0.0001f, 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(7, 5, 24), 1, 0, 0, 0.0001f, 2, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(7, 5, 32), 1, 0, 1, 0.0001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)

           // across-channel path, exercise short spatial tails
           || test_normalize(RandomMat(1, 1, 12), 0, 1, 0, 0.0001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(2, 1, 12), 0, 1, 1, 0.0001f, 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(3, 1, 12), 0, 1, 0, 0.0001f, 2, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(4, 1, 12), 0, 1, 1, 0.0001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(3, 1, 24), 0, 1, 0, 0.0001f, 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(3, 1, 32), 0, 1, 1, 0.0001f, 2, TEST_LAYER_DISABLE_GPU_TESTING)

           // across-spatial + across-channel
           || test_normalize(RandomMat(7, 5, 3), 1, 1, 1, 0.0001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(7, 5, 12), 1, 1, 0, 0.0001f, 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(7, 5, 24), 1, 1, 1, 0.0001f, 2, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(7, 5, 32), 1, 1, 0, 0.0001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)

           // dims=4 packed path
           || test_normalize(RandomMat(3, 5, 3, 12), 0, 1, 0, 0.0001f, 1, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(3, 5, 3, 24), 1, 0, 1, 0.0001f, 2, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_normalize(RandomMat(3, 5, 3, 32), 1, 1, 0, 0.0001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)

           // (0, 0) is a no-op, one case is sufficient
           || test_normalize(RandomMat(7, 5, 12), 0, 0, 0, 0.0001f, 0, TEST_LAYER_DISABLE_GPU_TESTING);
}

#if NCNN_VULKAN

// NOTE fp16 arithmetic is not covered: test_layer_gpu turns it off (testutil.cpp) and this
// layer output scale 1/sqrt(sum(x^2)+eps) makes a hand driven comparison meaningless.

static int test_normalize_gpu_subgroup(const ncnn::Mat& a, int across_spatial, int across_channel, float eps, int eps_mode)
{
    const int scale_data_size = a.c;

    ncnn::ParamDict pd;
    pd.set(0, across_spatial);
    pd.set(4, across_channel);
    pd.set(1, 0);
    pd.set(2, eps);
    pd.set(3, scale_data_size);
    pd.set(9, eps_mode);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(scale_data_size);

    const ncnn::GpuInfo& info = ncnn::get_gpu_device()->info;

    const bool fp16_ps[4] = {false, true, true, false};
    const bool fp16_arith[4] = {false, false, true, false};
    const bool bf16[4] = {false, false, false, true};
    const char* names[4] = {"fp32", "fp16ps", "fp16psa", "bf16ps"};
    const float epsilons[4] = {0.001f, 0.002f, 0.002f, 0.01f};

    for (int i = 0; i < 4; i++)
    {
        if (fp16_ps[i] && !info.support_fp16_storage())
            continue;
        if (bf16[i] && !info.support_bf16_storage())
            continue;

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_vulkan_compute = true;
        opt.use_fp16_packed = fp16_ps[i];
        opt.use_fp16_storage = fp16_ps[i];
        opt.use_fp16_arithmetic = fp16_arith[i];
        opt.use_bf16_packed = bf16[i];
        opt.use_bf16_storage = bf16[i];

        opt.use_subgroup_ops = false;
        ncnn::Mat b0;
        int ret = test_layer_gpu(ncnn::layer_to_index("Normalize"), pd, weights, opt, a, b0, a, 0);
        if (ret == 233)
        {
            // test_layer_gpu turns fp16 arithmetic off, that configuration is covered below
            continue;
        }
        if (ret != 0)
        {
            fprintf(stderr, "test_normalize_gpu_subgroup fallback failed a.dims=%d a=(%d %d %d %d) spatial=%d channel=%d %s ret=%d\n", a.dims, a.w, a.h, a.d, a.c, across_spatial, across_channel, names[i], ret);
            return -1;
        }

        opt.use_subgroup_ops = true;
        ncnn::Mat b1;
        ret = test_layer_gpu(ncnn::layer_to_index("Normalize"), pd, weights, opt, a, b1, a, 0);
        if (ret != 0)
        {
            fprintf(stderr, "test_normalize_gpu_subgroup subgroup failed a.dims=%d a=(%d %d %d %d) spatial=%d channel=%d %s ret=%d\n", a.dims, a.w, a.h, a.d, a.c, across_spatial, across_channel, names[i], ret);
            return -1;
        }

        if (CompareMat(b0, b1, epsilons[i]) != 0)
        {
            fprintf(stderr, "test_normalize_gpu_subgroup mismatch a.dims=%d a=(%d %d %d %d) spatial=%d channel=%d %s\n", a.dims, a.w, a.h, a.d, a.c, across_spatial, across_channel, names[i]);
            return -1;
        }
    }


    return 0;
}

static int test_normalize_subgroup()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    // shapes large enough for the single pass subgroup reduce to take over
    return 0
           || test_normalize_gpu_subgroup(RandomMat(64, 16, 8), 1, 0, 0.0001f, 0)
           || test_normalize_gpu_subgroup(RandomMat(56, 56, 64), 1, 0, 0.0001f, 1)
           || test_normalize_gpu_subgroup(RandomMat(128, 3, 24), 1, 0, 0.0001f, 2)
           || test_normalize_gpu_subgroup(RandomMat(33, 5, 12), 1, 0, 0.0001f, 0)
           || test_normalize_gpu_subgroup(RandomMat(8, 8, 16), 0, 1, 0.0001f, 0)
           || test_normalize_gpu_subgroup(RandomMat(32, 32, 64), 0, 1, 0.0001f, 1)
           || test_normalize_gpu_subgroup(RandomMat(7, 5, 3, 24), 0, 1, 0.0001f, 2)
           || test_normalize_gpu_subgroup(RandomMat(16, 4, 8, 32), 0, 1, 0.0001f, 0);
}
#endif // NCNN_VULKAN

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
           || test_normalize_6()
#if NCNN_VULKAN
           || test_normalize_subgroup()
#endif
           ;
}