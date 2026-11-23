// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

#if NCNN_VULKAN
#include "gpu.h"

#include <math.h>
#endif

static int test_groupnorm(const ncnn::Mat& a, int group, float eps, int affine)
{
    int channels = a.c;
    if (a.dims == 1)
    {
        channels = a.w;
    }
    else if (a.dims == 2)
    {
        channels = a.h;
    }

    ncnn::ParamDict pd;
    pd.set(0, group);
    pd.set(1, channels);
    pd.set(2, eps);
    pd.set(3, affine);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(channels);
    weights[1] = RandomMat(channels);

    int ret = test_layer("GroupNorm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_groupnorm failed a.dims=%d a=(%d %d %d %d) group=%d eps=%f affine=%d\n", a.dims, a.w, a.h, a.d, a.c, group, eps, affine);
    }

    return ret;
}

static int test_groupnorm_0()
{
    return 0
           || test_groupnorm(RandomMat(3, 6, 4, 2), 1, 0.01f, 0)
           || test_groupnorm(RandomMat(2, 3, 3, 8), 2, 0.002f, 0)
           || test_groupnorm(RandomMat(3, 4, 5, 6), 3, 0.01f, 0)
           || test_groupnorm(RandomMat(4, 5, 6, 12), 4, 0.02f, 0)
           || test_groupnorm(RandomMat(5, 6, 7, 24), 2, 0.001f, 0)
           || test_groupnorm(RandomMat(2, 8, 9, 24), 3, 0.0001f, 0)
           || test_groupnorm(RandomMat(3, 6, 4, 2), 1, 0.01f, 1)
           || test_groupnorm(RandomMat(2, 3, 3, 8), 2, 0.002f, 1)
           || test_groupnorm(RandomMat(3, 4, 5, 6), 3, 0.01f, 1)
           || test_groupnorm(RandomMat(4, 5, 6, 12), 4, 0.02f, 1)
           || test_groupnorm(RandomMat(5, 6, 7, 24), 2, 0.001f, 1)
           || test_groupnorm(RandomMat(2, 8, 9, 24), 3, 0.0001f, 1);
}

static int test_groupnorm_1()
{
    return 0
           || test_groupnorm(RandomMat(6, 4, 2), 1, 0.01f, 0)
           || test_groupnorm(RandomMat(3, 3, 8), 2, 0.002f, 0)
           || test_groupnorm(RandomMat(4, 5, 6), 3, 0.01f, 0)
           || test_groupnorm(RandomMat(5, 6, 12), 4, 0.02f, 0)
           || test_groupnorm(RandomMat(6, 7, 24), 2, 0.001f, 0)
           || test_groupnorm(RandomMat(8, 9, 24), 3, 0.0001f, 0)
           || test_groupnorm(RandomMat(6, 4, 2), 1, 0.01f, 1)
           || test_groupnorm(RandomMat(3, 3, 8), 2, 0.002f, 1)
           || test_groupnorm(RandomMat(4, 5, 6), 3, 0.01f, 1)
           || test_groupnorm(RandomMat(5, 6, 12), 4, 0.02f, 1)
           || test_groupnorm(RandomMat(6, 7, 24), 2, 0.001f, 1)
           || test_groupnorm(RandomMat(8, 9, 24), 3, 0.0001f, 1);
}

static int test_groupnorm_2()
{
    return 0
           || test_groupnorm(RandomMat(24, 2), 1, 0.01f, 0)
           || test_groupnorm(RandomMat(23, 8), 2, 0.002f, 0)
           || test_groupnorm(RandomMat(25, 6), 3, 0.01f, 0)
           || test_groupnorm(RandomMat(26, 12), 4, 0.02f, 0)
           || test_groupnorm(RandomMat(27, 24), 2, 0.001f, 0)
           || test_groupnorm(RandomMat(29, 24), 3, 0.0001f, 0)
           || test_groupnorm(RandomMat(24, 2), 1, 0.01f, 1)
           || test_groupnorm(RandomMat(23, 8), 2, 0.002f, 1)
           || test_groupnorm(RandomMat(25, 6), 3, 0.01f, 1)
           || test_groupnorm(RandomMat(26, 12), 4, 0.02f, 1)
           || test_groupnorm(RandomMat(27, 24), 2, 0.001f, 1)
           || test_groupnorm(RandomMat(29, 24), 3, 0.0001f, 1);
}

static int test_groupnorm_3()
{
    return 0
           || test_groupnorm(RandomMat(12), 1, 0.01f, 0)
           || test_groupnorm(RandomMat(18), 2, 0.002f, 0)
           || test_groupnorm(RandomMat(36), 3, 0.01f, 0)
           || test_groupnorm(RandomMat(212), 4, 0.02f, 0)
           || test_groupnorm(RandomMat(124), 2, 0.001f, 0)
           || test_groupnorm(RandomMat(324), 3, 0.0001f, 0)
           || test_groupnorm(RandomMat(12), 1, 0.01f, 1)
           || test_groupnorm(RandomMat(18), 2, 0.002f, 1)
           || test_groupnorm(RandomMat(36), 3, 0.01f, 1)
           || test_groupnorm(RandomMat(212), 4, 0.02f, 1)
           || test_groupnorm(RandomMat(124), 2, 0.001f, 1)
           || test_groupnorm(RandomMat(324), 3, 0.0001f, 1);
}

#if NCNN_VALIDATION
static int test_groupnorm_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 2);
    base.set(1, 8);
    if (test_layer_param(ncnn::LayerType::GroupNorm, base, 0) != 0)
        return -1;

    const int invalid[] = {0, -1, -8, INT_MIN};
    for (int i = 0; i < 4; i++)
    {
        if (test_layer_param(ncnn::LayerType::GroupNorm, base, 0, invalid[i], -1) != 0)
            return -1;
    }

    ncnn::ParamDict nonaffine = base;
    nonaffine.set(3, 0);

    return 0
           || test_layer_param(ncnn::LayerType::GroupNorm, base, 0, 3, -1)
           || test_layer_param(ncnn::LayerType::GroupNorm, base, 1, -1, -1)
           || test_layer_param(ncnn::LayerType::GroupNorm, nonaffine, 1, -1, -1)
           || test_layer_param(ncnn::LayerType::GroupNorm, nonaffine, 1, 7, -1)
           || test_layer_param(ncnn::LayerType::GroupNorm, nonaffine, 1, 8, 0);
}
#endif // NCNN_VALIDATION

#if NCNN_VULKAN
static ncnn::Mat groupnorm_reference(const ncnn::Mat& a, int group, float eps, int affine, const std::vector<ncnn::Mat>& weights)
{
    // use double precision instead of the generic cpu layer's serial float sums
    const int channels = a.dims == 1 ? a.w : a.dims == 2 ? a.h : a.c;
    const int size = a.dims == 1 ? 1 : a.dims == 2 ? a.w : a.w * a.h * a.d;
    const int channels_g = channels / group;
    ncnn::Mat expected = a.clone();
    for (int g = 0; g < group; g++)
    {
        double sum = 0.0;
        for (int q = g * channels_g; q < (g + 1) * channels_g; q++)
        {
            const float* ptr = a.dims < 3 ? (const float*)a + q * size : a.channel(q);
            for (int i = 0; i < size; i++)
                sum += ptr[i];
        }
        const double mean = sum / (channels_g * size);
        double sqsum = 0.0;
        for (int q = g * channels_g; q < (g + 1) * channels_g; q++)
        {
            const float* ptr = a.dims < 3 ? (const float*)a + q * size : a.channel(q);
            for (int i = 0; i < size; i++)
            {
                const double d = ptr[i] - mean;
                sqsum += d * d;
            }
        }
        const double inv_stddev = 1.0 / sqrt(sqsum / (channels_g * size) + eps);
        for (int q = g * channels_g; q < (g + 1) * channels_g; q++)
        {
            const float* ptr = a.dims < 3 ? (const float*)a + q * size : a.channel(q);
            float* outptr = a.dims < 3 ? (float*)expected + q * size : expected.channel(q);
            const float gamma = affine ? weights[0][q] : 1.f;
            const float beta = affine ? weights[1][q] : 0.f;
            for (int i = 0; i < size; i++)
                outptr[i] = (float)((ptr[i] - mean) * inv_stddev * gamma + beta);
        }
    }

    return expected;
}

static int test_groupnorm_gpu_reference(const ncnn::Mat& a, int group, int affine, float tolerance = 0.005f)
{
    const float eps = 1e-5f;
    const int channels = a.dims == 1 ? a.w : a.dims == 2 ? a.h : a.c;
    ncnn::ParamDict pd;
    pd.set(0, group);
    pd.set(1, channels);
    pd.set(2, eps);
    pd.set(3, affine);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(channels, 0.5f, 1.5f);
    weights[1] = RandomMat(channels, -0.5f, 0.5f);
    ncnn::Mat expected = groupnorm_reference(a, group, eps, affine, weights);

    ncnn::Option opt;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;
    for (int subgroup = 0; subgroup < 2; subgroup++)
    {
        opt.use_subgroup_ops = subgroup != 0;
        ncnn::Mat b;
        int ret = test_layer_gpu(ncnn::layer_to_index("GroupNorm"), pd, weights, opt, a, b, ncnn::Mat(), 0);
        // allow rounding in the existing float coeffs/norm shaders, including FMA
        if (ret != 233 && (ret != 0 || CompareMat(expected, b, tolerance) != 0))
        {
            fprintf(stderr, "test_groupnorm_gpu_reference failed a.dims=%d a=(%d %d %d %d) group=%d affine=%d subgroup=%d ret=%d\n", a.dims, a.w, a.h, a.d, a.c, group, affine, subgroup, ret);
            return -1;
        }
    }
    return 0;
}

static int test_groupnorm_gpu_numerics()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    const int channels[] = {1, 4};
    for (int c = 0; c < 2; c++)
    {
        ncnn::Mat a(128, 128, channels[c]);
        a.fill(100.1f);
        if (test_groupnorm_gpu_reference(a, channels[c], 0) != 0)
            return -1;
    }

    const int groups[] = {12, 6, 4, 2};
    for (int g = 0; g < 4; g++)
    {
        ncnn::Mat a(17, 19, 12);
        for (int q = 0; q < a.c; q++)
        {
            float* ptr = a.channel(q);
            for (int i = 0; i < a.w * a.h; i++)
                ptr[i] = 10.f + q * 0.5f + (i % 7 - 3) * 0.125f;
        }
        if (test_groupnorm_gpu_reference(a, groups[g], 1) != 0)
            return -1;
    }
    return 0;
}

static int test_groupnorm_gpu_variance()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    const int shapes[][3] = {{1024, 512, 4}, {1024, 128, 17}};
    for (int c = 0; c < 2; c++)
    {
        ncnn::Mat a(shapes[c][0], shapes[c][1], shapes[c][2]);
        for (int q = 0; q < a.c; q++)
        {
            float value = q % 2 == 0 ? 1.f / 64 : -1.f / 64;
            // the extra odd channel selects pack1 while keeping the group mean zero
            if (a.c % 2 == 1 && q == a.c - 1)
                value = 0.f;
            float* ptr = a.channel(q);
            for (int i = 0; i < a.w * a.h; i++)
                ptr[i] = value;
        }
        // large initial squares must not hide the later small squared differences
        float* ptr0 = a.channel(0);
        float* ptr1 = a.channel(1);
        for (int i = 0; i < 4; i++)
        {
            ptr0[i] = 64.f;
            ptr1[i] = -64.f;
        }
        if (test_groupnorm_gpu_reference(a, 1, 0, 0.0001f) != 0)
            return -1;
    }
    return 0;
}

static int test_groupnorm_gpu_dispatch()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    ncnn::Mat a(65536);
    a.fill(1.f);
    return test_groupnorm_gpu_reference(a, 65536, 0);
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

    return 0
           || test_groupnorm_0()
           || test_groupnorm_1()
           || test_groupnorm_2()
           || test_groupnorm_3()
#if NCNN_VULKAN
           || test_groupnorm_gpu_numerics()
           || test_groupnorm_gpu_variance()
           || test_groupnorm_gpu_dispatch()
#endif // NCNN_VULKAN
#if NCNN_VALIDATION
           || test_groupnorm_load_param()
#endif // NCNN_VALIDATION
           ;
}
