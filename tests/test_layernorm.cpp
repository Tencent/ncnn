// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_VULKAN
#include "gpu.h"

#include <math.h>
#endif

static int test_layernorm(const ncnn::Mat& a, int affine_size, float eps, int affine, int flag = 0)
{
    ncnn::ParamDict pd;
    pd.set(0, affine_size);
    pd.set(1, eps);
    pd.set(2, affine);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(affine_size);
    weights[1] = RandomMat(affine_size);

    int ret = test_layer("LayerNorm", pd, weights, a, 1e-4f, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_layernorm failed a.dims=%d a=(%d %d %d %d) affine_size=%d eps=%f affine=%d\n", a.dims, a.w, a.h, a.d, a.c, affine_size, eps, affine);
    }

    return ret;
}

static int test_layernorm_0()
{
    return 0
           || test_layernorm(RandomMat(6, 4, 2), 6, 0.01f, 0)
           || test_layernorm(RandomMat(4, 5, 6), 4, 0.01f, 0)
           || test_layernorm(RandomMat(3, 3, 8), 3, 0.002f, 0)
           || test_layernorm(RandomMat(5, 6, 12), 5, 0.02f, 0)
           || test_layernorm(RandomMat(4, 7, 16), 4, 0.02f, 0)
           || test_layernorm(RandomMat(6, 7, 24), 6, 0.001f, 0)
           || test_layernorm(RandomMat(5, 8, 32), 5, 0.001f, 0, TEST_LAYER_DISABLE_GPU_TESTING)
           || test_layernorm(RandomMat(6, 4, 2), 6, 0.01f, 1)
           || test_layernorm(RandomMat(4, 5, 6), 4, 0.01f, 1)
           || test_layernorm(RandomMat(3, 3, 8), 3, 0.002f, 1)
           || test_layernorm(RandomMat(5, 6, 12), 5, 0.02f, 1)
           || test_layernorm(RandomMat(4, 7, 16), 4, 0.02f, 1)
           || test_layernorm(RandomMat(6, 7, 24), 6, 0.001f, 1)
           || test_layernorm(RandomMat(5, 8, 32), 5, 0.001f, 1);
}

static int test_layernorm_1()
{
    return 0
           || test_layernorm(RandomMat(6, 4, 2), 24, 0.01f, 0)
           || test_layernorm(RandomMat(4, 5, 6), 20, 0.01f, 0)
           || test_layernorm(RandomMat(3, 3, 8), 9, 0.002f, 0)
           || test_layernorm(RandomMat(5, 6, 12), 30, 0.02f, 0)
           || test_layernorm(RandomMat(4, 7, 16), 28, 0.02f, 0)
           || test_layernorm(RandomMat(6, 7, 24), 42, 0.001f, 0)
           || test_layernorm(RandomMat(5, 8, 32), 40, 0.001f, 0)
           || test_layernorm(RandomMat(6, 4, 2), 24, 0.01f, 1)
           || test_layernorm(RandomMat(4, 5, 6), 20, 0.01f, 1)
           || test_layernorm(RandomMat(3, 3, 8), 9, 0.002f, 1)
           || test_layernorm(RandomMat(5, 6, 12), 30, 0.02f, 1)
           || test_layernorm(RandomMat(4, 7, 16), 28, 0.02f, 1)
           || test_layernorm(RandomMat(6, 7, 24), 42, 0.001f, 1)
           || test_layernorm(RandomMat(5, 8, 32), 40, 0.001f, 1);
}

static int test_layernorm_2()
{
    return 0
           || test_layernorm(RandomMat(4, 2), 4, 0.01f, 0)
           || test_layernorm(RandomMat(5, 6), 5, 0.01f, 0)
           || test_layernorm(RandomMat(3, 8), 3, 0.002f, 0)
           || test_layernorm(RandomMat(6, 12), 6, 0.02f, 0)
           || test_layernorm(RandomMat(4, 16), 4, 0.02f, 0)
           || test_layernorm(RandomMat(7, 24), 7, 0.001f, 0)
           || test_layernorm(RandomMat(8, 32), 8, 0.001f, 0)
           || test_layernorm(RandomMat(4, 2), 4, 0.01f, 1)
           || test_layernorm(RandomMat(5, 6), 5, 0.01f, 1)
           || test_layernorm(RandomMat(3, 8), 3, 0.002f, 1)
           || test_layernorm(RandomMat(6, 12), 6, 0.02f, 1)
           || test_layernorm(RandomMat(4, 16), 4, 0.02f, 1)
           || test_layernorm(RandomMat(7, 24), 7, 0.001f, 1)
           || test_layernorm(RandomMat(8, 32), 8, 0.001f, 1);
}

static int test_layernorm_3()
{
    return 0
           || test_layernorm(RandomMat(2), 2, 0.01f, 0)
           || test_layernorm(RandomMat(6), 6, 0.01f, 0)
           || test_layernorm(RandomMat(8), 8, 0.002f, 0)
           || test_layernorm(RandomMat(12), 12, 0.02f, 0)
           || test_layernorm(RandomMat(16), 16, 0.02f, 0)
           || test_layernorm(RandomMat(24), 24, 0.001f, 0)
           || test_layernorm(RandomMat(32), 32, 0.001f, 0)
           || test_layernorm(RandomMat(2), 2, 0.01f, 1)
           || test_layernorm(RandomMat(6), 6, 0.01f, 1)
           || test_layernorm(RandomMat(8), 8, 0.002f, 1)
           || test_layernorm(RandomMat(12), 12, 0.02f, 1)
           || test_layernorm(RandomMat(16), 16, 0.02f, 1)
           || test_layernorm(RandomMat(24), 24, 0.001f, 1)
           || test_layernorm(RandomMat(32), 32, 0.001f, 1);
}

static int test_layernorm_4()
{
    return 0
           || test_layernorm(RandomMat(5, 3, 2, 8), 5, 0.01f, 0)
           || test_layernorm(RandomMat(4, 3, 2, 5), 4, 0.01f, 1)
           || test_layernorm(RandomMat(5, 3, 2, 8), 15, 0.002f, 0)
           || test_layernorm(RandomMat(4, 3, 2, 5), 12, 0.002f, 1)
           || test_layernorm(RandomMat(5, 3, 2, 8), 30, 0.001f, 0)
           || test_layernorm(RandomMat(4, 3, 2, 5), 24, 0.001f, 1);
}

#if NCNN_VULKAN
static int test_layernorm_gpu_reference(const ncnn::Mat& a, int affine_size)
{
    const float eps = 1e-5f;
    ncnn::ParamDict pd;
    pd.set(0, affine_size);
    pd.set(1, eps);
    pd.set(2, 0);

    // Use double precision instead of the generic CPU layer's serial float sums.
    ncnn::Mat expected = a.clone();
    const int groups = a.w * a.h * a.d / affine_size;
    for (int q = 0; q < a.c; q++)
    {
        const float* ptr = a.channel(q);
        float* outptr = expected.channel(q);
        for (int g = 0; g < groups; g++)
        {
            double sum = 0;
            for (int i = 0; i < affine_size; i++)
                sum += ptr[g * affine_size + i];
            const double mean = sum / affine_size;

            double sqsum = 0;
            for (int i = 0; i < affine_size; i++)
            {
                const double v = ptr[g * affine_size + i] - mean;
                sqsum += v * v;
            }
            const double inv_stddev = 1.0 / sqrt(sqsum / affine_size + eps);

            for (int i = 0; i < affine_size; i++)
                outptr[g * affine_size + i] = (float)((ptr[g * affine_size + i] - mean) * inv_stddev);
        }
    }

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
        int ret = test_layer_gpu(ncnn::layer_to_index("LayerNorm"), pd, std::vector<ncnn::Mat>(), opt, a, b, ncnn::Mat(), 0);
        // Allow rounding in the existing float coeffs/norm shaders, including FMA.
        if (ret != 233 && (ret != 0 || CompareMat(expected, b, 0.005f) != 0))
        {
            fprintf(stderr, "test_layernorm_gpu_reference failed a=(%d %d %d %d) affine_size=%d subgroup=%d\n", a.w, a.h, a.d, a.c, affine_size, subgroup);
            return -1;
        }
    }

    return 0;
}

static int test_layernorm_gpu_numerics()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    const int subgroup_size = std::max(4, std::min(128, (int)ncnn::get_gpu_device()->info.subgroup_size()));
    const int sizes[] = {subgroup_size - 1, subgroup_size, subgroup_size + 1, subgroup_size * 2 + 1, 4096, 16384};
    const int channels[] = {1, 4};
    for (int c = 0; c < 2; c++)
    {
        for (int s = 0; s < 6; s++)
        {
            ncnn::Mat a(sizes[s], 1, channels[c]);
            for (int q = 0; q < a.c; q++)
            {
                float* ptr = a.channel(q);
                for (int i = 0; i < a.w; i++)
                    ptr[i] = 100.1f + q * 0.5f;
            }
            if (test_layernorm_gpu_reference(a, a.w) != 0)
                return -1;

            for (int q = 0; q < a.c; q++)
            {
                float* ptr = a.channel(q);
                for (int i = 0; i < a.w; i++)
                    ptr[i] = 100.f + q * 0.5f + (i % 7 - 3) * 0.125f;
            }
            if (test_layernorm_gpu_reference(a, a.w) != 0)
                return -1;
        }
    }

    return 0;
}

static int test_layernorm_gpu_dispatch()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    const int heights[] = {65535, 65536, 65537};
    for (int s = 0; s < 3; s++)
    {
        // Exercise pack1, pack4, and padding at the end of the tiled dispatch.
        ncnn::Mat a(s == 2 ? 8 : 32, heights[s], s == 2 ? 4 : 1);
        for (int q = 0; q < a.c; q++)
        {
            float* ptr = a.channel(q);
            for (int y = 0; y < a.h; y++)
            {
                for (int x = 0; x < a.w; x++)
                    ptr[y * a.w + x] = (x % 7 - 3) * (1.f + (y % 5) * 0.125f) + (y % 11) * 0.25f + q * 0.5f;
            }
        }
        if (test_layernorm_gpu_reference(a, a.w) != 0)
            return -1;
    }

    return 0;
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

    return 0
           || test_layernorm_0()
           || test_layernorm_1()
           || test_layernorm_2()
           || test_layernorm_3()
           || test_layernorm_4()
#if NCNN_VULKAN
           || test_layernorm_gpu_numerics()
           || test_layernorm_gpu_dispatch()
#endif
        ;
}
