// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_VULKAN
#include "gpu.h"

#include <math.h>
#endif

static int test_instancenorm(const ncnn::Mat& a, float eps, int affine)
{
    int channels = a.c;

    ncnn::ParamDict pd;
    pd.set(0, affine ? channels : 0);
    pd.set(1, eps);
    pd.set(2, affine);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(channels);
    weights[1] = RandomMat(channels);

    int ret = test_layer("InstanceNorm", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_instancenorm failed a.dims=%d a=(%d %d %d %d) eps=%f affine=%d\n", a.dims, a.w, a.h, a.d, a.c, eps, affine);
    }

    return ret;
}

static int test_instancenorm_0()
{
    return 0
           || test_instancenorm(RandomMat(6, 4, 2), 0.01f, 0)
           || test_instancenorm(RandomMat(3, 3, 12), 0.002f, 0)
           || test_instancenorm(RandomMat(5, 7, 16), 0.02f, 0)
           || test_instancenorm(RandomMat(6, 4, 2), 0.01f, 1)
           || test_instancenorm(RandomMat(3, 3, 12), 0.002f, 1)
           || test_instancenorm(RandomMat(5, 7, 16), 0.02f, 1);
}

static int test_instancenorm_1()
{
    return 0
           || test_instancenorm(RandomMat(5, 4, 3, 3), 0.01f, 0)
           || test_instancenorm(RandomMat(3, 3, 2, 8), 0.002f, 0)
           || test_instancenorm(RandomMat(5, 4, 3, 3), 0.01f, 1)
           || test_instancenorm(RandomMat(3, 3, 2, 8), 0.002f, 1);
}

#if NCNN_VULKAN
static int test_instancenorm_gpu_reference(const ncnn::Mat& a)
{
    const float eps = 1e-5f;
    const int size = a.w * a.h * a.d;
    ncnn::Mat expected = a.clone();
    for (int q = 0; q < a.c; q++)
    {
        const float* ptr = a.channel(q);
        float* outptr = expected.channel(q);
        double sum = 0.0;
        for (int i = 0; i < size; i++)
            sum += ptr[i];
        const double mean = sum / size;
        double sqsum = 0.0;
        for (int i = 0; i < size; i++)
        {
            const double d = ptr[i] - mean;
            sqsum += d * d;
        }
        const double inv_stddev = 1.0 / sqrt(sqsum / size + eps);
        for (int i = 0; i < size; i++)
            outptr[i] = (float)((ptr[i] - mean) * inv_stddev);
    }

    ncnn::ParamDict pd;
    pd.set(0, 0);
    pd.set(1, eps);
    pd.set(2, 0);

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
        int ret = test_layer_gpu(ncnn::layer_to_index("InstanceNorm"), pd, std::vector<ncnn::Mat>(), opt, a, b, ncnn::Mat(), 0);
        // allow rounding in the existing float coeffs/norm shaders, including FMA
        if (ret != 233 && (ret != 0 || CompareMat(expected, b, 0.005f) != 0))
        {
            fprintf(stderr, "test_instancenorm_gpu_reference failed a=(%d %d %d) subgroup=%d\n", a.w, a.h, a.c, subgroup);
            return -1;
        }
    }

    return 0;
}

static int test_instancenorm_gpu_numerics()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    const int channels[] = {1, 4};
    for (int c = 0; c < 2; c++)
    {
        ncnn::Mat a(256, 256, channels[c]);
        for (int q = 0; q < a.c; q++)
        {
            float* ptr = a.channel(q);
            for (int i = 0; i < a.w * a.h; i++)
                ptr[i] = 10.1f + q * 0.5f;
        }
        if (test_instancenorm_gpu_reference(a) != 0)
            return -1;

        a.create(257, 255, channels[c]);
        for (int q = 0; q < a.c; q++)
        {
            float* ptr = a.channel(q);
            for (int i = 0; i < a.w * a.h; i++)
                ptr[i] = 10.f + q * 0.5f + (i % 7 - 3) * 0.125f;
        }
        if (test_instancenorm_gpu_reference(a) != 0)
            return -1;
    }

    return 0;
}

static int test_instancenorm_gpu_dispatch()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    ncnn::Mat a(2, 1, 65537);
    for (int q = 0; q < a.c; q++)
    {
        float* ptr = a.channel(q);
        ptr[0] = -1.f;
        ptr[1] = 1.f;
    }
    return test_instancenorm_gpu_reference(a);
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

    return 0
           || test_instancenorm_0()
           || test_instancenorm_1()
#if NCNN_VULKAN
           || test_instancenorm_gpu_numerics()
           || test_instancenorm_gpu_dispatch()
#endif // NCNN_VULKAN
           ;
}
