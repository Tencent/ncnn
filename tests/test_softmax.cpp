// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_softmax(const ncnn::Mat& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); // axis
    pd.set(1, 1);    // fixbug0

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("Softmax", pd, weights, a, 0.001);
    if (ret != 0)
    {
        fprintf(stderr, "test_softmax failed a.dims=%d a=(%d %d %d %d) axis=%d\n", a.dims, a.w, a.h, a.d, a.c, axis);
    }

    return ret;
}

static int test_softmax_axes(const ncnn::Mat& a)
{
    for (int axis = 0; axis < a.dims; axis++)
    {
        if (test_softmax(a, axis) != 0)
            return -1;
    }

    return 0;
}

static int test_softmax_negative_axes(const ncnn::Mat& a)
{
    for (int axis = -a.dims; axis < 0; axis++)
    {
        if (test_softmax(a, axis) != 0)
            return -1;
    }

    return 0;
}

static int test_softmax_0()
{
    // channel counts exercise pack16, pack8, pack4 and scalar tails
    return 0
           || test_softmax_negative_axes(RandomMat(9, 5, 7, 32))
           || test_softmax_axes(RandomMat(10, 7, 6, 40))
           || test_softmax_axes(RandomMat(8, 7, 5, 28))
           || test_softmax_axes(RandomMat(9, 7, 5, 31));
}

static int test_softmax_1()
{
    return 0
           || test_softmax_negative_axes(RandomMat(25, 27, 32))
           || test_softmax_axes(RandomMat(22, 19, 40))
           || test_softmax_axes(RandomMat(27, 29, 28))
           || test_softmax_axes(RandomMat(23, 25, 31));
}

static int test_softmax_2()
{
    return 0
           || test_softmax_negative_axes(RandomMat(125, 32))
           || test_softmax_axes(RandomMat(147, 40))
           || test_softmax_axes(RandomMat(127, 28))
           || test_softmax_axes(RandomMat(129, 31));
}

static int test_softmax_3()
{
    return 0
           || test_softmax_negative_axes(RandomMat(128))
           || test_softmax_axes(RandomMat(120))
           || test_softmax_axes(RandomMat(124))
           || test_softmax_axes(RandomMat(127));
}

static int test_softmax_cpu_fp16a(ncnn::Mat a, int axis)
{
    // equal inputs, so every output is exactly 1/n and each sum adds n ones
    a.fill(0.5f);

    ncnn::ParamDict pd;
    pd.set(0, axis); // axis
    pd.set(1, 1);    // fixbug0

    std::vector<ncnn::Mat> weights(0);

    ncnn::Mat b;
    test_layer_naive(ncnn::layer_to_index("Softmax"), pd, weights, a, b, 0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_bf16_packed = false;
    opt.use_bf16_storage = false;

    ncnn::Mat c;
    int ret = test_layer_cpu(ncnn::layer_to_index("Softmax"), pd, weights, opt, a, c, ncnn::Mat(), 0);
    if (ret == 233)
    {
        // this build's Softmax does not support packing or fp16 storage, skip like test_layer does
        return 0;
    }
    if (ret != 0 || c.dims != b.dims || c.w != b.w || c.h != b.h || c.d != b.d || c.c != b.c)
    {
        fprintf(stderr, "test_softmax_cpu_fp16a failed a.dims=%d a=(%d %d %d %d) axis=%d ret=%d\n", a.dims, a.w, a.h, a.d, a.c, axis, ret);
        return -1;
    }

    // the outputs are small, CompareMat accepts any absolute error below its epsilon, so check the relative error
    for (int q = 0; q < b.c; q++)
    {
        const float* pb = b.channel(q);
        const float* pc = c.channel(q);
        for (int i = 0; i < b.w * b.h * b.d; i++)
        {
            if (fabs(pc[i] - pb[i]) > 0.001f * fabs(pb[i]))
            {
                fprintf(stderr, "test_softmax_cpu_fp16a failed a.dims=%d a=(%d %d %d %d) axis=%d got %e expect %e\n", a.dims, a.w, a.h, a.d, a.c, axis, pc[i], pb[i]);
                return -1;
            }
        }
    }

    return 0;
}

static int test_softmax_cpu_fp16a()
{
    // outer dims of 8, 4 and other sizes select elempack 8, 4 and 1
    // the reduced length is long enough to stall a fp16 sum in each path
    return 0
           || test_softmax_cpu_fp16a(ncnn::Mat(32768), 0)
           || test_softmax_cpu_fp16a(ncnn::Mat(32771), 0)
           || test_softmax_cpu_fp16a(ncnn::Mat(4097, 8), 1)
           || test_softmax_cpu_fp16a(ncnn::Mat(8193, 4), 1)
           || test_softmax_cpu_fp16a(ncnn::Mat(32771, 3), 1)
           || test_softmax_cpu_fp16a(ncnn::Mat(13, 32768), 0)
           || test_softmax_cpu_fp16a(ncnn::Mat(13, 16388), 0)
           || test_softmax_cpu_fp16a(ncnn::Mat(13, 4097), 0)
           || test_softmax_cpu_fp16a(ncnn::Mat(4097, 2, 8), 2)
           || test_softmax_cpu_fp16a(ncnn::Mat(13, 4097, 8), 1)
           || test_softmax_cpu_fp16a(ncnn::Mat(13, 1, 32768), 0)
           || test_softmax_cpu_fp16a(ncnn::Mat(4097, 2, 2, 8), 3)
           || test_softmax_cpu_fp16a(ncnn::Mat(13, 4097, 2, 8), 2)
           || test_softmax_cpu_fp16a(ncnn::Mat(13, 2, 4097, 8), 1);
}

static int test_softmax_large()
{
    // large allocations exercise staging reuse and multi-workgroup reductions
    return 0
           || test_softmax(RandomMat(23, 25, 27, 32), -4)
           || test_softmax(RandomMat(24, 27, 29, 28), 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_softmax_0()
           || test_softmax_1()
           || test_softmax_2()
           || test_softmax_3()
           || test_softmax_cpu_fp16a()
           || test_softmax_large()
           // pack8 rows include a full avx512 block, an eight-element tail and a scalar tail
           || test_softmax(RandomMat(25, 24), 0);
}
