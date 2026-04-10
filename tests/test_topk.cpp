// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_SIMPLESTL
// simplemath.h conflicts with system math.h; define only what we need
static const float TEST_INF = 1.f / 0.f;
static const float TEST_NAN = 0.f / 0.f;
#define INFINITY TEST_INF
#define NAN      TEST_NAN
#else
#include <math.h>
#endif

static int test_topk_cpu_forward(const ncnn::Mat& a, int axis, int k, int largest, int sorted, ncnn::Mat& values, ncnn::Mat& indices)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);
    pd.set(1, largest);
    pd.set(2, sorted);
    pd.set(3, k);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("TopK");
    if (!op)
        return -1;

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);

    op->create_pipeline(opt);

    std::vector<ncnn::Mat> bottom_blobs(1);
    bottom_blobs[0] = a;

    std::vector<ncnn::Mat> top_blobs(2);
    int ret = op->forward(bottom_blobs, top_blobs, opt);

    op->destroy_pipeline(opt);
    delete op;

    if (ret != 0)
        return ret;

    values = top_blobs[0];
    indices = top_blobs[1];

    return 0;
}

static int test_topk_cpu_forward_values_only(const ncnn::Mat& a, int axis, int k, int largest, int sorted, ncnn::Mat& values)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);
    pd.set(1, largest);
    pd.set(2, sorted);
    pd.set(3, k);

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer_cpu("TopK");
    if (!op)
        return -1;

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);

    op->create_pipeline(opt);

    std::vector<ncnn::Mat> bottom_blobs(1);
    bottom_blobs[0] = a;

    std::vector<ncnn::Mat> top_blobs(1);
    int ret = op->forward(bottom_blobs, top_blobs, opt);

    op->destroy_pipeline(opt);
    delete op;

    if (ret != 0)
        return ret;

    values = top_blobs[0];

    return 0;
}

static int test_topk(const ncnn::Mat& a, int axis, int k, int largest, int sorted)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);
    pd.set(1, largest);
    pd.set(2, sorted);
    pd.set(3, k);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> a0(1);
    a0[0] = a;

    int ret = test_layer("TopK", pd, weights, a0, 2, 0.01f, TEST_LAYER_DISABLE_AUTO_INPUT_CASTING);
    if (ret != 0)
    {
        fprintf(stderr, "test_topk failed a.dims=%d a=(%d %d %d %d) axis=%d k=%d largest=%d sorted=%d\n", a.dims, a.w, a.h, a.d, a.c, axis, k, largest, sorted);
    }

    return ret;
}

static int test_topk_0()
{
    ncnn::Mat a = RandomMat(13);

    return 0
           || test_topk(a, 0, 1, 1, 1)
           || test_topk(a, 0, 5, 1, 1)
           || test_topk(a, 0, 1, 0, 0)
           || test_topk(a, -1, 7, 0, 1)
           || test_topk(a, 0, 4, 1, 0)
           || test_topk(a, 0, 9, 1, 1);
}

static int test_topk_1()
{
    ncnn::Mat a = RandomMat(12, 17);

    return 0
           || test_topk(a, 0, 1, 1, 1)
           || test_topk(a, 0, 5, 1, 1)
           || test_topk(a, 1, 3, 1, 1)
           || test_topk(a, -1, 8, 0, 1)
           || test_topk(a, 1, 6, 0, 0)
           || test_topk(a, -2, 7, 1, 1);
}

static int test_topk_2()
{
    ncnn::Mat a = RandomMat(8, 9, 11);

    return 0
           || test_topk(a, 0, 3, 1, 1)
           || test_topk(a, 1, 4, 1, 1)
           || test_topk(a, 2, 2, 0, 1)
           || test_topk(a, 2, 5, 1, 0)
           || test_topk(a, -1, 6, 1, 1)
           || test_topk(a, -2, 5, 0, 1)
           || test_topk(a, -3, 7, 1, 1);
}

static int test_topk_3()
{
    ncnn::Mat a = RandomMat(5, 7, 9, 10);

    return 0
           || test_topk(a, 0, 2, 1, 1)
           || test_topk(a, 1, 3, 0, 1)
           || test_topk(a, 2, 4, 1, 1)
           || test_topk(a, 3, 4, 0, 0)
           || test_topk(a, 3, 5, 1, 1)
           || test_topk(a, -1, 6, 0, 1)
           || test_topk(a, -2, 3, 1, 1)
           || test_topk(a, -3, 4, 0, 1)
           || test_topk(a, -4, 2, 1, 1);
}

static int test_topk_inf_order()
{
    ncnn::Mat a(6);
    float* ptr = a;
    ptr[0] = 1.f;
    ptr[1] = INFINITY;
    ptr[2] = -2.f;
    ptr[3] = -INFINITY;
    ptr[4] = 0.5f;
    ptr[5] = 3.f;

    ncnn::Mat values;
    ncnn::Mat indices;

    int ret = test_topk_cpu_forward(a, 0, 2, 1, 1, values, indices);
    if (ret != 0)
    {
        fprintf(stderr, "test_topk_inf_order largest failed ret=%d\n", ret);
        return -1;
    }

    const float* vptr = values;
    const float* iptr = indices;
    if (values.w != 2 || indices.w != 2 || vptr[0] != INFINITY || vptr[1] != 3.f || (int)iptr[0] != 1 || (int)iptr[1] != 5)
    {
        fprintf(stderr, "test_topk_inf_order largest result mismatch\n");
        return -1;
    }

    ret = test_topk_cpu_forward(a, 0, 2, 0, 1, values, indices);
    if (ret != 0)
    {
        fprintf(stderr, "test_topk_inf_order smallest failed ret=%d\n", ret);
        return -1;
    }

    vptr = values;
    iptr = indices;
    if (values.w != 2 || indices.w != 2 || vptr[0] != -INFINITY || vptr[1] != -2.f || (int)iptr[0] != 3 || (int)iptr[1] != 2)
    {
        fprintf(stderr, "test_topk_inf_order smallest result mismatch\n");
        return -1;
    }

    return 0;
}

static int test_topk_nan_robust()
{
    ncnn::Mat a(4);
    float* ptr = a;
    ptr[0] = 1.f;
    ptr[1] = NAN;
    ptr[2] = 2.f;
    ptr[3] = -1.f;

    ncnn::Mat values;
    ncnn::Mat indices;

    int ret = test_topk_cpu_forward(a, 0, 2, 1, 1, values, indices);
    if (ret != 0)
    {
        fprintf(stderr, "test_topk_nan_robust sorted failed ret=%d\n", ret);
        return -1;
    }

    if (values.w != 2 || indices.w != 2)
    {
        fprintf(stderr, "test_topk_nan_robust sorted shape mismatch\n");
        return -1;
    }

    const float* vptr = values;
    const float* iptr = indices;
    if (vptr[0] != 2.f || vptr[1] != 1.f || (int)iptr[0] != 2 || (int)iptr[1] != 0)
    {
        fprintf(stderr, "test_topk_nan_robust sorted largest mismatch\n");
        return -1;
    }

    ret = test_topk_cpu_forward(a, 0, 2, 0, 1, values, indices);
    if (ret != 0)
    {
        fprintf(stderr, "test_topk_nan_robust sorted smallest failed ret=%d\n", ret);
        return -1;
    }

    if (values.w != 2 || indices.w != 2)
    {
        fprintf(stderr, "test_topk_nan_robust sorted smallest shape mismatch\n");
        return -1;
    }

    vptr = values;
    iptr = indices;
    if (vptr[0] != -1.f || vptr[1] != 1.f || (int)iptr[0] != 3 || (int)iptr[1] != 0)
    {
        fprintf(stderr, "test_topk_nan_robust sorted smallest mismatch\n");
        return -1;
    }

    ret = test_topk_cpu_forward(a, 0, 2, 1, 0, values, indices);
    if (ret != 0)
    {
        fprintf(stderr, "test_topk_nan_robust unsorted failed ret=%d\n", ret);
        return -1;
    }

    if (values.w != 2 || indices.w != 2)
    {
        fprintf(stderr, "test_topk_nan_robust unsorted shape mismatch\n");
        return -1;
    }

    iptr = indices;
    if ((int)iptr[0] < 0 || (int)iptr[0] >= 4 || (int)iptr[1] < 0 || (int)iptr[1] >= 4)
    {
        fprintf(stderr, "test_topk_nan_robust unsorted invalid indices\n");
        return -1;
    }

    return 0;
}

static int test_topk_values_only_fastpaths()
{
    ncnn::Mat a(5);
    float* ptr = a;
    ptr[0] = 1.f;
    ptr[1] = -2.f;
    ptr[2] = 4.f;
    ptr[3] = 3.f;
    ptr[4] = 0.f;

    ncnn::Mat values;

    int ret = test_topk_cpu_forward_values_only(a, 0, 1, 1, 0, values);
    if (ret != 0)
    {
        fprintf(stderr, "test_topk_values_only_fastpaths k1 failed ret=%d\n", ret);
        return -1;
    }

    if (values.w != 1 || ((const float*)values)[0] != 4.f)
    {
        fprintf(stderr, "test_topk_values_only_fastpaths k1 result mismatch\n");
        return -1;
    }

    ret = test_topk_cpu_forward_values_only(a, 0, 5, 1, 0, values);
    if (ret != 0)
    {
        fprintf(stderr, "test_topk_values_only_fastpaths fullk failed ret=%d\n", ret);
        return -1;
    }

    if (values.w != 5)
    {
        fprintf(stderr, "test_topk_values_only_fastpaths fullk shape mismatch\n");
        return -1;
    }

    const float* vptr = values;
    for (int i = 0; i < 5; i++)
    {
        if (vptr[i] != ptr[i])
        {
            fprintf(stderr, "test_topk_values_only_fastpaths fullk value mismatch\n");
            return -1;
        }
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_topk_0()
           || test_topk_1()
           || test_topk_2()
           || test_topk_3()
           || test_topk_inf_order()
           || test_topk_nan_robust()
           || test_topk_values_only_fastpaths();
}
