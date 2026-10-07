// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "net.h"
#include "testutil.h"

static int test_batch_expression_support()
{
    const char* expressions[] = {"-1,0n", "//(0w,1n),1n", "1w,*(2n,3n),-1"};
    for (int i = 0; i < 3; i++)
    {
        ncnn::Layer* layer = ncnn::create_layer("Reshape");
        ncnn::ParamDict pd;
        pd.set(6, expressions[i]);
        int ret = layer->load_param(pd);
        delete layer;
#if NCNN_BATCH
        if (ret != 0)
#else
        if (ret == 0)
#endif
        {
            fprintf(stderr, "test_batch_expression_support failed expr=%s ret=%d\n", expressions[i], ret);
            return -1;
        }
    }
    return 0;
}

#if NCNN_BATCH
// three logical dimensions, with one carried by native n
static ncnn::Mat logical_sequence(int a, int b, int c, int axis)
{
    const int shape[3] = {a, b, c};
    int physical[2];
    int j = 0;
    for (int i = 0; i < 3; i++)
        if (i != axis)
            physical[j++] = shape[i];

    ncnn::Mat m(physical[1], physical[0], (size_t)4, 1, shape[axis]);
    int suffix = 1;
    for (int i = axis + 1; i < 3; i++)
        suffix *= shape[i];
    for (int i = 0; i < a * b * c; i++)
    {
        const int batch = (i / suffix) % shape[axis];
        const int offset = (i / (suffix * shape[axis])) * suffix + i % suffix;
        float* ptr = m.batch(batch);
        ptr[offset] = (float)i;
    }
    return m;
}

static int test_partition(int a, int b, int c, int axis, int x, int y, int z, int out_axis)
{
    const ncnn::Mat input = logical_sequence(a, b, c, axis);
    const ncnn::Mat expected = logical_sequence(x, y, z, out_axis);
    ncnn::ParamDict pd;
    pd.set(0, z);
    pd.set(1, y);
    pd.set(2, x);
    pd.set(12, axis);
    pd.set(13, out_axis);

    std::vector<ncnn::Mat> weights;
    ncnn::Mat naive;
    int ret = test_layer_naive(ncnn::layer_to_index("Reshape"), pd, weights, input, naive, 0);
    if (ret != 0 || naive.n != expected.n || CompareMat(expected, naive, 0.f) != 0)
        return -1;

    for (int packing = 0; packing < 2; packing++)
    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = packing;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_storage = false;
        ncnn::Mat output;
        ret = test_layer_cpu(ncnn::layer_to_index("Reshape"), pd, weights, opt, input, output, ncnn::Mat(), 0);
        if (ret != 0 || output.n != expected.n || CompareMat(expected, output, 0.f) != 0)
            return -1;
#if NCNN_VULKAN
        if (packing && ncnn::get_gpu_count() > 0)
        {
            ret = test_layer_gpu(ncnn::layer_to_index("Reshape"), pd, weights, opt, input, output, ncnn::Mat(), 0);
            if (ret != 0 || output.n != expected.n || CompareMat(expected, output, 0.f) != 0)
                return -1;
        }
#endif
    }
    return 0;
}

static int test_shape_references(const char* expression, int data_batch, int width, int height, bool explicit_batch, bool vulkan, bool packing)
{
    char param[512];
    snprintf(param, sizeof(param), "7767517\n"
             "4 4\n"
             "Input data 0 1 data\n"
             "Input ref1 0 1 ref1\n"
             "Input ref2 0 1 ref2\n"
             "Reshape reshape 3 1 data ref1 ref2 out 6=\"%s\" %s\n",
             expression, explicit_batch ? "12=0 13=0" : "");

    ncnn::Net net;
    net.opt.num_threads = 1;
    net.opt.use_vulkan_compute = vulkan;
    net.opt.use_packing_layout = packing;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    if (net.load_param_mem(param) != 0 || net.load_model((const unsigned char*)"") != 0)
        return -1;

    ncnn::Mat data(24, (size_t)4, 1, data_batch);
    ncnn::Mat ref1(12, (size_t)4, 1, 2);
    ncnn::Mat ref2(2, (size_t)4, 1, 3);
    ref1.fill(0.f);
    ref2.fill(0.f);
    ncnn::Mat expected(width, height, (size_t)4, 1, data_batch);
    for (int b = 0; b < data_batch; b++)
    {
        float* ptr = data.batch(b);
        float* expected_ptr = expected.batch(b);
        for (int i = 0; i < 24; i++)
            expected_ptr[i] = ptr[i] = (float)(b * 24 + i);
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", data);
    ex.input("ref1", ref1);
    ex.input("ref2", ref2);
    ncnn::Mat output;
    int ret = ex.extract("out", output);
    if (ret != 0 || output.n != data_batch || CompareMat(expected, output, 0.f) != 0)
    {
        fprintf(stderr, "test_shape_references failed expr=%s data_batch=%d vulkan=%d packing=%d ret=%d output.n=%d\n", expression, data_batch, vulkan, packing, ret, output.n);
        return -1;
    }
    return 0;
}

static int test_batch_reshape()
{
    if (test_partition(3, 2, 8, 1, 4, 2, 6, 1)
            || test_partition(4, 2, 24, 1, 8, 2, 12, 1)
            || test_partition(16, 2, 24, 1, 8, 2, 48, 1)
            || test_partition(3, 1, 8, 1, 4, 1, 6, 1)
            || test_partition(2, 3, 4, 0, 1, 2, 12, 1)
            || test_partition(4, 3, 2, 2, 2, 2, 6, 1))
        return -1;

    for (int vulkan = 0; vulkan < 2; vulkan++)
    {
#if NCNN_VULKAN
        if (vulkan && ncnn::get_gpu_count() == 0)
            continue;
#else
        if (vulkan)
            continue;
#endif
        for (int packing = 0; packing < 2; packing++)
        {
            if (test_shape_references("1w,-1", 1, 12, 2, false, vulkan, packing)
                    || test_shape_references("1w,2w", 1, 12, 2, false, vulkan, packing)
                    || test_shape_references("*(1n,2n),-1", 1, 6, 4, false, vulkan, packing)
                    || test_shape_references("1w,2w", 2, 12, 2, false, vulkan, packing)
                    || test_shape_references("0n,-1", 2, 2, 12, false, vulkan, packing)
                    || test_shape_references("//(0w,1n),1n,0n", 2, 12, 2, true, vulkan, packing))
                return -1;
        }
    }
    return 0;
}
#endif // NCNN_BATCH

int main()
{
    if (test_batch_expression_support())
        return -1;
#if NCNN_BATCH
    return test_batch_reshape();
#else
    return 0;
#endif
}
