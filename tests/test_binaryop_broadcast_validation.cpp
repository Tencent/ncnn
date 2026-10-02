// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include "layer_type.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif // NCNN_VULKAN

#if NCNN_VALIDATION
static int test_invalid_cpu(const ncnn::Mat& a, const ncnn::Mat& b, bool naive)
{
    ncnn::Layer* op = naive ? ncnn::create_layer_naive(ncnn::LayerType::BinaryOp) : ncnn::create_layer_cpu(ncnn::LayerType::BinaryOp);
    if (!op)
        return -1;

    ncnn::ParamDict pd;
    pd.set(0, 0);
    ncnn::Option opt;
    opt.num_threads = 1;
    op->load_param(pd);
    op->create_pipeline(opt);
    std::vector<ncnn::Mat> bottom_blobs(2);
    bottom_blobs[0] = a;
    bottom_blobs[1] = b;
    std::vector<ncnn::Mat> top_blobs(1);
    const int ret = op->forward(bottom_blobs, top_blobs, opt);
    const bool output_empty = top_blobs[0].empty();
    op->destroy_pipeline(opt);
    delete op;
    if (ret != -1 || !output_empty)
    {
        fprintf(stderr, "incompatible BinaryOp shapes accepted (naive=%d ret=%d empty=%d)\n", naive, ret, output_empty);
        return -1;
    }
    return 0;
}

#if NCNN_VULKAN
static int test_invalid_vulkan(const ncnn::Mat& a, const ncnn::Mat& b)
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::Layer* op = ncnn::create_layer_vulkan(ncnn::LayerType::BinaryOp);
    if (!op)
        return -1;
    op->vkdev = vkdev;
    ncnn::VkBlobAllocator allocator(vkdev);
    ncnn::Option opt;
    opt.blob_vkallocator = &allocator;
    opt.workspace_vkallocator = &allocator;
    opt.use_fp16_storage = false;
    opt.use_fp16_packed = false;
    ncnn::ParamDict pd;
    pd.set(0, 0);
    op->load_param(pd);
    if (op->create_pipeline(opt) != 0)
    {
        op->destroy_pipeline(opt);
        delete op;
        return -1;
    }

    int ret;
    bool output_empty;
    {
        std::vector<ncnn::VkMat> bottom_blobs(2);
        bottom_blobs[0].create_like(a, &allocator);
        bottom_blobs[1].create_like(b, &allocator);
        if (bottom_blobs[0].empty() || bottom_blobs[1].empty())
        {
            op->destroy_pipeline(opt);
            delete op;
            return -1;
        }
        std::vector<ncnn::VkMat> top_blobs(1);
        ncnn::VkCompute cmd(vkdev);
        // Validation happens before command recording; no malformed dispatch is submitted.
        ret = op->forward(bottom_blobs, top_blobs, cmd, opt);
        output_empty = top_blobs[0].empty();
    }
    op->destroy_pipeline(opt);
    delete op;
    return ret == -1 && output_empty ? 0 : -1;
}
#endif // NCNN_VULKAN
#endif // NCNN_VALIDATION

static int test_valid_broadcast()
{
    // Scalar-like, explicit singleton axes, implicit inner axes, rank-one
    // outer-axis compatibility, and the ambiguous inner-axis preference.
    const ncnn::Mat a[] = {
        RandomMat(5), RandomMat(5, 3), RandomMat(5, 3, 4), RandomMat(5, 3, 2, 4),
        RandomMat(5, 3), RandomMat(5, 3, 4), RandomMat(5, 3, 4), RandomMat(5, 3, 2, 4),
        RandomMat(5, 3), RandomMat(5, 3, 4), RandomMat(5, 3, 4), RandomMat(5, 3, 2, 4),
        RandomMat(5, 3, 2, 4), RandomMat(5, 3, 2, 4),
        RandomMat(5, 3), RandomMat(5, 3, 4), RandomMat(5, 3, 2, 4), RandomMat(4, 4, 4)
    };
    const ncnn::Mat b[] = {
        RandomMat(1), RandomMat(1, 1), RandomMat(1, 1, 1), RandomMat(1, 1, 1, 1),
        RandomMat(1, 3), RandomMat(5, 1, 4), RandomMat(5, 3, 1), RandomMat(5, 3, 1, 4),
        RandomMat(3), RandomMat(4), RandomMat(3, 4), RandomMat(4),
        RandomMat(2, 4), RandomMat(3, 2, 4),
        RandomMat(5), RandomMat(5), RandomMat(5), RandomMat(4)
    };
    ncnn::ParamDict pd;
    pd.set(0, 1); // SUB also checks operand order.
    const std::vector<ncnn::Mat> weights;
    for (size_t i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = a[i];
        inputs[1] = b[i];
        if (test_layer("BinaryOp", pd, weights, inputs))
            return -1;
        inputs[0] = b[i];
        inputs[1] = a[i];
        if (test_layer("BinaryOp", pd, weights, inputs))
            return -1;
    }
    return 0;
}

int main()
{
    SRAND(7767517);
#if NCNN_VALIDATION
    const ncnn::Mat a[] = {
        RandomMat(17), RandomMat(5, 3), RandomMat(5, 3),
        RandomMat(5, 3, 4), RandomMat(5, 3, 4), RandomMat(5, 3, 4),
        RandomMat(5, 3, 2, 4), RandomMat(5, 3, 2, 4), RandomMat(5, 3, 2, 4), RandomMat(5, 3, 2, 4),
        RandomMat(7), RandomMat(7, 2), RandomMat(5, 4, 4), RandomMat(12, 7, 128)
    };
    const ncnn::Mat b[] = {
        RandomMat(129), RandomMat(6, 3), RandomMat(5, 4),
        RandomMat(6, 3, 4), RandomMat(5, 4, 4), RandomMat(5, 3, 5),
        RandomMat(6, 3, 2, 4), RandomMat(5, 4, 2, 4), RandomMat(5, 3, 3, 4), RandomMat(5, 3, 2, 5),
        RandomMat(5, 3), RandomMat(5, 3, 4), RandomMat(5, 3, 2, 4), RandomMat(7, 12, 128)
    };
    for (size_t i = 0; i < sizeof(a) / sizeof(a[0]); i++)
    {
        if (test_invalid_cpu(a[i], b[i], true) || test_invalid_cpu(b[i], a[i], true)
                || test_invalid_cpu(a[i], b[i], false) || test_invalid_cpu(b[i], a[i], false))
            return -1;
#if NCNN_VULKAN
        if (test_invalid_vulkan(a[i], b[i]) || test_invalid_vulkan(b[i], a[i]))
            return -1;
#endif // NCNN_VULKAN
    }

    ncnn::Mat a4;
    ncnn::Mat b4;
    ncnn::Option opt;
    ncnn::convert_packing(a[3], a4, 4, opt);
    ncnn::convert_packing(b[3], b4, 4, opt);
    if (a4.elempack != 4 || b4.elempack != 4
            || test_invalid_cpu(a4, b4, false) || test_invalid_cpu(b4, a4, false))
        return -1;
#if NCNN_VULKAN
    if (test_invalid_vulkan(a4, b4) || test_invalid_vulkan(b4, a4))
        return -1;
#endif // NCNN_VULKAN
#endif // NCNN_VALIDATION
    return test_valid_broadcast();
}
