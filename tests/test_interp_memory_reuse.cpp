// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include "layer_type.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"

static int test_interp_workspace_reuse()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    ncnn::VkBlobAllocator blobs(vkdev);
    ncnn::VkBlobAllocator workspace(vkdev);
    ncnn::VkStagingAllocator staging(vkdev);
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = opt.use_fp16_storage = opt.use_fp16_arithmetic = false;
    opt.use_bf16_packed = opt.use_bf16_storage = false;
    opt.blob_vkallocator = &blobs;
    opt.workspace_vkallocator = &workspace;
    opt.staging_vkallocator = &staging;

    ncnn::ParamDict pd;
    pd.set(0, 3);
    pd.set(3, 32);
    pd.set(4, 32);
    ncnn::Mat inputs[2] = {RandomMat(16, 16, 1), RandomMat(32, 16, 1)};
    ncnn::Mat expected[2];
    ncnn::Mat results[2];
    for (int i = 0; i < 2; i++)
        if (test_layer_naive(ncnn::LayerType::Interp, pd, std::vector<ncnn::Mat>(), inputs[i], expected[i], 0) != 0)
            return -1;

    ncnn::Layer* layer = ncnn::create_layer_vulkan(ncnn::LayerType::Interp);
    if (!layer)
        return -1;
    layer->vkdev = vkdev;
    int ret = layer->load_param(pd);
    if (ret == 0) ret = layer->create_pipeline(opt);
    if (ret == 0)
    {
        ncnn::VkCompute cmd(vkdev);
        ncnn::VkMat bottoms[2];
        ncnn::VkMat tops[2];
        cmd.record_clone(inputs[0], bottoms[0], opt);
        cmd.record_clone(inputs[1], bottoms[1], opt);
        ret = cmd.submit_and_wait();
        if (ret == 0) ret = cmd.reset();
        // The second forward reuses coefficient arrays still read by the first.
        if (ret == 0) ret = layer->forward(bottoms[0], tops[0], cmd, opt);
        if (ret == 0) ret = layer->forward(bottoms[1], tops[1], cmd, opt);
        if (ret == 0)
        {
            cmd.record_clone(tops[0], results[0], opt);
            cmd.record_clone(tops[1], results[1], opt);
            ret = cmd.submit_and_wait();
        }
    }
    layer->destroy_pipeline(opt);
    delete layer;
    if (ret == 0)
        ret = CompareMat(expected[0], results[0], 0.001) || CompareMat(expected[1], results[1], 0.001);
    if (ret != 0)
        fprintf(stderr, "test_interp_workspace_reuse failed ret=%d\n", ret);
    return ret;
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);
    int ret = 0;
#if NCNN_VULKAN
    ncnn::create_gpu_instance();
    if (ncnn::get_gpu_count() != 0)
        ret = test_interp_workspace_reuse();
    ncnn::destroy_gpu_instance();
#endif // NCNN_VULKAN
    return ret;
}
