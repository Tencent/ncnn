// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#include "layer_type.h"
#include "pipelinecache.h"

static int test_rotaryembed_forward(ncnn::Layer* op, const std::vector<ncnn::Mat>& a, ncnn::Mat& result, const ncnn::Option& opt)
{
    ncnn::VkCompute cmd(op->vkdev);
    std::vector<ncnn::VkMat> a_gpu(a.size());
    for (size_t i = 0; i < a.size(); i++)
    {
        cmd.record_upload(a[i], a_gpu[i], opt);

        int axis_length = a[i].w;
        if (a[i].dims == 2) axis_length = a[i].h;
        if (a[i].dims == 3) axis_length = a[i].c;
        const int expected_pack = axis_length * a[i].elempack % 4 == 0 ? 4 : 1;
        if (a_gpu[i].empty() || a_gpu[i].elempack != expected_pack)
        {
            fprintf(stderr, "RotaryEmbed upload failed input=%zu host pack=%d gpu pack=%d expected=%d\n", i, a[i].elempack, a_gpu[i].elempack, expected_pack);
            return -1;
        }
    }

    std::vector<ncnn::VkMat> b_gpu(1);
    int ret = op->forward(a_gpu, b_gpu, cmd, opt);
    if (ret != 0)
        return ret;

    const ncnn::VkMat& input = a_gpu[0];
    const ncnn::VkMat& output = b_gpu[0];
    // check the layout before record_download can repack the output
    if (output.empty() || output.elempack != input.elempack || output.elemsize != input.elemsize || output.dims != input.dims
        || output.w != input.w || output.h != input.h || output.c != input.c)
    {
        fprintf(stderr, "RotaryEmbed output layout mismatch input pack=%d cache pack=%d output pack=%d\n", input.elempack, a_gpu[1].elempack, output.elempack);
        return -1;
    }

    cmd.record_download(output, result, opt);
    return cmd.submit_and_wait();
}

static int test_rotaryembed_packing(const ncnn::Mat& a, const ncnn::Mat& cos_cache, int interleaved, ncnn::VulkanDevice* vkdev, const ncnn::Option& opt, int input_pack = 1, int cache_pack = 1)
{
    ncnn::ParamDict pd;
    pd.set(0, interleaved);

    std::vector<ncnn::Mat> inputs(3);
    inputs[0] = a;
    inputs[1] = cos_cache;
    inputs[2].create_like(cos_cache);
    Randomize(inputs[2]);

    const std::vector<ncnn::Mat> weights;
    std::vector<ncnn::Mat> expected;
    int ret = test_layer_naive(ncnn::LayerType::RotaryEmbed, pd, weights, inputs, 1, expected, 0);
    if (ret != 0)
        return ret;

    const int host_packs[] = {input_pack, cache_pack, cache_pack};
    std::vector<ncnn::Mat> packed = inputs;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        if (host_packs[i] == 1)
            continue;

        ncnn::convert_packing(inputs[i], packed[i], host_packs[i], opt);
        if (packed[i].empty() || packed[i].elempack != host_packs[i])
        {
            fprintf(stderr, "RotaryEmbed host packing failed input=%zu pack=%d\n", i, host_packs[i]);
            return -1;
        }
    }

    ncnn::Layer* op = ncnn::create_layer_vulkan(ncnn::LayerType::RotaryEmbed);
    op->vkdev = vkdev;
    op->load_param(pd);
    ret = op->create_pipeline(opt);

    ncnn::Mat result;
    if (ret == 0)
        ret = test_rotaryembed_forward(op, packed, result, opt);

    op->destroy_pipeline(opt);
    delete op;

    const float epsilon = opt.use_fp16_arithmetic ? 0.004f : 0.00001f;
    if (ret != 0 || CompareMat(expected[0], result, epsilon) != 0)
    {
        fprintf(stderr, "test_rotaryembed_packing failed dims=%d a=(%d %d %d) cache=(%d %d) host pack=(%d %d) interleaved=%d fp16p=%d fp16s=%d fp16a=%d\n",
                a.dims, a.w, a.h, a.c, cos_cache.w, cos_cache.h, input_pack, cache_pack, interleaved, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_fp16_arithmetic);
        return -1;
    }

    return 0;
}

static int test_rotaryembed_native_packing(ncnn::VulkanDevice* vkdev, const ncnn::Option& opt)
{
    // each pair has one pack1 and one pack4 input along its packing axis
    const ncnn::Mat inputs[] = {
        RandomMat(62),
        RandomMat(64),
        RandomMat(62, 5),
        RandomMat(62, 8),
        RandomMat(62, 8, 3),
        RandomMat(62, 8, 4),
    };
    const int cache_heights[] = {9, 8};
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++)
    {
        const ncnn::Mat& a = inputs[i];
        for (size_t j = 0; j < sizeof(cache_heights) / sizeof(cache_heights[0]); j++)
        {
            const ncnn::Mat cos_half = RandomMat(a.w / 2, cache_heights[j]);
            const ncnn::Mat cos_full = RandomMat(a.w, cache_heights[j]);
            for (int interleaved = 0; interleaved < 2; interleaved++)
            {
                if (test_rotaryembed_packing(a, cos_half, interleaved, vkdev, opt)
                    || test_rotaryembed_packing(a, cos_full, interleaved, vkdev, opt))
                    return -1;
            }
        }
    }

    return 0;
}

static int test_rotaryembed_host_packing(ncnn::VulkanDevice* vkdev, const ncnn::Option& opt)
{
    // upload normalizes all eligible host layouts to vulkan pack4
    const int packs[] = {1, 4, 8, 16, 32};
    const ncnn::Mat inputs[] = {RandomMat(64), RandomMat(64, 32), RandomMat(64, 32, 32)};
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++)
    {
        const ncnn::Mat& a = inputs[i];
        const ncnn::Mat cos_cache = a.dims == 1 ? RandomMat(32) : RandomMat(32, 32);
        for (size_t j = 0; j < sizeof(packs) / sizeof(packs[0]); j++)
        {
            for (size_t k = 0; k < sizeof(packs) / sizeof(packs[0]); k++)
            {
                for (int interleaved = 0; interleaved < 2; interleaved++)
                {
                    if (test_rotaryembed_packing(a, cos_cache, interleaved, vkdev, opt, packs[j], packs[k]) != 0)
                        return -1;
                }
            }
        }
    }

    return 0;
}

static int test_rotaryembed_vulkan()
{
    if (ncnn::get_gpu_count() == 0)
        return 0;

    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return -1;

    ncnn::VkBlobAllocator blob_allocator(vkdev);
    ncnn::VkStagingAllocator staging_allocator(vkdev);
    ncnn::PipelineCache pipeline_cache(vkdev);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_uniform = false;
    opt.use_fp16_arithmetic = false;
    opt.blob_vkallocator = &blob_allocator;
    opt.staging_vkallocator = &staging_allocator;
    opt.pipeline_cache = &pipeline_cache;

    if (test_rotaryembed_native_packing(vkdev, opt) || test_rotaryembed_host_packing(vkdev, opt))
        return -1;

    if (!vkdev->info.support_fp16_arithmetic())
        return 0;

    opt.use_fp16_arithmetic = true;
    if (vkdev->info.support_fp16_packed())
    {
        opt.use_fp16_packed = true;
        if (test_rotaryembed_native_packing(vkdev, opt) != 0)
            return -1;
    }
    if (vkdev->info.support_fp16_storage())
    {
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = true;
        return test_rotaryembed_native_packing(vkdev, opt);
    }

    return 0;
}
#endif // NCNN_VULKAN

int main()
{
#if NCNN_VULKAN
    SRAND(7767517);
    return test_rotaryembed_vulkan();
#else
    return 0;
#endif // NCNN_VULKAN
}
