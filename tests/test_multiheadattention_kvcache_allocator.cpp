// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif // NCNN_VULKAN

#include <float.h>

static int test_multiheadattention_kvcache_allocator(const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, int qdim, int max_seqlen_hint)
{
    ncnn::Layer* reference = ncnn::create_layer_naive("MultiHeadAttention");
    ncnn::Layer* op = ncnn::create_layer_cpu("MultiHeadAttention");
    if (!reference || !op)
    {
        delete reference;
        delete op;
        return -1;
    }

    reference->load_param(pd);
    reference->load_model(ncnn::ModelBinFromMatArray(weights.data()));
    op->load_param(pd);
    op->load_model(ncnn::ModelBinFromMatArray(weights.data()));

    ncnn::UnlockedPoolAllocator kvcache_allocator;
    ncnn::Option reference_opt;
    ncnn::Option opt;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.kvcache_allocator = &kvcache_allocator;
    opt.kvcache_max_seqlen_hint = max_seqlen_hint;

    int ret = reference->create_pipeline(reference_opt);
    if (ret == 0)
        ret = op->create_pipeline(opt);

    ncnn::Mat reference_key;
    ncnn::Mat reference_value;
    ncnn::Mat key_cache;
    ncnn::Mat value_cache;
    const int append_lengths[] = {15, 2, 1};
    void* key_data = 0;
    void* value_data = 0;

    for (int i = 0; ret == 0 && i < 3; i++)
    {
        ncnn::Mat q = RandomMat(qdim, append_lengths[i]);

        std::vector<ncnn::Mat> reference_bottoms(3);
        reference_bottoms[0] = q;
        reference_bottoms[1] = reference_key;
        reference_bottoms[2] = reference_value;
        reference_key.release();
        reference_value.release();

        std::vector<ncnn::Mat> reference_tops(3);
        ret = reference->forward(reference_bottoms, reference_tops, reference_opt);
        if (ret != 0)
            break;

        std::vector<ncnn::Mat> bottoms(3);
        bottoms[0] = q;
        bottoms[1] = key_cache;
        bottoms[2] = value_cache;
        key_cache.release();
        value_cache.release();

        std::vector<ncnn::Mat> tops(3);
        ret = op->forward(bottoms, tops, opt);
        if (ret != 0)
            break;

        if (CompareMat(reference_tops[0], tops[0], 0.001) != 0)
            ret = -1;
        if (tops[1].allocator != &kvcache_allocator || tops[2].allocator != &kvcache_allocator)
            ret = -1;

        if (i == 0 || (i == 1 && max_seqlen_hint == 0))
        {
            if (i == 1 && (tops[1].data == key_data || tops[2].data == value_data))
                ret = -1;
            key_data = tops[1].data;
            value_data = tops[2].data;
        }
        if (tops[1].data != key_data || tops[2].data != value_data)
            ret = -1;

        reference_key = reference_tops[1];
        reference_value = reference_tops[2];
        reference_tops[1].release();
        reference_tops[2].release();

        key_cache = tops[1];
        value_cache = tops[2];
        tops[1].release();
        tops[2].release();
    }

    reference->destroy_pipeline(reference_opt);
    op->destroy_pipeline(opt);
    delete reference;
    delete op;

    if (ret != 0)
        fprintf(stderr, "test_multiheadattention_kvcache_allocator failed ret=%d\n", ret);

    return ret;
}

static int test_multiheadattention_kvcache_allocator()
{
    const int qdim = 12;
    const int embed_dim = 16;
    const int num_heads = 4;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));
    pd.set(7, 1); // kv_cache

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * qdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * qdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);

    return 0
           || test_multiheadattention_kvcache_allocator(pd, weights, qdim, 0)
           || test_multiheadattention_kvcache_allocator(pd, weights, qdim, 32);
}

#if NCNN_VULKAN
static int test_multiheadattention_vulkan_kvcache_allocator()
{
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device();
    if (!vkdev)
        return 0;

    const int qdim = 12;
    const int embed_dim = 16;
    const int num_heads = 4;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));
    pd.set(7, 1); // kv_cache

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * qdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * qdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);

    ncnn::VkBlobAllocator blob_vkallocator(vkdev);
    ncnn::VkBlobAllocator kvcache_vkallocator(vkdev);
    ncnn::VkStagingAllocator staging_vkallocator(vkdev);
    ncnn::VkWeightAllocator weight_vkallocator(vkdev);
    ncnn::VkWeightStagingAllocator weight_staging_vkallocator(vkdev);

    ncnn::Layer* reference = ncnn::create_layer_naive("MultiHeadAttention");
    ncnn::Layer* op = ncnn::create_layer_vulkan("MultiHeadAttention");
    if (!reference || !op)
    {
        delete reference;
        delete op;
        return -1;
    }

    ncnn::Option reference_opt;
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.blob_vkallocator = &blob_vkallocator;
    opt.workspace_vkallocator = &blob_vkallocator;
    opt.staging_vkallocator = &staging_vkallocator;
    opt.kvcache_vkallocator = &kvcache_vkallocator;
    opt.kvcache_max_seqlen_hint = 0;

    reference->load_param(pd);
    reference->load_model(ncnn::ModelBinFromMatArray(weights.data()));
    op->vkdev = vkdev;
    op->load_param(pd);
    op->load_model(ncnn::ModelBinFromMatArray(weights.data()));

    int ret = reference->create_pipeline(reference_opt);
    if (ret == 0)
        ret = op->create_pipeline(opt);
    if (ret == 0)
    {
        ncnn::VkTransfer upload_cmd(vkdev);
        ncnn::Option upload_opt = opt;
        upload_opt.blob_vkallocator = &weight_vkallocator;
        upload_opt.workspace_vkallocator = &weight_vkallocator;
        upload_opt.staging_vkallocator = &weight_staging_vkallocator;
        op->upload_model(upload_cmd, upload_opt);
        ret = upload_cmd.submit_and_wait();
    }

    ncnn::Mat reference_key;
    ncnn::Mat reference_value;
    ncnn::VkMat key_cache;
    ncnn::VkMat value_cache;
    const int append_lengths[] = {15, 2, 1};
    ncnn::VkBufferMemory* key_data = 0;
    ncnn::VkBufferMemory* value_data = 0;

    for (int i = 0; ret == 0 && i < 3; i++)
    {
        ncnn::Mat q = RandomMat(qdim, append_lengths[i]);

        std::vector<ncnn::Mat> reference_bottoms(3);
        reference_bottoms[0] = q;
        reference_bottoms[1] = reference_key;
        reference_bottoms[2] = reference_value;
        reference_key.release();
        reference_value.release();
        std::vector<ncnn::Mat> reference_tops(3);
        ret = reference->forward(reference_bottoms, reference_tops, reference_opt);
        if (ret != 0)
            break;

        ncnn::VkCompute cmd(vkdev);
        ncnn::VkMat q_gpu;
        cmd.record_clone(q, q_gpu, opt);
        std::vector<ncnn::VkMat> bottoms(3);
        bottoms[0] = q_gpu;
        bottoms[1] = key_cache;
        bottoms[2] = value_cache;
        key_cache.release();
        value_cache.release();

        std::vector<ncnn::VkMat> tops(3);
        ret = op->forward(bottoms, tops, cmd, opt);

        ncnn::Mat output;
        if (ret == 0)
            cmd.record_clone(tops[0], output, opt);
        if (ret == 0)
            ret = cmd.submit_and_wait();
        if (ret != 0)
            break;

        if (CompareMat(reference_tops[0], output, 0.005f) != 0)
            ret = -1;
        if (tops[1].allocator != &kvcache_vkallocator || tops[2].allocator != &kvcache_vkallocator)
            ret = -1;

        if (i == 0)
        {
            key_data = tops[1].data;
            value_data = tops[2].data;
        }
        if (i == 1)
        {
            if (tops[1].data == key_data || tops[2].data == value_data)
                ret = -1;

            key_data = tops[1].data;
            value_data = tops[2].data;
        }
        if (i == 2 && (tops[1].data != key_data || tops[2].data != value_data))
            ret = -1;

        reference_key = reference_tops[1];
        reference_value = reference_tops[2];
        reference_tops[1].release();
        reference_tops[2].release();

        key_cache = tops[1];
        value_cache = tops[2];
        tops[1].release();
        tops[2].release();
    }

    reference->destroy_pipeline(reference_opt);
    op->destroy_pipeline(opt);
    delete reference;
    delete op;
    if (ret != 0)
        fprintf(stderr, "test_multiheadattention_vulkan_kvcache_allocator failed ret=%d\n", ret);

    return ret;
}
#endif // NCNN_VULKAN

int main()
{
    SRAND(7767517);

    return 0
           || test_multiheadattention_kvcache_allocator()
#if NCNN_VULKAN
           || test_multiheadattention_vulkan_kvcache_allocator()
#endif
           ;
}
