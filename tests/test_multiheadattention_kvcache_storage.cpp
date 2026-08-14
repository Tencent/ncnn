// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "kvcache_storage.h"

#if NCNN_VULKAN
#include "command.h"
#include "gpu.h"
#endif // NCNN_VULKAN

#include <float.h>

static int test_multiheadattention_kvcache_storage(const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, int qdim)
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

    ncnn::CPUKVCacheStorage storage;
    ncnn::Option reference_opt;
    ncnn::Option opt;
    opt.kvcache_storage = &storage;

    int ret = reference->create_pipeline(reference_opt);
    if (ret == 0)
        ret = op->create_pipeline(opt);
    if (ret != 0)
    {
        reference->destroy_pipeline(reference_opt);
        op->destroy_pipeline(opt);
        delete reference;
        delete op;
        return ret;
    }

    ncnn::Mat reference_key;
    ncnn::Mat reference_value;
    ncnn::Mat key_cache;
    ncnn::Mat value_cache;
    const int append_lengths[] = {15, 2, 1};
    int test_ret = 0;

    for (int i = 0; i < 3; i++)
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
        {
            reference_key = reference_bottoms[1];
            reference_value = reference_bottoms[2];
            reference_bottoms[1].release();
            reference_bottoms[2].release();
            test_ret = ret;
            break;
        }

        std::vector<ncnn::Mat> bottoms(3);
        bottoms[0] = q;
        bottoms[1] = key_cache;
        bottoms[2] = value_cache;
        key_cache.release();
        value_cache.release();

        std::vector<ncnn::Mat> tops(3);
        ret = op->forward(bottoms, tops, opt);
        if (ret != 0)
            test_ret = ret;
        if (test_ret == 0 && CompareMat(reference_tops[0], tops[0], 0.001) != 0)
            test_ret = -1;
        if (test_ret == 0 && (!storage.owns(tops[1]) || !storage.owns(tops[2])))
            test_ret = -1;

        reference_key = reference_tops[1];
        reference_value = reference_tops[2];
        reference_tops[1].release();
        reference_tops[2].release();
        reference_bottoms[1].release();
        reference_bottoms[2].release();

        if (ret == 0)
        {
            key_cache = tops[1];
            value_cache = tops[2];
            tops[1].release();
            tops[2].release();
        }
        else
        {
            key_cache = bottoms[1];
            value_cache = bottoms[2];
        }
        bottoms[1].release();
        bottoms[2].release();

        if (test_ret != 0)
            break;
    }

    reference->destroy_pipeline(reference_opt);
    op->destroy_pipeline(opt);
    delete reference;
    delete op;

    reference_key.release();
    reference_value.release();
    storage.destroy(key_cache);
    storage.destroy(value_cache);

    if (test_ret != 0)
        fprintf(stderr, "test_multiheadattention_kvcache_storage failed ret=%d\n", test_ret);

    return test_ret;
}

static int test_multiheadattention_kvcache_storage()
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

    return test_multiheadattention_kvcache_storage(pd, weights, qdim);
}

static int test_multiheadattention_cross_kvcache_storage()
{
    const int qdim = 12;
    const int kdim = 10;
    const int vdim = 8;
    const int embed_dim = 16;
    const int num_heads = 4;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kdim);
    pd.set(4, vdim);
    pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));
    pd.set(7, 1); // kv_cache

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * kdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * vdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);

    ncnn::ParamDict reference_pd = pd;
    reference_pd.set(7, 0);

    ncnn::Layer* reference = ncnn::create_layer_naive("MultiHeadAttention");
    ncnn::Layer* op = ncnn::create_layer_cpu("MultiHeadAttention");
    if (!reference || !op)
    {
        delete reference;
        delete op;
        return -1;
    }

    reference->load_param(reference_pd);
    reference->load_model(ncnn::ModelBinFromMatArray(weights.data()));
    op->load_param(pd);
    op->load_model(ncnn::ModelBinFromMatArray(weights.data()));

    ncnn::CPUKVCacheStorage storage;
    ncnn::Option reference_opt;
    ncnn::Option opt;
    opt.kvcache_storage = &storage;

    int test_ret = reference->create_pipeline(reference_opt);
    if (test_ret == 0)
        test_ret = op->create_pipeline(opt);

    const ncnn::Mat encoder_key = RandomMat(kdim, 5);
    const ncnn::Mat encoder_value = RandomMat(vdim, 5);
    ncnn::Mat key_cache;
    ncnn::Mat value_cache;

    for (int i = 0; test_ret == 0 && i < 2; i++)
    {
        ncnn::Mat query = RandomMat(qdim, i == 0 ? 3 : 1);

        std::vector<ncnn::Mat> reference_bottoms(3);
        reference_bottoms[0] = query;
        reference_bottoms[1] = encoder_key;
        reference_bottoms[2] = encoder_value;
        std::vector<ncnn::Mat> reference_tops(1);
        test_ret = reference->forward(reference_bottoms, reference_tops, reference_opt);

        std::vector<ncnn::Mat> bottoms(5);
        bottoms[0] = query;
        bottoms[1] = i == 0 ? encoder_key : RandomMat(kdim, 5);
        bottoms[2] = i == 0 ? encoder_value : RandomMat(vdim, 5);
        bottoms[3] = key_cache;
        bottoms[4] = value_cache;
        key_cache.release();
        value_cache.release();

        std::vector<ncnn::Mat> tops(3);
        int ret = test_ret == 0 ? op->forward(bottoms, tops, opt) : test_ret;
        if (ret != 0)
            test_ret = ret;
        if (test_ret == 0 && CompareMat(reference_tops[0], tops[0], 0.001) != 0)
            test_ret = -1;
        if (test_ret == 0 && (!storage.owns(tops[1]) || !storage.owns(tops[2])))
            test_ret = -1;

        if (ret == 0)
        {
            key_cache = tops[1];
            value_cache = tops[2];
            tops[1].release();
            tops[2].release();
        }
        else
        {
            key_cache = bottoms[3];
            value_cache = bottoms[4];
        }
        bottoms[3].release();
        bottoms[4].release();
    }

    reference->destroy_pipeline(reference_opt);
    op->destroy_pipeline(opt);
    delete reference;
    delete op;

    storage.destroy(key_cache);
    storage.destroy(value_cache);

    if (test_ret != 0)
        fprintf(stderr, "test_multiheadattention_cross_kvcache_storage failed ret=%d\n", test_ret);

    return test_ret;
}

#if NCNN_VULKAN
static int test_multiheadattention_vulkan_kvcache_storage()
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

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    int test_ret = 0;
    {
        ncnn::VkWeightAllocator weight_vkallocator(vkdev);
        ncnn::VkWeightStagingAllocator weight_staging_vkallocator(vkdev);
        ncnn::VkKVCacheStorage storage(vkdev);

        ncnn::Layer* reference = ncnn::create_layer_naive("MultiHeadAttention");
        ncnn::Layer* op = ncnn::create_layer_vulkan("MultiHeadAttention");
        if (!reference || !op)
            test_ret = -1;

        ncnn::Option reference_opt;
        ncnn::Option opt;
        opt.use_vulkan_compute = true;
        opt.use_packing_layout = false;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.blob_vkallocator = blob_vkallocator;
        opt.workspace_vkallocator = blob_vkallocator;
        opt.staging_vkallocator = staging_vkallocator;
        opt.kvcache_storage = &storage;

        if (reference)
        {
            reference->load_param(pd);
            reference->load_model(ncnn::ModelBinFromMatArray(weights.data()));
            if (reference->create_pipeline(reference_opt) != 0)
                test_ret = -1;
        }
        if (op)
        {
            op->vkdev = vkdev;
            op->load_param(pd);
            op->load_model(ncnn::ModelBinFromMatArray(weights.data()));
            if (op->create_pipeline(opt) != 0)
                test_ret = -1;
        }

        if (test_ret == 0)
        {
            ncnn::VkTransfer upload_cmd(vkdev);
            ncnn::Option upload_opt = opt;
            upload_opt.blob_vkallocator = &weight_vkallocator;
            upload_opt.workspace_vkallocator = &weight_vkallocator;
            upload_opt.staging_vkallocator = &weight_staging_vkallocator;
            op->upload_model(upload_cmd, upload_opt);
            test_ret = upload_cmd.submit_and_wait();
        }

        ncnn::Mat reference_key;
        ncnn::Mat reference_value;
        ncnn::VkMat key_cache;
        ncnn::VkMat value_cache;
        const int append_lengths[] = {15, 2, 1};
        for (int i = 0; test_ret == 0 && i < 3; i++)
        {
            ncnn::Mat q = RandomMat(qdim, append_lengths[i]);

            std::vector<ncnn::Mat> reference_bottoms(3);
            reference_bottoms[0] = q;
            reference_bottoms[1] = reference_key;
            reference_bottoms[2] = reference_value;
            reference_key.release();
            reference_value.release();
            std::vector<ncnn::Mat> reference_tops(3);
            int ret = reference->forward(reference_bottoms, reference_tops, reference_opt);

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
            if (ret == 0)
                ret = op->forward(bottoms, tops, cmd, opt);

            ncnn::Mat output;
            if (ret == 0)
                cmd.record_clone(tops[0], output, opt);
            if (ret == 0)
                ret = cmd.submit_and_wait();
            if (ret != 0 || CompareMat(reference_tops[0], output, 0.005f) != 0)
                test_ret = -1;
            if (test_ret == 0 && (!storage.owns(tops[1]) || !storage.owns(tops[2])))
                test_ret = -1;

            if (!reference_tops[1].empty())
            {
                reference_key = reference_tops[1];
                reference_value = reference_tops[2];
                reference_tops[1].release();
                reference_tops[2].release();
            }
            else
            {
                reference_key = reference_bottoms[1];
                reference_value = reference_bottoms[2];
            }
            reference_bottoms[1].release();
            reference_bottoms[2].release();

            if (!tops[1].empty())
            {
                key_cache = tops[1];
                value_cache = tops[2];
                tops[1].release();
                tops[2].release();
            }
            else
            {
                key_cache = bottoms[1];
                value_cache = bottoms[2];
            }
            bottoms[1].release();
            bottoms[2].release();
        }

        if (reference)
        {
            reference->destroy_pipeline(reference_opt);
            delete reference;
        }
        if (op)
        {
            op->destroy_pipeline(opt);
            delete op;
        }

        reference_key.release();
        reference_value.release();
        storage.destroy(key_cache);
        storage.destroy(value_cache);
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    if (test_ret != 0)
        fprintf(stderr, "test_multiheadattention_vulkan_kvcache_storage failed ret=%d\n", test_ret);

    return test_ret;
}
#endif // NCNN_VULKAN

#if NCNN_INT8
static int test_multiheadattention_int8_kvcache_storage()
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
    pd.set(7, 1);  // kv_cache
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(12);
    weights[0] = RandomS8Mat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomS8Mat(embed_dim * qdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomS8Mat(embed_dim * qdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomS8Mat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);
    weights[8] = RandomMat(embed_dim, 160.f, 200.f);
    weights[9] = RandomMat(embed_dim, 160.f, 200.f);
    weights[10] = RandomMat(embed_dim, 160.f, 200.f);
    weights[11] = RandomMat(1, 160.f, 200.f);

    return test_multiheadattention_kvcache_storage(pd, weights, qdim);
}
#endif // NCNN_INT8

int main()
{
    SRAND(7767517);

    return 0
           || test_multiheadattention_kvcache_storage()
           || test_multiheadattention_cross_kvcache_storage()
#if NCNN_VULKAN
           || test_multiheadattention_vulkan_kvcache_storage()
#endif
#if NCNN_INT8
           || test_multiheadattention_int8_kvcache_storage()
#endif
           ;
}
