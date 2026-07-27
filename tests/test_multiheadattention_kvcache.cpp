// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include <float.h>

static int test_multiheadattention_cross_kvcache(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int embed_dim, int num_heads, int attn_mask, int input_kvcache, int mask_channels = 0)
{
    const int qdim = q.w;
    const int kdim = k.w;
    const int vdim = v.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kdim);
    pd.set(4, vdim);
    pd.set(5, attn_mask);
    pd.set(6, 1.f / sqrtf(embed_dim / num_heads));
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

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (attn_mask)
    {
        as.push_back(mask_channels > 0 ? RandomMat(k.h, q.h, mask_channels) : RandomMat(k.h, q.h));
    }

    if (input_kvcache)
    {
        as.push_back(RandomMat(k.h, embed_dim));
        as.push_back(RandomMat(k.h, embed_dim));
    }
    else
    {
        as.push_back(ncnn::Mat());
        as.push_back(ncnn::Mat());
    }

    float epsilon = 0.005;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_cross_kvcache failed q=(%d %d) k=(%d %d) v=(%d %d) embed_dim=%d num_heads=%d kdim=%d vdim=%d attn_mask=%d input_kvcache=%d\n", q.w, q.h, k.w, k.h, v.w, v.h, embed_dim, num_heads, kdim, vdim, attn_mask, input_kvcache);
    }

    return ret;
}

static int test_multiheadattention_cross_kvcache(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int embed_dim, int num_heads, int attn_mask)
{
    return test_multiheadattention_cross_kvcache(q, k, v, embed_dim, num_heads, attn_mask, 0) || test_multiheadattention_cross_kvcache(q, k, v, embed_dim, num_heads, attn_mask, 1);
}

static int test_multiheadattention_self_kvcache_prefill(const ncnn::Mat& q, int embed_dim, int num_heads)
{
    const int qdim = q.w;
    const int dst_seqlen = q.h;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));
    pd.set(5, 1); // attn_mask
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

    ncnn::Mat attn_mask(dst_seqlen, dst_seqlen);
    attn_mask.fill(0.f);
    for (int i = 0; i < dst_seqlen; i++)
    {
        for (int j = i + 1; j < dst_seqlen; j++)
        {
            attn_mask.row(i)[j] = -60000.f;
        }
    }

    std::vector<ncnn::Mat> as(4);
    as[0] = q;
    as[1] = attn_mask;
    as[2] = ncnn::Mat();
    as[3] = ncnn::Mat();

    float epsilon = 0.005;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_self_kvcache_prefill failed q=(%d %d) embed_dim=%d num_heads=%d\n", q.w, q.h, embed_dim, num_heads);
    }

    return ret;
}

static int test_multiheadattention_self_kvcache_decode(int qdim, int past_seqlen, int cur_seqlen, int embed_dim, int num_heads, int mask_channels = 0)
{
    const int dst_seqlen = past_seqlen + cur_seqlen;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));
    pd.set(5, mask_channels > 0); // attn_mask
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

    std::vector<ncnn::Mat> as(mask_channels > 0 ? 4 : 3);
    as[0] = RandomMat(qdim, cur_seqlen);
    if (mask_channels > 0)
    {
        as[1] = RandomMat(dst_seqlen, cur_seqlen, mask_channels);
        as[2] = RandomMat(past_seqlen, embed_dim);
        as[3] = RandomMat(past_seqlen, embed_dim);
    }
    else
    {
        as[1] = RandomMat(past_seqlen, embed_dim);
        as[2] = RandomMat(past_seqlen, embed_dim);
    }

    float epsilon = 0.005;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_self_kvcache_decode failed qdim=%d past_seqlen=%d cur_seqlen=%d embed_dim=%d num_heads=%d mask_channels=%d\n", qdim, past_seqlen, cur_seqlen, embed_dim, num_heads, mask_channels);
    }

    return ret;
}

static int test_multiheadattention_self_kvcache_rolling(int qdim, int embed_dim, int num_heads, int steps, int tokens_per_step)
{
    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(5, 0); // attn_mask
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

    ncnn::PoolAllocator reference_blob_allocator;
    ncnn::UnlockedPoolAllocator reference_workspace_allocator;
    ncnn::PoolAllocator cpu_blob_allocator;
    ncnn::UnlockedPoolAllocator cpu_workspace_allocator;

    ncnn::Option reference_opt;
    reference_opt.num_threads = 1;
    reference_opt.lightmode = true;
    reference_opt.use_packing_layout = false;
    reference_opt.blob_allocator = &reference_blob_allocator;
    reference_opt.workspace_allocator = &reference_workspace_allocator;

    ncnn::Option cpu_opt = reference_opt;
    cpu_opt.blob_allocator = &cpu_blob_allocator;
    cpu_opt.workspace_allocator = &cpu_workspace_allocator;

    const int typeindex = ncnn::layer_to_index("MultiHeadAttention");
    ncnn::Layer* reference_op = ncnn::create_layer_naive(typeindex);
    ncnn::Layer* cpu_op = ncnn::create_layer_cpu(typeindex);

    reference_op->load_param(pd);
    cpu_op->load_param(pd);
    reference_op->load_model(ncnn::ModelBinFromMatArray(weights.data()));
    cpu_op->load_model(ncnn::ModelBinFromMatArray(weights.data()));
    reference_op->create_pipeline(reference_opt);
    cpu_op->create_pipeline(cpu_opt);

    std::vector<ncnn::Mat> reference_cache(2);
    std::vector<ncnn::Mat> cpu_cache(2);
    int ret = 0;

    for (int i = 0; i < steps; i++)
    {
        const ncnn::Mat token = RandomMat(qdim, tokens_per_step);

        std::vector<ncnn::Mat> reference_bottoms(3);
        reference_bottoms[0] = token;
        reference_bottoms[1] = reference_cache[0];
        reference_bottoms[2] = reference_cache[1];
        std::vector<ncnn::Mat> reference_tops(3);

        std::vector<ncnn::Mat> cpu_bottoms(3);
        cpu_bottoms[0] = token;
        cpu_bottoms[1] = cpu_cache[0];
        cpu_bottoms[2] = cpu_cache[1];
        std::vector<ncnn::Mat> cpu_tops(3);

        const int reference_ret = reference_op->forward(reference_bottoms, reference_tops, reference_opt);
        const int cpu_ret = cpu_op->forward(cpu_bottoms, cpu_tops, cpu_opt);
        if (reference_ret != 0 || cpu_ret != 0 || CompareMat(reference_tops, cpu_tops, 0.005) != 0)
        {
            fprintf(stderr, "test_multiheadattention_self_kvcache_rolling output mismatch at step=%d\n", i);
            ret = -1;
            break;
        }

        const int expected_cache_len = (i + 1) * tokens_per_step;
        if (cpu_tops[1].w != expected_cache_len || cpu_tops[2].w != expected_cache_len
            || reference_tops[1].allocator != &reference_blob_allocator
            || reference_tops[2].allocator != &reference_blob_allocator
            || cpu_tops[1].allocator != &cpu_blob_allocator
            || cpu_tops[2].allocator != &cpu_blob_allocator)
        {
            fprintf(stderr, "test_multiheadattention_self_kvcache_rolling invalid cache at step=%d k_w=%d v_w=%d\n", i, cpu_tops[1].w, cpu_tops[2].w);
            ret = -1;
            break;
        }

        reference_cache[0] = reference_tops[1];
        reference_cache[1] = reference_tops[2];
        cpu_cache[0] = cpu_tops[1];
        cpu_cache[1] = cpu_tops[2];
    }

    reference_op->destroy_pipeline(reference_opt);
    cpu_op->destroy_pipeline(cpu_opt);
    delete reference_op;
    delete cpu_op;

    return ret;
}

static int test_multiheadattention_0()
{
    return 0
           || test_multiheadattention_cross_kvcache(RandomMat(62, 66), RandomMat(32, 66), RandomMat(20, 66), 62, 2, 0)
           || test_multiheadattention_cross_kvcache(RandomMat(26, 64), RandomMat(32, 64), RandomMat(18, 64), 26, 2, 1)
           || test_multiheadattention_cross_kvcache(RandomMat(64, 128), RandomMat(64, 128), RandomMat(64, 128), 64, 4, 0)
           || test_multiheadattention_cross_kvcache(RandomMat(48, 127), RandomMat(64, 127), RandomMat(64, 127), 64, 16, 1)
           || test_multiheadattention_cross_kvcache(RandomMat(16, 128), RandomMat(44, 128), RandomMat(55, 128), 16, 2, 0)
           || test_multiheadattention_cross_kvcache(RandomMat(12, 128), RandomMat(44, 127), RandomMat(55, 127), 16, 4, 1)
           || test_multiheadattention_cross_kvcache(RandomMat(12, 17), RandomMat(28, 127), RandomMat(32, 127), 12, 3, 0)
           || test_multiheadattention_cross_kvcache(RandomMat(12, 17), RandomMat(28, 32), RandomMat(11, 32), 12, 3, 1);
}

static int test_multiheadattention_1()
{
    return 0
           || test_multiheadattention_self_kvcache_prefill(RandomMat(64, 128), 64, 4)
           || test_multiheadattention_self_kvcache_prefill(RandomMat(48, 127), 64, 8)
           || test_multiheadattention_self_kvcache_prefill(RandomMat(48, 127), 64, 8);
}

static int test_multiheadattention_2()
{
    return 0
           || test_multiheadattention_self_kvcache_decode(64, 128, 1, 64, 4)
           || test_multiheadattention_self_kvcache_decode(48, 127, 1, 64, 8)
           || test_multiheadattention_self_kvcache_decode(64, 256, 1, 64, 4)
           || test_multiheadattention_self_kvcache_decode(64, 1024, 1, 64, 4)
           || test_multiheadattention_self_kvcache_decode(64, 2560, 1, 64, 4)
           || test_multiheadattention_self_kvcache_decode(64, 1024, 1, 64, 4, 1)
           || test_multiheadattention_self_kvcache_decode(64, 127, 3, 64, 4)
           || test_multiheadattention_self_kvcache_decode(64, 1024, 3, 64, 4, 1)
           || test_multiheadattention_self_kvcache_decode(64, 2560, 3, 64, 4, 1)
           || test_multiheadattention_self_kvcache_decode(64, 1024, 4, 64, 4, 1)
           || test_multiheadattention_self_kvcache_decode(64, 1024, 4, 64, 4, 4)
           || test_multiheadattention_cross_kvcache(RandomMat(64, 1), RandomMat(64, 1024), RandomMat(64, 1024), 64, 4, 1, 1, 1)
           || test_multiheadattention_self_kvcache_rolling(64, 64, 4, 257, 1)
           || test_multiheadattention_self_kvcache_rolling(64, 64, 4, 257, 3);
}

#if NCNN_INT8
static int test_multiheadattention_int8_cross_kvcache(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int embed_dim, int num_heads, int attn_mask, int input_kvcache)
{
    const int qdim = q.w;
    const int kdim = k.w;
    const int vdim = v.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kdim);
    pd.set(4, vdim);
    pd.set(5, attn_mask);
    pd.set(6, 1.f / sqrtf(embed_dim / num_heads));
    pd.set(7, 1);  // kv_cache
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(12);
    weights[0] = RandomS8Mat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomS8Mat(embed_dim * kdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomS8Mat(embed_dim * vdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomS8Mat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);
    weights[8] = RandomMat(embed_dim, 160.f, 200.f);
    weights[9] = RandomMat(embed_dim, 160.f, 200.f);
    weights[10] = RandomMat(embed_dim, 160.f, 200.f);
    weights[11] = RandomMat(1, 160.f, 200.f);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (attn_mask)
    {
        as.push_back(RandomMat(k.h, q.h));
    }

    if (input_kvcache)
    {
        as.push_back(RandomMat(k.h, embed_dim));
        as.push_back(RandomMat(k.h, embed_dim));
    }
    else
    {
        as.push_back(ncnn::Mat());
        as.push_back(ncnn::Mat());
    }

    float epsilon = 0.1;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_int8_cross_kvcache failed q=(%d %d) k=(%d %d) v=(%d %d) embed_dim=%d num_heads=%d kdim=%d vdim=%d attn_mask=%d input_kvcache=%d\n", q.w, q.h, k.w, k.h, v.w, v.h, embed_dim, num_heads, kdim, vdim, attn_mask, input_kvcache);
    }

    return ret;
}

static int test_multiheadattention_int8_cross_kvcache(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int embed_dim, int num_heads, int attn_mask)
{
    return test_multiheadattention_int8_cross_kvcache(q, k, v, embed_dim, num_heads, attn_mask, 0) || test_multiheadattention_int8_cross_kvcache(q, k, v, embed_dim, num_heads, attn_mask, 1);
}

static int test_multiheadattention_int8_self_kvcache_prefill(const ncnn::Mat& q, int embed_dim, int num_heads)
{
    const int qdim = q.w;
    const int dst_seqlen = q.h;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));
    pd.set(5, 1);  // attn_mask
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

    ncnn::Mat attn_mask(dst_seqlen, dst_seqlen);
    attn_mask.fill(0.f);
    for (int i = 0; i < dst_seqlen; i++)
    {
        for (int j = i + 1; j < dst_seqlen; j++)
        {
            attn_mask.row(i)[j] = -60000.f;
        }
    }

    std::vector<ncnn::Mat> as(4);
    as[0] = q;
    as[1] = attn_mask;
    as[2] = ncnn::Mat();
    as[3] = ncnn::Mat();

    float epsilon = 0.1;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_int8_self_kvcache_prefill failed q=(%d %d) embed_dim=%d num_heads=%d\n", q.w, q.h, embed_dim, num_heads);
    }

    return ret;
}

static int test_multiheadattention_int8_self_kvcache_decode(int qdim, int past_seqlen, int cur_seqlen, int embed_dim, int num_heads, int mask_channels = 0)
{
    const int dst_seqlen = past_seqlen + cur_seqlen;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));
    pd.set(5, mask_channels > 0); // attn_mask
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

    std::vector<ncnn::Mat> as(mask_channels > 0 ? 4 : 3);
    as[0] = RandomMat(qdim, cur_seqlen);
    if (mask_channels > 0)
    {
        as[1] = RandomMat(dst_seqlen, cur_seqlen, mask_channels);
        as[2] = RandomMat(past_seqlen, embed_dim);
        as[3] = RandomMat(past_seqlen, embed_dim);
    }
    else
    {
        as[1] = RandomMat(past_seqlen, embed_dim);
        as[2] = RandomMat(past_seqlen, embed_dim);
    }

    float epsilon = 0.1;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_int8_self_kvcache_decode failed qdim=%d past_seqlen=%d cur_seqlen=%d embed_dim=%d num_heads=%d mask_channels=%d\n", qdim, past_seqlen, cur_seqlen, embed_dim, num_heads, mask_channels);
    }

    return ret;
}

static int test_multiheadattention_3()
{
    return 0
           || test_multiheadattention_int8_cross_kvcache(RandomMat(62, 66), RandomMat(32, 66), RandomMat(20, 66), 62, 2, 0)
           || test_multiheadattention_int8_cross_kvcache(RandomMat(26, 64), RandomMat(32, 64), RandomMat(18, 64), 26, 2, 1)
           || test_multiheadattention_int8_cross_kvcache(RandomMat(64, 128), RandomMat(64, 128), RandomMat(64, 128), 64, 4, 0)
           || test_multiheadattention_int8_cross_kvcache(RandomMat(48, 127), RandomMat(64, 127), RandomMat(64, 127), 64, 16, 1)
           || test_multiheadattention_int8_cross_kvcache(RandomMat(16, 128), RandomMat(44, 128), RandomMat(55, 128), 16, 2, 0)
           || test_multiheadattention_int8_cross_kvcache(RandomMat(12, 128), RandomMat(44, 127), RandomMat(55, 127), 16, 4, 1)
           || test_multiheadattention_int8_cross_kvcache(RandomMat(12, 17), RandomMat(28, 127), RandomMat(32, 127), 12, 3, 0)
           || test_multiheadattention_int8_cross_kvcache(RandomMat(12, 17), RandomMat(28, 32), RandomMat(11, 32), 12, 3, 1);
}

static int test_multiheadattention_4()
{
    return 0
           || test_multiheadattention_int8_self_kvcache_prefill(RandomMat(64, 128), 64, 4)
           || test_multiheadattention_int8_self_kvcache_prefill(RandomMat(48, 127), 64, 8);
}

static int test_multiheadattention_5()
{
    return 0
           || test_multiheadattention_int8_self_kvcache_decode(64, 128, 1, 64, 4)
           || test_multiheadattention_int8_self_kvcache_decode(48, 127, 1, 64, 8)
           || test_multiheadattention_int8_self_kvcache_decode(64, 1024, 3, 64, 4, 1);
}
#endif

int main()
{
    SRAND(7767517);

    return 0
           || test_multiheadattention_0()
           || test_multiheadattention_1()
           || test_multiheadattention_2()
#if NCNN_INT8
           || test_multiheadattention_3()
           || test_multiheadattention_4()
           || test_multiheadattention_5()
#endif
           ;
}
