// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

class TestKVCacheOOMAllocator : public ncnn::Allocator
{
public:
    TestKVCacheOOMAllocator()
    {
        oom = 0;
    }

    virtual void* fastMalloc(size_t size)
    {
        if (oom)
            return 0;

        return ncnn::fastMalloc(size);
    }

    virtual void fastFree(void* ptr)
    {
        ncnn::fastFree(ptr);
    }

public:
    int oom;
};

static int test_sdpa_oom(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask, float scale = 0.f, int flag = 0)
{
    const int src_seqlen = q.h;
    const int dst_seqlen = k.h;

    ncnn::ParamDict pd;
    pd.set(5, attn_mask);
    pd.set(6, scale);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (attn_mask)
    {
        as.push_back(RandomMat(dst_seqlen, src_seqlen));
    }

    int ret = test_layer_oom("SDPA", pd, weights, as, 1, flag);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_oom failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d scale=%f\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, scale);
    }

    return ret;
}

static int test_sdpa_kvcache_oom(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask, int past_seqlen)
{
    const int embed_dim = q.w;
    const int out_embed_dim = v.w;
    const int src_seqlen = q.h;
    const int cur_seqlen = k.h;
    const int dst_seqlen = past_seqlen + cur_seqlen;

    ncnn::ParamDict pd;
    pd.set(5, attn_mask);
    pd.set(7, 1); // kv_cache

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (attn_mask)
    {
        as.push_back(RandomMat(dst_seqlen, src_seqlen));
    }

    as.push_back(RandomMat(embed_dim, past_seqlen, k.c));
    as.push_back(RandomMat(out_embed_dim, past_seqlen, v.c));

    int ret = test_layer_oom("SDPA", pd, weights, as, 3);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_kvcache_oom failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d past_seqlen=%d\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, past_seqlen);
    }

    return ret;
}

static int test_sdpa_kvcache_allocator_oom()
{
    ncnn::Layer* op = ncnn::create_layer_cpu("SDPA");
    if (!op)
        return -1;

    ncnn::ParamDict pd;
    pd.set(7, 1); // kv_cache
    op->load_param(pd);
    op->load_model(ncnn::ModelBinFromMatArray(0));

    TestKVCacheOOMAllocator kvcache_allocator;
    ncnn::Option opt;
    opt.num_threads = 1;
    opt.kvcache_allocator = &kvcache_allocator;
    opt.kvcache_max_seqlen_hint = 16;

    int ret = op->create_pipeline(opt);

    std::vector<ncnn::Mat> bottoms(5);
    bottoms[0] = RandomMat(7, 16, 4);
    bottoms[1] = RandomMat(7, 16, 2);
    bottoms[2] = RandomMat(5, 16, 2);

    std::vector<ncnn::Mat> tops(3);
    if (ret == 0)
    {
        kvcache_allocator.oom = 1;
        ret = op->forward(bottoms, tops, opt) == -100 ? 0 : -1;
        tops.clear();
    }

    if (ret == 0)
    {
        kvcache_allocator.oom = 0;
        tops.resize(3);
        ret = op->forward(bottoms, tops, opt);
    }

    if (ret == 0)
    {
        bottoms[0] = RandomMat(7, 1, 4);
        bottoms[1] = RandomMat(7, 1, 2);
        bottoms[2] = RandomMat(5, 1, 2);
        bottoms[3] = tops[1];
        bottoms[4] = tops[2];
        tops.clear();
        tops.resize(3);

        kvcache_allocator.oom = 1;
        ret = op->forward(bottoms, tops, opt) == -100 ? 0 : -1;
    }

    bottoms.clear();
    tops.clear();

    op->destroy_pipeline(opt);
    delete op;

    if (ret != 0)
        fprintf(stderr, "test_sdpa_kvcache_allocator_oom failed\n");

    return ret;
}

static int test_sdpa_0()
{
    return 0
           || test_sdpa_oom(RandomMat(63, 1, 8), RandomMat(63, 513, 1), RandomMat(37, 513, 1), 0)
           || test_sdpa_oom(RandomMat(63, 2, 1), RandomMat(63, 513, 1), RandomMat(37, 513, 1), 0, 0.f, TEST_LAYER_ENABLE_THREADING)
           || test_sdpa_oom(RandomMat(32, 66, 8), RandomMat(32, 66, 8), RandomMat(20, 66, 8), 0)
           || test_sdpa_oom(RandomMat(26, 64, 8), RandomMat(26, 61, 8), RandomMat(18, 61, 8), 1)
           || test_sdpa_oom(RandomMat(40, 62, 7), RandomMat(40, 61, 7), RandomMat(24, 61, 7), 0)
           || test_sdpa_oom(RandomMat(24, 22, 6), RandomMat(24, 19, 6), RandomMat(16, 19, 6), 1)
           || test_sdpa_oom(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 0.1f)
           || test_sdpa_oom(RandomMat(28, 17, 15), RandomMat(28, 32, 5), RandomMat(11, 32, 5), 1, -0.4f)
           || test_sdpa_kvcache_oom(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 0)
           || test_sdpa_kvcache_allocator_oom();
}

#if NCNN_INT8
static int test_sdpa_int8_oom(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask, float scale = 0.f)
{
    const int src_seqlen = q.h;
    const int dst_seqlen = k.h;

    ncnn::ParamDict pd;
    pd.set(5, attn_mask);
    pd.set(6, scale);
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (attn_mask)
    {
        as.push_back(RandomMat(dst_seqlen, src_seqlen));
    }

    float epsilon = 0.001;

    int ret = test_layer_oom("SDPA", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_int8_oom failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d scale=%f\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, scale);
    }

    return ret;
}

static int test_sdpa_int8_kvcache_oom(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask, int past_seqlen)
{
    const int embed_dim = q.w;
    const int out_embed_dim = v.w;
    const int src_seqlen = q.h;
    const int cur_seqlen = k.h;
    const int dst_seqlen = past_seqlen + cur_seqlen;

    ncnn::ParamDict pd;
    pd.set(5, attn_mask);
    pd.set(7, 1);  // kv_cache
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (attn_mask)
    {
        as.push_back(RandomMat(dst_seqlen, src_seqlen));
    }

    as.push_back(RandomMat(embed_dim, past_seqlen, k.c));
    as.push_back(RandomMat(out_embed_dim, past_seqlen, v.c));

    int ret = test_layer_oom("SDPA", pd, weights, as, 3);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_int8_kvcache_oom failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d past_seqlen=%d\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, past_seqlen);
    }

    return ret;
}

static int test_sdpa_1()
{
    return 0
           || test_sdpa_int8_oom(RandomMat(32, 66, 8), RandomMat(32, 66, 8), RandomMat(20, 66, 8), 0)
           || test_sdpa_int8_oom(RandomMat(26, 64, 8), RandomMat(26, 61, 8), RandomMat(18, 61, 8), 1)
           || test_sdpa_int8_oom(RandomMat(40, 62, 7), RandomMat(40, 61, 7), RandomMat(24, 61, 7), 0)
           || test_sdpa_int8_oom(RandomMat(24, 22, 6), RandomMat(24, 19, 6), RandomMat(16, 19, 6), 1)
           || test_sdpa_int8_oom(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 0.1f)
           || test_sdpa_int8_oom(RandomMat(28, 17, 15), RandomMat(28, 32, 5), RandomMat(11, 32, 5), 1, -0.4f)
           || test_sdpa_int8_kvcache_oom(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 0);
}
#endif

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return 0
           || test_sdpa_0()
           || test_sdpa_1();
#else
    return 0
           || test_sdpa_0();
#endif
}
