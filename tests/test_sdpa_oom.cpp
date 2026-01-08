// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_sdpa_oom(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask, float scale = 0.f)
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

    float epsilon = 0.001;

    int ret = test_layer_oom("SDPA", pd, weights, as, 1, epsilon);
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

static int test_sdpa_0()
{
    return 0
           || test_sdpa_oom(RandomMat(32, 66, 8), RandomMat(32, 66, 8), RandomMat(20, 66, 8), 0)
           || test_sdpa_oom(RandomMat(26, 64, 8), RandomMat(26, 61, 8), RandomMat(18, 61, 8), 1)
           || test_sdpa_oom(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 0.1f)
           || test_sdpa_oom(RandomMat(28, 17, 15), RandomMat(28, 32, 5), RandomMat(11, 32, 5), 1, -0.4f)
           || test_sdpa_kvcache_oom(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 3);
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
           || test_sdpa_int8_oom(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 0.1f)
           || test_sdpa_int8_oom(RandomMat(28, 17, 15), RandomMat(28, 32, 5), RandomMat(11, 32, 5), 1, -0.4f)
           || test_sdpa_int8_kvcache_oom(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 3);
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
