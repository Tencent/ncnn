// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include <float.h>

static ncnn::Mat CausalMask(int src_seqlen, int past_seqlen)
{
    const int total = past_seqlen + src_seqlen;
    ncnn::Mat mask(total, src_seqlen);
    mask.fill(0.f);
    for (int i = 0; i < src_seqlen; i++)
    {
        float* row = mask.row(i);
        for (int j = past_seqlen + i + 1; j < total; j++)
            row[j] = -1e38f;
    }
    return mask;
}

static int test_sdpa_kvcache(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask, int past_seqlen)
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

    int ret = test_layer("SDPA", pd, weights, as, 3);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_kvcache failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d past_seqlen=%d\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, past_seqlen);
    }

    return ret;
}

static int test_sdpa_kvcache2(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask, int past_seqlen, int max_seqlen)
{
    const int embed_dim = q.w;
    const int out_embed_dim = v.w;
    const int src_seqlen = q.h;
    const int cur_seqlen = k.h;
    const int dst_seqlen = past_seqlen + cur_seqlen;

    ncnn::ParamDict pd;
    pd.set(5, attn_mask);
    pd.set(7, 2); // kv_cache

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (attn_mask)
    {
        as.push_back(CausalMask(src_seqlen, past_seqlen));
    }

    as.push_back(RandomMat(embed_dim, max_seqlen, k.c));
    as.push_back(RandomMat(out_embed_dim, max_seqlen, v.c));
    as.push_back(RandomMat(1, (float)past_seqlen));

    int ret = test_layer("SDPA", pd, weights, as, 3);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_kvcache2 failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d past_seqlen=%d max_seqlen=%d\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, past_seqlen, max_seqlen);
    }

    return ret;
}

static int test_sdpa_kvcache2_invalid_past_len()
{
    ncnn::ParamDict pd;
    pd.set(7, 2); // kv_cache

    ncnn::Layer* op = ncnn::create_layer_cpu("SDPA");
    op->load_param(pd);

    ncnn::Option opt;
    opt.num_threads = 1;

    std::vector<ncnn::Mat> as(5);
    as[0] = RandomMat(32, 1, 4);
    as[1] = RandomMat(32, 1, 4);
    as[2] = RandomMat(20, 1, 4);
    as[3] = RandomMat(32, 8, 4);
    as[4] = RandomMat(20, 8, 4);

    std::vector<ncnn::Mat> top_blobs(3);
    int ret = op->forward(as, top_blobs, opt);
    if (ret == 0)
    {
        fprintf(stderr, "test_sdpa_kvcache2_invalid_past_len failed missing past_len\n");
        delete op;
        return -1;
    }

    as.push_back(RandomMat(1, -1.f));
    ret = op->forward(as, top_blobs, opt);
    if (ret == 0)
    {
        fprintf(stderr, "test_sdpa_kvcache2_invalid_past_len failed negative past_len\n");
        delete op;
        return -1;
    }

    delete op;
    return 0;
}

static int test_sdpa_0()
{
    return 0
           || test_sdpa_kvcache(RandomMat(32, 66, 8), RandomMat(32, 66, 8), RandomMat(20, 66, 8), 0, 11)
           || test_sdpa_kvcache(RandomMat(26, 64, 8), RandomMat(26, 61, 8), RandomMat(18, 61, 8), 1, 11)
           || test_sdpa_kvcache(RandomMat(40, 62, 7), RandomMat(40, 61, 7), RandomMat(24, 61, 7), 0, 9)
           || test_sdpa_kvcache(RandomMat(24, 22, 6), RandomMat(24, 19, 6), RandomMat(16, 19, 6), 1, 9)
           || test_sdpa_kvcache(RandomMat(64, 128, 12), RandomMat(64, 128, 2), RandomMat(64, 128, 2), 0, 1)
           || test_sdpa_kvcache(RandomMat(64, 122, 12), RandomMat(64, 127, 2), RandomMat(48, 127, 2), 1, 1)
           || test_sdpa_kvcache(RandomMat(44, 128, 4), RandomMat(44, 123, 4), RandomMat(55, 123, 4), 0, 0)
           || test_sdpa_kvcache(RandomMat(12, 127, 4), RandomMat(12, 127, 4), RandomMat(55, 127, 4), 1, 0)
           || test_sdpa_kvcache(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 3)
           || test_sdpa_kvcache(RandomMat(28, 17, 15), RandomMat(28, 32, 5), RandomMat(11, 32, 5), 1, 5)
           || test_sdpa_kvcache2(RandomMat(32, 1, 8), RandomMat(32, 1, 8), RandomMat(20, 1, 8), 0, 11, 32)
           || test_sdpa_kvcache2(RandomMat(64, 1, 12), RandomMat(64, 1, 2), RandomMat(64, 1, 2), 0, 1, 8)
           || test_sdpa_kvcache2(RandomMat(64, 1, 12), RandomMat(64, 1, 2), RandomMat(96, 1, 2), 0, 1, 8)
           || test_sdpa_kvcache2(RandomMat(28, 1, 15), RandomMat(28, 1, 5), RandomMat(32, 1, 5), 0, 3, 16)
           || test_sdpa_kvcache2(RandomMat(32, 16, 8), RandomMat(32, 16, 8), RandomMat(20, 16, 8), 0, 0, 32)
           || test_sdpa_kvcache2(RandomMat(64, 17, 12), RandomMat(64, 17, 2), RandomMat(64, 17, 2), 0, 0, 32)
           || test_sdpa_kvcache2_invalid_past_len()
           || test_sdpa_kvcache2(RandomMat(32, 16, 8), RandomMat(32, 16, 8), RandomMat(20, 16, 8), 1, 0, 32)
           || test_sdpa_kvcache2(RandomMat(64, 17, 12), RandomMat(64, 17, 2), RandomMat(64, 17, 2), 1, 0, 32);
}

#if NCNN_INT8
static int test_sdpa_int8_kvcache(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask, int past_seqlen)
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

    float epsilon = 0.01;

    int ret = test_layer("SDPA", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_int8_kvcache failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d past_seqlen=%d\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, past_seqlen);
    }

    return ret;
}

static int test_sdpa_int8_kvcache2(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask, int past_seqlen, int max_seqlen)
{
    const int embed_dim = q.w;
    const int out_embed_dim = v.w;
    const int src_seqlen = q.h;

    ncnn::ParamDict pd;
    pd.set(5, attn_mask);
    pd.set(7, 2);  // kv_cache
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = q;
    as[1] = k;
    as[2] = v;

    if (attn_mask)
    {
        as.push_back(CausalMask(src_seqlen, past_seqlen));
    }

    as.push_back(RandomMat(embed_dim, max_seqlen, k.c));
    as.push_back(RandomMat(out_embed_dim, max_seqlen, v.c));
    as.push_back(RandomMat(1, (float)past_seqlen));

    float epsilon = 0.01;

    int ret = test_layer("SDPA", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_int8_kvcache2 failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d past_seqlen=%d max_seqlen=%d\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, past_seqlen, max_seqlen);
    }

    return ret;
}

static int test_sdpa_1()
{
    return 0
           || test_sdpa_int8_kvcache(RandomMat(32, 66, 8), RandomMat(32, 66, 8), RandomMat(20, 66, 8), 0, 11)
           || test_sdpa_int8_kvcache(RandomMat(26, 64, 8), RandomMat(26, 61, 8), RandomMat(18, 61, 8), 1, 11)
           || test_sdpa_int8_kvcache(RandomMat(40, 62, 7), RandomMat(40, 61, 7), RandomMat(24, 61, 7), 0, 9)
           || test_sdpa_int8_kvcache(RandomMat(24, 22, 6), RandomMat(24, 19, 6), RandomMat(16, 19, 6), 1, 9)
           || test_sdpa_int8_kvcache(RandomMat(64, 128, 12), RandomMat(64, 128, 2), RandomMat(64, 128, 2), 0, 1)
           || test_sdpa_int8_kvcache(RandomMat(48, 122, 12), RandomMat(64, 127, 2), RandomMat(64, 127, 2), 1, 1)
           || test_sdpa_int8_kvcache(RandomMat(44, 128, 4), RandomMat(44, 123, 4), RandomMat(55, 123, 4), 0, 0)
           || test_sdpa_int8_kvcache(RandomMat(12, 127, 4), RandomMat(12, 127, 4), RandomMat(55, 127, 4), 1, 0)
           || test_sdpa_int8_kvcache(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 3)
           || test_sdpa_int8_kvcache(RandomMat(28, 17, 15), RandomMat(28, 32, 5), RandomMat(11, 32, 5), 1, 5)
           || test_sdpa_int8_kvcache2(RandomMat(32, 1, 8), RandomMat(32, 1, 8), RandomMat(20, 1, 8), 0, 11, 32)
           || test_sdpa_int8_kvcache2(RandomMat(64, 1, 12), RandomMat(64, 1, 2), RandomMat(64, 1, 2), 0, 1, 8)
           || test_sdpa_int8_kvcache2(RandomMat(32, 16, 8), RandomMat(32, 16, 8), RandomMat(20, 16, 8), 1, 0, 32)
           || test_sdpa_int8_kvcache2(RandomMat(64, 17, 12), RandomMat(64, 17, 2), RandomMat(64, 17, 2), 1, 0, 32);
}
#endif

int main()
{
    SRAND(7767517);

#if NCNN_INT8
    return test_sdpa_0() || test_sdpa_1();
#else
    return test_sdpa_0();
#endif
}
