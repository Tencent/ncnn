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

    ncnn::Mat past_key_full = RandomMat(embed_dim, max_seqlen, k.c);
    ncnn::Mat past_value_full = RandomMat(out_embed_dim, max_seqlen, v.c);
    ncnn::Mat past_key = past_key_full;
    ncnn::Mat past_value = past_value_full;
    past_key.h = past_seqlen;
    past_value.h = past_seqlen;

    ncnn::Mat mask;
    if (attn_mask)
        mask = CausalMask(src_seqlen, past_seqlen);

    ncnn::ParamDict pd1;
    pd1.set(5, attn_mask);
    pd1.set(7, 1); // kv_cache

    ncnn::Layer* op1 = ncnn::create_layer_cpu("SDPA");
    op1->load_param(pd1);

    ncnn::ParamDict pd2;
    pd2.set(5, attn_mask);
    pd2.set(7, 2); // kv_cache

    ncnn::Layer* op2 = ncnn::create_layer_cpu("SDPA");
    op2->load_param(pd2);

    ncnn::Option opt;
    opt.num_threads = 1;

    op1->create_pipeline(opt);
    op2->create_pipeline(opt);

    std::vector<ncnn::Mat> as1(attn_mask ? 6 : 5);
    std::vector<ncnn::Mat> as2(attn_mask ? 6 : 5);
    as1[0] = q;
    as1[1] = k;
    as1[2] = v;
    as2[0] = q;
    as2[1] = k;
    as2[2] = v;
    int offset = 3;
    if (attn_mask)
    {
        as1[offset] = mask;
        as2[offset] = mask;
        offset++;
    }
    as1[offset] = past_key;
    as1[offset + 1] = past_value;
    as2[offset] = past_key;
    as2[offset + 1] = past_value;

    std::vector<ncnn::Mat> out1(3);
    std::vector<ncnn::Mat> out2(3);
    int ret1 = op1->forward(as1, out1, opt);
    int ret2 = op2->forward(as2, out2, opt);

    int ret = 0;
    if (ret1 != 0 || ret2 != 0 || CompareMat(out1[0], out2[0], 0.001f) != 0)
        ret = -1;
    if (ret == 0 && (out2[1].data != past_key_full.data || out2[2].data != past_value_full.data || out2[1].h != dst_seqlen || out2[2].h != dst_seqlen))
        ret = -1;

    op1->destroy_pipeline(opt);
    op2->destroy_pipeline(opt);
    delete op1;
    delete op2;

    if (ret != 0)
        fprintf(stderr, "test_sdpa_kvcache2 failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d past_seqlen=%d max_seqlen=%d\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, past_seqlen, max_seqlen);

    return ret;
}

static int test_sdpa_kvcache2_invalid_view()
{
    ncnn::ParamDict pd;
    pd.set(7, 2); // kv_cache

    ncnn::Layer* op = ncnn::create_layer_cpu("SDPA");
    op->load_param(pd);

    ncnn::Option opt;
    opt.num_threads = 1;

    std::vector<ncnn::Mat> as(4);
    as[0] = RandomMat(32, 1, 4);
    as[1] = RandomMat(32, 1, 4);
    as[2] = RandomMat(20, 1, 4);
    as[3] = RandomMat(32, 8, 4);

    std::vector<ncnn::Mat> top_blobs(3);
    int ret = op->forward(as, top_blobs, opt);
    if (ret == 0)
    {
        fprintf(stderr, "test_sdpa_kvcache2_invalid_view failed missing past_value\n");
        delete op;
        return -1;
    }

    as.push_back(RandomMat(20, 7, 4));
    ret = op->forward(as, top_blobs, opt);
    if (ret == 0)
    {
        fprintf(stderr, "test_sdpa_kvcache2_invalid_view failed mismatched cache views\n");
        delete op;
        return -1;
    }

    delete op;
    return 0;
}

static int test_sdpa_kvcache2_persistent_buffer()
{
    const int embed_dim = 4;
    const int out_embed_dim = 3;
    const int past_seqlen = 2;
    const int cur_seqlen = 2;
    const int max_seqlen = 5;

    ncnn::Mat query(embed_dim, cur_seqlen, 2);
    query.fill(0.01f);

    ncnn::Mat cur_key(embed_dim, cur_seqlen, 1);
    ncnn::Mat cur_value(out_embed_dim, cur_seqlen, 1);
    ncnn::Mat past_key_full(embed_dim, max_seqlen, 1);
    ncnn::Mat past_value_full(out_embed_dim, max_seqlen, 1);

    for (int y = 0; y < cur_seqlen; y++)
    {
        float* kptr = cur_key.row(y);
        for (int x = 0; x < embed_dim; x++)
            kptr[x] = 100.f + y * 10 + x;

        float* vptr = cur_value.row(y);
        for (int x = 0; x < out_embed_dim; x++)
            vptr[x] = 200.f + y * 10 + x;
    }

    for (int y = 0; y < max_seqlen; y++)
    {
        float* kptr = past_key_full.row(y);
        for (int x = 0; x < embed_dim; x++)
            kptr[x] = 300.f + y * 10 + x;

        float* vptr = past_value_full.row(y);
        for (int x = 0; x < out_embed_dim; x++)
            vptr[x] = 400.f + y * 10 + x;
    }

    ncnn::Mat past_key = past_key_full;
    ncnn::Mat past_value = past_value_full;
    past_key.h = past_seqlen;
    past_value.h = past_seqlen;

    ncnn::ParamDict pd;
    pd.set(7, 2); // kv_cache

    ncnn::Layer* op = ncnn::create_layer_cpu("SDPA");
    op->load_param(pd);

    ncnn::Option opt;
    opt.num_threads = 1;

    op->create_pipeline(opt);

    std::vector<ncnn::Mat> bottom_blobs(5);
    bottom_blobs[0] = query;
    bottom_blobs[1] = cur_key;
    bottom_blobs[2] = cur_value;
    bottom_blobs[3] = past_key;
    bottom_blobs[4] = past_value;

    std::vector<ncnn::Mat> top_blobs(3);
    int ret = op->forward(bottom_blobs, top_blobs, opt);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_kvcache2_persistent_buffer failed forward\n");
        op->destroy_pipeline(opt);
        delete op;
        return -1;
    }

    if (top_blobs[1].data != past_key_full.data || top_blobs[2].data != past_value_full.data)
    {
        fprintf(stderr, "test_sdpa_kvcache2_persistent_buffer failed buffer identity\n");
        op->destroy_pipeline(opt);
        delete op;
        return -1;
    }

    if (top_blobs[1].h != past_seqlen + cur_seqlen || top_blobs[2].h != past_seqlen + cur_seqlen || top_blobs[1].cstep != past_key_full.cstep || top_blobs[2].cstep != past_value_full.cstep)
    {
        fprintf(stderr, "test_sdpa_kvcache2_persistent_buffer failed output view shape\n");
        op->destroy_pipeline(opt);
        delete op;
        return -1;
    }

    for (int y = 0; y < past_seqlen; y++)
    {
        const float* kptr = past_key_full.row(y);
        for (int x = 0; x < embed_dim; x++)
        {
            if (kptr[x] != 300.f + y * 10 + x)
            {
                fprintf(stderr, "test_sdpa_kvcache2_persistent_buffer clobbered past_key\n");
                op->destroy_pipeline(opt);
                delete op;
                return -1;
            }
        }

        const float* vptr = past_value_full.row(y);
        for (int x = 0; x < out_embed_dim; x++)
        {
            if (vptr[x] != 400.f + y * 10 + x)
            {
                fprintf(stderr, "test_sdpa_kvcache2_persistent_buffer clobbered past_value\n");
                op->destroy_pipeline(opt);
                delete op;
                return -1;
            }
        }
    }

    for (int y = 0; y < cur_seqlen; y++)
    {
        const float* kptr = past_key_full.row(past_seqlen + y);
        for (int x = 0; x < embed_dim; x++)
        {
            if (kptr[x] != 100.f + y * 10 + x)
            {
                fprintf(stderr, "test_sdpa_kvcache2_persistent_buffer failed appended key\n");
                op->destroy_pipeline(opt);
                delete op;
                return -1;
            }
        }

        const float* vptr = past_value_full.row(past_seqlen + y);
        for (int x = 0; x < out_embed_dim; x++)
        {
            if (vptr[x] != 200.f + y * 10 + x)
            {
                fprintf(stderr, "test_sdpa_kvcache2_persistent_buffer failed appended value\n");
                op->destroy_pipeline(opt);
                delete op;
                return -1;
            }
        }
    }

    past_key.h = max_seqlen - cur_seqlen + 1;
    past_value.h = max_seqlen - cur_seqlen + 1;
    bottom_blobs[3] = past_key;
    bottom_blobs[4] = past_value;
    ret = op->forward(bottom_blobs, top_blobs, opt);
    if (ret == 0)
    {
        fprintf(stderr, "test_sdpa_kvcache2_persistent_buffer failed overflow check\n");
        op->destroy_pipeline(opt);
        delete op;
        return -1;
    }

    op->destroy_pipeline(opt);
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
           || test_sdpa_kvcache2_invalid_view()
           || test_sdpa_kvcache2_persistent_buffer()
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
    const int cur_seqlen = k.h;
    const int dst_seqlen = past_seqlen + cur_seqlen;

    ncnn::Mat past_key_full = RandomMat(embed_dim, max_seqlen, k.c);
    ncnn::Mat past_value_full = RandomMat(out_embed_dim, max_seqlen, v.c);
    ncnn::Mat past_key = past_key_full;
    ncnn::Mat past_value = past_value_full;
    past_key.h = past_seqlen;
    past_value.h = past_seqlen;

    ncnn::Mat mask;
    if (attn_mask)
        mask = CausalMask(src_seqlen, past_seqlen);

    ncnn::ParamDict pd1;
    pd1.set(5, attn_mask);
    pd1.set(7, 1);  // kv_cache
    pd1.set(18, 2); // int8_scale_term

    ncnn::ParamDict pd2;
    pd2.set(5, attn_mask);
    pd2.set(7, 2);  // kv_cache
    pd2.set(18, 2); // int8_scale_term

    ncnn::Layer* op2 = ncnn::create_layer_cpu("SDPA");
    op2->load_param(pd2);
    std::vector<ncnn::Mat> weights(0);
    ncnn::ModelBinFromMatArray mb(weights.data());
    op2->load_model(mb);

    float epsilon = 0.01;

    ncnn::Option opt;
    opt.num_threads = 1;

    op2->create_pipeline(opt);

    std::vector<ncnn::Mat> as1(attn_mask ? 6 : 5);
    std::vector<ncnn::Mat> as2(attn_mask ? 6 : 5);
    as1[0] = q;
    as1[1] = k;
    as1[2] = v;
    as2[0] = q;
    as2[1] = k;
    as2[2] = v;
    int offset = 3;
    if (attn_mask)
    {
        as1[offset] = mask;
        as2[offset] = mask;
        offset++;
    }
    as1[offset] = past_key;
    as1[offset + 1] = past_value;
    as2[offset] = past_key;
    as2[offset + 1] = past_value;

    std::vector<ncnn::Mat> out1(3);
    std::vector<ncnn::Mat> out2(3);
    int ret1 = test_layer_naive(ncnn::layer_to_index("SDPA"), pd1, weights, as1, 3, out1, 0);
    int ret2 = op2->forward(as2, out2, opt);

    int ret = 0;
    if (ret1 != 0 || ret2 != 0 || CompareMat(out1[0], out2[0], epsilon) != 0)
        ret = -1;
    if (ret == 0 && (out2[1].data != past_key_full.data || out2[2].data != past_value_full.data || out2[1].h != dst_seqlen || out2[2].h != dst_seqlen))
        ret = -1;

    op2->destroy_pipeline(opt);
    delete op2;

    if (ret != 0)
        fprintf(stderr, "test_sdpa_int8_kvcache2 failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d past_seqlen=%d max_seqlen=%d\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, past_seqlen, max_seqlen);

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
