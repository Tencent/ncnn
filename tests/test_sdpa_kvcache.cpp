// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include <float.h>

static int test_sdpa_cross_kvcache(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask, int input_kvcache)
{
    const int embed_dim = q.w;
    const int out_embed_dim = v.w;
    const int src_seqlen = q.h;
    const int dst_seqlen = k.h;

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

    if (input_kvcache)
    {
        as.push_back(RandomMat(embed_dim, k.h, k.c));
        as.push_back(RandomMat(out_embed_dim, v.h, v.c));
    }
    else
    {
        as.push_back(ncnn::Mat());
        as.push_back(ncnn::Mat());
    }

    float epsilon = 0.001;

    int ret = test_layer("SDPA", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_cross_kvcache failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d input_kvcache=%d\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, input_kvcache);
    }

    return ret;
}

static int test_sdpa_cross_kvcache(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask)
{
    return test_sdpa_cross_kvcache(q, k, v, attn_mask, 0) || test_sdpa_cross_kvcache(q, k, v, attn_mask, 1);
}

static int test_sdpa_self_kvcache_prefill(const ncnn::Mat& a)
{
    const int seqlen = a.h;

    ncnn::ParamDict pd;
    pd.set(5, 1); // attn_mask
    pd.set(7, 1); // kv_cache

    std::vector<ncnn::Mat> weights(0);

    ncnn::Mat attn_mask(seqlen, seqlen);
    attn_mask.fill(0.f);
    for (int i = 0; i < seqlen; i++)
    {
        for (int j = i + 1; j < seqlen; j++)
        {
            attn_mask.row(i)[j] = -60000.f;
        }
    }

    std::vector<ncnn::Mat> as(4);
    as[0] = a;
    as[1] = attn_mask;
    as[2] = ncnn::Mat();
    as[3] = ncnn::Mat();

    float epsilon = 0.001;

    int ret = test_layer("SDPA", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_self_kvcache_prefill failed a=(%d %d %d)\n", a.w, a.h, a.c);
    }

    return ret;
}

static int test_sdpa_self_kvcache_decode(const ncnn::Mat& a)
{
    const int embed_dim = a.w;
    const int past_seqlen = a.h;
    const int cur_seqlen = 1;

    ncnn::ParamDict pd;
    pd.set(5, 0); // attn_mask
    pd.set(7, 1); // kv_cache

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = RandomMat(embed_dim, 1, a.c);
    as[1] = RandomMat(embed_dim, past_seqlen, a.c);
    as[2] = RandomMat(embed_dim, past_seqlen, a.c);

    float epsilon = 0.001;

    int ret = test_layer("SDPA", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_self_kvcache_decode failed a=(%d %d %d)\n", a.w, a.h, a.c);
    }

    return ret;
}

static int test_sdpa_0()
{
    return 0
           || test_sdpa_cross_kvcache(RandomMat(32, 66, 8), RandomMat(32, 66, 8), RandomMat(20, 66, 8), 0)
           || test_sdpa_cross_kvcache(RandomMat(26, 64, 8), RandomMat(26, 61, 8), RandomMat(18, 61, 8), 1)
           || test_sdpa_cross_kvcache(RandomMat(64, 128, 12), RandomMat(64, 128, 2), RandomMat(64, 128, 2), 0)
           || test_sdpa_cross_kvcache(RandomMat(48, 122, 12), RandomMat(64, 127, 2), RandomMat(64, 127, 2), 1)
           || test_sdpa_cross_kvcache(RandomMat(44, 128, 4), RandomMat(44, 123, 4), RandomMat(55, 123, 4), 0, 1.f)
           || test_sdpa_cross_kvcache(RandomMat(12, 127, 4), RandomMat(12, 127, 4), RandomMat(55, 127, 4), 1, 1.f)
           || test_sdpa_cross_kvcache(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 0.1f)
           || test_sdpa_cross_kvcache(RandomMat(28, 17, 15), RandomMat(28, 32, 5), RandomMat(11, 32, 5), 1, -0.4f);
}

static int test_sdpa_1()
{
    return 0
           || test_sdpa_self_kvcache_prefill(RandomMat(32, 66, 8))
           || test_sdpa_self_kvcache_prefill(RandomMat(64, 128, 12))
           || test_sdpa_self_kvcache_prefill(RandomMat(12, 127, 4))
           || test_sdpa_self_kvcache_prefill(RandomMat(28, 17, 15));
}

static int test_sdpa_2()
{
    return 0
           || test_sdpa_self_kvcache_decode(RandomMat(32, 66, 8))
           || test_sdpa_self_kvcache_decode(RandomMat(64, 128, 12))
           || test_sdpa_self_kvcache_decode(RandomMat(12, 127, 4))
           || test_sdpa_self_kvcache_decode(RandomMat(28, 17, 15));
}

#if NCNN_INT8
static int test_sdpa_int8_cross_kvcache(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask, int input_kvcache)
{
    const int embed_dim = q.w;
    const int out_embed_dim = v.w;
    const int src_seqlen = q.h;
    const int dst_seqlen = k.h;

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

    if (input_kvcache)
    {
        as.push_back(RandomMat(embed_dim, k.h, k.c));
        as.push_back(RandomMat(out_embed_dim, v.h, v.c));
    }
    else
    {
        as.push_back(ncnn::Mat());
        as.push_back(ncnn::Mat());
    }

    float epsilon = 0.001;

    int ret = test_layer("SDPA", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_int8_cross_kvcache failed q=(%d %d %d) k=(%d %d %d) v=(%d %d %d) attn_mask=%d input_kvcache=%d\n", q.w, q.h, q.c, k.w, k.h, k.c, v.w, v.h, v.c, attn_mask, input_kvcache);
    }

    return ret;
}

static int test_sdpa_int8_cross_kvcache(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int attn_mask)
{
    return test_sdpa_int8_cross_kvcache(q, k, v, attn_mask, 0) || test_sdpa_int8_cross_kvcache(q, k, v, attn_mask, 1);
}

static int test_sdpa_int8_self_kvcache_prefill(const ncnn::Mat& a)
{
    const int seqlen = a.h;

    ncnn::ParamDict pd;
    pd.set(5, 1);  // attn_mask
    pd.set(7, 1);  // kv_cache
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(0);

    ncnn::Mat attn_mask(seqlen, seqlen);
    attn_mask.fill(0.f);
    for (int i = 0; i < seqlen; i++)
    {
        for (int j = i + 1; j < seqlen; j++)
        {
            attn_mask.row(i)[j] = -60000.f;
        }
    }

    std::vector<ncnn::Mat> as(4);
    as[0] = a;
    as[1] = attn_mask;
    as[2] = ncnn::Mat();
    as[3] = ncnn::Mat();

    float epsilon = 0.001;

    int ret = test_layer("SDPA", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_int8_self_kvcache_prefill failed a=(%d %d %d)\n", a.w, a.h, a.c);
    }

    return ret;
}

static int test_sdpa_int8_self_kvcache_decode(const ncnn::Mat& a)
{
    const int embed_dim = a.w;
    const int past_seqlen = a.h;
    const int cur_seqlen = 1;

    ncnn::ParamDict pd;
    pd.set(5, 0);  // attn_mask
    pd.set(7, 1);  // kv_cache
    pd.set(18, 2); // int8_scale_term

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> as(3);
    as[0] = RandomMat(embed_dim, 1, a.c);
    as[1] = RandomMat(embed_dim, past_seqlen, a.c);
    as[2] = RandomMat(embed_dim, past_seqlen, a.c);

    float epsilon = 0.001;

    int ret = test_layer("SDPA", pd, weights, as, 3, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_sdpa_int8_self_kvcache_decode failed a=(%d %d %d)\n", a.w, a.h, a.c);
    }

    return ret;
}

static int test_sdpa_3()
{
    return 0
           || test_sdpa_int8_cross_kvcache(RandomMat(32, 66, 8), RandomMat(32, 66, 8), RandomMat(20, 66, 8), 0)
           || test_sdpa_int8_cross_kvcache(RandomMat(26, 64, 8), RandomMat(26, 61, 8), RandomMat(18, 61, 8), 1)
           || test_sdpa_int8_cross_kvcache(RandomMat(64, 128, 12), RandomMat(64, 128, 2), RandomMat(64, 128, 2), 0)
           || test_sdpa_int8_cross_kvcache(RandomMat(48, 122, 12), RandomMat(64, 127, 2), RandomMat(64, 127, 2), 1)
           || test_sdpa_int8_cross_kvcache(RandomMat(44, 128, 4), RandomMat(44, 123, 4), RandomMat(55, 123, 4), 0, 1.f)
           || test_sdpa_int8_cross_kvcache(RandomMat(12, 127, 4), RandomMat(12, 127, 4), RandomMat(55, 127, 4), 1, 1.f)
           || test_sdpa_int8_cross_kvcache(RandomMat(28, 17, 15), RandomMat(28, 127, 5), RandomMat(32, 127, 5), 0, 0.1f)
           || test_sdpa_int8_cross_kvcache(RandomMat(28, 17, 15), RandomMat(28, 32, 5), RandomMat(11, 32, 5), 1, -0.4f);
}

static int test_sdpa_4()
{
    return 0
           || test_sdpa_int8_self_kvcache_prefill(RandomMat(32, 66, 8))
           || test_sdpa_int8_self_kvcache_prefill(RandomMat(64, 128, 12))
           || test_sdpa_int8_self_kvcache_prefill(RandomMat(12, 127, 4))
           || test_sdpa_int8_self_kvcache_prefill(RandomMat(28, 17, 15));
}

static int test_sdpa_5()
{
    return 0
           || test_sdpa_int8_self_kvcache_decode(RandomMat(32, 66, 8))
           || test_sdpa_int8_self_kvcache_decode(RandomMat(64, 128, 12))
           || test_sdpa_int8_self_kvcache_decode(RandomMat(12, 127, 4))
           || test_sdpa_int8_self_kvcache_decode(RandomMat(28, 17, 15));
}
#endif

int main()
{
    SRAND(7767517);

    return 0
           || test_sdpa_0()
           || test_sdpa_1()
           || test_sdpa_2()
#if NCNN_INT8
           || test_sdpa_3()
           || test_sdpa_4()
           || test_sdpa_5()
#endif
           ;
}
