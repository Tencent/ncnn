// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer_type.h"

#include <limits.h>

static int test_multiheadattention(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int embed_dim, int num_heads, int attn_mask)
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
        as.push_back(RandomMat(k.h, q.h));
    }

    float epsilon = 0.005;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention failed q=(%d %d) k=(%d %d) v=(%d %d) embed_dim=%d num_heads=%d kdim=%d vdim=%d attn_mask=%d\n", q.w, q.h, k.w, k.h, v.w, v.h, embed_dim, num_heads, kdim, vdim, attn_mask);
    }

    return ret;
}

static int test_multiheadattention_samekv(const ncnn::Mat& q, const ncnn::Mat& kv, int embed_dim, int num_heads, const ncnn::Mat& attn_mask = ncnn::Mat())
{
    const int qdim = q.w;
    const int kvdim = kv.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(5, !attn_mask.empty());
    pd.set(2, embed_dim * qdim);
    pd.set(3, kvdim);
    pd.set(4, kvdim);

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * kvdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * kvdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);

    std::vector<ncnn::Mat> as(attn_mask.empty() ? 2 : 3);
    as[0] = q;
    as[1] = kv;

    if (!attn_mask.empty())
    {
        as[2] = attn_mask;
    }

    float epsilon = 0.005;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_samekv failed q=(%d %d) kv=(%d %d) embed_dim=%d num_heads=%d kvdim=%d\n", q.w, q.h, kv.w, kv.h, embed_dim, num_heads, kvdim);
    }

    return ret;
}

static int test_multiheadattention_sameqkv(const ncnn::Mat& a, int embed_dim, int num_heads, const ncnn::Mat& attn_mask = ncnn::Mat())
{
    const int qdim = a.w;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(5, !attn_mask.empty());
    pd.set(2, embed_dim * qdim);
    pd.set(3, qdim);
    pd.set(4, qdim);
    pd.set(6, 0.7f / sqrtf(embed_dim / num_heads));

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * qdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * qdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);

    std::vector<ncnn::Mat> as(attn_mask.empty() ? 1 : 2);
    as[0] = a;

    if (!attn_mask.empty())
    {
        as[1] = attn_mask;
    }

    float epsilon = 0.005;

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, epsilon);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_sameqkv failed a=(%d %d) embed_dim=%d num_heads=%d\n", a.w, a.h, embed_dim, num_heads);
    }

    return ret;
}

static int test_multiheadattention_0()
{
    return 0
           || test_multiheadattention(RandomMat(62, 66), RandomMat(32, 66), RandomMat(20, 66), 62, 2, 0)
           || test_multiheadattention(RandomMat(26, 64), RandomMat(32, 64), RandomMat(18, 64), 26, 2, 1)
           || test_multiheadattention(RandomMat(64, 128), RandomMat(64, 128), RandomMat(64, 128), 64, 4, 0)
           || test_multiheadattention(RandomMat(48, 127), RandomMat(64, 127), RandomMat(64, 127), 64, 16, 1)
           || test_multiheadattention(RandomMat(16, 128), RandomMat(44, 128), RandomMat(55, 128), 16, 2, 0)
           || test_multiheadattention(RandomMat(12, 128), RandomMat(44, 127), RandomMat(55, 127), 16, 4, 1)
           || test_multiheadattention(RandomMat(12, 17), RandomMat(28, 127), RandomMat(32, 127), 12, 3, 0)
           || test_multiheadattention(RandomMat(12, 17), RandomMat(28, 32), RandomMat(11, 32), 12, 3, 1);
}

static int test_multiheadattention_1()
{
    return 0
           || test_multiheadattention_samekv(RandomMat(64, 4), RandomMat(64, 128), 64, 4)
           || test_multiheadattention_samekv(RandomMat(12, 17), RandomMat(28, 127), 12, 3)
           || test_multiheadattention_samekv(RandomMat(12, 17), RandomMat(11, 7), 12, 3);
}

static int test_multiheadattention_2()
{
    return 0
           || test_multiheadattention_sameqkv(RandomMat(64, 128), 64, 4);
}

#if NCNN_VALIDATION
static int test_multiheadattention_load_param()
{
    ncnn::ParamDict base;
    base.set(0, 8);
    base.set(1, 2);
    base.set(2, 64);
    if (test_layer_param(ncnn::LayerType::MultiHeadAttention, base, 0) != 0)
        return -1;

    // explicit scale must not bypass dimension validation
    ncnn::ParamDict scaled = base;
    scaled.set(6, 1.f);

    const int invalid[] = {0, -1, INT_MIN, 3, 16};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); i++)
    {
        if (test_layer_param(ncnn::LayerType::MultiHeadAttention, scaled, 1, invalid[i], -1) != 0)
            return -1;
    }

    const int invalid_dims[] = {0, -1, INT_MIN};
    for (int id = 2; id <= 4; id++)
    {
        for (int i = 0; i < 3; i++)
        {
            if (test_layer_param(ncnn::LayerType::MultiHeadAttention, scaled, id, invalid_dims[i], -1) != 0)
                return -1;
        }
    }

    for (int id = 3; id <= 4; id++)
    {
        if (test_layer_param(ncnn::LayerType::MultiHeadAttention, scaled, id, 1, 0)
                || test_layer_param(ncnn::LayerType::MultiHeadAttention, scaled, id, INT_MAX / 8, 0)
                || test_layer_param(ncnn::LayerType::MultiHeadAttention, scaled, id, INT_MAX / 8 + 1, -1)
                || test_layer_param(ncnn::LayerType::MultiHeadAttention, scaled, id, INT_MAX, -1))
            return -1;
    }

    const int invalid_quantize[] = {4, 5, 6, 403, 420, 500, 700, 813, INT_MAX};
    for (size_t i = 0; i < sizeof(invalid_quantize) / sizeof(invalid_quantize[0]); i++)
    {
        if (test_layer_param(ncnn::LayerType::MultiHeadAttention, scaled, 18, invalid_quantize[i], -1) != 0)
            return -1;
    }

    const int valid_quantize[] = {400, 412, 600, 612, 800, 812};
    for (size_t i = 0; i < sizeof(valid_quantize) / sizeof(valid_quantize[0]); i++)
    {
#if NCNN_WEIGHT_QUANT
        const int expected_ret = 0;
#else
        const int expected_ret = -1;
#endif
        if (test_layer_param(ncnn::LayerType::MultiHeadAttention, scaled, 18, valid_quantize[i], expected_ret) != 0)
            return -1;
    }

    return 0
           || test_layer_param(ncnn::LayerType::MultiHeadAttention, scaled, 0, 0, -1)
           || test_layer_param(ncnn::LayerType::MultiHeadAttention, scaled, 2, 8, 0)
           || test_layer_param(ncnn::LayerType::MultiHeadAttention, scaled, 2, INT_MAX / 8 * 8, 0)
           || test_layer_param(ncnn::LayerType::MultiHeadAttention, scaled, 2, 65, -1)
           || test_layer_param(ncnn::LayerType::MultiHeadAttention, scaled, 2, INT_MAX, -1);
}
#endif // NCNN_VALIDATION

// per-head mask interfaces are orthogonal to key packing and query-row packing
static int test_multiheadattention_mask_layout(std::vector<ncnn::Mat> as)
{
    const int embed_dim = as[0].w;
    const int num_heads = 4;
    const int query_len = as[0].h;
    const int key_len = as.back().h;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * embed_dim);
    pd.set(3, embed_dim);
    pd.set(4, embed_dim);
    pd.set(5, 1);

    std::vector<ncnn::Mat> weights(8);
    for (int i = 0; i < 8; i++)
    {
        weights[i].create(i % 2 == 0 ? embed_dim * embed_dim : embed_dim);
        weights[i].fill(0.f);
    }
    float* v_weight = (float*)weights[4].data;
    float* out_weight = (float*)weights[6].data;
    for (int i = 0; i < embed_dim; i++)
    {
        v_weight[i * embed_dim + i] = 1.f;
        out_weight[i * embed_dim + i] = 1.f;
    }

    // zero query and key projections leave the per-head mask visible in the output
    // self-attention shares the value signal with the query and key input
    for (size_t input = 0; input < as.size(); input++)
    {
        for (int y = 0; y < as[input].h; y++)
        {
            float* ptr = as[input].row(y);
            for (int x = 0; x < embed_dim; x++)
                ptr[x] = (float)y;
        }
    }

    ncnn::Mat mask(key_len, query_len, num_heads);
    float* maskptr = (float*)mask.data;
    for (size_t i = 0; i < mask.cstep * num_heads; i++)
        maskptr[i] = 0.f;
    for (int head = 0; head < num_heads; head++)
        for (int y = 0; y < query_len; y++)
            for (int x = 0; x < key_len; x++)
                maskptr[head * mask.cstep + y * key_len + x] = (head * 5 + y + 1) * (x - key_len / 2) * 0.125f;
    as.push_back(mask);

    int ret = test_layer("MultiHeadAttention", pd, weights, as, 1, 0.005f);
    if (ret != 0)
        fprintf(stderr, "test_multiheadattention_mask_layout failed embed_dim=%d num_heads=%d query_len=%d key_len=%d input_count=%zu\n", embed_dim, num_heads, query_len, key_len, as.size());
    return ret;
}

static int test_multiheadattention_mask()
{
    std::vector<ncnn::Mat> self_vector(1);
    self_vector[0].create(16, 5);

    std::vector<ncnn::Mat> shared_scalar(2);
    shared_scalar[0].create(4, 5);
    shared_scalar[1].create(4, 7);

    std::vector<ncnn::Mat> shared_vector_packed_rows(2);
    shared_vector_packed_rows[0].create(16, 4);
    shared_vector_packed_rows[1].create(16, 7);

    std::vector<ncnn::Mat> shared_scalar_packed_rows(2);
    shared_scalar_packed_rows[0].create(4, 4);
    shared_scalar_packed_rows[1].create(4, 7);

    return 0
           || test_multiheadattention_mask_layout(self_vector)
           || test_multiheadattention_mask_layout(shared_scalar)
           || test_multiheadattention_mask_layout(shared_vector_packed_rows)
           || test_multiheadattention_mask_layout(shared_scalar_packed_rows);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_multiheadattention_0()
           || test_multiheadattention_1()
           || test_multiheadattention_2()
           || test_multiheadattention_mask()
#if NCNN_VALIDATION
           || test_multiheadattention_load_param()
#endif // NCNN_VALIDATION
           ;
}
