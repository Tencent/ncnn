// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_multiheadattention_oom(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v, int embed_dim, int num_heads, int attn_mask)
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

    int ret = test_layer_oom("MultiHeadAttention", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_oom failed q=(%d %d) k=(%d %d) v=(%d %d) embed_dim=%d num_heads=%d kdim=%d vdim=%d attn_mask=%d\n", q.w, q.h, k.w, k.h, v.w, v.h, embed_dim, num_heads, kdim, vdim, attn_mask);
    }

    return ret;
}

static int test_multiheadattention_0()
{
    return 0
           || test_multiheadattention_oom(RandomMat(62, 66), RandomMat(32, 66), RandomMat(20, 66), 62, 2, 0)
           || test_multiheadattention_oom(RandomMat(26, 64), RandomMat(32, 64), RandomMat(18, 64), 26, 2, 1)
           || test_multiheadattention_oom(RandomMat(12, 17), RandomMat(28, 127), RandomMat(32, 127), 12, 3, 0)
           || test_multiheadattention_oom(RandomMat(12, 17), RandomMat(28, 32), RandomMat(11, 32), 12, 3, 1);
}

#if NCNN_WEIGHT_QUANT
static int test_multiheadattention_w8a8_oom(int qdim, int kdim, int vdim, int embed_dim, int num_heads, int block_size, int input_scale)
{
    const int block_size_code = block_size == 32 ? 0 : block_size == 64 ? 1 : 2;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kdim);
    pd.set(4, vdim);
    pd.set(5, 0);
    pd.set(18, 800 + input_scale * 10 + block_size_code);

    std::vector<ncnn::Mat> weights(input_scale ? 16 : 12);
    weights[0] = RandomS8Mat(qdim, embed_dim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomS8Mat(kdim, embed_dim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomS8Mat(vdim, embed_dim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomS8Mat(embed_dim, qdim);
    weights[7] = RandomMat(qdim);
    weights[8] = RandomMat((qdim + block_size - 1) / block_size, embed_dim, 10.f, 20.f);
    weights[9] = RandomMat((kdim + block_size - 1) / block_size, embed_dim, 10.f, 20.f);
    weights[10] = RandomMat((vdim + block_size - 1) / block_size, embed_dim, 10.f, 20.f);
    weights[11] = RandomMat((embed_dim + block_size - 1) / block_size, qdim, 10.f, 20.f);

    if (input_scale)
    {
        weights[12] = RandomMat(qdim, 0.5f, 1.5f);
        weights[13] = RandomMat(kdim, 0.5f, 1.5f);
        weights[14] = RandomMat(vdim, 0.5f, 1.5f);
        weights[15] = RandomMat(embed_dim, 0.5f, 1.5f);
    }

    std::vector<ncnn::Mat> inputs(3);
    inputs[0] = RandomMat(qdim, 3);
    inputs[1] = RandomMat(kdim, 5);
    inputs[2] = RandomMat(vdim, 5);

    int ret = test_layer_oom("MultiHeadAttention", pd, weights, inputs, 1, TEST_LAYER_ENABLE_THREADING);
    if (ret != 0)
    {
        fprintf(stderr, "test_multiheadattention_w8a8_oom failed qdim=%d kdim=%d vdim=%d embed_dim=%d num_heads=%d block_size=%d input_scale=%d\n", qdim, kdim, vdim, embed_dim, num_heads, block_size, input_scale);
    }

    return ret;
}
#endif // NCNN_WEIGHT_QUANT

int main()
{
    SRAND(7767517);

    return 0
           || test_multiheadattention_0()
#if NCNN_WEIGHT_QUANT
           || test_multiheadattention_w8a8_oom(33, 35, 37, 8, 2, 32, 0)
           || test_multiheadattention_w8a8_oom(65, 67, 69, 8, 2, 64, 1)
#endif
        ;
}
