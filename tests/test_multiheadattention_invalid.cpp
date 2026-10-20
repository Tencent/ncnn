// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

// Runs one layer through load_param, load_model, create_pipeline and, when
// inputs are given, forward, and returns the first non-zero return code, or 0.
static int run_layer(int naive, const char* type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& inputs, int top_count)
{
    ncnn::Layer* op = naive ? ncnn::create_layer_naive(type) : ncnn::create_layer_cpu(type);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;

    int ret = op->load_param(pd);
    if (ret == 0)
    {
        ncnn::ModelBinFromMatArray mb(weights.data());
        ret = op->load_model(mb);
    }

    if (ret == 0)
    {
        ret = op->create_pipeline(opt);
        if (ret == 0 && !inputs.empty())
        {
            std::vector<ncnn::Mat> tops(top_count);
            ret = op->forward(inputs, tops, opt);
        }
        op->destroy_pipeline(opt);
    }

    delete op;
    return ret;
}

// weights are made for embed_dim 16, qdim 16, kdim 16 and vdim 16
static int test_multiheadattention_shape(int naive, int q_w, int k_w, int v_w, int k_h, int v_h, int expected_ret)
{
    const int embed_dim = 16;
    const int num_heads = 2;
    const int qdim = 16;
    const int kdim = 16;
    const int vdim = 16;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, num_heads);
    pd.set(2, embed_dim * qdim);
    pd.set(3, kdim);
    pd.set(4, vdim);

    std::vector<ncnn::Mat> weights(8);
    weights[0] = RandomMat(embed_dim * qdim);
    weights[1] = RandomMat(embed_dim);
    weights[2] = RandomMat(embed_dim * kdim);
    weights[3] = RandomMat(embed_dim);
    weights[4] = RandomMat(embed_dim * vdim);
    weights[5] = RandomMat(embed_dim);
    weights[6] = RandomMat(qdim * embed_dim);
    weights[7] = RandomMat(qdim);

    std::vector<ncnn::Mat> inputs(3);
    inputs[0] = RandomMat(q_w, 4);
    inputs[1] = RandomMat(k_w, k_h);
    inputs[2] = RandomMat(v_w, v_h);

    int ret = run_layer(naive, "MultiHeadAttention", pd, weights, inputs, 1);
    if (ret != expected_ret)
    {
        fprintf(stderr, "test_multiheadattention_shape failed naive=%d q_w=%d k_w=%d v_w=%d k_h=%d v_h=%d ret=%d expected=%d\n", naive, q_w, k_w, v_w, k_h, v_h, ret, expected_ret);
        return -1;
    }

    return 0;
}

#if NCNN_INT8
// the int8 weight scales are read by forward, so a model without them must not load
static int test_multiheadattention_int8_scales(int naive, int with_scales, int expected_ret)
{
    const int embed_dim = 16;

    ncnn::ParamDict pd;
    pd.set(0, embed_dim);
    pd.set(1, 2);
    pd.set(2, embed_dim * embed_dim);
    pd.set(18, 2);

    std::vector<ncnn::Mat> weights(12);
    for (int i = 0; i < 4; i++)
    {
        weights[i * 2] = RandomS8Mat(embed_dim * embed_dim);
        weights[i * 2 + 1] = RandomMat(embed_dim);
    }
    if (with_scales)
    {
        weights[8] = RandomMat(embed_dim, 1.f, 10.f);
        weights[9] = RandomMat(embed_dim, 1.f, 10.f);
        weights[10] = RandomMat(embed_dim, 1.f, 10.f);
        weights[11] = RandomMat(1, 1.f, 10.f);
    }

    std::vector<ncnn::Mat> inputs;

    int ret = run_layer(naive, "MultiHeadAttention", pd, weights, inputs, 1);
    if ((ret == 0) != (expected_ret == 0))
    {
        fprintf(stderr, "test_multiheadattention_int8_scales failed naive=%d with_scales=%d ret=%d expected=%d\n", naive, with_scales, ret, expected_ret);
        return -1;
    }

    return 0;
}
#endif // NCNN_INT8

static int test_multiheadattention_invalid_0(int naive)
{
    return 0
           || test_multiheadattention_shape(naive, 16, 16, 16, 5, 5, 0)
           || test_multiheadattention_shape(naive, 1, 16, 16, 5, 5, -1)
           || test_multiheadattention_shape(naive, 4096, 16, 16, 5, 5, -1)
           || test_multiheadattention_shape(naive, 16, 1, 16, 5, 5, -1)
           || test_multiheadattention_shape(naive, 16, 16, 1, 5, 5, -1)
           || test_multiheadattention_shape(naive, 16, 16, 16, 64, 1, -1);
}

int main()
{
    SRAND(7767517);

#if NCNN_VALIDATION
    if (test_multiheadattention_invalid_0(1) || test_multiheadattention_invalid_0(0))
        return -1;
#endif // NCNN_VALIDATION

#if NCNN_INT8
    return 0
           || test_multiheadattention_int8_scales(1, 1, 0)
           || test_multiheadattention_int8_scales(0, 1, 0)
           || test_multiheadattention_int8_scales(1, 0, -100)
           || test_multiheadattention_int8_scales(0, 0, -100);
#else
    return 0;
#endif // NCNN_INT8
}
