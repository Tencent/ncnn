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

// size: the input width the weights were made for; in_w: the runtime input width;
// state_w: width of the explicit state blobs (0 for none)
static int test_gru_shape(int naive, int size, int in_w, int outch, int direction, int state_w, int state_h, int expected_ret)
{
    const int num_directions = direction == 2 ? 2 : 1;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, outch * size * 3 * num_directions);
    pd.set(2, direction);

    std::vector<ncnn::Mat> weights(3);
    weights[0] = RandomMat(outch * size * 3 * num_directions);
    weights[1] = RandomMat(outch * 4 * num_directions);
    weights[2] = RandomMat(outch * outch * 3 * num_directions);

    std::vector<ncnn::Mat> inputs(1);
    inputs[0] = RandomMat(in_w, 4);
    if (state_w > 0)
    {
        inputs.resize(1 + 1);
        for (int i = 1; i <= 1; i++)
            inputs[i] = RandomMat(state_w, state_h);
    }

    int ret = run_layer(naive, "GRU", pd, weights, inputs, 1);
    if (ret != expected_ret)
    {
        fprintf(stderr, "test_gru_shape failed naive=%d size=%d in_w=%d outch=%d direction=%d state_w=%d state_h=%d ret=%d expected=%d\n", naive, size, in_w, outch, direction, state_w, state_h, ret, expected_ret);
        return -1;
    }

    return 0;
}

#if NCNN_INT8
// the int8 weight scales are read by forward, so a model without them must not load
static int test_gru_int8_scales(int naive, int with_scales, int expected_ret)
{
    const int size = 8;
    const int outch = 16;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, outch * size * 3);
    pd.set(2, 0);
    pd.set(8, 2);

    std::vector<ncnn::Mat> weights(5);
    weights[0] = RandomS8Mat(outch * size * 3);
    weights[1] = RandomMat(outch * 4);
    weights[2] = RandomS8Mat(outch * outch * 3);
    if (with_scales)
    {
        weights[3] = RandomMat(outch * 3, 100.f, 200.f);
        weights[4] = RandomMat(outch * 3, 100.f, 200.f);
    }

    std::vector<ncnn::Mat> inputs;

    int ret = run_layer(naive, "GRU", pd, weights, inputs, 1);
    if ((ret == 0) != (expected_ret == 0))
    {
        fprintf(stderr, "test_gru_int8_scales failed naive=%d with_scales=%d ret=%d expected=%d\n", naive, with_scales, ret, expected_ret);
        return -1;
    }

    return 0;
}
#endif // NCNN_INT8

static int test_gru_invalid_0(int naive)
{
    return 0
           || test_gru_shape(naive, 8, 8, 16, 0, 0, 0, 0)
           || test_gru_shape(naive, 8, 8, 16, 2, 0, 0, 0)
           || test_gru_shape(naive, 8, 1024, 16, 0, 0, 0, -1)
           || test_gru_shape(naive, 8, 1024, 16, 2, 0, 0, -1)
           || test_gru_shape(naive, 8, 9, 16, 1, 0, 0, -1)
           || test_gru_shape(naive, 8, 8, 256, 0, 256, 1, 0)
           || test_gru_shape(naive, 8, 8, 16, 2, 16, 2, 0)
           || test_gru_shape(naive, 8, 8, 256, 0, 1, 1, -1)
           || test_gru_shape(naive, 8, 8, 256, 0, 255, 1, -1)
           || test_gru_shape(naive, 8, 8, 16, 2, 16, 1, -1);
}

int main()
{
    SRAND(7767517);

#if NCNN_VALIDATION
    if (test_gru_invalid_0(1) || test_gru_invalid_0(0))
        return -1;
#endif // NCNN_VALIDATION

#if NCNN_INT8
    return 0
           || test_gru_int8_scales(1, 1, 0)
           || test_gru_int8_scales(0, 1, 0)
           || test_gru_int8_scales(1, 0, -100)
           || test_gru_int8_scales(0, 0, -100);
#else
    return 0;
#endif // NCNN_INT8
}
