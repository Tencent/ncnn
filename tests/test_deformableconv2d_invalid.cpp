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

// weights are made for inch input channels and a kernel x kernel window;
// the runtime input, offset and mask blobs are given explicitly
static int test_deformableconv2d_shape(int naive, int inch, int in_c, int off_w, int off_c, int mask_c, int expected_ret)
{
    const int outch = 8;
    const int kernel = 3;
    const int maxk = kernel * kernel;

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(5, 1);
    pd.set(6, outch * inch * maxk);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(outch * inch * maxk);
    weights[1] = RandomMat(outch);

    // 16x16 input and 3x3 kernel give a 14x14 output
    std::vector<ncnn::Mat> inputs(mask_c > 0 ? 3 : 2);
    inputs[0] = RandomMat(16, 16, in_c);
    inputs[1] = RandomMat(off_w, off_w, off_c);
    if (mask_c > 0)
        inputs[2] = RandomMat(14, 14, mask_c);

    int ret = run_layer(naive, "DeformableConv2D", pd, weights, inputs, 1);
    if (ret != expected_ret)
    {
        fprintf(stderr, "test_deformableconv2d_shape failed naive=%d inch=%d in_c=%d off_w=%d off_c=%d mask_c=%d ret=%d expected=%d\n", naive, inch, in_c, off_w, off_c, mask_c, ret, expected_ret);
        return -1;
    }

    return 0;
}

static int test_deformableconv2d_invalid_0(int naive)
{
    return 0
           || test_deformableconv2d_shape(naive, 4, 4, 14, 18, 0, 0)
           || test_deformableconv2d_shape(naive, 4, 4, 14, 18, 9, 0)
           || test_deformableconv2d_shape(naive, 4, 4, 14, 1, 0, -1)
           || test_deformableconv2d_shape(naive, 4, 4, 1, 18, 0, -1)
           || test_deformableconv2d_shape(naive, 4, 4, 14, 18, 1, -1)
           || test_deformableconv2d_shape(naive, 4, 256, 14, 18, 0, -1)
           || test_deformableconv2d_shape(naive, 64, 1, 14, 18, 0, -1);
}

int main()
{
    SRAND(7767517);

#if NCNN_VALIDATION
    return test_deformableconv2d_invalid_0(1) || test_deformableconv2d_invalid_0(0);
#else
    return 0;
#endif // NCNN_VALIDATION
}
