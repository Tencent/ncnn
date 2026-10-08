// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layer_type.h"
#include "testutil.h"

static ncnn::Mat make_input(int dims, int k, bool transposed)
{
    if (dims == 1)
        return RandomMat(k);

    const int w = transposed ? 3 : k;
    const int h = transposed ? k : 3;
    if (dims == 2)
        return RandomMat(w, h);
    if (dims == 3)
        return RandomMat(w, h, 2);
    return RandomMat(w, h, 2, 2);
}

static int test_invalid(const ncnn::Mat& a, const ncnn::Mat& b, int transB, bool naive)
{
    ncnn::Layer* op = naive ? ncnn::create_layer_naive(ncnn::LayerType::MatMul) : ncnn::create_layer_cpu(ncnn::LayerType::MatMul);
    if (!op)
        return -1;

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_storage = false;
    opt.use_bf16_storage = false;

    ncnn::ParamDict pd;
    pd.set(0, transB);
    int ret = op->load_param(pd);
    if (ret == 0)
        ret = op->load_model(ncnn::ModelBinFromMatArray(0));
    if (ret == 0)
        ret = op->create_pipeline(opt);
    if (ret != 0)
    {
        op->destroy_pipeline(opt);
        delete op;
        return -1;
    }

    std::vector<ncnn::Mat> inputs(2);
    inputs[0] = a;
    inputs[1] = b;
    std::vector<ncnn::Mat> outputs(1);
    ret = op->forward(inputs, outputs, opt);
    op->destroy_pipeline(opt);
    delete op;

    if (ret != -1 || !outputs[0].empty())
    {
        fprintf(stderr, "test_matmul_invalid failed naive=%d transB=%d A.dims=%d B.dims=%d A.w=%d B.w=%d B.h=%d ret=%d\n", naive, transB, a.dims, b.dims, a.w, b.w, b.h, ret);
        return -1;
    }
    return 0;
}

int main(int argc, char** argv)
{
    (void)argv;
    SRAND(7767517);

    // The optional argument starts with a shorter B for sanitizer reproduction on unpatched code.
    const int bk[2] = {argc > 1 ? 1 : 65, argc > 1 ? 65 : 1};
    for (int i = 0; i < 2; i++)
    {
        for (int adims = 1; adims <= 4; adims++)
        {
            for (int bdims = 1; bdims <= 4; bdims++)
            {
                for (int transB = 0; transB <= 1; transB++)
                {
                    const ncnn::Mat a = make_input(adims, 33, false);
                    const ncnn::Mat b = make_input(bdims, bk[i], transB == 0);
                    if (test_invalid(a, b, transB, true) || test_invalid(a, b, transB, false))
                        return -1;
                }
            }
        }
    }
    return 0;
}
