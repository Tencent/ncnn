// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

static int test_convolution(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad, int bias, bool use_sgemm, bool use_winograd)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, bias);
    pd.set(6, outch * c * kernel * kernel);

    int activation_type = RAND() % 7; // 0 1 2 3 4 5 6
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = RandomMat(outch * c * kernel * kernel);
    if (bias)
        weights[1] = RandomMat(outch);

    float epsilon = 0.001;

    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = false;
        opt.use_bf16_storage = false;
        opt.use_sgemm_convolution = use_sgemm;
        opt.use_winograd_convolution = use_winograd;

        int ret = test_layer_opt("Convolution", pd, weights, opt, a, epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d sgemm=%d winograd=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, use_sgemm, use_winograd, activation_type, activation_params[0], activation_params[1]);
            return ret;
        }
    }

    {
        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = true;
        opt.use_fp16_packed = false;
        opt.use_fp16_storage = false;
        opt.use_fp16_arithmetic = false;
        opt.use_bf16_packed = true;
        opt.use_bf16_storage = true;
        opt.use_sgemm_convolution = use_sgemm;
        opt.use_winograd_convolution = use_winograd;

        int ret = test_layer_opt("Convolution", pd, weights, opt, a, epsilon);
        if (ret != 0)
        {
            fprintf(stderr, "test_convolution failed w=%d h=%d c=%d outch=%d kernel=%d dilation=%d stride=%d pad=%d bias=%d sgemm=%d winograd=%d act=%d actparams=[%f,%f]\n", w, h, c, outch, kernel, dilation, stride, pad, bias, use_sgemm, use_winograd, activation_type, activation_params[0], activation_params[1]);
            return ret;
        }
    }

    return 0;
}

// Target uncovered paths in convolution_packed_bf16s.h
// All tests use sgemm=false, winograd=false to force the packed convolution path
static int test_convolution_packed()
{
    static const int kdsp[3][4] = {
        {3, 1, 2, 1},
        {5, 1, 1, -234},
        {3, 2, 1, -234},
    };

    for (int i = 0; i < 3; i++)
    {
        const int k = kdsp[i][0];
        const int d = kdsp[i][1];
        const int s = kdsp[i][2];
        const int p = kdsp[i][3];

        int ret = 0
                  || test_convolution(11, 10, 16, 2, k, d, s, p, 1, false, false)
                  || test_convolution(11, 10, 16, 3, k, d, s, p, 0, false, false)
                  || test_convolution(11, 10, 16, 1, k, d, s, p, 1, false, false)
                  || test_convolution(11, 10, 1, 16, k, d, s, p, 0, false, false)
                  || test_convolution(11, 10, 2, 16, k, d, s, p, 1, false, false)
                  || test_convolution(11, 10, 3, 16, k, d, s, p, 0, false, false)
                  || test_convolution(11, 10, 16, 20, k, d, s, p, 1, false, false);

        if (ret != 0)
            return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return test_convolution_packed();
}
