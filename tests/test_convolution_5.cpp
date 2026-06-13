// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"
#include "layer_type.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int test_convolution_int8_1d(int num_input, int num_output)
{
    ncnn::Mat a = RandomMat(num_input);
    // scale up so int8 quantization rounding error is negligible
    for (int i = 0; i < num_input; i++)
        a[i] = roundf(a[i] * 10.f);

    ncnn::ParamDict pd;
    pd.set(0, num_output);
    pd.set(1, 1);
    pd.set(11, 1);
    pd.set(2, 1);
    pd.set(12, 1);
    pd.set(3, 1);
    pd.set(13, 1);
    pd.set(4, 0);
    pd.set(5, 1);
    pd.set(6, num_input * num_output);
    pd.set(8, 1); // int8_scale_term

    // int8 weights: weight, bias, per-output weight scales, input scale
    std::vector<ncnn::Mat> weights_int8(4);
    weights_int8[0] = RandomS8Mat(num_input * num_output);
    weights_int8[1] = RandomMat(num_output);
    weights_int8[2] = RandomMat(num_output);
    for (int i = 0; i < num_output; i++)
        weights_int8[2][i] = 1.f;
    weights_int8[3] = RandomMat(1);
    weights_int8[3][0] = 1.f;

    // fp32 reference weights, converted from the same int8 values
    std::vector<ncnn::Mat> weights_fp32(2);
    weights_fp32[0] = ncnn::Mat(num_input * num_output);
    for (int i = 0; i < num_input * num_output; i++)
        weights_fp32[0][i] = (float)((signed char*)weights_int8[0])[i];
    weights_fp32[1] = weights_int8[1];

    // fp32 reference path: Convolution::forward will redirect to InnerProduct
    ncnn::Mat ref;
    {
        ncnn::ParamDict pd_fp32 = pd;
        pd_fp32.set(8, 0);

        ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution);
        op->load_param(pd_fp32);
        op->load_model(ncnn::ModelBinFromMatArray(weights_fp32.data()));

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_int8_inference = false;
        opt.use_packing_layout = false;

        int ret = op->create_pipeline(opt);
        if (ret != 0)
            return ret;
        ret = op->forward(a, ref, opt);
        op->destroy_pipeline(opt);
        delete op;
        if (ret != 0)
            return ret;
    }

    // int8 path: was missing the flattened blob handling before the fix
    ncnn::Mat out;
    {
        ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution);
        op->load_param(pd);
        op->load_model(ncnn::ModelBinFromMatArray(weights_int8.data()));

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_int8_inference = true;
        opt.use_packing_layout = false;

        int ret = op->create_pipeline(opt);
        if (ret != 0)
            return ret;
        ret = op->forward(a, out, opt);
        op->destroy_pipeline(opt);
        delete op;
        if (ret != 0)
            return ret;
    }

    // compare shape and values against fp32 reference
    if (ref.dims != out.dims || ref.w != out.w || ref.h != out.h || ref.c != out.c)
    {
        fprintf(stderr, "test_convolution_int8_1d shape mismatch num_input=%d num_output=%d ref(dims=%d,w=%d,h=%d,c=%d) out(dims=%d,w=%d,h=%d,c=%d)\n",
                num_input, num_output,
                ref.dims, ref.w, ref.h, ref.c,
                out.dims, out.w, out.h, out.c);
        return -1;
    }

    float maxerr = 0.f;
    for (int i = 0; i < ref.w; i++)
    {
        float err = fabsf(ref[i] - out[i]);
        if (err > maxerr)
            maxerr = err;
    }

    if (maxerr > 0.01f)
    {
        fprintf(stderr, "test_convolution_int8_1d failed num_input=%d num_output=%d maxerr=%f\n", num_input, num_output, maxerr);
        return -1;
    }

    return 0;
}

int main()
{
    SRAND(7767517);

    return 0
           || test_convolution_int8_1d(8, 8)
           || test_convolution_int8_1d(16, 8)
           || test_convolution_int8_1d(17, 5);
}
