// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "testutil.h"

#include "layer.h"

static int test_convolutiondepthwise_invalid_group(int group)
{
    ncnn::ParamDict pd;
    pd.set(0, 8);     // num_output
    pd.set(1, 3);     // kernel_w
    pd.set(2, 1);     // dilation_w
    pd.set(3, 1);     // stride_w
    pd.set(4, 0);     // pad_left
    pd.set(5, 0);     // bias_term
    pd.set(6, 8);     // weight_data_size
    pd.set(7, group); // group

    int typeindex = ncnn::layer_to_index("ConvolutionDepthWise");
    if (typeindex == -1)
        return -1;

    ncnn::Layer* op = ncnn::create_layer_cpu(typeindex);

    int ret = op->load_param(pd);

    delete op;

    if (ret != -100)
    {
        fprintf(stderr, "test_convolutiondepthwise_invalid_group failed group=%d ret=%d (expected -100)\n", group, ret);
        return -1;
    }

    return 0;
}

static int test_convolutiondepthwise_invalid_0()
{
    return 0
           || test_convolutiondepthwise_invalid_group(0)
           || test_convolutiondepthwise_invalid_group(-1)
           || test_convolutiondepthwise_invalid_group(-8);
}

int main()
{
    return test_convolutiondepthwise_invalid_0();
}