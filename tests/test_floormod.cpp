// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "layer.h"
#include "layer_type.h"

static int test_floormod(int reverse, int scalar)
{
    ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::FloorMod);
    if (!op) return -1;
    ncnn::ParamDict pd;
    pd.set(0, reverse);
    pd.set(1, scalar);
    pd.set(2, -2.f);
    ncnn::Option opt;
    opt.num_threads = 1;
    int ret = op->load_param(pd);
    if (ret == 0) ret = op->create_pipeline(opt);

    ncnn::Mat x(4), y(4), output;
    float* xp = x;
    float* yp = y;
    xp[0] = -1.f; xp[1] = -3.f; xp[2] = 1.f; xp[3] = 3.f;
    yp[0] = -2.f; yp[1] = 2.f; yp[2] = -2.f; yp[3] = 2.f;
    if (ret == 0 && scalar)
    {
        output = x.clone();
        ret = op->forward_inplace(output, opt);
    }
    else if (ret == 0)
    {
        std::vector<ncnn::Mat> inputs(2), outputs(1);
        inputs[0] = x;
        inputs[1] = y;
        ret = op->forward(inputs, outputs, opt);
        output = outputs[0];
    }

    const float expected[2][2][4] = {
        {{-1.f, 1.f, -1.f, 1.f}, {-1.f, -1.f, -1.f, -1.f}},
        {{0.f, -1.f, 0.f, 2.f}, {0.f, -2.f, 0.f, 1.f}}
    };
    if (ret == 0)
    {
        const float* out = output;
        for (int i = 0; i < 4; i++)
            if (out[i] != expected[reverse][scalar][i]) ret = -1;
    }
    op->destroy_pipeline(opt);
    delete op;
    return ret;
}

int main()
{
    return test_floormod(0, 0)
           || test_floormod(0, 1)
           || test_floormod(1, 0)
           || test_floormod(1, 1);
}
