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
    xp[0] = -1.f;
    xp[1] = -3.f;
    xp[2] = 1.f;
    xp[3] = 3.f;
    yp[0] = -2.f;
    yp[1] = 2.f;
    yp[2] = -2.f;
    yp[3] = 2.f;
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
        {{0.f, -1.f, 0.f, 2.f}, {0.f, -2.f, 0.f, 1.f}}};
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

static float test_float_from_bits(unsigned int bits)
{
    union
    {
        unsigned int i;
        float f;
    } u;
    u.i = bits;
    return u.f;
}

static unsigned int test_float_bits(float value)
{
    union
    {
        float f;
        unsigned int i;
    } u;
    u.f = value;
    return u.i;
}

static int test_floormod_special_values()
{
    ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::FloorMod);
    if (!op) return -1;
    ncnn::ParamDict pd;
    pd.set(0, 0);
    ncnn::Option opt;
    opt.num_threads = 1;
    int ret = op->load_param(pd);
    if (ret == 0) ret = op->create_pipeline(opt);

    const float inf = test_float_from_bits(0x7f800000u);
    const float nan = test_float_from_bits(0x7fc00000u);
    const float negzero = test_float_from_bits(0x80000000u);
    const float sub1 = test_float_from_bits(1u);
    const float sub2 = test_float_from_bits(2u);
    const float sub3 = test_float_from_bits(3u);
    const float sub7 = test_float_from_bits(7u);
    const float minnormal = test_float_from_bits(0x00800000u);
    const float nextnormal = test_float_from_bits(0x00800001u);
    const float maxsub = test_float_from_bits(0x007fffffu);
    const float maxfinite = test_float_from_bits(0x7f7fffffu);
    const float values[20] = {1e8f, -1e8f, 1.f, -1.f, 1.f, -1.f, negzero, 0.f, inf, -inf, 1.f, nan, 1.f, -1.f, sub7, minnormal, nextnormal, maxfinite, -maxfinite, 1.f};
    const float divisors[20] = {3.f, 3.f, inf, inf, -inf, -inf, 3.f, -3.f, 3.f, 3.f, 0.f, 3.f, sub3, sub3, sub3, maxsub, minnormal, 1.f, 1.f, nan};
    const float expected[20] = {1.f, 2.f, 1.f, inf, -inf, -1.f, negzero, 0.f, nan, nan, nan, nan, sub2, sub1, sub1, sub1, sub1, 0.f, negzero, nan};
    ncnn::Mat x(20), y(20);
    for (int i = 0; i < 20; i++)
    {
        ((float*)x)[i] = values[i];
        ((float*)y)[i] = divisors[i];
    }
    std::vector<ncnn::Mat> inputs(2), outputs(1);
    inputs[0] = x;
    inputs[1] = y;
    if (ret == 0) ret = op->forward(inputs, outputs, opt);
    if (ret == 0)
    {
        const float* out = outputs[0];
        for (int i = 0; i < 20; i++)
        {
            const unsigned int actual_bits = test_float_bits(out[i]);
            const unsigned int expected_bits = test_float_bits(expected[i]);
            if ((expected_bits & 0x7fffffffu) > 0x7f800000u)
            {
                if ((actual_bits & 0x7fffffffu) <= 0x7f800000u) ret = -1;
            }
            else if (actual_bits != expected_bits)
                ret = -1;
        }
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
           || test_floormod(1, 1)
           || test_floormod_special_values();
}
