// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "../tools/modelwriter.h"

#include <cmath>
#include <cstdio>
#include <string>

static int test_floormod_roundtrip(int reverse, int scalar)
{
    std::string param = "7767517\n";
    param += scalar ? "2 2\n" : "3 3\n";
    param += "Input input0 0 1 in0\n";
    if (!scalar)
        param += "Input input1 0 1 in1\n";
    param += scalar ? "FloorMod mod 1 1 in0 out0" : "FloorMod mod 2 1 in0 in1 out0";
    param += reverse ? " 0=1" : " 0=0";
    if (scalar)
        param += " 1=1 2=-2";
    param += "\n";

    const char* input_bin = "test_floormod_input.bin";
    const char* output_param = "test_floormod_saved.param";
    const char* output_bin = "test_floormod_saved.bin";
    FILE* fp = fopen(input_bin, "wb");
    if (!fp) return -1;
    fclose(fp);

    ModelWriter writer;
    writer.opt.num_threads = 1;
    if (writer.load_param_mem(param.c_str()) != 0 || writer.load_model(input_bin) != 0)
        return -1;
    if (writer.save(output_param, output_bin) != 0)
        return -1;

    ncnn::Net net;
    net.opt.num_threads = 1;
    if (net.load_param(output_param) != 0 || net.load_model(output_bin) != 0)
        return -1;

    const ncnn::BinaryOp* op = (const ncnn::BinaryOp*)net.layers().back();
    if (op->op_type != (reverse ? ncnn::BinaryOp::Operation_RREMAINDER : ncnn::BinaryOp::Operation_REMAINDER)
            || op->with_scalar != scalar || (scalar && op->b != -2.f))
        return -1;

    ncnn::Mat x(4), y(4);
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
    const float expected[2][2][4] = {
        {{-1.f, 1.f, -1.f, 1.f}, {-1.f, -1.f, -1.f, -1.f}},
        {{-0.f, -1.f, -0.f, 2.f}, {-0.f, -2.f, -0.f, 1.f}}
    };

    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("in0", x) != 0 || (!scalar && ex.input("in1", y) != 0))
        return -1;
    ncnn::Mat output;
    if (ex.extract("out0", output) != 0)
        return -1;
    const float* out = output;
    for (int i = 0; i < 4; i++)
    {
        float reference = expected[reverse][scalar][i];
        if (out[i] != reference || (reference == 0.f && std::signbit(out[i]) != std::signbit(reference)))
            return -1;
    }
    return 0;
}

int main()
{
    return test_floormod_roundtrip(0, 0)
           || test_floormod_roundtrip(0, 1)
           || test_floormod_roundtrip(1, 0)
           || test_floormod_roundtrip(1, 1);
}
