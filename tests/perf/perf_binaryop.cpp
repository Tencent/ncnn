// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static const char* op_type_name(int op_type)
{
    static const char* names[] = {"Add", "Sub", "Mul", "Div", "Max", "Min"};
    if (op_type >= 0 && op_type < 6)
        return names[op_type];
    return "Op";
}

static void perf_binaryop(const ncnn::Mat& a, const ncnn::Mat& b, int op_type)
{
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 0);
    pd.set(2, 0.f);

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> inputs(2);
    inputs[0] = a;
    inputs[1] = b;

    perf_layer("BinaryOp", pd, weights, inputs, 1, "%s", op_type_name(op_type));
}

static void perf_binaryop_scalar(const ncnn::Mat& a, int op_type)
{
    ncnn::ParamDict pd;
    pd.set(0, op_type);
    pd.set(1, 1);
    pd.set(2, 0.5f);

    std::vector<ncnn::Mat> weights(0);

    perf_layer("BinaryOp", pd, weights, a, "%s scalar", op_type_name(op_type));
}

int main()
{
    ncnn::Mat m1 = PerfMat(56, 56, 64);
    ncnn::Mat m2 = PerfMat(28, 28, 128);
    ncnn::Mat m3 = PerfMat(14, 14, 256);

    perf_binaryop(m1, m1, 0);
    perf_binaryop(m2, m2, 0);
    perf_binaryop(m3, m3, 0);
    perf_binaryop(m1, m1, 2);
    perf_binaryop_scalar(m1, 0);

    return 0;
}
