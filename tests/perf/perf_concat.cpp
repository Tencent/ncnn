// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_concat(int axis, const std::vector<ncnn::Mat>& inputs)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);

    std::vector<ncnn::Mat> weights(0);

    perf_layer("Concat", pd, weights, inputs, 1, "x%d axis=%d", (int)inputs.size(), axis);
}

int main()
{
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = PerfMat(56, 56, 64);
        inputs[1] = PerfMat(56, 56, 64);
        perf_concat(0, inputs);
    }
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = PerfMat(28, 28, 128);
        inputs[1] = PerfMat(28, 28, 128);
        perf_concat(0, inputs);
    }
    {
        std::vector<ncnn::Mat> inputs(2);
        inputs[0] = PerfMat(14, 14, 256);
        inputs[1] = PerfMat(14, 14, 256);
        perf_concat(0, inputs);
    }
    {
        std::vector<ncnn::Mat> inputs(4);
        for (int i = 0; i < 4; i++) inputs[i] = PerfMat(56, 56, 64);
        perf_concat(0, inputs);
    }

    return 0;
}
