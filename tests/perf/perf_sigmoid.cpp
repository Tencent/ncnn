// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_sigmoid(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;
    std::vector<ncnn::Mat> weights(0);

    perf_layer("Sigmoid", pd, weights, a, NULL);
}

int main()
{
    perf_sigmoid(PerfMat(56, 56, 64));
    perf_sigmoid(PerfMat(28, 28, 128));
    perf_sigmoid(PerfMat(14, 14, 256));
    perf_sigmoid(PerfMat(224, 224, 3));
    perf_sigmoid(PerfMat(100000));

    return 0;
}
