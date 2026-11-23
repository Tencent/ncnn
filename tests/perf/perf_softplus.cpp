// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_softplus(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    perf_layer("Softplus", pd, weights, a, "");
}

int main()
{
    perf_softplus(PerfMat(56, 56, 64));
    perf_softplus(PerfMat(28, 28, 128));
    perf_softplus(PerfMat(14, 14, 256));
    perf_softplus(PerfMat(7, 7, 512));
    perf_softplus(PerfMat(224, 224, 3));
    perf_softplus(PerfMat(3072, 145));

    return 0;
}
