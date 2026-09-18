// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_relu(const ncnn::Mat& a, float slope)
{
    ncnn::ParamDict pd;
    pd.set(0, slope);

    std::vector<ncnn::Mat> weights(0);

    perf_layer("ReLU", pd, weights, a, "slope=%.1f", slope);
}

int main()
{
    perf_relu(PerfMat(56, 56, 64), 0.f);
    perf_relu(PerfMat(28, 28, 128), 0.f);
    perf_relu(PerfMat(14, 14, 256), 0.f);
    perf_relu(PerfMat(7, 7, 512), 0.f);
    perf_relu(PerfMat(112, 112, 32), 0.f);
    perf_relu(PerfMat(224, 224, 3), 0.f);
    perf_relu(PerfMat(56, 56, 64), 0.1f);

    return 0;
}
