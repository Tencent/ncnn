// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_softmax(const ncnn::Mat& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);
    pd.set(1, 1);

    std::vector<ncnn::Mat> weights(0);

    perf_layer("Softmax", pd, weights, a, "axis=%d", axis);
}

int main()
{
    perf_softmax(PerfMat(1000), 0);
    perf_softmax(PerfMat(100000), 0);
    perf_softmax(PerfMat(56, 56, 64), 0);
    perf_softmax(PerfMat(56, 56, 64), 1);
    perf_softmax(PerfMat(56, 56, 64), 2);
    perf_softmax(PerfMat(1, 1, 1000), 2);

    return 0;
}
