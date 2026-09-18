// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_batchnorm(int w, int h, int c)
{
    ncnn::ParamDict pd;
    pd.set(0, c);

    std::vector<ncnn::Mat> weights(4);
    weights[0] = PerfMat(c, 1.0f);
    weights[1] = PerfMat(c, 0.0f);
    weights[2] = PerfMat(c, 1.0f);
    weights[3] = PerfMat(c, 0.0f);

    perf_layer("BatchNorm", pd, weights, PerfMat(w, h, c), NULL);
}

int main()
{
    perf_batchnorm(224, 224, 3);
    perf_batchnorm(112, 112, 32);
    perf_batchnorm(56, 56, 64);
    perf_batchnorm(28, 28, 128);
    perf_batchnorm(14, 14, 256);

    return 0;
}
