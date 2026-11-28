// Copyright 2026 <pchar.cn> futz12
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_bnll(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    perf_layer("BNLL", pd, weights, a, "");
}

int main()
{
    perf_bnll(PerfMat(56, 56, 64));
    perf_bnll(PerfMat(28, 28, 128));
    perf_bnll(PerfMat(14, 14, 256));
    perf_bnll(PerfMat(7, 7, 512));
    perf_bnll(PerfMat(224, 224, 3));
    perf_bnll(PerfMat(3072, 145));

    return 0;
}
