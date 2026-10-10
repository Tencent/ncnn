// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_pooling(int w, int h, int c, int pooling_type, int kernel, int stride, int pad, int global_pooling)
{
    ncnn::ParamDict pd;
    pd.set(0, pooling_type);
    pd.set(1, kernel);
    pd.set(2, stride);
    pd.set(3, pad);
    pd.set(4, global_pooling);
    pd.set(5, 0);
    pd.set(6, 1);

    std::vector<ncnn::Mat> weights(0);
    const char* pool_name = (pooling_type == 0) ? "max" : "avg";

    if (global_pooling)
        perf_layer("Pooling", pd, weights, PerfMat(w, h, c), "%s global", pool_name);
    else
        perf_layer("Pooling", pd, weights, PerfMat(w, h, c), "%s k=%d s=%d p=%d", pool_name, kernel, stride, pad);
}

int main()
{
    perf_pooling(112, 112, 64, 0, 3, 2, 1, 0);
    perf_pooling(56, 56, 128, 0, 3, 2, 1, 0);
    perf_pooling(28, 28, 256, 0, 3, 2, 1, 0);
    perf_pooling(14, 14, 512, 0, 3, 2, 1, 0);
    perf_pooling(56, 56, 128, 1, 3, 2, 1, 0);
    perf_pooling(7, 7, 512, 1, 0, 0, 0, 1);

    return 0;
}
