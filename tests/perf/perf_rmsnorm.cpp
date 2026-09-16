// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_rmsnorm(int w, int h, int c, int affine_size)
{
    ncnn::ParamDict pd;
    pd.set(0, affine_size);
    pd.set(1, 1e-5f);
    pd.set(2, 1);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = PerfMat(affine_size, 1.0f);

    perf_layer("RMSNorm", pd, weights, PerfMat(w, h, c), "affine_size=%d", affine_size);
}

int main()
{
    // typical LLM feature dimensions
    perf_rmsnorm(4096, 1, 1, 4096);
    perf_rmsnorm(4096, 1, 32, 4096);
    perf_rmsnorm(16384, 1, 1, 16384);
    perf_rmsnorm(5120, 1, 1, 5120);
    perf_rmsnorm(4096, 512, 1, 4096);

    // smaller dims for comparison
    perf_rmsnorm(1024, 1, 1, 1024);
    perf_rmsnorm(768, 1, 1, 768);

    return 0;
}
