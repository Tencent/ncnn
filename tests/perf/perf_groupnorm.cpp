// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_groupnorm(int w, int h, int c, int group)
{
    ncnn::ParamDict pd;
    pd.set(0, group);
    pd.set(1, c);
    pd.set(2, 1e-5f);
    pd.set(3, 1);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = PerfMat(c, 1.0f);
    weights[1] = PerfMat(c, 0.0f);

    perf_layer("GroupNorm", pd, weights, PerfMat(w, h, c), NULL);
}

int main()
{
    // Stable Diffusion-like shapes
    perf_groupnorm(64, 64, 128, 32);
    perf_groupnorm(32, 32, 256, 32);
    perf_groupnorm(16, 16, 512, 32);
    perf_groupnorm(8, 8, 512, 32);

    // Image-like shapes
    perf_groupnorm(224, 224, 3, 3);
    perf_groupnorm(224, 224, 64, 32);

    // 1D / LLM-like
    perf_groupnorm(4096, 1, 1, 1);
    perf_groupnorm(512, 1, 1, 1);

    return 0;
}
