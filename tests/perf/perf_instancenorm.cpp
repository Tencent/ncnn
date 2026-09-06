// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_instancenorm(int w, int h, int c)
{
    ncnn::ParamDict pd;
    pd.set(0, c);
    pd.set(1, 1e-5f);
    pd.set(2, 1);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = PerfMat(c, 1.0f);
    weights[1] = PerfMat(c, 0.0f);

    perf_layer("InstanceNorm", pd, weights, PerfMat(w, h, c), "channels=%d", c);
}

int main()
{
    // StyleGAN / diffusion representative shapes
    perf_instancenorm(64, 64, 128);
    perf_instancenorm(32, 32, 256);
    perf_instancenorm(16, 16, 512);
    perf_instancenorm(8, 8, 512);

    // Larger spatial
    perf_instancenorm(224, 224, 64);
    perf_instancenorm(224, 224, 3);

    // LLM-style degenerate case
    perf_instancenorm(4096, 1, 1);
    perf_instancenorm(512, 1, 1);

    return 0;
}
