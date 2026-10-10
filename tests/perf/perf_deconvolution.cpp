// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_deconvolution(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad)
{
    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, 1);
    pd.set(6, outch * c * kernel * kernel);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = PerfMat(outch * c * kernel * kernel);
    weights[1] = PerfMat(outch);

    perf_layer("Deconvolution", pd, weights, PerfMat(w, h, c),
               "out=%d k=%d d=%d s=%d p=%d", outch, kernel, dilation, stride, pad);
}

int main()
{
    // typical decoder / generator upsampling configurations
    perf_deconvolution(7, 7, 512, 256, 4, 1, 2, 1);
    perf_deconvolution(14, 14, 256, 128, 4, 1, 2, 1);
    perf_deconvolution(28, 28, 128, 64, 4, 1, 2, 1);
    perf_deconvolution(56, 56, 64, 32, 4, 1, 2, 1);

    // kernel=2 stride=2 (sub-pixel style)
    perf_deconvolution(14, 14, 256, 128, 2, 1, 2, 0);
    perf_deconvolution(28, 28, 128, 64, 2, 1, 2, 0);

    // kernel=3 stride=1 (no upsampling, dilated)
    perf_deconvolution(14, 14, 256, 256, 3, 1, 1, 1);
    perf_deconvolution(14, 14, 256, 256, 3, 2, 1, 2);

    // small spatial, large channel
    perf_deconvolution(4, 4, 512, 512, 4, 1, 2, 1);

    // non-pack4-aligned channel counts
    perf_deconvolution(14, 14, 13, 8, 3, 1, 1, 1);
    perf_deconvolution(14, 14, 64, 13, 4, 1, 2, 1);

    return 0;
}
