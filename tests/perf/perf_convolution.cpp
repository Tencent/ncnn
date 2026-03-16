// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_convolution(int w, int h, int c, int outch, int kernel, int dilation, int stride, int pad)
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

    perf_layer("Convolution", pd, weights, PerfMat(w, h, c),
               "out=%d k=%d d=%d s=%d p=%d", outch, kernel, dilation, stride, pad);
}

int main()
{
    perf_convolution(224, 224, 3, 64, 7, 1, 2, 3);
    perf_convolution(56, 56, 64, 64, 3, 1, 1, 1);
    perf_convolution(28, 28, 128, 128, 3, 1, 1, 1);
    perf_convolution(14, 14, 256, 256, 3, 1, 1, 1);
    perf_convolution(7, 7, 512, 512, 3, 1, 1, 1);
    perf_convolution(56, 56, 64, 128, 3, 1, 2, 1);
    perf_convolution(56, 56, 64, 256, 1, 1, 1, 0);

    return 0;
}
