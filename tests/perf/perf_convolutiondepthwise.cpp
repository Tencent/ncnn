// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_convolutiondepthwise(int w, int h, int c, int kernel, int dilation, int stride, int pad, int group)
{
    ncnn::ParamDict pd;
    pd.set(0, c);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, 1);
    pd.set(6, c * kernel * kernel);
    pd.set(7, group);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = PerfMat(c * kernel * kernel);
    weights[1] = PerfMat(c);

    perf_layer("ConvolutionDepthWise", pd, weights, PerfMat(w, h, c),
               "k=%d s=%d g=%d", kernel, stride, group);
}

int main()
{
    perf_convolutiondepthwise(112, 112, 32, 3, 1, 1, 1, 32);
    perf_convolutiondepthwise(56, 56, 64, 3, 1, 1, 1, 64);
    perf_convolutiondepthwise(28, 28, 128, 3, 1, 1, 1, 128);
    perf_convolutiondepthwise(14, 14, 256, 3, 1, 1, 1, 256);
    perf_convolutiondepthwise(7, 7, 512, 3, 1, 1, 1, 512);
    perf_convolutiondepthwise(56, 56, 64, 5, 1, 1, 2, 64);

    return 0;
}
