// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_convolution1d(int w, int h, int outch, int kernel, int dilation, int stride, int pad)
{
    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, kernel);
    pd.set(2, dilation);
    pd.set(3, stride);
    pd.set(4, pad);
    pd.set(5, 1);
    pd.set(6, outch * h * kernel);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = PerfMat(outch * h * kernel);
    weights[1] = PerfMat(outch);

    perf_layer("Convolution1D", pd, weights, PerfMat(w, h),
               "out=%d k=%d d=%d s=%d p=%d", outch, kernel, dilation, stride, pad);
}

int main()
{
    perf_convolution1d(512, 64, 64, 3, 1, 1, 1);
    perf_convolution1d(256, 128, 128, 3, 1, 1, 1);
    perf_convolution1d(128, 256, 256, 3, 1, 1, 1);
    perf_convolution1d(64, 512, 512, 3, 1, 1, 1);
    perf_convolution1d(512, 64, 128, 3, 1, 2, 1);
    perf_convolution1d(512, 64, 256, 1, 1, 1, 0);
    perf_convolution1d(1024, 32, 64, 5, 1, 1, 2);
    perf_convolution1d(256, 256, 256, 7, 1, 2, 3);

    return 0;
}
