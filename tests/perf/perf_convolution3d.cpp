// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_convolution3d(int w, int h, int d, int c, int outch, int kernel, int dilation, int stride, int pad)
{
    ncnn::ParamDict pd;
    pd.set(0, outch);    // num_output
    pd.set(1, kernel);   // kernel_w
    pd.set(2, dilation); // dilation_w
    pd.set(3, stride);   // stride_w
    pd.set(4, pad);      // pad_w
    pd.set(5, 1);        // bias_term
    pd.set(6, outch * c * kernel * kernel * kernel);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = PerfMat(outch * c * kernel * kernel * kernel);
    weights[1] = PerfMat(outch);

    perf_layer("Convolution3D", pd, weights, PerfMat(w, h, d, c),
               "out=%d k=%d d=%d s=%d p=%d", outch, kernel, dilation, stride, pad);
}

int main()
{
    // direct convolution
    perf_convolution3d(24, 24, 16, 4, 16, 3, 1, 1, 1);
    perf_convolution3d(16, 16, 16, 8, 16, 3, 1, 2, 1);

    // winograd222 (3x3x3 stride 1)
    perf_convolution3d(16, 16, 16, 16, 16, 3, 1, 1, 1);
    perf_convolution3d(16, 16, 16, 32, 32, 3, 1, 1, 1);
    perf_convolution3d(12, 12, 12, 64, 64, 3, 1, 1, 1);

    // gemm (3x3x3 stride 2)
    perf_convolution3d(16, 16, 16, 32, 64, 3, 1, 2, 1);
    perf_convolution3d(12, 12, 12, 64, 64, 3, 1, 2, 1);

    // 1x1x1
    perf_convolution3d(16, 16, 16, 64, 64, 1, 1, 1, 0);
    perf_convolution3d(12, 12, 12, 128, 128, 1, 1, 1, 0);

    return 0;
}
