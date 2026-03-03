// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_innerproduct(int w, int h, int c, int outch, int bias)
{
    ncnn::Mat input;
    int weight_data_size;
    if (h == 0 && c == 0)
    {
        input = PerfMat(w);
        weight_data_size = outch * w;
    }
    else if (c == 0)
    {
        input = PerfMat(w, h);
        weight_data_size = outch * w * h;
    }
    else
    {
        input = PerfMat(w, h, c);
        weight_data_size = outch * w * h * c;
    }

    ncnn::ParamDict pd;
    pd.set(0, outch);
    pd.set(1, bias);
    pd.set(2, weight_data_size);

    std::vector<ncnn::Mat> weights(bias ? 2 : 1);
    weights[0] = PerfMat(weight_data_size);
    if (bias)
        weights[1] = PerfMat(outch);

    perf_layer("InnerProduct", pd, weights, input, "out=%d bias=%d", outch, bias);
}

int main()
{
    perf_innerproduct(25088, 0, 0, 4096, 1);
    perf_innerproduct(4096, 0, 0, 1000, 1);
    perf_innerproduct(2048, 0, 0, 1000, 1);
    perf_innerproduct(7, 7, 512, 4096, 1);

    return 0;
}
