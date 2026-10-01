// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_normalize(const ncnn::Mat& a, int across_spatial, int across_channel, int channel_shared, float eps, int eps_mode)
{
    const int scale_data_size = channel_shared ? 1 : a.c;

    ncnn::ParamDict pd;
    pd.set(0, across_spatial);
    pd.set(4, across_channel);
    pd.set(1, channel_shared);
    pd.set(2, eps);
    pd.set(3, scale_data_size);
    pd.set(9, eps_mode);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = PerfMat(scale_data_size, 0.75f);

    perf_layer("Normalize", pd, weights, a, "spatial=%d channel=%d shared=%d eps=%.4g mode=%d", across_spatial, across_channel, channel_shared, eps, eps_mode);
}

int main()
{
    perf_normalize(PerfMat(56, 56, 64), 1, 0, 0, 0.001f, 0);
    perf_normalize(PerfMat(56, 56, 64), 0, 1, 0, 0.001f, 0);
    perf_normalize(PerfMat(56, 56, 64), 1, 1, 1, 0.001f, 0);

    perf_normalize(PerfMat(28, 28, 128), 1, 0, 0, 0.001f, 1);
    perf_normalize(PerfMat(28, 28, 128), 0, 1, 1, 0.001f, 1);
    perf_normalize(PerfMat(28, 28, 128), 1, 1, 0, 0.001f, 1);

    perf_normalize(PerfMat(14, 14, 8, 64), 1, 0, 0, 0.0001f, 2);
    perf_normalize(PerfMat(14, 14, 8, 64), 0, 1, 1, 0.0001f, 2);
    perf_normalize(PerfMat(14, 14, 8, 64), 1, 1, 1, 0.0001f, 2);

    return 0;
}
