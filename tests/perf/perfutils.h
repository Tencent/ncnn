// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PERFUTILS_H
#define PERFUTILS_H

#include "benchmark.h"
#include "layer.h"
#include "mat.h"
#include "option.h"
#include "paramdict.h"

#include <stdio.h>
#include <string.h>

#include <vector>

#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

// default benchmark parameters
#define PERF_WARMUP_COUNT 10
#define PERF_RUN_COUNT    20
#define PERF_MIN_TOTAL_MS 100.0

// benchmark result for a single test case
struct PerfResult
{
    double time_min;
    double time_max;
    double time_avg;
    double time_median;
    int loop_count; // inner loops per iteration (for short ops)
};

// fill mat with constant value for reproducible benchmarks
void FillMat(ncnn::Mat& m, float v = 0.01f);

ncnn::Mat PerfMat(int w, float v = 0.01f);
ncnn::Mat PerfMat(int w, int h, float v = 0.01f);
ncnn::Mat PerfMat(int w, int h, int c, float v = 0.01f);
ncnn::Mat PerfMat(int w, int h, int d, int c, float v = 0.01f);

// high-level perf entry point: benchmark a layer across all precision and GPU variations
// layer_type: ncnn layer type name (e.g. "Convolution")
// pd, weights, input(s): layer configuration
// top_blob_count: number of output blobs (for multi-input overload)
// param_fmt: printf-style format for layer-specific params (e.g. "k=%d s=%d"), can be NULL
void perf_layer(const char* layer_type, const ncnn::ParamDict& pd,
                const std::vector<ncnn::Mat>& weights,
                const ncnn::Mat& input, const char* param_fmt, ...);

void perf_layer(const char* layer_type, const ncnn::ParamDict& pd,
                const std::vector<ncnn::Mat>& weights,
                const std::vector<ncnn::Mat>& inputs, int top_blob_count,
                const char* param_fmt, ...);

#endif // PERFUTILS_H
