// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PERFUTIL_H
#define PERFUTIL_H

#include "layer.h"
#include "mat.h"
#include "option.h"
#include "paramdict.h"

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

#endif // PERFUTIL_H
