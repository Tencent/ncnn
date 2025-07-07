// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_BENCHMARK_H
#define NCNN_BENCHMARK_H

#include "layer.h"
#include "mat.h"
#include "platform.h"

namespace ncnn {

// get now timestamp in ms
NCNN_EXPORT double get_current_time();

// sleep milliseconds
NCNN_EXPORT void sleep(unsigned long long int milliseconds = 1000);

#if NCNN_BENCHMARK

NCNN_EXPORT void benchmark(const Layer* layer, double start, double end);
NCNN_EXPORT void benchmark(const Layer* layer, const Mat& bottom_blob, Mat& top_blob, double start, double end);

#endif // NCNN_BENCHMARK

} // namespace ncnn

#endif // NCNN_BENCHMARK_H
