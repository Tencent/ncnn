// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PERFTESTUTIL_H
#define PERFTESTUTIL_H

#include "benchmark.h"
#include "cpu.h"
#include "layer.h"
#include "mat.h"
#include "option.h"
#include "paramdict.h"

#include <float.h>
#include <stdio.h>
#include <string.h>

#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN

// default benchmark parameters
#define PERF_WARMUP_COUNT  10
#define PERF_RUN_COUNT     20
#define PERF_MIN_TOTAL_MS  100.0

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

// simple insertion sort for small arrays
void sort_doubles(double* arr, int n);

// benchmark a layer on CPU
// input/output format conversions (packing, fp16/bf16 cast) are excluded from timing
// for fast ops (<10ms), inner loop count is automatically increased
// returns 0 on success
int perf_layer_cpu(const char* layer_type, const ncnn::ParamDict& pd,
                   const std::vector<ncnn::Mat>& weights,
                   const std::vector<ncnn::Mat>& inputs,
                   const ncnn::Option& opt,
                   int warmup_count, int run_count,
                   PerfResult& result);

// convenience overload for single-input layers
int perf_layer_cpu(const char* layer_type, const ncnn::ParamDict& pd,
                   const std::vector<ncnn::Mat>& weights,
                   const ncnn::Mat& input,
                   const ncnn::Option& opt,
                   int warmup_count, int run_count,
                   PerfResult& result);

#if NCNN_VULKAN
// benchmark a layer on GPU (Vulkan)
// host-to-device upload/download is excluded from timing
// only the GPU pipeline execution is measured
int perf_layer_gpu(const char* layer_type, const ncnn::ParamDict& pd,
                   const std::vector<ncnn::Mat>& weights,
                   const std::vector<ncnn::Mat>& inputs,
                   const ncnn::Option& opt,
                   ncnn::VulkanDevice* vkdev,
                   int warmup_count, int run_count,
                   PerfResult& result);

// convenience overload for single-input layers
int perf_layer_gpu(const char* layer_type, const ncnn::ParamDict& pd,
                   const std::vector<ncnn::Mat>& weights,
                   const ncnn::Mat& input,
                   const ncnn::Option& opt,
                   ncnn::VulkanDevice* vkdev,
                   int warmup_count, int run_count,
                   PerfResult& result);
#endif // NCNN_VULKAN

// print a formatted benchmark result line
// tag: descriptive string for the test case
void print_perf_result(const char* tag, const PerfResult& result);

// helper to format shape as string (writes into buf, returns buf)
char* format_shape(char* buf, int bufsize, const ncnn::Mat& m);

// build a default ncnn::Option for perf testing
ncnn::Option make_perf_option(int num_threads, bool use_packing = true,
                              bool use_fp16 = false, bool use_bf16 = false);

#endif // PERFTESTUTIL_H
