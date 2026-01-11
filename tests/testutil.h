// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef TESTUTIL_H
#define TESTUTIL_H

#include "cpu.h"
#include "layer.h"
#include "mat.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define TEST_LAYER_DISABLE_AUTO_INPUT_PACKING (1 << 0)
#define TEST_LAYER_DISABLE_AUTO_INPUT_CASTING (1 << 1)
#define TEST_LAYER_DISABLE_GPU_TESTING        (1 << 2)
#define TEST_LAYER_ENABLE_FORCE_INPUT_PACK8   (1 << 3)
#define TEST_LAYER_ENABLE_THREADING           (1 << 4)

void SRAND(int seed);

uint64_t RAND();

float RandomFloat(float a = -1.2f, float b = 1.2f);

int RandomInt(int a = -10000, int b = 10000);

signed char RandomS8();

void Randomize(ncnn::Mat& m, float a = -1.2f, float b = 1.2f);

void RandomizeInt(ncnn::Mat& m, int a = -10000, int b = 10000);

void RandomizeS8(ncnn::Mat& m);

ncnn::Mat RandomMat(int w, float a = -1.2f, float b = 1.2f);

ncnn::Mat RandomMat(int w, int h, float a = -1.2f, float b = 1.2f);

ncnn::Mat RandomMat(int w, int h, int c, float a = -1.2f, float b = 1.2f);

ncnn::Mat RandomMat(int w, int h, int d, int c, float a = -1.2f, float b = 1.2f);

ncnn::Mat RandomIntMat(int w);

ncnn::Mat RandomIntMat(int w, int h);

ncnn::Mat RandomIntMat(int w, int h, int c);

ncnn::Mat RandomIntMat(int w, int h, int d, int c);

ncnn::Mat RandomS8Mat(int w);

ncnn::Mat RandomS8Mat(int w, int h);

ncnn::Mat RandomS8Mat(int w, int h, int c);

ncnn::Mat RandomS8Mat(int w, int h, int d, int c);

ncnn::Mat scales_mat(const ncnn::Mat& mat, int m, int k, int ldx);

bool NearlyEqual(float a, float b, float epsilon);

int Compare(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001);

int CompareMat(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001);

int CompareMat(const std::vector<ncnn::Mat>& a, const std::vector<ncnn::Mat>& b, float epsilon = 0.001);

int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& b, int flag);

int test_layer_cpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& c, const std::vector<ncnn::Mat>& top_shapes, int flag);

#if NCNN_VULKAN
int test_layer_gpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& d, const std::vector<ncnn::Mat>& top_shapes, int flag);
#endif // NCNN_VULKAN

int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, int top_blob_count, const std::vector<ncnn::Mat>& top_shapes = std::vector<ncnn::Mat>(), float epsilon = 0.001, int flag = 0);

int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, ncnn::Mat& b, int flag);

int test_layer_cpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, ncnn::Mat& c, const ncnn::Mat& top_shape, int flag);

#if NCNN_VULKAN
int test_layer_gpu(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, ncnn::Mat& d, const ncnn::Mat& top_shape, int flag);
#endif // NCNN_VULKAN

int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, const ncnn::Mat& top_shape = ncnn::Mat(), float epsilon = 0.001, int flag = 0);

int test_layer_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const std::vector<ncnn::Mat>& a, int top_blob_count = 1, float epsilon = 0.001, int flag = 0);

int test_layer_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const ncnn::Mat& a, float epsilon = 0.001, int flag = 0);

int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count = 1, float epsilon = 0.001, int flag = 0);

int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, float epsilon = 0.001, int flag = 0);

// oom test

int test_layer_oom_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const std::vector<ncnn::Mat>& a, int top_blob_count = 1, int flag = 0);

int test_layer_oom_opt(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& opt, const ncnn::Mat& a, int flag = 0);

int test_layer_oom(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count = 1, int flag = 0);

int test_layer_oom(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, int flag = 0);

#endif // TESTUTIL_H
