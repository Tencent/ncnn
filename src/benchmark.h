// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

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
