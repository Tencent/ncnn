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

#include "platform.h"

#if NCNN_BENCHMARK

#include "mat.h"
#include "layer.h"

namespace ncnn {

struct timeval
{
    long tv_sec;
    long tv_usec;
};

// get now timestamp
struct timeval get_current_time();

// get the time elapsed in ms
double time_elapsed(struct timeval start, struct timeval end);

void benchmark(const Layer* layer, struct timeval start, struct timeval end);
void benchmark(const Layer* layer, const Mat& bottom_blob, Mat& top_blob, struct timeval start, struct timeval end);

} // namespace ncnn

#endif // NCNN_BENCHMARK

#endif // NCNN_BENCHMARK_H
