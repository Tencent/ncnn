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

#include "mat.h"
#include "layer.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64

typedef struct timeval 
{
    long tv_sec;
    long tv_usec;
} timeval;

int gettimeofday(struct timeval * tp, struct timezone * tzp);
#elif defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <sys/time.h>
#endif // _WIN32

namespace ncnn {

void benchmark(const ncnn::Layer* layer, struct timeval start, struct timeval end);
void benchmark(const ncnn::Layer* layer, const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, struct timeval start, struct timeval end);

}

#endif // NCNN_BENCHMARK_H