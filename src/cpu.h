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

#ifndef NCNN_CPU_H
#define NCNN_CPU_H

#include <stddef.h>

namespace ncnn {

// test optional cpu features
// neon = armv7 neon or aarch64 asimd
int cpu_support_arm_neon();
// vfpv4 = armv7 fp16 + fma
int cpu_support_arm_vfpv4();
// asimdhp = aarch64 asimd half precision
int cpu_support_arm_asimdhp();

// avx2 = x86_64 avx2 + fma + f16c
int cpu_support_x86_avx2();

// cpu info
int get_cpu_count();

// bind all threads on little clusters if powersave enabled
// affacts HMP arch cpu like ARM big.LITTLE
// only implemented on android at the moment
// switching powersave is expensive and not thread-safe
// 0 = all cores enabled(default)
// 1 = only little clusters enabled
// 2 = only big clusters enabled
// return 0 if success for setter function
int get_cpu_powersave();
int set_cpu_powersave(int powersave);

// convenient wrapper
size_t get_cpu_thread_affinity_mask(int powersave);

// set explicit thread affinity
int set_cpu_thread_affinity(size_t thread_affinity_mask);

// misc function wrapper for openmp routines
int get_omp_num_threads();
void set_omp_num_threads(int num_threads);

int get_omp_dynamic();
void set_omp_dynamic(int dynamic);

int get_omp_thread_num();

int get_kmp_blocktime();
void set_kmp_blocktime(int time_ms);

} // namespace ncnn

#endif // NCNN_CPU_H
