// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef NCNN_SIMPLEOMP_H
#define NCNN_SIMPLEOMP_H

#include "platform.h"

#if NCNN_SIMPLEOMP

#include <stdint.h>

// This minimal openmp runtime implementation only supports the llvm openmp abi
// and only supports #pragma omp parallel for num_threads(X)

#ifdef __cplusplus
extern "C" {
#endif

int omp_get_max_threads();

void omp_set_num_threads(int num_threads);

int omp_get_dynamic();

void omp_set_dynamic(int dynamic);

int omp_get_num_threads();

int omp_get_thread_num();

int kmp_get_blocktime();

void kmp_set_blocktime(int blocktime);

#ifdef __cplusplus
}
#endif

#endif // NCNN_SIMPLEOMP

#endif // NCNN_SIMPLEOMP_H
