// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

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

NCNN_EXPORT int omp_get_max_threads();

NCNN_EXPORT void omp_set_num_threads(int num_threads);

NCNN_EXPORT int omp_get_dynamic();

NCNN_EXPORT void omp_set_dynamic(int dynamic);

NCNN_EXPORT int omp_get_num_threads();

NCNN_EXPORT int omp_get_thread_num();

NCNN_EXPORT int kmp_get_blocktime();

NCNN_EXPORT void kmp_set_blocktime(int blocktime);

#ifdef __cplusplus
}
#endif

#endif // NCNN_SIMPLEOMP

#endif // NCNN_SIMPLEOMP_H
