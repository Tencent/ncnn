// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_SIMPLEOMP_H
#define NCNN_SIMPLEOMP_H

#include "platform.h"

#if NCNN_SIMPLEOMP

#include <stdint.h>

// This minimal openmp runtime implementation supports the two OpenMP compiler
// ABIs used by ncnn:
// - LLVM openmp abi, emitted by clang and clang-cl
// - GNU libgomp abi, emitted by gcc
// (msvc cl /openmp uses the incompatible vcomp abi)
// It implements a practical subset of the OpenMP runtime ABI used by ncnn and
// user code: parallel / parallel for (static / static,chunk / dynamic / guided
// / runtime / auto / collapse), if clause, reduction, critical, barrier,
// master, single, ordered, sections, atomic and flush

#ifdef __cplusplus
extern "C" {
#endif

NCNN_EXPORT int omp_get_max_threads();

NCNN_EXPORT int omp_get_num_procs();

NCNN_EXPORT int omp_in_parallel();

NCNN_EXPORT int omp_get_level();

NCNN_EXPORT int omp_get_active_level();

NCNN_EXPORT int omp_get_thread_limit();

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
