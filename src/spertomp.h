// Copyright 2026 SpacemiT
// SPDX-License-Identifier: BSD-3-Clause
//
// spertomp: a drop-in replacement for simpleomp that routes ncnn's OpenMP
// parallel regions onto the spert (spine-runtime) Tile-SPMD scheduler.
//
// Enabled by NCNN_SPERTOMP (mutually exclusive with NCNN_SIMPLEOMP).
// It exports the same OpenMP runtime ABI surface as simpleomp so that the
// 3500+ `#pragma omp parallel for` sites in ncnn compile and run unchanged.

#ifndef NCNN_SPERTOMP_H
#define NCNN_SPERTOMP_H

#include "platform.h"

#if NCNN_SPERTOMP

#include <stdint.h>

#ifdef __cplusplus
namespace spert {
class Stream;
}

namespace ncnn {
class Option;

// Own a stream for one extraction or pipeline initialization. Nested scopes
// reuse the current stream, including OpenMP regions inside runtime tiles.
class SpertStreamScope
{
public:
    SpertStreamScope();
    ~SpertStreamScope();
    bool valid() const;

private:
    SpertStreamScope(const SpertStreamScope&);
    SpertStreamScope& operator=(const SpertStreamScope&);
    spert::Stream* stream;
    bool owner;
};

// Execute CPU work on the backend; nested calls stay in their current tile.
// The callback and its arguments remain alive until this synchronous call ends.
int spert_run(int (*fn)(void*), void* userdata, const Option& opt);

template<typename Fn>
int spert_run(Fn fn, const Option& opt)
{
    return spert_run([](void* userdata) { return (*static_cast<Fn*>(userdata))(); }, &fn, opt);
}
}
#endif

// Same ABI surface as simpleomp.h. GCC lowers `#pragma omp parallel for` to
// GOMP_parallel (libgomp ABI); clang lowers it to __kmpc_fork_call (libomp ABI).
// Both families are provided here.

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

#endif // NCNN_SPERTOMP

#endif // NCNN_SPERTOMP_H
