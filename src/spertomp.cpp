// Copyright 2026 SpacemiT
// SPDX-License-Identifier: BSD-3-Clause
//
// spertomp: routes ncnn's OpenMP parallel regions onto the spert
// (spine-runtime) Tile-SPMD scheduler.
//
// Contract (identical to simpleomp.cpp):
//   * GCC   lowers `#pragma omp parallel for num_threads(N)` to GOMP_parallel
//   * clang lowers it to __kmpc_fork_call + __kmpc_for_static_init_*
//
// Both entry points must BLOCK until every participant has finished: the
// shared `data` captured by the outlined function lives on the caller's
// stack, so returning early would dangle it. That matches spert's
// launch()->sync() shape exactly.

#include "spertomp.h"

#if NCNN_SPERTOMP

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <algorithm>

#include <spert.hpp>

#include "option.h"
#include "cpu.h"

// ============================================================
//  TLS mirroring simpleomp's semantics
//  tls_num_threads: team size, set on region entry (also the value
//                   omp_set_num_threads() writes)
//  tls_thread_num : this thread's position in the team
// ============================================================
namespace {
thread_local int g_tls_num_threads = 0;
thread_local int g_tls_thread_num = 0;
thread_local spert::Context* g_tls_spert_context = 0;
thread_local spert::Stream* g_tls_spert_stream = 0;

// A suspended tile may resume on a different worker. Keep TLS writes behind
// a call boundary so the compiler cannot reuse a TLS address from before join.
__attribute__((noinline)) void spert_restore_thread_state(spert::Context* ctx, spert::Stream* stream, int num_threads, int thread_num)
{
    g_tls_spert_context = ctx;
    g_tls_spert_stream = stream;
    g_tls_num_threads = num_threads;
    g_tls_thread_num = thread_num;
}

struct SpertTileScope
{
    SpertTileScope(spert::Context* ctx, spert::Stream* stream, int num_threads, int thread_num)
    {
        g_tls_spert_context = ctx;
        g_tls_spert_stream = stream;
        g_tls_num_threads = num_threads;
        g_tls_thread_num = thread_num;
    }

    ~SpertTileScope()
    {
        // Return a clean worker to the scheduler. A suspended parent restores
        // its own state after join, possibly on a different worker.
        spert_restore_thread_state(0, 0, 0, 0);
    }
};

__attribute__((noinline)) spert::Status spert_wait(spert::Future& future)
{
    spert::Context* ctx = g_tls_spert_context;
    spert::Stream* stream = g_tls_spert_stream;
    // Clang's push_num_threads may have overwritten TLS with the next team's
    // size. The current tile's grid still describes the enclosing team.
    const int num_threads = ctx ? (int)ctx->grid_dim(0) : g_tls_num_threads;
    const int thread_num = ctx ? (int)ctx->program_id(0) : g_tls_thread_num;
    // A tile must yield its worker while waiting for nested OpenMP regions.
    const spert::Status status = ctx ? ctx->join(future) : future.sync();
    // Other tiles may have used this worker's TLS while the caller was parked.
    spert_restore_thread_state(ctx, stream, num_threads, thread_num);
    return status;
}

int spert_core_count()
{
    // Querying OpenMP capabilities must not check out the backend's cores.
    const int cores = (int)(g_tls_spert_stream ? g_tls_spert_stream->core_count() : spert::backend_info().num_cores);
    return cores > 0 ? cores : 1;
}

// The tile body. spert hands us the tile's coordinates; we translate the
// grid position into the OpenMP thread number and run the outlined function.
void spert_tile_body(spert::Context* ctx, spert::Stream* stream, void (*fn)(void*), void* data, int num_threads)
{
    SpertTileScope scope(ctx, stream, num_threads, (int)ctx->program_id(0));
    fn(data);
}
} // namespace

namespace ncnn {
SpertStreamScope::SpertStreamScope()
    : stream(g_tls_spert_stream), owner(!stream)
{
    if (owner)
    {
        stream = new spert::Stream;
        g_tls_spert_stream = stream;
        if (!stream->valid())
            NCNN_LOGE("spert stream creation failed");
    }
}

SpertStreamScope::~SpertStreamScope()
{
    if (owner)
    {
        g_tls_spert_stream = 0;
        delete stream;
    }
}

bool SpertStreamScope::valid() const
{
    return stream->valid();
}

int spert_run(int (*fn)(void*), void* userdata, const Option& opt)
{
    if (g_tls_spert_context)
        return fn(userdata);

    // Root calls own an independent stream, released before returning.
    // Nested calls reuse the enclosing stream for cooperative joins.
    SpertStreamScope stream_scope;
    if (!stream_scope.valid())
        return -1;

    int ret = -1;
    auto fut = g_tls_spert_stream->launch(spert::Grid{1}, [](spert::Context* ctx, spert::Stream* stream, int (*fn)(void*), void* userdata, const Option* opt, int* result) {
        SpertTileScope scope(ctx, stream, 1, 0);
        const int old_flush_denormals = get_flush_denormals();
        set_flush_denormals(opt->flush_denormals);
        *result = fn(userdata);
        set_flush_denormals(old_flush_denormals);
    }, g_tls_spert_stream, fn, userdata, &opt, &ret);

    const spert::Status status = spert_wait(fut);
    if (status != spert::Status::Ok)
    {
        NCNN_LOGE("spert execution failed: %s", spert::to_string(status));
        return -1;
    }

    return ret;
}

} // namespace ncnn

// ============================================================
//  Common dispatcher
// ============================================================
static void spert_run_parallel(void (*fn)(void*), void* data, unsigned num_threads)
{
    // Calls outside Extractor::extract() receive a temporary stream too.
    ncnn::SpertStreamScope stream_scope;
    if (!stream_scope.valid())
        abort(); // The OpenMP ABI cannot return a stream creation error.

    if (num_threads == 0)
        num_threads = (unsigned)omp_get_max_threads();

    // Cap at the number of cores spert will actually schedule onto. Extra
    // tiles would merely queue behind them (correct, but pointless).
    const int cores = spert_core_count();
    if (num_threads > (unsigned)cores)
        num_threads = (unsigned)cores;

    // One SPMD tile per OpenMP thread. Launch is asynchronous, but the
    // contract above requires us to block before returning, so sync here.
    //
    // Even num_threads==1 goes through launch: on K3 the tiles must execute
    // on the A100 cluster for the packn baked into the pipeline (create_pipeline
    // ran on an A100-placed thread) to match the core that actually runs the
    // kernel.
    auto fut = g_tls_spert_stream->launch(spert::Grid{(uint32_t)num_threads}, spert_tile_body, g_tls_spert_stream, fn, data, (int)num_threads);
    spert_wait(fut);
}

// ============================================================
//  omp_* entry points (ABI surface shared with simpleomp)
// ============================================================
#ifdef __cplusplus
extern "C" {
#endif

int omp_get_max_threads()
{
    return spert_core_count();
}

int omp_get_dynamic()
{
    return 1;
}

void omp_set_dynamic(int /*dynamic*/)
{
    // always dynamic, ignore
}

void omp_set_num_threads(int num_threads)
{
    g_tls_num_threads = std::max(num_threads, 1);
}

int omp_get_num_threads()
{
    return std::max(g_tls_num_threads, 1);
}

int omp_get_thread_num()
{
    return g_tls_thread_num;
}

int kmp_get_blocktime()
{
    return 0;
}

void kmp_set_blocktime(int /*blocktime*/)
{
    // always passive, ignore
}

#ifdef __cplusplus
} // extern "C"
#endif

// ============================================================
//  GCC / libgomp ABI  (this is the path K3's gcc 15 builds use)
// ============================================================
#ifndef __clang__
extern "C" {

    void GOMP_parallel(void (*fn)(void*), void* data, unsigned num_threads, unsigned int /*flags*/)
    {
        spert_run_parallel(fn, data, num_threads);
    }

// Legacy split pair, kept for ABI completeness (GCC >= 4.9 emits
// GOMP_parallel instead; nothing in ncnn should reach these).
    void GOMP_parallel_start(void (*fn)(void*), void* data, unsigned num_threads)
    {
        spert_run_parallel(fn, data, num_threads);
    }

    void GOMP_parallel_end()
    {
    }

} // extern "C"
#endif // !__clang__

// ============================================================
//  clang / libomp ABI
// ============================================================
#ifdef __clang__
extern "C" typedef void (*kmpc_micro)(int32_t* gtid, int32_t* tid, ...);

namespace {
// Team state for one clang parallel region.
struct KmpRegion
{
    kmpc_micro fn;
    int argc;
    void** argv;
    int num_threads;
};

void kmp_tile_body(spert::Context* ctx, spert::Stream* stream, KmpRegion r)
{
    int32_t tid = (int32_t)ctx->program_id(0);
    int32_t gtid = tid;
    SpertTileScope scope(ctx, stream, r.num_threads, (int)tid);

    switch (r.argc)
    {
    case 0:
        r.fn(&gtid, &tid);
        break;
    case 1:
        ((void (*)(int32_t*, int32_t*, void*))r.fn)(&gtid, &tid, r.argv[0]);
        break;
    case 2:
        ((void (*)(int32_t*, int32_t*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1]);
        break;
    case 3:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2]);
        break;
    case 4:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2], r.argv[3]);
        break;
    case 5:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2], r.argv[3], r.argv[4]);
        break;
    case 6:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2], r.argv[3], r.argv[4], r.argv[5]);
        break;
    case 7:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2], r.argv[3], r.argv[4], r.argv[5], r.argv[6]);
        break;
    case 8:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*, void*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2], r.argv[3], r.argv[4], r.argv[5], r.argv[6], r.argv[7]);
        break;
    case 9:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*, void*, void*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2], r.argv[3], r.argv[4], r.argv[5], r.argv[6], r.argv[7], r.argv[8]);
        break;
    case 10:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2], r.argv[3], r.argv[4], r.argv[5], r.argv[6], r.argv[7], r.argv[8], r.argv[9]);
        break;
    case 11:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2], r.argv[3], r.argv[4], r.argv[5], r.argv[6], r.argv[7], r.argv[8], r.argv[9], r.argv[10]);
        break;
    case 12:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2], r.argv[3], r.argv[4], r.argv[5], r.argv[6], r.argv[7], r.argv[8], r.argv[9], r.argv[10], r.argv[11]);
        break;
    case 13:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2], r.argv[3], r.argv[4], r.argv[5], r.argv[6], r.argv[7], r.argv[8], r.argv[9], r.argv[10], r.argv[11], r.argv[12]);
        break;
    case 14:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2], r.argv[3], r.argv[4], r.argv[5], r.argv[6], r.argv[7], r.argv[8], r.argv[9], r.argv[10], r.argv[11], r.argv[12], r.argv[13]);
        break;
    case 15:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2], r.argv[3], r.argv[4], r.argv[5], r.argv[6], r.argv[7], r.argv[8], r.argv[9], r.argv[10], r.argv[11], r.argv[12], r.argv[13], r.argv[14]);
        break;
    case 16:
        ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*, void*))r.fn)(&gtid, &tid, r.argv[0], r.argv[1], r.argv[2], r.argv[3], r.argv[4], r.argv[5], r.argv[6], r.argv[7], r.argv[8], r.argv[9], r.argv[10], r.argv[11], r.argv[12], r.argv[13], r.argv[14], r.argv[15]);
        break;
    default:
        fprintf(stderr, "spertomp: unsupported microtask argc %d\n", r.argc);
        break;
    }
}
} // namespace

extern "C" {

    int32_t __kmpc_global_thread_num(void* /*loc*/)
    {
        return omp_get_thread_num();
    }

    void __kmpc_push_num_threads(void* /*loc*/, int32_t /*gtid*/, int32_t num_threads)
    {
        omp_set_num_threads(num_threads);
    }

    void __kmpc_fork_call(void* /*loc*/, int32_t argc, kmpc_micro fn, ...)
    {
        ncnn::SpertStreamScope stream_scope;
        if (!stream_scope.valid())
            abort(); // The OpenMP ABI cannot return a stream creation error.

        unsigned num_threads = (unsigned)omp_get_num_threads();
        if (num_threads == 0)
            num_threads = (unsigned)omp_get_max_threads();

        // The tile closure captures a copy of argv[0..argc), so the va_list's
        // stack storage only has to survive this call -- and it does, because
        // we block in sync() below.
        void* argv[32];
        {
            va_list ap;
            va_start(ap, fn);
            for (int i = 0; i < argc && i < 32; i++)
                argv[i] = va_arg(ap, void*);
            va_end(ap);
        }

        const int cores = spert_core_count();
        if (num_threads > (unsigned)cores)
            num_threads = (unsigned)cores;

        // Even a single participant must run in a runtime tile, with the same
        // A100 placement and RVV layout as multi-participant regions.
        KmpRegion r{fn, (int)argc, argv, (int)num_threads};
        auto fut = g_tls_spert_stream->launch(spert::Grid{(uint32_t)num_threads}, kmp_tile_body, g_tls_spert_stream, r);
        spert_wait(fut);
    }

    void __kmpc_for_static_init_4(void* /*loc*/, int32_t gtid, int32_t /*sched*/, int32_t* last, int32_t* lower, int32_t* upper, int32_t* /*stride*/, int32_t /*incr*/, int32_t /*chunk*/)
    {
        int num_threads = omp_get_num_threads();
        int32_t count = *upper - *lower + 1;
        int32_t threads = std::min(count, (int32_t)num_threads);
        int32_t count_per_thread = count / threads;
        int32_t remain = count % threads;
        *last = gtid == (int32_t)(threads - 1);
        *lower = gtid * count_per_thread + std::min(remain, gtid);
        *upper = std::min((gtid + 1) * count_per_thread + std::min(remain, gtid + 1) - 1, *upper);
    }

    void __kmpc_for_static_init_4u(void* /*loc*/, int32_t gtid, int32_t /*sched*/, int32_t* last, uint32_t* lower, uint32_t* upper, int32_t* /*stride*/, int32_t /*incr*/, int32_t /*chunk*/)
    {
        int num_threads = omp_get_num_threads();
        uint32_t count = *upper - *lower + 1;
        uint32_t threads = std::min(count, (uint32_t)num_threads);
        uint32_t count_per_thread = count / threads;
        uint32_t remain = count % threads;
        *last = gtid == (int32_t)(threads - 1);
        *lower = gtid * count_per_thread + std::min(remain, (uint32_t)gtid);
        *upper = std::min((gtid + 1) * count_per_thread + std::min(remain, (uint32_t)gtid + 1) - 1, *upper);
    }

    void __kmpc_for_static_init_8(void* /*loc*/, int32_t gtid, int32_t /*sched*/, int32_t* last, int64_t* lower, int64_t* upper, int64_t* /*stride*/, int64_t /*incr*/, int64_t /*chunk*/)
    {
        int num_threads = omp_get_num_threads();
        int64_t count = *upper - *lower + 1;
        int64_t threads = std::min(count, (int64_t)num_threads);
        int64_t count_per_thread = count / threads;
        int64_t remain = count % threads;
        *last = gtid == (int64_t)(threads - 1);
        *lower = gtid * count_per_thread + std::min(remain, (int64_t)gtid);
        *upper = std::min((gtid + 1) * count_per_thread + std::min(remain, (int64_t)gtid + 1) - 1, *upper);
    }

    void __kmpc_for_static_init_8u(void* /*loc*/, int32_t gtid, int32_t /*sched*/, int32_t* last, uint64_t* lower, uint64_t* upper, int64_t* /*stride*/, int64_t /*incr*/, int64_t /*chunk*/)
    {
        int num_threads = omp_get_num_threads();
        uint64_t count = *upper - *lower + 1;
        uint64_t threads = std::min(count, (uint64_t)num_threads);
        uint64_t count_per_thread = count / threads;
        uint64_t remain = count % threads;
        *last = gtid == (int64_t)(threads - 1);
        *lower = gtid * count_per_thread + std::min(remain, (uint64_t)gtid);
        *upper = std::min((gtid + 1) * count_per_thread + std::min(remain, (uint64_t)gtid + 1) - 1, *upper);
    }

    void __kmpc_for_static_fini(void* /*loc*/, int32_t gtid)
    {
        (void)gtid;
    }

} // extern "C"
#endif // __clang__

#endif // NCNN_SPERTOMP
