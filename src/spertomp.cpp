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
#include <string.h>
#include <algorithm>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

#include <spert.hpp>

// ============================================================
//  A100 placement (K3 only).
//
//  ncnn bakes the RVV element-pack width into its pipeline layout at
//  create_pipeline() time:  packn = csrr_vlenb() / 4.
//
//  On K3 the two clusters have DIFFERENT vector lengths:
//      X100  cpu 0-7   vlenb= 32 B  VLEN= 256 bit  packn= 8
//      A100  cpu 8-15  vlenb=128 B  VLEN=1024 bit  packn=32
//
//  spert executes every tile on the A100 cluster, so any pipeline packed
//  while the calling thread sits on X100 is sized 4x too small for the
//  core that runs it -> out-of-bounds writes -> SIGSEGV.
//
//  The fix is therefore not in the RVV kernels: it is to make the thread
//  that builds the pipelines live on A100 as well.  Writing "0" to
//  /proc/set_ai_thread migrates the *calling* thread onto the A100 set.
//
//  That must happen before the first create_pipeline(), so it is done from
//  a static constructor that runs at load time, before main().
//  Set SPERTOMP_NO_A100=1 to disable (debugging / data layout checks).
// ============================================================
static void spert_place_on_a100()
{
#if defined(__linux__)
    if (getenv("SPERTOMP_NO_A100"))
        return;
    int fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd < 0)
        return; // not a K3 / no CC cluster: nothing to do (e.g. x86 build)
    ssize_t w = write(fd, "0", 1);
    close(fd);
    (void)w;
#endif
}

namespace {
struct SpertA100Init
{
    SpertA100Init()
    {
        spert_place_on_a100();
    }
} g_spert_a100_init;
} // namespace

// ============================================================
//  TLS mirroring simpleomp's semantics
//  tls_num_threads: team size, set on region entry (also the value
//                   omp_set_num_threads() writes)
//  tls_thread_num : this thread's position in the team
// ============================================================
namespace {
thread_local int g_tls_num_threads = 0;
thread_local int g_tls_thread_num = 0;

// Lazily created; one spert stream for the process. The first Stream for a
// backend checks out that backend's CC cores and starts its worker pool.
spert::Stream& spert_stream()
{
    static spert::Stream s;
    return s;
}

int spert_core_count()
{
    static int n = [] {
        int c = (int)spert_stream().core_count();
        return c > 0 ? c : 1;
    }();
    return n;
}

// Debug probe: report the (requested, clamped) team size and flag any
// program_id that falls outside the team, so a mismatch with ncnn's
// nT-sized scratch Mat is visible instead of a silent out-of-bounds access.
int spert_debug_enabled()
{
    static int e = [] { const char* v = getenv("SPERTOMP_DEBUG"); return v && v[0] == '1'; }();
    return e;
}

void spert_debug_launch(unsigned requested, unsigned used)
{
    if (!spert_debug_enabled())
        return;
    static int n = 0;
    if (n++ < 12)
        fprintf(stderr, "[spertomp] requested=%u clamped=%u cores=%d\n", requested, used, spert_core_count());
}

// The tile body. spert hands us the tile's coordinates; we translate the
// grid position into the OpenMP thread number and run the outlined function.
void spert_tile_body(spert::Context* ctx, void (*fn)(void*), void* data, int num_threads)
{
    g_tls_num_threads = num_threads;
    g_tls_thread_num = (int)ctx->program_id(0);
    if (spert_debug_enabled() && g_tls_thread_num >= num_threads)
        fprintf(stderr, "[spertomp] OUT OF RANGE program_id=%d num_threads=%d\n", g_tls_thread_num, num_threads);
    fn(data);
}
} // namespace

// ============================================================
//  Common dispatcher
// ============================================================
static void spert_run_parallel(void (*fn)(void*), void* data, unsigned num_threads)
{
    const unsigned requested = num_threads;
    if (num_threads == 0)
        num_threads = (unsigned)omp_get_max_threads();

    // Cap at the number of cores spert will actually schedule onto. Extra
    // tiles would merely queue behind them (correct, but pointless).
    const int cores = spert_core_count();
    if (num_threads > (unsigned)cores)
        num_threads = (unsigned)cores;

    spert_debug_launch(requested, num_threads);

    g_tls_num_threads = (int)num_threads;
    g_tls_thread_num = 0;

    // One SPMD tile per OpenMP thread. Launch is asynchronous, but the
    // contract above requires us to block before returning, so sync here.
    //
    // Even num_threads==1 goes through launch: on K3 the tiles must execute
    // on the A100 cluster for the packn baked into the pipeline (create_pipeline
    // ran on an A100-placed thread) to match the core that actually runs the
    // kernel.
    auto fut = spert_stream().launch(spert::Grid{(uint32_t)num_threads}, spert_tile_body, fn, data, (int)num_threads);
    fut.sync();
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
//  clang / libomp ABI  (not exercised on K3; kept for parity)
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

void kmp_tile_body(spert::Context* ctx, KmpRegion r)
{
    int32_t tid = (int32_t)ctx->program_id(0);
    int32_t gtid = 0;
    g_tls_num_threads = r.num_threads;
    g_tls_thread_num = (int)tid;

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
    default:
        fprintf(stderr, "spertomp: unsupported microtask argc %d\n", r.argc);
        break;
    }
}
} // namespace

extern "C" {

    int32_t __kmpc_global_thread_num(void* /*loc*/)
    {
        return 0;
    }

    void __kmpc_push_num_threads(void* /*loc*/, int32_t /*gtid*/, int32_t num_threads)
    {
        omp_set_num_threads(num_threads);
    }

    void __kmpc_fork_call(void* /*loc*/, int32_t argc, kmpc_micro fn, ...)
    {
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

        if (num_threads <= 1)
        {
            // Serial: invoke the microtask directly with tid 0. Cannot go through
            // kmp_tile_body because there is no Context to read program_id from.
            int32_t gtid = 0;
            int32_t tid = 0;
            g_tls_num_threads = 1;
            g_tls_thread_num = 0;
            switch (argc)
            {
            case 0:
                fn(&gtid, &tid);
                break;
            case 1:
                ((void (*)(int32_t*, int32_t*, void*))fn)(&gtid, &tid, argv[0]);
                break;
            case 2:
                ((void (*)(int32_t*, int32_t*, void*, void*))fn)(&gtid, &tid, argv[0], argv[1]);
                break;
            case 3:
                ((void (*)(int32_t*, int32_t*, void*, void*, void*))fn)(&gtid, &tid, argv[0], argv[1], argv[2]);
                break;
            case 4:
                ((void (*)(int32_t*, int32_t*, void*, void*, void*, void*))fn)(&gtid, &tid, argv[0], argv[1], argv[2], argv[3]);
                break;
            default:
                fprintf(stderr, "spertomp: unsupported serial microtask argc %d\n", argc);
                break;
            }
            return;
        }

        KmpRegion r{fn, (int)argc, argv, (int)num_threads};
        auto fut = spert_stream().launch(spert::Grid{(uint32_t)num_threads}, kmp_tile_body, r);
        fut.sync();
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
