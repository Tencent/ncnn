// SpacemiT K3 A100 IME2 (smt.vfwmadot) kernels for SDPA prefill.
//
// Same origin and constraints as gemm_riscv_ime2.h:
//   - Can only execute on A100 clusters (VLEN=1024); vfwmadot is SIGILL on X100.
//   - The whole process must be pinned to A100 (ai-run / /proc/set_ai_thread).
// Difference: inputs here are fp32 (SDPA q/k/v are all fp32 Mat), converted to fp16 at packing time;
//       output is written as fp32 (attention scores/output feed softmax and the following fp32 graph).
//
// This file is #include'd by sdpa_riscv_zfh.cpp (the zfh variant TU carries xsmtvdotii assembly support).

#ifndef SDPA_RISCV_IME2_H
#define SDPA_RISCV_IME2_H

#include "platform.h"
#include <float.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#if defined __linux__
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <poll.h>
#include <errno.h>
#endif
#include <stdlib.h>
#include <string.h>
#if defined __linux__
#include <sched.h>
#endif

#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2 && __riscv_v

namespace ncnn_ime2_sdpa {

/* Tail room for the packed buffers, matching the over-read of the K-direction
 * prefetch in ime2_tile_16x16p (128 bytes = one prefetch). Twice that is kept. */
#define IME2_TILE_SLACK 128

typedef __fp16 ime2_fp16;

/* ---------------- Runtime probe (A100?) ---------------- */

__attribute__((noinline)) static void sdpa_ime2_probe_instruction()
{
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vmv.v.i v16, 0\n\t"
        "vsetvli t0, zero, e16, m1, tu, mu\n\t"
        "vmv.v.i v0, 0\n\t"
        "vmv.v.i v1, 0\n\t"
        "smt.vfwmadot v16, v0, v1\n\t"
        :
        :
        : "t0", "v0", "v1", "v16", "v17", "memory");
}

#if defined __linux__
/* On the heterogeneous dual-cluster (K3: A100 has VLEN=1024, X100 has VLEN=512), the
 * "core the probe runs on" and the "core the OpenMP workers actually land on during
 * forward" may differ: if the former happens to be A100, true gets cached, and the
 * latter landing on X100 gets SIGILL. So the probe must cover **every** core in the
 * affinity mask: move the calling thread to each in turn and read vlenb; only 128B
 * (1024bit) counts. */
static int sdpa_ime2_probe_affinity(int nthreads)
{
    cpu_set_t cs;
    if (sched_getaffinity(0, sizeof(cs), &cs) != 0)
        return 0; /* cannot confirm affinity: conservatively disable */
    const cpu_set_t orig = cs;
    int all_ok = 1;
    for (int c = 0; c < CPU_SETSIZE && all_ok; c++)
    {
        if (!CPU_ISSET(c, &cs))
            continue;
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(c, &one);
        if (sched_setaffinity(0, sizeof(one), &one) != 0)
        {
            all_ok = 0; /* if we cannot move there, that core cannot be confirmed */
            break;
        }
        unsigned long vlenb = 0;
        __asm__ volatile("csrr %0, vlenb"
                         : "=r"(vlenb));
        if (vlenb != 128)
            all_ok = 0;
    }
    sched_setaffinity(0, sizeof(orig), &orig); /* restore */

    /* (a2) OpenMP workers may have **per-thread** bindings (OMP_PROC_BIND /
     * GOMP_CPU_AFFINITY / host runtime); the mask walk above only looks at the calling
     * thread and cannot see them. Also, forward's parallel regions all carry
     * num_threads(opt.num_threads): if it exceeds the default team, the runtime assigns
     * forward extra workers that have **never been probed**. So open the region here
     * with exactly the same team size as forward.
     * Reading vlenb once is not enough either: if a worker's mask covers both clusters,
     * the probe may happen to land on A100 and pass, while the scheduler may move it to
     * X100 during forward. So each worker must walk **its own** affinity mask core by
     * core, confirming vlenb==128 on each (restoring the worker's original binding
     * afterwards). If any worker's mask mixes in X100 cores, conservatively disable IME2. */
    if (all_ok)
    {
        int workers_ok = 1;
        #pragma omp parallel num_threads(nthreads)
        {
            cpu_set_t wcs;
            if (sched_getaffinity(0, sizeof(wcs), &wcs) != 0)
            {
                #pragma omp atomic write
                workers_ok = 0;
            }
            else
            {
                const cpu_set_t worig = wcs;
                for (int c = 0; c < CPU_SETSIZE; c++)
                {
                    if (!CPU_ISSET(c, &wcs))
                        continue;
                    cpu_set_t one;
                    CPU_ZERO(&one);
                    CPU_SET(c, &one);
                    if (sched_setaffinity(0, sizeof(one), &one) != 0)
                    {
                        #pragma omp atomic write
                        workers_ok = 0;
                        break;
                    }
                    unsigned long vlenb = 0;
                    __asm__ volatile("csrr %0, vlenb"
                                     : "=r"(vlenb));
                    if (vlenb != 128)
                    {
                        #pragma omp atomic write
                        workers_ok = 0;
                    }
                }
                sched_setaffinity(0, sizeof(worig), &worig); /* restore the worker's original binding */
            }
        }
        all_ok = workers_ok;
    }
    return all_ok;
}
#else
static int sdpa_ime2_probe_affinity(int /*nthreads*/)
{
    return 0;
}
#endif // __linux__

/* Hardware support probe, by construction free of every signal-state race:
 * run the instruction in a FORKED CHILD (same design as the gemm probe).
 * If the CPU lacks xsmtvdotii, only the child dies of SIGILL and the parent
 * observes a non-zero exit. No process-wide handler, no saved disposition,
 * no forwarding, no mask manipulation, nothing to restore -- a custom
 * handler, a one-shot action, sigaltstack, or a blocked SIGILL in the
 * parent are all irrelevant because the child resets its own disposition
 * and unblocks SIGILL before executing the instruction. The child pins
 * itself to an A100 CPU (vlenb==128); the result is a hardware property,
 * cached process-wide (one fork per process, under the probe mutex). */
static int ime2_hw_supported()
{
#if defined __linux__
    static int cached = -1;
    if (cached != -1)
        return cached;

    int cpu = -1;
    cpu_set_t cs;
    if (sched_getaffinity(0, sizeof(cs), &cs) == 0)
    {
        const cpu_set_t orig = cs;
        for (int c = 0; c < CPU_SETSIZE; c++)
        {
            if (!CPU_ISSET(c, &cs))
                continue;
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(c, &one);
            if (sched_setaffinity(0, sizeof(one), &one) != 0)
                continue;
            unsigned long vlenb = 0;
            __asm__ volatile("csrr %0, vlenb"
                             : "=r"(vlenb));
            if (vlenb == 128)
            {
                cpu = c;
                break;
            }
        }
        sched_setaffinity(0, sizeof(orig), &orig);
    }
    if (cpu < 0)
    {
        cached = 0;
        return 0;
    }

    /* The child reports success over a pipe (same as the gemm probe): with
     * SIGCHLD ignored, SA_NOCLDWAIT, or a reaping handler, waitpid can fail
     * with ECHILD and a zero-initialized status would read as 'supported'.
     * Only an explicit byte from the child is accepted as success. */
    /* O_CLOEXEC on both ends (same as the gemm probe): another thread's
     * fork+exec'd subprocess must not inherit fds[1] and delay this read's
     * EOF; our own probe child never execs, so its write end is unaffected. */
    int fds[2];
    if (pipe2(fds, O_CLOEXEC) != 0)
    {
        cached = 0;
        return 0;
    }
    pid_t pid = fork();
    if (pid < 0)
    {
        close(fds[0]);
        close(fds[1]);
        cached = 0;
        return 0;
    }
    if (pid == 0)
    {
        close(fds[0]);
        /* No model-sized core dump for the intentional SIGILL. */
        struct rlimit rl;
        rl.rlim_cur = 0;
        rl.rlim_max = 0;
        setrlimit(RLIMIT_CORE, &rl);
        struct sigaction sa;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = SIG_DFL;
        if (sigaction(SIGILL, &sa, NULL) != 0)
        {
            /* Could not reset (seccomp): exit without writing so the parent
             * judges unsupported and nothing hangs or reports a false success. */
            _exit(1);
        }
        sigset_t m;
        sigemptyset(&m);
        sigaddset(&m, SIGILL);
        sigprocmask(SIG_UNBLOCK, &m, NULL);
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpu, &one);
        sched_setaffinity(0, sizeof(one), &one);
        sdpa_ime2_probe_instruction();
        (void)!write(fds[1], "1", 1);
        _exit(0);
    }
    close(fds[1]);
    /* Reap first, then read with a bounded poll (same as the gemm probe):
     * a fork-without-exec sibling can still hold fds[1], so a blocking EOF
     * read could hang; after the child is reaped (or auto-reaped, ECHILD),
     * anything it wrote is already buffered. */
    int st = 0;
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR)
    {
    }
    struct pollfd pfd;
    memset(&pfd, 0, sizeof pfd);
    pfd.fd = fds[0];
    pfd.events = POLLIN;
    char b = 0;
    ssize_t n = -1;
    if (poll(&pfd, 1, 100) > 0)
    {
        do
        {
            n = read(fds[0], &b, 1);
        } while (n < 0 && errno == EINTR);
    }
    close(fds[0]);
    cached = (n == 1 && b == '1') ? 1 : 0;
    return cached;
#else
    return 0;
#endif
}

static int sdpa_ime2_probe_locked(int nthreads)
{
    /* (a) Affinity: every core allowed for this process + every worker of the forward team must have vlenb == 1024bit */
    if (!sdpa_ime2_probe_affinity(nthreads))
        return 0;

    return ime2_hw_supported();
}

/* Returns 1 = every core allowed in the current affinity context can execute
 * smt.vfwmadot; 0 = cannot.
 * The instruction itself is probed in a forked child (ime2_hw_supported), so no
 * process-wide signal state is touched at all; the mutex only serializes the
 * affinity walk and the cached result.
 * Note: no process-wide caching: affinity is per-thread; the probe result is only valid
 * for the affinity context of this call. If a process first binds a thread to the A100
 * cluster to build a pipeline, then binds to the X100 cluster to build a second
 * pipeline, caching would wrongly reuse on the latter an answer that only holds for the
 * former (forward on X100 = SIGILL).
 * nthreads must match forward's num_threads: the probed team is the team that executes
 * the kernels.
 * The probe is called only once at pipeline build time; re-probing overhead is
 * negligible. */
static int ime2_probe(int nthreads)
{
    static ncnn::Mutex lock;
    ncnn::MutexLockGuard g(lock);
    return sdpa_ime2_probe_locked(nthreads);
}

/* One-shot determination of which cores among those currently allowed for this process
 * are A100 (vlenb==128). A hardware property, cached process-wide; if the mask at
 * determination time covers only part of the cores, the resulting set is a subset of
 * the true A100 set — the bias direction is conservative (can only wrongly disable
 * IME2, never wrongly enable it). */
#if defined __linux__
/* force_collect=1: forced re-collection when team_ok finds the cache incomplete (bypasses the caller-mask fast path) */
static int sdpa_ime2_a100_set_impl(int nthreads, int force_collect, cpu_set_t* out)
{
    static ncnn::Mutex lock;
    ncnn::MutexLockGuard g(lock);
    static int probed = 0;
    static cpu_set_t cached;     /* set of cores confirmed vlenb==128 (grows only) */
    static cpu_set_t seen_union; /* union of team masks seen so far */

    /* Fast path: the cache covers the caller's mask and re-collection is not forced —
     * use the cache directly (one syscall, no omp region; decoding has many layers per
     * token, so overhead must stay bounded). */
    if (probed && !force_collect)
    {
        cpu_set_t wcs;
        if (sched_getaffinity(0, sizeof(wcs), &wcs) == 0)
        {
            int covered = 1;
            for (int c = 0; c < CPU_SETSIZE; c++)
                if (CPU_ISSET(c, &wcs) && !CPU_ISSET(c, &seen_union))
                {
                    covered = 0;
                    break;
                }
            if (covered)
            {
                *out = cached;
                return 1;
            }
        }
    }

    /* 1) Collect the union of the current team on every call (inside the region only
     *    sched_getaffinity is done; no privileged instructions, runs anywhere). The
     *    cache is only valid for "cores already seen": if a small team is probed first
     *    and a larger team is used later, the newcomer's legitimate A100 cores are not
     *    in the cache and team_ok would misjudge — so the union must accumulate. */
    cpu_set_t uni;
    CPU_ZERO(&uni);
    #pragma omp parallel num_threads(nthreads)
    {
        cpu_set_t wcs;
        if (sched_getaffinity(0, sizeof(wcs), &wcs) == 0)
        {
            #pragma omp critical
            {
                for (int c = 0; c < CPU_SETSIZE; c++)
                    if (CPU_ISSET(c, &wcs))
                        CPU_SET(c, &uni);
            }
        }
    }
    if (CPU_COUNT(&uni) == 0)
        return probed ? (*out = cached, 1) : 0;

    /* 2) Classify only cores not seen before (vlenb walk); seen ones use the cache directly */
    cpu_set_t fresh;
    CPU_ZERO(&fresh);
    for (int c = 0; c < CPU_SETSIZE; c++)
        if (CPU_ISSET(c, &uni) && (!probed || !CPU_ISSET(c, &seen_union)))
            CPU_SET(c, &fresh);

    if (CPU_COUNT(&fresh) > 0)
    {
        cpu_set_t orig;
        if (sched_getaffinity(0, sizeof(orig), &orig) != 0)
            return probed ? (*out = cached, 1) : 0;
        for (int c = 0; c < CPU_SETSIZE; c++)
        {
            if (!CPU_ISSET(c, &fresh))
                continue;
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(c, &one);
            if (sched_setaffinity(0, sizeof(one), &one) != 0)
                continue;
            unsigned long vlenb = 0;
            __asm__ volatile("csrr %0, vlenb"
                             : "=r"(vlenb));
            if (vlenb == 128)
                CPU_SET(c, &cached);
        }
        sched_setaffinity(0, sizeof(orig), &orig);
        for (int c = 0; c < CPU_SETSIZE; c++)
            if (CPU_ISSET(c, &uni))
                CPU_SET(c, &seen_union);
        probed = 1;
    }

    *out = cached;
    return probed;
}

static int sdpa_ime2_a100_set(int nthreads, cpu_set_t* out)
{
    return sdpa_ime2_a100_set_impl(nthreads, 0, out);
}

/* In-place check of the CURRENT worker against the cached A100 set — no region
 * spawn, for in-region validation (validate inside the same parallel region that
 * runs the smt instructions). */
static int sdpa_ime2_worker_in_set_now()
{
#if defined __linux__
    cpu_set_t a100;
    if (!sdpa_ime2_a100_set(1, &a100))
        return 0;
    cpu_set_t wcs;
    if (sched_getaffinity(0, sizeof(wcs), &wcs) != 0)
        return 0;
    for (int c = 0; c < CPU_SETSIZE; c++)
        if (CPU_ISSET(c, &wcs) && !CPU_ISSET(c, &a100))
            return 0;
    return 1;
#else
    return 0;
#endif
}
#endif

/* Lightweight validation at forward entry: this forward's team **cannot possibly**
 * land on non-A100 cores.
 * The pipeline-build probe (ime2_probe) only covers the context at the pipeline-build
 * moment: the caller may later change bindings with set_cpu_thread_affinity(), or call
 * forward with a larger num_threads (the runtime assigns unprobed workers). And looking
 * only at a worker's "current" vlenb is not enough either: a worker whose mask spans
 * clusters may be on A100 at sampling time but migrated to X100 mid-kernel. So here we
 * check per worker whether its affinity mask is a subset of the A100 set (if a subset,
 * the scheduler has nowhere to migrate it). One sched_getaffinity + bit test per
 * worker, µs-level, negligible relative to each SDPA layer.
 * On return 0 the caller falls back to the non-IME2 path; result semantics unchanged. */
static int ime2_team_ok(int nthreads)
{
#if defined __linux__
    cpu_set_t a100;
    if (!sdpa_ime2_a100_set(nthreads, &a100))
        return 0;
    int ok = 1;
    #pragma omp parallel num_threads(nthreads)
    {
        cpu_set_t wcs;
        if (sched_getaffinity(0, sizeof(wcs), &wcs) != 0)
        {
            #pragma omp atomic write
            ok = 0;
        }
        else
        {
            for (int c = 0; c < CPU_SETSIZE; c++)
            {
                if (CPU_ISSET(c, &wcs) && !CPU_ISSET(c, &a100))
                {
                    #pragma omp atomic write
                    ok = 0;
                    break;
                }
            }
        }
    }
    if (ok)
        return 1;
    /* Failure: possibly the cache lacks cores newly seen by the team (per-worker
     * binding expanded the mask). Force one re-collection and re-verify; only a
     * still-failing result counts as a real failure. */
    if (!sdpa_ime2_a100_set_impl(nthreads, 1, &a100))
        return 0;
    ok = 1;
    #pragma omp parallel num_threads(nthreads)
    {
        cpu_set_t wcs;
        if (sched_getaffinity(0, sizeof(wcs), &wcs) != 0)
        {
            #pragma omp atomic write
            ok = 0;
        }
        else
        {
            for (int c = 0; c < CPU_SETSIZE; c++)
            {
                if (CPU_ISSET(c, &wcs) && !CPU_ISSET(c, &a100))
                {
                    #pragma omp atomic write
                    ok = 0;
                    break;
                }
            }
        }
    }
    return ok;
#else
    (void)nthreads;
    return 0;
#endif
}

/* fp16 packing range check: finite values with |v| > 65504 pack into ±inf, while the
 * upstream fp32 path stays finite. IME2 prefill scans all inputs once before packing
 * (before any state change); out of range means the caller falls back wholesale to the
 * upstream GEMM combination path. Per channel, scan only valid elements (the cstep
 * padding area is uninitialized garbage and must not be hit). max is an exact
 * operation, order-independent, so an unordered reduction is fine. */
/* When out_mx is non-null, write back the maximum absolute value of valid elements (for range-scaling related gates) */
static int ime2_fits_fp16(const ncnn::Mat& m, float* out_mx = 0)
{
    if (m.empty())
        return 1;
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    const int nch = m.dims == 3 ? m.c : 1;
    for (int q = 0; q < nch; q++)
    {
        const float* p = m.dims == 3 ? (const float*)m.channel(q).data : (const float*)m.data;
        const size_t n = (size_t)m.w * m.h;
        size_t i = 0;
        vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(0.f, vlmax);
        vfloat32m1_t vmin = __riscv_vfmv_v_f_f32m1(FLT_MAX, vlmax);
        int has_nan = 0;
        while (i < n)
        {
            const size_t vl = __riscv_vsetvl_e32m1(n - i);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(p + i, vl);
            /* NaN slips through every comparison: |NaN| > 65504 is false, and
             * vfmin/vfmax absorb a single NaN (numeric operand wins), so mx/vmin
             * stay finite. Causal block skipping then diverges from the fallback:
             * the fallback still multiplies P by NaN V (0*NaN = NaN) while the
             * flash path omits the block entirely. Reject non-finite content. */
            vbool32_t isnan = __riscv_vmfne_vv_f32m1_b32(v, v, vl);
            if (__riscv_vfirst_m_b32(isnan, vl) != -1)
                has_nan = 1;
            vfloat32m1_t vabs = __riscv_vfabs_v_f32m1(v, vl);
            vmax = __riscv_vfmax_vv_f32m1_tu(vmax, vmax, vabs, vl);
            /* Only nonzero values participate in the lower bound: 0 is legitimate
             * content, not underflow. Zero lanes are set to FLT_MAX (vfmerge: zero
             * lanes pick the latter), then a plain _tu min. */
            vbool32_t iszero = __riscv_vmfeq_vf_f32m1_b32(v, 0.f, vl);
            vfloat32m1_t vsel = __riscv_vfmerge_vfm_f32m1(vabs, FLT_MAX, iszero, vl);
            vmin = __riscv_vfmin_vv_f32m1_tu(vmin, vmin, vsel, vl);
            i += vl;
        }
        if (has_nan)
            return 0;
        const float mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(vmax, __riscv_vfmv_v_f_f32m1(0.f, 1), vlmax));
        if (out_mx && mx > *out_mx)
            *out_mx = mx;
        if (mx > 65504.f)
            return 0;
        /* Underflow to zero: finite values with |v| <= 2^-25 pack into +0 (under RNE a
         * value exactly 2^-25 ties to the even side = 0; the decimal literal 2.98e-8f
         * is smaller than the exact boundary and would miss the (2.98e-8, 2^-25]
         * interval). Use the exact hexadecimal boundary, inclusive.
         * Such values in Q/K get magnified by scale into non-negligible dot-product
         * loss; fall back just as for overflow. */
        const float mn = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(vmin, __riscv_vfmv_v_f_f32m1(FLT_MAX, 1), vlmax));
        if (mn <= 0x1p-25f)
            return 0;
    }
    return 1;
}

/* ---------------- Packing: fp32 -> 8x8 fp16 tile contiguous layout ---------------- */

/* src: [R, K] fp32 row-major (row stride ld elements) -> tiles [rt][kt][64] fp16 */
static int ime2_pack_fp32(const float* src, size_t ld, ncnn::Mat& T, int R, int K, ncnn::Allocator* alloc)
{
    const int rt = (R + 7) / 8, kt = (K + 7) / 8;
    /* IME2_TILE_SLACK fp16 of tail room: the K-direction pipeline in
     * ime2_tile_16x16p prefetches the next tile pair, and with an even kt the
     * final prefetch reads 128 bytes past the last tile row. The data is never
     * consumed, but the read has to stay inside the allocation. */
    /* Reuse a sufficient existing buffer: with causal long prefill (nend advances
     * per 32-row block) or non-multiple KV lengths, the exact-size create would
     * release and reallocate on every block — hundreds of alloc/free pairs per
     * head per layer. Only recreate when the capacity is insufficient. */
    const size_t need = (size_t)rt * kt * 64 + IME2_TILE_SLACK;
    if (T.dims != 1 || T.elemsize != (size_t)2u || T.total() < need)
    {
        T.release();
        T.create(need, (size_t)2u, 1, alloc);
        if (T.empty())
            return -100;
    }
    ime2_fp16* dst = (ime2_fp16*)T;

    for (int i = 0; i < rt; i++)
    {
        const int gr0 = i * 8;
        for (int j = 0; j < kt; j++)
        {
            ime2_fp16* t = dst + ((size_t)i * kt + j) * 64;
            const int gk0 = j * 8;
            if (gr0 + 8 <= R && gk0 + 8 <= K)
            {
                // Fast path: whole tile in bounds, explicit RVV (vl=8: f32m2 loads 8 -> vfncvt -> f16m1 stores 16B)
                const float* s0 = src + (size_t)gr0 * ld + gk0;
                for (int r = 0; r < 8; r++)
                {
                    const float* s = s0 + (size_t)r * ld;
                    ime2_fp16* d = t + r * 8;
                    vfloat32m2_t v = __riscv_vle32_v_f32m2(s, 8);
                    __riscv_vse16_v_f16m1(d, __riscv_vfncvt_f_f_w_f16m1(v, 8), 8);
                }
            }
            else
            {
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                    {
                        const int gr = gr0 + r, gk = gk0 + c;
                        float v = 0.f;
                        if (gr < R && gk < K)
                            v = src[(size_t)gr * ld + gk];
                        t[r * 8 + c] = (ime2_fp16)v;
                    }
            }
        }
    }
    return 0;
}

/* Transposed packing: src is [K, R] fp32 row-major (row stride ld); produces tiles of
 * B[R,K], i.e. tile element B[r][k] = src[k*ld + r]. Used for V^T in P·V. */
static int ime2_pack_fp32_T(const float* src, size_t ld, ncnn::Mat& T, int R, int K, ncnn::Allocator* alloc)
{
    const int rt = (R + 7) / 8, kt = (K + 7) / 8;
    /* IME2_TILE_SLACK fp16 of tail room: the K-direction pipeline in
     * ime2_tile_16x16p prefetches the next tile pair, and with an even kt the
     * final prefetch reads 128 bytes past the last tile row. The data is never
     * consumed, but the read has to stay inside the allocation. */
    T.create((size_t)rt * kt * 64 + IME2_TILE_SLACK, (size_t)2u, 1, alloc);
    if (T.empty())
        return -100;
    ime2_fp16* dst = (ime2_fp16*)T;

    for (int i = 0; i < rt; i++)
    {
        const int gr0 = i * 8;
        for (int j = 0; j < kt; j++)
        {
            ime2_fp16* t = dst + ((size_t)i * kt + j) * 64;
            const int gk0 = j * 8;
            if (gr0 + 8 <= R && gk0 + 8 <= K)
            {
                // Transposed fast path: source strides by column (stride ld); one vlse32 instruction fetches 8
                for (int r = 0; r < 8; r++)
                {
                    const float* s = src + (size_t)gk0 * ld + (gr0 + r);
                    ime2_fp16* d = t + r * 8;
                    vfloat32m2_t v = __riscv_vlse32_v_f32m2(s, (ptrdiff_t)ld * 4, 8);
                    __riscv_vse16_v_f16m1(d, __riscv_vfncvt_f_f_w_f16m1(v, 8), 8);
                }
            }
            else
            {
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                    {
                        const int gr = gr0 + r, gk = gk0 + c;
                        float v = 0.f;
                        if (gr < R && gk < K)
                            v = src[(size_t)gk * ld + gr];
                        t[r * 8 + c] = (ime2_fp16)v;
                    }
            }
        }
    }
    return 0;
}

/* ---------------- Inner kernels (same vfwmadot tile as the gemm version) ---------------- */

static void ime2_tile_8x8(const ime2_fp16* At, const ime2_fp16* Bt, float* Ctmp, int kt)
{
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vmv.v.i v16, 0\n\t"
        "1:\n\t"
        "vsetvli t0, zero, e16, m1, tu, mu\n\t"
        "vle16.v v2, (%[A])\n\t"
        "vle16.v v8, (%[B])\n\t"
        "smt.vfwmadot v16, v2, v8\n\t"
        "addi %[A], %[A], 128\n\t"
        "addi %[B], %[B], 128\n\t"
        "addi %[k], %[k], -1\n\t"
        "bnez %[k], 1b\n\t"
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vse32.v v16, (%[C])\n\t"
        : [A] "+r"(At), [B] "+r"(Bt), [k] "+r"(kt)
        : [C] "r"(Ctmp)
        : "t0", "v2", "v8", "v16", "v17", "memory", "cc");
}

static void ime2_tile_16x16(const ime2_fp16* At, const ime2_fp16* Bt,
                            float* C00, float* C10, float* C01, float* C11, int kt)
{
    const ime2_fp16* A1 = At + (size_t)kt * 64;
    const ime2_fp16* B1 = Bt + (size_t)kt * 64;
    int k = kt;
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vmv.v.i v16, 0\n\t"
        "vmv.v.i v18, 0\n\t"
        "vmv.v.i v20, 0\n\t"
        "vmv.v.i v22, 0\n\t"
        "1:\n\t"
        "vsetvli t0, zero, e16, m1, tu, mu\n\t"
        "vle16.v v2, (%[A0])\n\t"
        "vle16.v v8, (%[B0])\n\t"
        "smt.vfwmadot v16, v2, v8\n\t"
        "vle16.v v4, (%[A1])\n\t"
        "smt.vfwmadot v18, v4, v8\n\t"
        "vle16.v v10, (%[B1])\n\t"
        "smt.vfwmadot v20, v2, v10\n\t"
        "smt.vfwmadot v22, v4, v10\n\t"
        "addi %[A0], %[A0], 128\n\t"
        "addi %[A1], %[A1], 128\n\t"
        "addi %[B0], %[B0], 128\n\t"
        "addi %[B1], %[B1], 128\n\t"
        "addi %[k], %[k], -1\n\t"
        "bnez %[k], 1b\n\t"
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vse32.v v16, (%[C00])\n\t"
        "vse32.v v18, (%[C10])\n\t"
        "vse32.v v20, (%[C01])\n\t"
        "vse32.v v22, (%[C11])\n\t"
        : [A0] "+r"(At), [A1] "+r"(A1), [B0] "+r"(Bt), [B1] "+r"(B1), [k] "+r"(k)
        : [C00] "r"(C00), [C10] "r"(C10), [C01] "r"(C01), [C11] "r"(C11)
        : "t0", "v2", "v4", "v8", "v10",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "memory", "cc");
}

/* ---------------- fp32-output GEMM (single-threaded, for per-head calls) ---------------- */

/* C[M,N] fp32 (row stride ldc) = alpha * (A[M,K] · B[N,K]^T). AT/BT are packed tiles. */
static void ime2_gemm_fp32out(const ime2_fp16* at, const ime2_fp16* bt,
                              float* C, size_t ldc, int M, int N, int K, float alpha)
{
    const int mt = (M + 7) / 8, nt = (N + 7) / 8, kt = (K + 7) / 8;

    for (int i = 0; i < mt; i += 2)
    {
        for (int j = 0; j < nt; j += 2)
        {
            float t00[64], t10[64], t01[64], t11[64];
            const bool has_i1 = (i + 1 < mt);
            const bool has_j1 = (j + 1 < nt);
            bool valid[4] = {true, has_j1, has_i1, has_i1 && has_j1};

            if (has_i1 && has_j1)
            {
                ime2_tile_16x16(at + (size_t)i * kt * 64, bt + (size_t)j * kt * 64,
                                t00, t10, t01, t11, kt);
            }
            else
            {
                ime2_tile_8x8(at + (size_t)i * kt * 64, bt + (size_t)j * kt * 64, t00, kt);
                if (has_j1)
                    ime2_tile_8x8(at + (size_t)i * kt * 64, bt + (size_t)(j + 1) * kt * 64, t01, kt);
                if (has_i1)
                    ime2_tile_8x8(at + (size_t)(i + 1) * kt * 64, bt + (size_t)j * kt * 64, t10, kt);
                if (has_i1 && has_j1)
                    ime2_tile_8x8(at + (size_t)(i + 1) * kt * 64, bt + (size_t)(j + 1) * kt * 64, t11, kt);
            }

            const float* tiles[4] = {t00, t01, t10, t11};
            const int dm[4] = {0, 0, 8, 8};
            const int dn[4] = {0, 8, 0, 8};
            for (int q = 0; q < 4; q++)
            {
                if (!valid[q])
                    continue;
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                    {
                        const int gm = i * 8 + dm[q] + r;
                        const int gn = j * 8 + dn[q] + c;
                        if (gm >= M || gn >= N)
                            continue;
                        C[(size_t)gm * ldc + gn] = tiles[q][r * 8 + c] * alpha;
                    }
            }
        }
    }
}

/* ==================================================================
 * Flash-style SDPA (port of the official spine-triton ops/08 online-softmax recipe)
 * Streams K/V blocks: QK^T (IME2) → row max/exp/sum → rescale acc → P·V (IME2 accumulate)
 * All state stays f32 (m_i / l_i / acc); only P is converted to fp16 to feed the second GEMM.
 * ================================================================== */

/* ---- k+1 software-pipelined tile kernel (same as gemm_riscv_ime2.h, explicit pointers to allow striding) ---- */

static void ime2_tile_8x8p(const ime2_fp16* At, const ime2_fp16* Bt, float* Ctmp, int kt)
{
    int k = kt - 1;
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vmv.v.i v16, 0\n\t"
        "vsetvli t0, zero, e16, m1, tu, mu\n\t"
        "vle16.v v2, (%[A])\n\t"
        "vle16.v v8, (%[B])\n\t"
        "addi %[A], %[A], 128\n\t"
        "addi %[B], %[B], 128\n\t"
        "beqz %[k], 2f\n\t" /* zero iterations when kt==1, go straight to the epilogue (else do-while over-reads) */
        "1:\n\t"
        "vle16.v v12, (%[A])\n\t"
        "smt.vfwmadot v16, v2, v8\n\t"
        "vle16.v v13, (%[B])\n\t"
        "vmv.v.v v2, v12\n\t"
        "vmv.v.v v8, v13\n\t"
        "addi %[A], %[A], 128\n\t"
        "addi %[B], %[B], 128\n\t"
        "addi %[k], %[k], -1\n\t"
        "bnez %[k], 1b\n\t"
        "2:\n\t"
        "smt.vfwmadot v16, v2, v8\n\t"
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vse32.v v16, (%[C])\n\t"
        : [A] "+r"(At), [B] "+r"(Bt), [k] "+r"(k)
        : [C] "r"(Ctmp)
        : "t0", "v2", "v8", "v12", "v13", "v16", "v17", "memory", "cc");
}

/* 16x16 (2x2 of 8x8 tiles) pipelined version; A0/A1/B0/B1 passed explicitly (supports B striding over k blocks) */
static void ime2_tile_16x16p(const ime2_fp16* A0, const ime2_fp16* A1,
                             const ime2_fp16* B0, const ime2_fp16* B1,
                             float* C00, float* C10, float* C01, float* C11, int kt)
{
    /* Unroll 2 steps in the K direction + register rotation: removes the old version's
     * 4 vmv per step (they were on the MAC dependency critical path).
     * Microbenchmark streaming fetch: 210 -> 373 GFLOP/s/core (+77%, see REPORT 7.17/7.18). */
    int k = kt >> 1;
    const int odd = kt & 1;
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vmv.v.i v16, 0\n\t vmv.v.i v18, 0\n\t vmv.v.i v20, 0\n\t vmv.v.i v22, 0\n\t"
        "vsetvli t0, zero, e16, m1, tu, mu\n\t"
        "vle16.v v2, (%[A0])\n\t vle16.v v4, (%[A1])\n\t"
        "vle16.v v8, (%[B0])\n\t vle16.v v10, (%[B1])\n\t"
        "addi %[A0], %[A0], 128\n\t addi %[A1], %[A1], 128\n\t"
        "addi %[B0], %[B0], 128\n\t addi %[B1], %[B1], 128\n\t"
        "blez %[k], 2f\n\t"
        "1:\n\t"
        "vle16.v v12, (%[A0])\n\t"
        "smt.vfwmadot v16, v2, v8\n\t"
        "vle16.v v13, (%[B0])\n\t"
        "smt.vfwmadot v18, v4, v8\n\t"
        "vle16.v v14, (%[A1])\n\t"
        "smt.vfwmadot v20, v2, v10\n\t"
        "vle16.v v15, (%[B1])\n\t"
        "smt.vfwmadot v22, v4, v10\n\t"
        "addi %[A0], %[A0], 128\n\t addi %[A1], %[A1], 128\n\t"
        "addi %[B0], %[B0], 128\n\t addi %[B1], %[B1], 128\n\t"
        "vle16.v v2, (%[A0])\n\t"
        "smt.vfwmadot v16, v12, v13\n\t"
        "vle16.v v8, (%[B0])\n\t"
        "smt.vfwmadot v18, v14, v13\n\t"
        "vle16.v v4, (%[A1])\n\t"
        "smt.vfwmadot v20, v12, v15\n\t"
        "vle16.v v10, (%[B1])\n\t"
        "smt.vfwmadot v22, v14, v15\n\t"
        "addi %[A0], %[A0], 128\n\t addi %[A1], %[A1], 128\n\t"
        "addi %[B0], %[B0], 128\n\t addi %[B1], %[B1], 128\n\t"
        "addi %[k], %[k], -1\n\t bnez %[k], 1b\n\t"
        "2:\n\t"
        "beqz %[odd], 3f\n\t"
        "smt.vfwmadot v16, v2, v8\n\t"
        "smt.vfwmadot v18, v4, v8\n\t"
        "smt.vfwmadot v20, v2, v10\n\t"
        "smt.vfwmadot v22, v4, v10\n\t"
        "3:\n\t"
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vse32.v v16, (%[C00])\n\t vse32.v v18, (%[C10])\n\t"
        "vse32.v v20, (%[C01])\n\t vse32.v v22, (%[C11])\n\t"
        : [A0] "+r"(A0), [A1] "+r"(A1), [B0] "+r"(B0), [B1] "+r"(B1), [k] "+r"(k)
        : [C00] "r"(C00), [C10] "r"(C10), [C01] "r"(C01), [C11] "r"(C11), [odd] "r"(odd)
        : "t0", "v2", "v4", "v8", "v10", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "memory", "cc");
}

/* Pipelined GEMM driver: the kt dimension can stride (at_stride/bt_stride = the
 * respective tile row strides, in elements).
 * beta==0: C = alpha·A·B^T; beta==1: C += A·B^T (alpha treated as 1). */
static void ime2_gemm_fp32out_p(const ime2_fp16* at, const ime2_fp16* bt,
                                float* C, size_t ldc, int M, int N, int K,
                                float alpha, int beta,
                                size_t at_stride, size_t bt_stride)
{
    const int mt = (M + 7) / 8, nt = (N + 7) / 8, kt = (K + 7) / 8;

    for (int i = 0; i < mt; i += 2)
    {
        for (int j = 0; j < nt; j += 2)
        {
            float t00[64], t10[64], t01[64], t11[64];
            const bool has_i1 = (i + 1 < mt);
            const bool has_j1 = (j + 1 < nt);
            bool valid[4] = {true, has_j1, has_i1, has_i1 && has_j1};

            const ime2_fp16* A0 = at + (size_t)i * at_stride;
            const ime2_fp16* B0 = bt + (size_t)j * bt_stride;
            const ime2_fp16* A1 = A0 + at_stride;
            const ime2_fp16* B1 = B0 + bt_stride;

            if (has_i1 && has_j1)
            {
                ime2_tile_16x16p(A0, A1, B0, B1, t00, t10, t01, t11, kt);
            }
            else
            {
                ime2_tile_8x8p(A0, B0, t00, kt);
                if (has_j1)
                    ime2_tile_8x8p(A0, B1, t01, kt);
                if (has_i1)
                    ime2_tile_8x8p(A1, B0, t10, kt);
                if (has_i1 && has_j1)
                    ime2_tile_8x8p(A1, B1, t11, kt);
            }

            const float* tiles[4] = {t00, t01, t10, t11};
            const int dm[4] = {0, 0, 8, 8};
            const int dn[4] = {0, 8, 0, 8};
            for (int q = 0; q < 4; q++)
            {
                if (!valid[q])
                    continue;
                for (int r = 0; r < 8; r++)
                {
                    const int gm = i * 8 + dm[q] + r;
                    if (gm >= M)
                        continue;
                    float* crow = C + (size_t)gm * ldc;
                    const float* trow = tiles[q] + r * 8;
                    const int gn0 = j * 8 + dn[q];
                    if (gn0 + 8 <= N)
                    {
                        /* Fast path: all 8 columns of the row in bounds — explicit RVV (vl=8, e32m1)
                         * With flash tiling the GEMM is small and write-back share is high; must use fixed-length vector load/store. */
                        if (__riscv_vsetvlmax_e32m1() >= 8)
                        {
                            if (beta)
                            {
                                vfloat32m1_t v = __riscv_vle32_v_f32m1(crow + gn0, 8);
                                v = __riscv_vfadd_vv_f32m1(v, __riscv_vle32_v_f32m1(trow, 8), 8);
                                __riscv_vse32_v_f32m1(crow + gn0, v, 8);
                            }
                            else
                            {
                                vfloat32m1_t v = __riscv_vle32_v_f32m1(trow, 8);
                                if (alpha != 1.f)
                                    v = __riscv_vfmul_vf_f32m1(v, alpha, 8);
                                __riscv_vse32_v_f32m1(crow + gn0, v, 8);
                            }
                        }
                        else
                        {
                            for (int c = 0; c < 8; c++)
                            {
                                if (beta)
                                    crow[gn0 + c] += trow[c];
                                else
                                    crow[gn0 + c] = trow[c] * alpha;
                            }
                        }
                    }
                    else
                    {
                        for (int c = 0; c < 8; c++)
                        {
                            const int gn = gn0 + c;
                            if (gn >= N)
                                continue;
                            if (beta)
                                crow[gn] += trow[c];
                            else
                                crow[gn] = trow[c] * alpha;
                        }
                    }
                }
            }
        }
    }
}

/* flash tiling sizes: BM=64 rows of Q / BN=128 columns of KV (official recommendation
 * tile ≤ 8192 elements; S=64×128×4B=32KB, acc=64×256×4B=64KB resident on stack,
 * L1/L2 friendly) */
#define IME2_FLASH_BM    32
#define IME2_FLASH_BN    256 /* scan-measured optimum (see below): 1024/2048/4096 are
                              * 7.2%/3.9%/3.0% faster than 512 respectively (K3 A100,
                              * E=64/H=16/G=8, causal).
                              * Methodology references spine-triton's FlashAttention tiling scan. */
#define IME2_FLASH_MAXEV 256

/* ---------------- RVV vector helpers (VLEN=1024; explicit vsetvl, no VL overrun) ---------------- */

#include <riscv_vector.h>

/* s[j] = exp(s[j] - mx), j < n. x≤0 always holds (mx is the row max).
 * 2^k integer-bit construction + 5th-order Taylor on [-ln2/2, ln2/2] (max relative
 * error ~2e-6, far below the fp16 packing rounding of 5e-4); clamped at x < -87 (exp
 * has already underflowed to 0 for fp32). */
static void ime2_row_exp(float* s, int n, float mx)
{
    const float log2e = 1.4426950408889634f;
    const float ln2_hi = 0.693359375f;
    const float ln2_lo = -2.12194440e-4f; /* ln2 - ln2_hi */
    int j = 0;
    while (j < n)
    {
        size_t vl = __riscv_vsetvl_e32m8(n - j);
        vfloat32m8_t x = __riscv_vle32_v_f32m8(s + j, vl);
        x = __riscv_vfsub_vf_f32m8(x, mx, vl);
        /* Zero on underflow: when x-mx < -87.34 the true exp < 2^-126 (construction
         * lower bound); clamping would lift it to nonzero — a hard mask / deeply
         * negative logit multiplied by a large V would leak. Record these lanes and
         * zero them at the end. */
        vbool4_t vund = __riscv_vmflt_vf_f32m8_b4(x, -87.34f, vl);
        /* Clamp upper first, matching the fallback exp_ps (rvv_mathfun.h): RVV
         * vfmin/vfmax return the numeric operand when one is NaN, so a NaN score
         * maps to the upper endpoint (+huge exponential) exactly as upstream —
         * without this, vfmax(NaN, -87) mapped it to the opposite endpoint and
         * the lane packed to fp16 zero. */
        x = __riscv_vfmin_vf_f32m8(x, 88.3762626647949f, vl);
        x = __riscv_vfmax_vf_f32m8(x, -87.0f, vl);
        vint32m8_t ni = __riscv_vfcvt_x_f_v_i32m8(__riscv_vfmul_vf_f32m8(x, log2e, vl), vl); /* default RNE */
        vfloat32m8_t nf = __riscv_vfcvt_f_x_v_f32m8(ni, vl);
        vfloat32m8_t r = __riscv_vfsub_vv_f32m8(x, __riscv_vfmul_vf_f32m8(nf, ln2_hi, vl), vl);
        r = __riscv_vfsub_vv_f32m8(r, __riscv_vfmul_vf_f32m8(nf, ln2_lo, vl), vl);
        /* Horner: (((((1/120)r + 1/24)r + 1/6)r + 1/2)r + 1)r + 1 */
        vfloat32m8_t p = __riscv_vfmv_v_f_f32m8(1.f / 120.f, vl);
        p = __riscv_vfmacc_vv_f32m8(__riscv_vfmv_v_f_f32m8(1.f / 24.f, vl), p, r, vl);
        p = __riscv_vfmacc_vv_f32m8(__riscv_vfmv_v_f_f32m8(1.f / 6.f, vl), p, r, vl);
        p = __riscv_vfmacc_vv_f32m8(__riscv_vfmv_v_f_f32m8(0.5f, vl), p, r, vl);
        p = __riscv_vfmacc_vv_f32m8(__riscv_vfmv_v_f_f32m8(1.f, vl), p, r, vl);
        p = __riscv_vfmacc_vv_f32m8(__riscv_vfmv_v_f_f32m8(1.f, vl), p, r, vl);
        vint32m8_t ei = __riscv_vsll_vx_i32m8(__riscv_vadd_vx_i32m8(ni, 127, vl), 23, vl);
        vfloat32m8_t y = __riscv_vfmul_vv_f32m8(p, __riscv_vreinterpret_v_i32m8_f32m8(ei), vl);
        y = __riscv_vfmerge_vfm_f32m8(y, 0.f, vund, vl);
        __riscv_vse32_v_f32m8(s + j, y, vl);
        j += vl;
    }
}

/* Row max (over scores with mask already added) */
static float ime2_row_max(const float* s, int n)
{
    vfloat32m1_t vm = __riscv_vfmv_v_f_f32m1(-FLT_MAX, 1);
    int j = 0;
    while (j < n)
    {
        size_t vl = __riscv_vsetvl_e32m8(n - j);
        vm = __riscv_vfredmax_vs_f32m8_f32m1(__riscv_vle32_v_f32m8(s + j, vl), vm, vl);
        j += vl;
    }
    return __riscv_vfmv_f_s_f32m1_f32(vm);
}

/* Row sum (unordered reduction; only feeds l_i normalization, not per-element results) */
static float ime2_row_sum(const float* s, int n)
{
    vfloat32m1_t vs = __riscv_vfmv_v_f_f32m1(0.f, 1);
    int j = 0;
    while (j < n)
    {
        size_t vl = __riscv_vsetvl_e32m8(n - j);
        vs = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vle32_v_f32m8(s + j, vl), vs, vl);
        j += vl;
    }
    return __riscv_vfmv_f_s_f32m1_f32(vs);
}

/* acc row *= alpha */
static void ime2_row_scale(float* a, int n, float alpha)
{
    int j = 0;
    while (j < n)
    {
        size_t vl = __riscv_vsetvl_e32m8(n - j);
        __riscv_vse32_v_f32m8(a + j, __riscv_vfmul_vf_f32m8(__riscv_vle32_v_f32m8(a + j, vl), alpha, vl), vl);
        j += vl;
    }
}

/* Row add (mask): a[j] += b[j] */
static void ime2_row_add(float* a, const float* b, int n)
{
    int j = 0;
    while (j < n)
    {
        size_t vl = __riscv_vsetvl_e32m8(n - j);
        __riscv_vse32_v_f32m8(a + j, __riscv_vfadd_vv_f32m8(__riscv_vle32_v_f32m8(a + j, vl), __riscv_vle32_v_f32m8(b + j, vl), vl), vl);
        j += vl;
    }
}

/* Single-head flash forward: QT/KT/VT are packed tiles (Q already pre-scaled); mask
 * may be 0 (must be passed when non-causal).
 * Out: [L, Ev] fp32 with row stride ldo. Returns 0 on success; -1 means Ev exceeds the
 * on-stack budget (caller takes the old path). */
static int ime2_flash_head_tiles(const ime2_fp16* QT, const ime2_fp16* KT, const ime2_fp16* VT,
                                 const float* mask, size_t ldm, float* Out, size_t ldo,
                                 int L, int dst, int E, int Ev, int past, int causal, float scale,
                                 ncnn::Mat& PT, ncnn::Allocator* alloc)
{
    if (Ev > IME2_FLASH_MAXEV)
        return -1;

    const int ktE = (E + 7) / 8;
    const int ktD = (dst + 7) / 8;
    /* Shape-aware tiling: BN=512 is chosen for long sequences (≥512); for very short
     * sequences, softmax/P packing/write-back would do plenty of wasted work at 512
     * columns (at 115 tokens QK is only 13.5 MFLOP, yet still processed as 512
     * columns). Choose the block size by actual length, keeping it contiguous (for
     * PT's tile layout). */
    int BN = IME2_FLASH_BN;
    while (BN > 64 && BN / 2 >= dst) BN /= 2;

    float S[IME2_FLASH_BM * IME2_FLASH_BN];
    float acc[IME2_FLASH_BM * IME2_FLASH_MAXEV];
    float mi[IME2_FLASH_BM], li[IME2_FLASH_BM];

    for (int m0 = 0; m0 < L; m0 += IME2_FLASH_BM)
    {
        const int rows = IME2_FLASH_BM < L - m0 ? IME2_FLASH_BM : L - m0;
        memset(acc, 0, (size_t)rows * Ev * sizeof(float));
        for (int r = 0; r < rows; r++)
        {
            mi[r] = -FLT_MAX;
            li[r] = 0.f;
        }

        /* causal: the visible right end of rows m0..m0+rows-1 = past+m0+rows-1 (inclusive); blocks only need to scan up to here */
        const int nend = causal ? (past + m0 + rows < dst ? past + m0 + rows : dst) : dst;

        for (int n0 = 0; n0 < nend; n0 += BN)
        {
            const int cols = BN < nend - n0 ? BN : nend - n0;

            /* S = scale · Q_blk · K_blk^T: QT is **not** pre-multiplied by scale
             * (multiplying by scale before fp16 packing would overflow otherwise
             * representable Q values into inf); instead multiply alpha=scale onto the
             * write-back of the fp32 accumulated result — consistent with the
             * non-flash IME2 path and the upstream fallback. */
            ime2_gemm_fp32out_p(QT + (size_t)(m0 / 8) * ktE * 64,
                                KT + (size_t)(n0 / 8) * ktE * 64,
                                S, BN, rows, cols, E, scale, 0,
                                (size_t)ktE * 64, (size_t)ktE * 64);

            /* Only the diagonal region (or the catch-all for arbitrary masks) needs the mask added */
            if (!causal || n0 + cols > past + m0)
            {
                for (int r = 0; r < rows; r++)
                    ime2_row_add(S + r * BN, mask + (size_t)(m0 + r) * ldm + n0, cols);
            }

            /* online softmax: row max → exp → row sum → rescale factor (RVV vectorized) */
            for (int r = 0; r < rows; r++)
            {
                float* s = S + r * BN;
                const float bmax = ime2_row_max(s, cols);
                const float mx = mi[r] > bmax ? mi[r] : bmax;
                const float alpha = expf(mi[r] - mx); /* first block mi=-FLT_MAX → alpha=0 */
                ime2_row_scale(acc + r * Ev, Ev, alpha);
                ime2_row_exp(s, cols, mx);
                li[r] = li[r] * alpha + ime2_row_sum(s, cols);
                mi[r] = mx;
            }

            /* Convert the P block to fp16 tiles (zero-padded to a multiple of 8; the
             * extra padded K columns correspond to P=0 and contribute nothing to PV).
             * Scale up by 2^15 before packing: tiny probabilities (e.g. 2.5e-8) packed
             * directly would underflow to 0 and the aggregate mass (li sums in fp32)
             * would be lost; after scaling, normal fp16 precision preserves them.
             * Compensate with li·2^15 at normalization. Upper bound p≤1 → ×32768 ≤ 32768 < 65504, no overflow. */
            /* A clamped NaN score maps to ~exp(88.376) ~ 2.4e38 (the upper endpoint,
             * matching the fallback exp_ps): multiplying it by 32768 for the fp16
             * pack overflows fp32 to +inf (and li*32768 at normalization
             * likewise), producing NaN where the fallback gives a finite output.
             * Bail this head to the old path instead (the caller recomputes the
             * layer). li < FLT_MAX/32768 is required for the final scaling to
             * stay finite; normal rows (P <= 1, cols small) never approach it. */
            for (int r = 0; r < rows; r++)
                if (!(li[r] * 32768.f < 3.4028234663852886e+38f))
                    return -1;
            for (int r = 0; r < rows; r++)
                ime2_row_scale(S + r * BN, cols, 32768.f);
            if (ime2_pack_fp32(S, BN, PT, rows, cols, alloc) != 0)
                return -100;

            /* acc += P · V_blk (k-block offset n0/8 into the V^T tiles, row stride ktD) */
            const int ktP = (cols + 7) / 8;
            ime2_gemm_fp32out_p((const ime2_fp16*)PT, VT + (size_t)(n0 / 8) * 64,
                                acc, Ev, rows, Ev, ktP * 8, 1.f, 1,
                                (size_t)ktP * 64, (size_t)ktD * 64);
        }

        /* Normalize only after the scan completes (one multiply); li·32768 compensates the pre-packing scale-up of P */
        for (int r = 0; r < rows; r++)
        {
            const float inv = 1.f / (li[r] * 32768.f);
            ime2_row_scale(acc + r * Ev, Ev, inv);
            const float* arow = acc + r * Ev;
            float* o = Out + (size_t)(m0 + r) * ldo;
            for (int e = 0; e < Ev; e++)
                o[e] = arow[e];
        }
    }
    return 0;
}

} // namespace ncnn_ime2_sdpa

#endif // NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2 && __riscv_v
#endif // SDPA_RISCV_IME2_H
