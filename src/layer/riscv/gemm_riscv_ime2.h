// SpacemiT K3 A100 IME2 (smt.vfwmadot) fp16 GEMM path for ncnn Gemm layer.
//
// Background:
//   - The K3 A100 cluster (cpu8-15, VLEN=1024) supports the IME2 instruction smt.vfwmadot:
//       C[8x8]fp32 += A[8x8]fp16 x B^T[8x8]fp16   (i.e. C[i][j] += sum_k A[i][k]*B[j][k])
//   - This is exactly the semantics of ncnn Gemm with transB=1.
//   - A single A100 core measures ~77 GFLOP/s fp16 (see k3-ncnn/ime/ime2_gemm_fp16_2x2.c).
//
// Constraints (must be obeyed):
//   - Can only run on the A100 cluster (vfwmadot is SIGILL on X100).
//     The whole process must be pinned to A100 (e.g. ai-run), because ncnn packs weights
//     at load time according to vlenb; running across clusters silently computes wrong results.
//   - Only supported: transA=0, transB=1, no C term, alpha=1, no output_transpose/N1M.
//     Otherwise fall back to the original ncnn path.
//
// This file is #included at the end of gemm_riscv.cpp (following ncnn's gemm_fp16s.h style).

#ifndef GEMM_RISCV_IME2_H
#define GEMM_RISCV_IME2_H

#include "platform.h"
#include <math.h>
#include <signal.h>
#include <string.h>
#if defined __linux__
#include <sched.h>
#endif
#include <stdio.h>
#if defined __linux__
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <poll.h>
#include <errno.h>
#endif

#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2 && __riscv_v

namespace ncnn_ime2 {

typedef __fp16 ime2_fp16;

/* ---------------- Runtime probe ---------------- */

__attribute__((noinline)) static void ime2_probe_instruction()
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
/* On the heterogeneous dual-cluster (K3: A100 has VLEN=1024, X100 has VLEN=512),
 * "the core pair seen at probe time" and "the core pair the OpenMP workers actually
 * land on at forward time" may differ: if the former happens to be A100, true gets
 * cached, and the latter landing on X100 eats a SIGILL. So the probe must cover
 * **every** core in the affinity mask: move the calling thread to each in turn and
 * read vlenb; only 128B (1024bit) counts. */
static int ime2_probe_affinity(int nthreads)
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
     * GOMP_CPU_AFFINITY / the host runtime); the mask walk above only looks at the
     * calling thread and cannot see them. And forward's parallel regions all carry
     * num_threads(opt.num_threads): if it exceeds the default team, the runtime
     * assigns forward **never-probed** extra workers. So open the region here with
     * exactly the same team size as forward.
     * Reading vlenb once is not enough either: if a worker's mask covers both
     * clusters, it may happen to land on A100 at probe time and pass, while the
     * scheduler may move it to X100 at forward time. So each worker must walk every
     * core in **its own** affinity mask, confirming vlenb==128 per core (restoring
     * the worker's original binding after probing). If any worker's mask mixes in
     * X100 cores, conservatively disable IME2. */
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
static int ime2_probe_affinity(int /*nthreads*/)
{
    return 0;
}
#endif // __linux__

/* Hardware support probe, by construction free of every signal-state race:
 * run the instruction in a FORKED CHILD. If the CPU lacks xsmtvdotii, the
 * child alone dies of SIGILL and the parent merely observes a non-zero exit.
 * There is no process-wide handler, no saved disposition, no forwarding, no
 * mask manipulation, and nothing to restore -- a custom handler, a one-shot
 * action, sigaltstack, or a blocked SIGILL in the parent are all simply
 * irrelevant, because the child resets its own disposition to SIG_DFL and
 * unblocks SIGILL before executing the instruction.
 *
 * The child pins itself to an A100 CPU (vlenb==128) found by the same walk as
 * the affinity check, so the answer is a hardware property of the machine and
 * can be cached process-wide. fork happens once per process (under the probe
 * mutex), not once per layer. In the child only async-signal-safe calls are
 * used (sigaction/sigprocmask/sched_setaffinity and the instruction itself),
 * then _exit. */
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

    /* The child reports success over a pipe: with SIGCHLD ignored, SA_NOCLDWAIT,
     * or a handler that reaps children, waitpid can fail with ECHILD and a
     * zero-initialized status would read as "supported" -- exactly backwards on
     * hardware that lacks the instruction. The pipe byte is the only accepted
     * positive result; a dead child writes nothing and the read hits EOF. */
    /* O_CLOEXEC on both ends: another thread's fork+exec'd subprocess must not
     * inherit fds[1] -- if it did and the probe child died without writing, this
     * read would wait for EOF until that unrelated process exits (pipeline
     * creation could hang indefinitely). Our own probe child does not exec, so
     * the flag does not affect its write end. */
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
        /* The intentional SIGILL must not emit a core dump: the child inherits
         * the inference process's address space, and a model-sized dump would
         * stall pipeline creation or fill crash storage. */
        struct rlimit rl;
        rl.rlim_cur = 0;
        rl.rlim_max = 0;
        setrlimit(RLIMIT_CORE, &rl);
        struct sigaction sa;
        memset(&sa, 0, sizeof sa);
        sa.sa_handler = SIG_DFL;
        if (sigaction(SIGILL, &sa, NULL) != 0)
        {
            /* Could not reset (seccomp): the inherited custom handler could trap
             * this child forever (the parent would block on the pipe) or emulate
             * the instruction (false success). Exit without writing -- the parent
             * judges unsupported and no one hangs. */
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
        ime2_probe_instruction();
        (void)!write(fds[1], "1", 1);
        _exit(0);
    }
    close(fds[1]);
    /* Reap FIRST: once waitpid returns (or fails with ECHILD on a SIGCHLD-
     * ignoring host, meaning the child was auto-reaped = already exited),
     * anything the child wrote is already in the pipe buffer. Then read with
     * a BOUNDED poll instead of blocking on EOF: O_CLOEXEC only closes on a
     * successful exec, so an unrelated thread's fork WITHOUT exec during the
     * window still inherits fds[1] and a blocking read could wait for that
     * process's exit (pipeline creation would hang). */
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
            /* EINTR (handled signal without SA_RESTART) must not read as failure */
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

static int ime2_probe_locked(int nthreads)
{
    /* (a) affinity: the vlenb of every core this process is allowed on + every worker of the forward team must be 1024bit */
    if (!ime2_probe_affinity(nthreads))
        return 0;

    return ime2_hw_supported();
}

/* Returns 1 = every allowed core in the current affinity context can execute
 * smt.vfwmadot; 0 = cannot. The instruction itself is probed in a forked child
 * (ime2_hw_supported), so no process-wide signal state is touched at all; the
 * mutex only serializes the affinity walk and the cached result.
 * Note: no process-wide caching: affinity is per-thread, and the probe result is
 * valid only for the affinity context of this call. If a process first binds a
 * thread to the A100 cluster to create a pipeline, then binds to the X100 cluster
 * to create a second pipeline, a cache would wrongly reuse the answer valid only
 * for the former on the latter (forward on X100 means SIGILL).
 * nthreads must match forward's num_threads: the probed team is the team that runs
 * the kernel. The probe is called only once at pipeline creation; re-probing cost
 * is negligible. */
static int ime2_probe(int nthreads)
{
    static ncnn::Mutex lock;
    ncnn::MutexLockGuard g(lock);
    return ime2_probe_locked(nthreads);
}

/* One-time determination of which cores the team may land on are A100
 * (vlenb==128). A hardware property, cached process-wide. The determination range
 * must be the **union of all team worker masks**, not just the calling thread:
 * under OMP_PROC_BIND / GOMP_CPU_AFFINITY the initial thread's mask often contains
 * only its own place; looking at it alone would misjudge other A100 workers as
 * unusable, effectively silently disabling IME2 (a performance regression). If the
 * team does not cover all A100 cores at determination time, the resulting set is a
 * subset of the true set — the bias direction is conservative (may wrongly
 * disable, never wrongly enable). */
#if defined __linux__
/* force_collect=1: forced re-collection when team_ok finds the cache incomplete (bypasses the caller-mask fast path) */
static int ime2_a100_set_impl(int nthreads, int force_collect, cpu_set_t* out)
{
    static ncnn::Mutex lock;
    ncnn::MutexLockGuard g(lock);
    static int probed = 0;
    static cpu_set_t cached;     /* set of cores confirmed vlenb==128 (grows only) */
    static cpu_set_t seen_union; /* union of team masks seen so far */

    /* Fast path: the cache covers the caller's mask and no forced re-collection is
     * requested — use the cache directly.
     * (Each worker mask ⊆ the process mask at its start; when the caller is covered
     * the team cannot see wider cores — unless there is per-worker independent
     * binding, in which case team_ok fails first and then force_collect takes the
     * slow path to extend.) */
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

    /* 1) Collect the union of the current team (inside the region only
     *    sched_getaffinity runs, no privileged instructions, can run anywhere). The
     *    cache is valid only for "cores already seen": if a small team was probed
     *    first and a larger team is used later, the latter's legitimate A100 cores
     *    are not in the cache and team_ok would misjudge — so the union must
     *    accumulate. */
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

    /* 2) Classify only cores not seen before (vlenb walk); seen cores use the cache directly */
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

static int ime2_a100_set(int nthreads, cpu_set_t* out)
{
    return ime2_a100_set_impl(nthreads, 0, out);
}
#endif

/* Lightweight check at forward entry: this forward's team **cannot** land on any
 * non-A100 core. The pipeline-creation probe (ime2_probe) only covers the context
 * at that moment: the caller may later change bindings with
 * set_cpu_thread_affinity(), or call forward with a larger num_threads (the
 * runtime assigns un-probed extra workers). And looking only at a worker's
 * "current" vlenb is not enough either: a worker whose mask spans clusters may be
 * sampled on A100 and migrated to X100 while the kernel runs. So here we check
 * per worker whether its affinity mask is a subset of the A100 set (if a subset,
 * the scheduler has nowhere to migrate it). One sched_getaffinity + bit test per
 * worker, µs-level, negligible relative to each GEMM layer.
 * Returning 0 makes the caller fall back to the non-IME2 path; result semantics
 * unchanged. */
/* Per-worker self-check: its own affinity mask ⊆ the given set. */
#if defined __linux__
static int ime2_worker_in_set(const cpu_set_t* a100)
{
    cpu_set_t wcs;
    if (sched_getaffinity(0, sizeof(wcs), &wcs) != 0)
        return 0;
    for (int c = 0; c < CPU_SETSIZE; c++)
        if (CPU_ISSET(c, &wcs) && !CPU_ISSET(c, a100))
            return 0;
    return 1;
}
#endif

static int ime2_team_ok(int nthreads)
{
#if defined __linux__
    cpu_set_t a100;
    if (!ime2_a100_set(nthreads, &a100))
        return 0;
    int ok = 1;
    #pragma omp parallel num_threads(nthreads)
    {
        if (!ime2_worker_in_set(&a100))
        {
            #pragma omp atomic write
            ok = 0;
        }
    }
    if (ok)
        return 1;
    /* Failure: possibly the cache lacks cores newly seen by the team (per-worker
     * binding extended a mask). Force one re-collection and re-verify; only failure
     * now counts as real failure. */
    if (!ime2_a100_set_impl(nthreads, 1, &a100))
        return 0;
    ok = 1;
    #pragma omp parallel num_threads(nthreads)
    {
        if (!ime2_worker_in_set(&a100))
        {
            #pragma omp atomic write
            ok = 0;
        }
    }
    return ok;
#else
    (void)nthreads;
    return 0;
#endif
}

/* Weight range check: once a |w| > 65504 packs to ±inf, the IME2 path (including
 * the VLEN-independent fallback — it also reads weights from tiles) is corrupted
 * thereafter. Such a layer does not arm IME2 (falls back to the upstream
 * pipeline; the semantics of out-of-range weights in upstream fp16s storage is
 * upstream's own concern). Both fp16/fp32 sources must be scanned. */
static int ime2_weights_fit_fp16(const ncnn::Mat& B)
{
    if (B.elembits() == 16)
    {
        /* Finite values of an fp16 source are naturally in range (at most 65504); ±inf
         * stays inf after packing, consistent with upstream fp16s semantics — not treated
         * as out of range. */
        return 1;
    }
    /* Scan only logical elements: Mat::total() includes tail alignment padding (with
     * dims<=2 there may also be one extra element), whose bytes are uninitialized —
     * reading them into the scan would non-deterministically judge a legitimate layer
     * as out of range. Scan only w*h per channel (skip cstep padding for 3D). */
    const int nch = B.dims == 3 ? B.c : 1;
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(0.f, vlmax);
    for (int q = 0; q < nch; q++)
    {
        const float* p = B.dims == 3 ? (const float*)B.channel(q).data : (const float*)B.data;
        const size_t n = (size_t)B.w * B.h;
        size_t i = 0;
        while (i < n)
        {
            const size_t vl = __riscv_vsetvl_e32m1(n - i);
            vmax = __riscv_vfmax_vv_f32m1_tu(vmax, vmax, __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(p + i, vl), vl), vl);
            i += vl;
        }
    }
    const float mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(vmax, __riscv_vfmv_v_f_f32m1(0.f, 1), vlmax));
    return mx <= 65504.f ? 1 : 0;
}

/* ---------------- Packing ---------------- */

/* B: [N,K] (fp16 or fp32, elempack=1) -> 8x8 tile contiguous layout [nt][kt][64] fp16 */
/* Tile stride padding: tiles were originally kt*64 fp16 apart (exactly 16KB at
 * K=1024, a power of two), so adjacent tile rows map to the same L1 set, causing
 * conflict misses. Microbenchmark: stride +256B lifts the kernel from 200.9 to
 * 252.9 GFLOP/s/core (+26%); +16/32/64B has no effect or is worse (see REPORT
 * 7.16). */
#define IME2_TILE_PAD    128 /* unit: fp16 elements (= 256B) */
#define IME2_TILE_LD(kt) ((size_t)(kt)*64 + IME2_TILE_PAD)

static int ime2_pack_B(const ncnn::Mat& B, ncnn::Mat& BT, int N, int K)
{
    const int nt = (N + 7) / 8, kt = (K + 7) / 8;
    const size_t ldt = IME2_TILE_LD(kt);
    BT.create((size_t)nt * ldt, (size_t)2u, 1, (ncnn::Allocator*)0);
    if (BT.empty())
        return -100;
    ime2_fp16* dst = (ime2_fp16*)BT;

    const int is_fp16 = B.elembits() == 16;

    for (int i = 0; i < nt; i++)
        for (int j = 0; j < kt; j++)
        {
            ime2_fp16* t = dst + (size_t)i * ldt + (size_t)j * 64;
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                {
                    const int gn = i * 8 + r, gk = j * 8 + c;
                    float v = 0.f;
                    if (gn < N && gk < K)
                    {
                        if (is_fp16)
                            v = ((const ime2_fp16*)B)[(size_t)gn * K + gk];
                        else
                            v = ((const float*)B)[(size_t)gn * K + gk];
                    }
                    t[r * 8 + c] = (ime2_fp16)v;
                }
        }
    return 0;
}

/* A: [M,K] fp16 (any elempack) -> 8x8 tile contiguous layout [mt][kt][64] fp16
 * ncnn packed layout: element (m,k) is at ((m/p)*K + k)*p + m%p */
static int ime2_pack_A(const ncnn::Mat& A, ncnn::Mat& AT, int M, int K, int nT = 1, ncnn::Allocator* alloc = 0)
{
    const int mt = (M + 7) / 8, kt = (K + 7) / 8;
    const size_t ldat = IME2_TILE_LD(kt);
    AT.create((size_t)mt * ldat, (size_t)2u, 1, alloc);
    if (AT.empty())
        return -100;
    ime2_fp16* d = (ime2_fp16*)AT;

    const ime2_fp16* a = (const ime2_fp16*)A;
    const int p = A.elempack;
    /* For 3-D A, the inter-channel stride (in elements) is cstep*p: Mat pads each
     * plane to 16B and cstep records the number of pack-groups. E.g. with elempack==1
     * and K==9, cstep=16; advancing by K*p would read into padding from the second
     * row on. Same convention as the upstream RVV packer. */
    const size_t as = A.dims == 3 ? (size_t)A.cstep * p : (size_t)K * p;

    // Parallelize by tile row: under prefill, pack_A is pure data movement (~1MB per
    // call); single-threaded it takes about 6% of GEMM wall clock (196 calls, ~78ms total).
    /* With M=1 (decode), mt=1: opening an 8-thread omp region for one row of data is
     * pure overhead. Measured (NCNN_IME2_I8_PROF): driver prefix 105µs/call × 196
     * calls = 20.6ms/token (24% of decode), mostly from here. ⇒ Parallelize only
     * when there are enough tile rows.
     * Note: the same guard was added earlier, but a polluted measurement (a
     * concurrent background process) misjudged it as no-gain and it was reverted. */
    /* Decide parallelism by **work amount** (tile rows × k blocks), not by mt alone:
     * with M=1 (decode), 1×kt ≈ 128 ⇒ serial; with M=27 (short prefill), 4×128 =
     * 512 ⇒ should be parallel. (An mt>=8 test used to also serialize the packing of
     * 27-token prefills — measured 14% loss.) */
    const bool pack_par = (nT > 1 && (long long)mt * kt >= 512);
    const int pack_nT = pack_par ? nT : 1;
    #pragma omp parallel for num_threads(pack_nT)
    for (int i = 0; i < mt; i++)
    {
        const int gm0 = i * 8;
        for (int j = 0; j < kt; j++)
        {
            ime2_fp16* t = d + (size_t)i * ldat + (size_t)j * 64;
            const int gk0 = j * 8;
            // Fast path: the whole block is in bounds.
            // With p == 1, the 8 k values of a row are contiguous; copy 16 bytes directly.
            // With p > 1, ncnn's layout is [m/p][k][p]: adjacent k values of the same m row
            // are p fp16 apart, and the 8 immediately adjacent fp16 are neighboring pack
            // lanes of the same k column, i.e. **other m rows**; so gather one by one with
            // stride p — no whole-block memcpy.
            if (gm0 + 8 <= M && gk0 + 8 <= K)
            {
                if (p == 1)
                {
                    for (int r = 0; r < 8; r++)
                        memcpy(t + r * 8, a + (size_t)(gm0 + r) * as + gk0, 16);
                }
                else
                {
                    for (int r = 0; r < 8; r++)
                    {
                        const int gm = gm0 + r;
                        const ime2_fp16* src = a + (size_t)(gm / p) * as + gk0 * p + gm % p;
                        for (int c = 0; c < 8; c++)
                            t[r * 8 + c] = src[(size_t)c * p];
                    }
                }
            }
            else
            {
                for (int r = 0; r < 8; r++)
                    for (int c = 0; c < 8; c++)
                    {
                        const int gm = gm0 + r, gk = gk0 + c;
                        ime2_fp16 v = (ime2_fp16)0.f;
                        if (gm < M && gk < K)
                        {
                            if (p == 1)
                                v = a[(size_t)gm * as + gk];
                            else
                                v = a[(size_t)(gm / p) * as + gk * p + gm % p];
                        }
                        t[r * 8 + c] = v;
                    }
            }
        }
    }
    return 0;
}

/* ---------------- Inner kernels ---------------- */

/* Single 8x8 output tile: Ctmp(64 fp32) = At(8x8) x Bt(8x8)^T, looping kt times along K */
/* X100 (VLEN=256)-specific 8x8 output tile kernel: pure RVV intrinsics, no IME2
 * instructions. Verified by bench/ime2_tile_verify.cpp against a scalar reference
 * computed element by element per the packed layout:
 *   scalar reference = IME2 kernel = this kernel (max|d| 1.5e-8 / 5.2e-8
 *   respectively)
 * Conclusion: this kernel's tile indexing matches the IME2 kernel; the earlier
 * dual-cluster work-split failure was in the "wiring", not the kernel.
 * Currently not wired in (wiring requires extracting the 16x16 block body into a
 * named function, see REPORT 7.33). */
static void ime2_rvv_tile_8x8(const ime2_fp16* At, const ime2_fp16* Bt, float* Ctmp, int kt)
{
    vfloat32m1_t a0 = __riscv_vfmv_v_f_f32m1(0.f, 8), a1 = __riscv_vfmv_v_f_f32m1(0.f, 8);
    vfloat32m1_t a2 = __riscv_vfmv_v_f_f32m1(0.f, 8), a3 = __riscv_vfmv_v_f_f32m1(0.f, 8);
    vfloat32m1_t a4 = __riscv_vfmv_v_f_f32m1(0.f, 8), a5 = __riscv_vfmv_v_f_f32m1(0.f, 8);
    vfloat32m1_t a6 = __riscv_vfmv_v_f_f32m1(0.f, 8), a7 = __riscv_vfmv_v_f_f32m1(0.f, 8);
    for (int j = 0; j < kt; j++)
    {
        const ime2_fp16* ap = At + (size_t)j * 64;
        const ime2_fp16* bp = Bt + (size_t)j * 64;
        for (int c = 0; c < 8; c++)
        {
            vfloat16mf2_t vb = __riscv_vlse16_v_f16mf2(bp + c, 16, 8);
            a0 = __riscv_vfwmacc_vf_f32m1(a0, (_Float16)ap[0 * 8 + c], vb, 8);
            a1 = __riscv_vfwmacc_vf_f32m1(a1, (_Float16)ap[1 * 8 + c], vb, 8);
            a2 = __riscv_vfwmacc_vf_f32m1(a2, (_Float16)ap[2 * 8 + c], vb, 8);
            a3 = __riscv_vfwmacc_vf_f32m1(a3, (_Float16)ap[3 * 8 + c], vb, 8);
            a4 = __riscv_vfwmacc_vf_f32m1(a4, (_Float16)ap[4 * 8 + c], vb, 8);
            a5 = __riscv_vfwmacc_vf_f32m1(a5, (_Float16)ap[5 * 8 + c], vb, 8);
            a6 = __riscv_vfwmacc_vf_f32m1(a6, (_Float16)ap[6 * 8 + c], vb, 8);
            a7 = __riscv_vfwmacc_vf_f32m1(a7, (_Float16)ap[7 * 8 + c], vb, 8);
        }
    }
    __riscv_vse32_v_f32m1(Ctmp + 0, a0, 8);
    __riscv_vse32_v_f32m1(Ctmp + 8, a1, 8);
    __riscv_vse32_v_f32m1(Ctmp + 16, a2, 8);
    __riscv_vse32_v_f32m1(Ctmp + 24, a3, 8);
    __riscv_vse32_v_f32m1(Ctmp + 32, a4, 8);
    __riscv_vse32_v_f32m1(Ctmp + 40, a5, 8);
    __riscv_vse32_v_f32m1(Ctmp + 48, a6, 8);
    __riscv_vse32_v_f32m1(Ctmp + 56, a7, 8);
}

static void ime2_tile_8x8(const ime2_fp16* At, const ime2_fp16* Bt, float* Ctmp, int kt)
{
    /* Pipelined version: prefetch step k+1's A/B while computing the current step —
     * hides L2 load latency. Measured ~2x faster than the same structure
     * non-pipelined (microbenchmark 119->195 GFLOP/s/core). */
    int k = kt - 1;
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vmv.v.i v16, 0\n\t"
        "vsetvli t0, zero, e16, m1, tu, mu\n\t"
        "vle16.v v2, (%[A])\n\t"
        "vle16.v v8, (%[B])\n\t"
        "addi %[A], %[A], 128\n\t"
        "addi %[B], %[B], 128\n\t"
        "beqz %[k], 2f\n\t" /* kt==1 guard */
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

static void ime2_tile_16x16(const ime2_fp16* At, const ime2_fp16* Bt,
                            float* C00, float* C10, float* C01, float* C11, int kt)
{
    /* 16x16 output = 4 8x8 MICRO tiles; K direction unrolled by 2 steps + register
     * rotation: each step's loads are issued one step ahead, and **vmv is no longer
     * used to move prefetch registers into current registers** (the old version's 4
     * vmv per step sat exactly on the MAC dependency critical path).
     * Microbenchmark (K=1024): before padding 199 / padding only 210 / padding +
     * 2-step unroll **373** GFLOP/s/core. */
    const size_t ld = IME2_TILE_LD(kt);
    const ime2_fp16* A1 = At + ld;
    const ime2_fp16* B1 = Bt + ld;
    int k = kt >> 1;
    const int odd = kt & 1;

    __asm__ volatile(
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vmv.v.i v16, 0\n\t vmv.v.i v18, 0\n\t vmv.v.i v20, 0\n\t vmv.v.i v22, 0\n\t"
        "vsetvli t0, zero, e16, m1, tu, mu\n\t"
        /* k=0 -> v2(A0) v4(A1) v8(B0) v10(B1) */
        "vle16.v v2, (%[A0])\n\t vle16.v v4, (%[A1])\n\t"
        "vle16.v v8, (%[B0])\n\t vle16.v v10, (%[B1])\n\t"
        "addi %[A0], %[A0], 128\n\t addi %[A1], %[A1], 128\n\t"
        "addi %[B0], %[B0], 128\n\t addi %[B1], %[B1], 128\n\t"
        "blez %[k], 2f\n\t"
        "1:\n\t"
        /* Phase A: load k+1 -> v12..v15; meanwhile do 4 MACs for k */
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
        /* Phase B: load k+2 -> v2..v10; meanwhile do 4 MACs for k+1 */
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
        /* when kt is odd, the last step's data is already in v2/v4/v8/v10 */
        "beqz %[odd], 3f\n\t"
        "smt.vfwmadot v16, v2, v8\n\t"
        "smt.vfwmadot v18, v4, v8\n\t"
        "smt.vfwmadot v20, v2, v10\n\t"
        "smt.vfwmadot v22, v4, v10\n\t"
        "3:\n\t"
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vse32.v v16, (%[C00])\n\t vse32.v v18, (%[C10])\n\t"
        "vse32.v v20, (%[C01])\n\t vse32.v v22, (%[C11])\n\t"
        : [A0] "+r"(At), [A1] "+r"(A1), [B0] "+r"(Bt), [B1] "+r"(B1), [k] "+r"(k)
        : [C00] "r"(C00), [C10] "r"(C10), [C01] "r"(C01), [C11] "r"(C11), [odd] "r"(odd)
        : "t0", "v2", "v4", "v8", "v10", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "memory", "cc");
}

/* 2x4 register blocking: computes 16(M)x32(N) at once. k+1 prefetch pipeline
 * (all vmv after the 8 MACs, with independent preload registers — moving vmv
 * earlier would make columns c2/c3 use the next step's A operand; verified by
 * differential testing). */
static void ime2_tile_16x32(const ime2_fp16* At, const ime2_fp16* Bt,
                            float* C00, float* C10,
                            float* C01, float* C11,
                            float* C02, float* C12,
                            float* C03, float* C13, int kt)
{
    const size_t ld = IME2_TILE_LD(kt);
    const ime2_fp16* A1 = At + ld;
    const ime2_fp16 *B1 = Bt + ld, *B2 = B1 + ld, *B3 = B2 + ld;
    int k = kt - 1;
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vmv.v.i v16, 0\n\t vmv.v.i v18, 0\n\t vmv.v.i v20, 0\n\t vmv.v.i v22, 0\n\t"
        "vmv.v.i v24, 0\n\t vmv.v.i v26, 0\n\t vmv.v.i v28, 0\n\t vmv.v.i v30, 0\n\t"
        "vsetvli t0, zero, e16, m1, tu, mu\n\t"
        "vle16.v v2, (%[A0])\n\t vle16.v v4, (%[A1])\n\t"
        "vle16.v v8, (%[B0])\n\t vle16.v v9, (%[B1])\n\t vle16.v v10, (%[B2])\n\t vle16.v v11, (%[B3])\n\t"
        "addi %[A0], %[A0], 128\n\t addi %[A1], %[A1], 128\n\t"
        "addi %[B0], %[B0], 128\n\t addi %[B1], %[B1], 128\n\t addi %[B2], %[B2], 128\n\t addi %[B3], %[B3], 128\n\t"
        "beqz %[k], 2f\n\t" /* kt==1 guard */
        "1:\n\t"
        "vle16.v v12, (%[A0])\n\t"
        "vle16.v v14, (%[A1])\n\t"
        "vle16.v v15, (%[B0])\n\t"
        "vle16.v v13, (%[B1])\n\t"
        "vle16.v v1, (%[B2])\n\t"
        "vle16.v v3, (%[B3])\n\t"
        "smt.vfwmadot v16, v2, v8\n\t"
        "smt.vfwmadot v18, v4, v8\n\t"
        "smt.vfwmadot v20, v2, v9\n\t"
        "smt.vfwmadot v22, v4, v9\n\t"
        "smt.vfwmadot v24, v2, v10\n\t"
        "smt.vfwmadot v26, v4, v10\n\t"
        "smt.vfwmadot v28, v2, v11\n\t"
        "smt.vfwmadot v30, v4, v11\n\t"
        "addi %[A0], %[A0], 128\n\t addi %[A1], %[A1], 128\n\t"
        "addi %[B0], %[B0], 128\n\t addi %[B1], %[B1], 128\n\t addi %[B2], %[B2], 128\n\t addi %[B3], %[B3], 128\n\t"
        "vmv.v.v v2, v12\n\t vmv.v.v v4, v14\n\t vmv.v.v v8, v15\n\t"
        "vmv.v.v v9, v13\n\t vmv.v.v v10, v1\n\t vmv.v.v v11, v3\n\t"
        "addi %[k], %[k], -1\n\t bnez %[k], 1b\n\t"
        "2:\n\t"
        "smt.vfwmadot v16, v2, v8\n\t smt.vfwmadot v18, v4, v8\n\t"
        "smt.vfwmadot v20, v2, v9\n\t smt.vfwmadot v22, v4, v9\n\t"
        "smt.vfwmadot v24, v2, v10\n\t smt.vfwmadot v26, v4, v10\n\t"
        "smt.vfwmadot v28, v2, v11\n\t smt.vfwmadot v30, v4, v11\n\t"
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vse32.v v16, (%[C00])\n\t vse32.v v18, (%[C10])\n\t"
        "vse32.v v20, (%[C01])\n\t vse32.v v22, (%[C11])\n\t"
        "vse32.v v24, (%[C02])\n\t vse32.v v26, (%[C12])\n\t"
        "vse32.v v28, (%[C03])\n\t vse32.v v30, (%[C13])\n\t"
        : [A0] "+r"(At), [A1] "+r"(A1),
        [B0] "+r"(Bt), [B1] "+r"(B1), [B2] "+r"(B2), [B3] "+r"(B3), [k] "+r"(k)
        : [C00] "r"(C00), [C10] "r"(C10), [C01] "r"(C01), [C11] "r"(C11),
        [C02] "r"(C02), [C12] "r"(C12), [C03] "r"(C03), [C13] "r"(C13)
        : "t0", "v1", "v2", "v3", "v4", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
        "memory", "cc");
}

/* ---------------- Full GEMM ---------------- */

/* M=1 (decode GEMV) specific kernel: one k loop advances 4 N-tiles in parallel
 * (32 output columns), 4 independent accumulation chains + k+1 software prefetch
 * (A and 4-way B double-buffered).
 * Background: with M=1 the 8x8 kernel only reaches 11.9 GB/s on 512 columns of
 * weights, while M=8 (4 chains) reaches 23.6 GB/s — decode GEMM is stuck on
 * vfwmadot load latency; here multiple chains + prefetch hide that latency.
 * The result takes only row 0 of each 8x8 tile (M=1), written into row[32] (tile
 * t -> row[t*8..t*8+8)). */
#if __riscv_vector
#include <time.h>
static inline double ime2_i8_now()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
#endif

static void ime2_quantize_B_int8_col(const ncnn::Mat& B, ncnn::Mat& BT_i8, ncnn::Mat& BT_wscale, int N, int K)
{
    const int is_fp16 = B.elembits() == 16;
    BT_i8.create((size_t)N * K, (size_t)1u, 1, (ncnn::Allocator*)0);
    /* scale must be stored in fp32: amax/127 underflows to fp16 zero when amax <
     * ~7.6e-6, while the panel was quantized by the fp32 sc into non-zero values —
     * reading scale back as 0 would zero out the whole column's dot product. */
    BT_wscale.create((size_t)N, (size_t)4u, 1, (ncnn::Allocator*)0);
    if (BT_i8.empty() || BT_wscale.empty())
        return;

    signed char* dst = (signed char*)BT_i8;
    float* ws = (float*)BT_wscale;

    for (int n = 0; n < N; n++)
    {
        const ime2_fp16* row16 = 0;
        const float* row32 = 0;
        if (is_fp16)
            row16 = (const ime2_fp16*)B + (size_t)n * K;
        else
            row32 = (const float*)B + (size_t)n * K;

        float amax = 0.f;
        for (int k = 0; k < K; k++)
        {
            const float v = is_fp16 ? (float)row16[k] : row32[k];
            const float a = fabsf(v);
            if (a > amax)
                amax = a;
        }
        const float sc = amax > 0.f ? amax / 127.f : 1.f;
        ws[n] = sc;
        signed char* d = dst + (size_t)n * K;
        for (int k = 0; k < K; k++)
        {
            const float v = is_fp16 ? (float)row16[k] : row32[k];
            int q = (int)lrintf(v / sc);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            d[k] = (signed char)q;
        }
    }
}

/* Quantize one row of fp16 activations to int8 (one scale for the whole row), for the integer-domain dot product. */
static float ime2_quantize_a_int8(const ime2_fp16* a, int K, signed char* a8, int nT)
{
    /* K is usually only 1024: opening an omp region (2 per call) is slower than
     * doing it serially, and decode has ~196 Gemm calls per token ⇒ this must stay
     * serial. */
    (void)nT;
    float amax = 0.f;
    for (int k = 0; k < K; k++)
    {
        const float v = fabsf((float)a[k]);
        if (v > amax)
            amax = v;
    }
    const float sc = amax > 0.f ? amax / 127.f : 1.f;
    for (int k = 0; k < K; k++)
    {
        int q = (int)lrintf((float)a[k] / sc);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        a8[k] = (signed char)q;
    }
    return sc;
}

/* Integer-domain M=1 GEMV: one int32 dot product per column (128 MACs / vector instruction), finally multiplied by both sides' scales. */
static void ime2_gemv_m1_i8c(const signed char* a8, float a_scale,
                             const signed char* W, const float* wscale,
                             float* out, int n0, int n1, int K)
{
    /* Cold-cache microbenchmark (bench/i8_gemv_cold_bench.c, 50MB working set far
     * beyond LLC):
     *   e8m1 (old) 12.99 GB/s → e8m2 16.54 → e8m2+prefetch 16.99 GB/s (+31%).
     * At decode time each layer's int8 panel is read once, 0.46GB total, all from
     * DRAM, so tuning must follow "cold" conditions. */
    const size_t vlmax = __riscv_vsetvlmax_e32m8();
    for (int n = n0; n < n1; n++)
    {
        const signed char* w = W + (size_t)n * K;
        vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vlmax);
        int k = 0;
        while (k < K)
        {
            const size_t vl = __riscv_vsetvl_e8m2((size_t)(K - k));
            __builtin_prefetch(w + k + 512); /* pull into cache 512B ahead (zicbop) */
            __builtin_prefetch(w + k + 576);
            vint8m2_t av = __riscv_vle8_v_i8m2(a8 + k, vl);
            vint8m2_t wv = __riscv_vle8_v_i8m2(w + k, vl);
            vint16m4_t a16 = __riscv_vwadd_vx_i16m4(av, 0, vl);
            vint16m4_t w16 = __riscv_vwadd_vx_i16m4(wv, 0, vl);
            /* _tu: when K is not a multiple of 256 (the vlmax of e32m8 at VLEN=1024), the
             * last short vl's vwmacc tail is agnostic, while the reduction below reads all
             * vlmax lanes. */
            acc = __riscv_vwmacc_vv_i32m8_tu(acc, a16, w16, vl);
            k += (int)vl;
        }
        /* Widening reduction: the i32 total overflows when K > 133145 (127·127·133145 >
         * INT32_MAX), with unsigned wraparound — the sign may flip even before
         * dequantization. First sign-extend the two lane halves to i64, then reduce (the
         * per-lane i32 partial sums don't overflow until K > 3.4e7, far beyond any
         * realistic K). */
        vint32m4_t acc_lo = __riscv_vget_v_i32m8_i32m4(acc, 0);
        vint32m4_t acc_hi = __riscv_vget_v_i32m8_i32m4(acc, 1);
        const size_t vlmax64 = __riscv_vsetvlmax_e64m8();
        vint64m8_t acc64 = __riscv_vadd_vv_i64m8(__riscv_vsext_vf2_i64m8(acc_lo, vlmax64), __riscv_vsext_vf2_i64m8(acc_hi, vlmax64), vlmax64);
        const long long isum = __riscv_vmv_x_s_i64m1_i64(__riscv_vredsum_vs_i64m8_i64m1(acc64, __riscv_vmv_v_x_i64m1(0, 1), vlmax64));
        out[n] = a_scale * wscale[n] * (float)isum;
    }
}

static void ime2_gemv_m1_x4(const ime2_fp16* At, const ime2_fp16* B0, size_t bt_stride, float* row, int kt)
{
    /* K direction unrolled by 2 steps + register rotation: removed the old version's
     * 5 vmv per step (they sat on the MAC dependency critical path).
     * Same technique as the 16x16 kernel (see REPORT 7.17): +77% under streaming
     * loads in the microbenchmark. */
    const ime2_fp16* B1 = B0 + bt_stride;
    const ime2_fp16* B2 = B1 + bt_stride;
    const ime2_fp16* B3 = B2 + bt_stride;
    const size_t vl8 = 8;
    int k = kt >> 1;
    const int odd = kt & 1;
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vmv.v.i v24, 0\n\t vmv.v.i v26, 0\n\t vmv.v.i v28, 0\n\t vmv.v.i v30, 0\n\t"
        "vsetvli t0, zero, e16, m1, tu, mu\n\t"
        /* k=0 -> v2(A) v8..v11(B0..B3) */
        "vle16.v v2, (%[A])\n\t"
        "vle16.v v8, (%[B0])\n\t vle16.v v9, (%[B1])\n\t vle16.v v10, (%[B2])\n\t vle16.v v11, (%[B3])\n\t"
        "addi %[A], %[A], 128\n\t"
        "addi %[B0], %[B0], 128\n\t addi %[B1], %[B1], 128\n\t"
        "addi %[B2], %[B2], 128\n\t addi %[B3], %[B3], 128\n\t"
        "blez %[k], 2f\n\t"
        "1:\n\t"
        /* Phase A: load k+1 -> v3/v12..v15; meanwhile do 4 MACs for k */
        "vle16.v v3, (%[A])\n\t"
        "vle16.v v12, (%[B0])\n\t vle16.v v13, (%[B1])\n\t vle16.v v14, (%[B2])\n\t vle16.v v15, (%[B3])\n\t"
        "smt.vfwmadot v24, v2, v8\n\t"
        "smt.vfwmadot v26, v2, v9\n\t"
        "smt.vfwmadot v28, v2, v10\n\t"
        "smt.vfwmadot v30, v2, v11\n\t"
        "addi %[A], %[A], 128\n\t"
        "addi %[B0], %[B0], 128\n\t addi %[B1], %[B1], 128\n\t"
        "addi %[B2], %[B2], 128\n\t addi %[B3], %[B3], 128\n\t"
        /* Phase B: load k+2 -> v2/v8..v11; meanwhile do 4 MACs for k+1 */
        "vle16.v v2, (%[A])\n\t"
        "vle16.v v8, (%[B0])\n\t vle16.v v9, (%[B1])\n\t vle16.v v10, (%[B2])\n\t vle16.v v11, (%[B3])\n\t"
        "smt.vfwmadot v24, v3, v12\n\t"
        "smt.vfwmadot v26, v3, v13\n\t"
        "smt.vfwmadot v28, v3, v14\n\t"
        "smt.vfwmadot v30, v3, v15\n\t"
        "addi %[A], %[A], 128\n\t"
        "addi %[B0], %[B0], 128\n\t addi %[B1], %[B1], 128\n\t"
        "addi %[B2], %[B2], 128\n\t addi %[B3], %[B3], 128\n\t"
        "addi %[k], %[k], -1\n\t bnez %[k], 1b\n\t"
        "2:\n\t"
        "beqz %[odd], 3f\n\t"
        "smt.vfwmadot v24, v2, v8\n\t"
        "smt.vfwmadot v26, v2, v9\n\t"
        "smt.vfwmadot v28, v2, v10\n\t"
        "smt.vfwmadot v30, v2, v11\n\t"
        "3:\n\t"
        /* take only row 0 of each 8x8 tile */
        "vsetvli t0, %[vl8], e32, m2, tu, mu\n\t"
        "vse32.v v24, (%[R0])\n\t"
        "vse32.v v26, (%[R1])\n\t"
        "vse32.v v28, (%[R2])\n\t"
        "vse32.v v30, (%[R3])\n\t"
        : [A] "+r"(At), [B0] "+r"(B0), [B1] "+r"(B1), [B2] "+r"(B2), [B3] "+r"(B3), [k] "+r"(k)
        : [R0] "r"(row), [R1] "r"(row + 8), [R2] "r"(row + 16), [R3] "r"(row + 24),
        [vl8] "r"(vl8), [odd] "r"(odd)
        : "t0", "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
        "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
        "memory", "cc");
}

/* Compute + writeback of one 16x16 output block (IB x JB tile pairs).
 * Why a named function: the single-cluster path and the dual-cluster (A100+X100)
 * path must share the same implementation. */
static void ime2_compute_block_16x16(const ime2_fp16* at, const ime2_fp16* bt, ime2_fp16* out,
                                     int ibb, int jbb, int mt, int nt, int M, int N,
                                     int ldat, int ldt, int kt, int p, float alpha)
{
    const int IB = 8, JB = 8;

    const int i_end = (ibb + 1) * IB * 2 < mt ? (ibb + 1) * IB * 2 : mt;
    const int j_end = (jbb + 1) * JB * 2 < nt ? (jbb + 1) * JB * 2 : nt;
    for (int j = jbb * JB * 2; j < j_end; j += 2)
    {
        for (int i = ibb * IB * 2; i < i_end; i += 2)
        {
            float t00[64], t10[64], t01[64], t11[64];

            const bool full2x2 = (i + 1 < mt) && (j + 1 < nt);

            if (full2x2)
            {
                ime2_tile_16x16(at + (size_t)i * ldat, bt + (size_t)j * ldt,
                                t00, t10, t01, t11, kt);
            }
            else
            {
                ime2_tile_8x8(at + (size_t)i * ldat, bt + (size_t)j * ldt, t00, kt);
                if (j + 1 < nt)
                    ime2_tile_8x8(at + (size_t)i * ldat, bt + (size_t)(j + 1) * ldt, t01, kt);
                if (i + 1 < mt)
                    ime2_tile_8x8(at + (size_t)(i + 1) * ldat, bt + (size_t)j * ldt, t10, kt);
                if (i + 1 < mt && j + 1 < nt)
                    ime2_tile_8x8(at + (size_t)(i + 1) * ldat, bt + (size_t)(j + 1) * ldt, t11, kt);
            }

            /* Scatter + fp32->fp16 conversion, written into the ncnn packed layout.
             * p==8 (the A100 fp16 norm): the same 8-row tile is exactly one elempack group;
             * contiguous 16B writes along m (vlse32 strided reads of the temp tile + vfncvt
             * + vse16); row base computed only once (eliminating per-element gm/p, gm%p
             * division).
             * p==1: contiguous writes within the row along n. */
            {
                const float* tiles[4] = {t00, t01, t10, t11};
                const int dm[4] = {0, 0, 8, 8};
                const int dn[4] = {0, 8, 0, 8};
                for (int q = 0; q < 4; q++)
                {
                    const int gm0 = i * 8 + dm[q];
                    if (gm0 >= M)
                        continue;
                    const int rmax = (gm0 + 8 <= M) ? 8 : (M - gm0);
                    const float* tq = tiles[q];
                    if (p == 8)
                    {
                        for (int c = 0; c < 8; c++)
                        {
                            const int gn = j * 8 + dn[q] + c;
                            if (gn >= N)
                                continue;
                            ime2_fp16* o = out + ((size_t)(gm0 / 8) * N + gn) * 8;
                            if (rmax == 8)
                            {
                                /* explicit vl=8 (with VLEN=1024, vsetvli zero would give 32 — out of bounds!) */
                                vfloat32m2_t v = __riscv_vlse32_v_f32m2(tq + c, 32, 8);
                                v = __riscv_vfmul_vf_f32m2(v, alpha, 8);
                                __riscv_vse16_v_f16m1(o, __riscv_vfncvt_f_f_w_f16m1(v, 8), 8);
                            }
                            else
                            {
                                for (int r = 0; r < rmax; r++)
                                    o[r] = (ime2_fp16)(tq[r * 8 + c] * alpha);
                            }
                        }
                    }
                    else if (p == 1)
                    {
                        for (int r = 0; r < rmax; r++)
                        {
                            const int gm = gm0 + r;
                            const int gn0 = j * 8 + dn[q];
                            ime2_fp16* orow = out + (size_t)gm * N + gn0;
                            const float* trow = tq + r * 8;
                            if (gn0 + 8 <= N)
                            {
                                for (int c = 0; c < 8; c++)
                                    orow[c] = (ime2_fp16)(trow[c] * alpha);
                            }
                            else
                            {
                                for (int c = 0; c < 8; c++)
                                {
                                    if (gn0 + c < N)
                                        orow[c] = (ime2_fp16)(trow[c] * alpha);
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int r = 0; r < rmax; r++)
                        {
                            const int gm = gm0 + r;
                            for (int c = 0; c < 8; c++)
                            {
                                const int gn = j * 8 + dn[q] + c;
                                if (gn >= N)
                                    continue;
                                out[((size_t)(gm / p) * N + gn) * p + gm % p] = (ime2_fp16)(tq[r * 8 + c] * alpha);
                            }
                        }
                    }
                }
            }
        }
    }
}

static int gemm_ime2_fp16(const ncnn::Mat& A, const ncnn::Mat& BT, ncnn::Mat& top_blob,
                          int M, int N, int K, float alpha, int out_elempack, int nT, const ncnn::Option& opt,
                          const ncnn::Mat& BT_i8c = ncnn::Mat(), const ncnn::Mat& BT_wscale = ncnn::Mat())
{
    const int mt = (M + 7) / 8, nt = (N + 7) / 8, kt = (K + 7) / 8;
    const int p = out_elempack;
    const size_t ldt = IME2_TILE_LD(kt);
    const size_t ldat = ldt;

    ime2_fp16* out0 = (ime2_fp16*)top_blob;

    /* ---- int8 weight mode + M=1: **packing-free GEMV** ----
     * A is a [1,K] contiguous fp16 row; the integer-domain kernel only needs
     * "quantized activations + per-column weights" — no IME2 tile packing at all.
     * Measured: pack_A at M=1 costs 105µs/call (196 calls/token = 20.6ms, ~24% of
     * decode); saved entirely here. */
    /* Note A.elempack == 1 is required: under a packed layout (elempack = 8/16) the
     * K values of the same row are stored with a stride ([m/p][k][p]); reading them
     * as a contiguous array would compute wrong results. Packed inputs still take
     * the general path below. */
    if (!BT_i8c.empty() && !BT_wscale.empty() && M == 1 && p == 1 && A.elembits() == 16 && A.elempack == 1)
    {
        const signed char* w8c = (const signed char*)BT_i8c;
        const float* wsc = (const float*)BT_wscale;
        /* Keep the on-stack scratch small: a caller thread with a small configured
         * stack (e.g. a 32 KiB embedded pthread) must not hit its guard page from
         * our frame before the workspace checks run. 1 KiB int8 + 4 KiB fp32 is
         * comfortably small; anything larger uses the checked workspace buffer. */
        static const int IME2_I8_STK = 1024;
        signed char a8stk[IME2_I8_STK];
        float ostk[IME2_I8_STK];
        /* Large scratch goes through the workspace allocator (null-checkable, consistent
         * with the caller's memory budget) instead of a per-token process-heap vector:
         * bad_alloc would propagate or directly terminate in exceptions-disabled builds,
         * while the workspace pool actually has capacity. The -100 return semantics is
         * unchanged. */
        ncnn::Mat a8ws, ows;
        signed char* a8p = a8stk;
        float* op = ostk;
        if (K > IME2_I8_STK)
        {
            a8ws.create(K, (size_t)1u, opt.workspace_allocator);
            if (a8ws.empty())
                return -100;
            a8p = (signed char*)a8ws.data;
        }
        if (N > IME2_I8_STK)
        {
            ows.create(N, (size_t)4u, opt.workspace_allocator);
            if (ows.empty())
                return -100;
            op = (float*)ows.data;
        }

        const bool iprof = (getenv("NCNN_IME2_I8_PROF") != nullptr);
        const double tA = iprof ? ime2_i8_now() : 0.0;
        const float a_scale = ime2_quantize_a_int8((const ime2_fp16*)A, K, a8p, nT);
        const double tB = iprof ? ime2_i8_now() : 0.0;

        #pragma omp parallel for num_threads(nT) schedule(static)
        for (int nb = 0; nb < nT; nb++)
        {
            const int lo = (int)((long long)N * nb / nT), hi = (int)((long long)N * (nb + 1) / nT);
            ime2_gemv_m1_i8c(a8p, a_scale, w8c, wsc, op, lo, hi, K);
        }
        const double tC = iprof ? ime2_i8_now() : 0.0;

        int c2 = 0;
        while (c2 < N)
        {
            const size_t vl = __riscv_vsetvl_e32m8(N - c2);
            vfloat32m8_t v = __riscv_vle32_v_f32m8(op + c2, vl);
            if (alpha != 1.f)
                v = __riscv_vfmul_vf_f32m8(v, alpha, vl);
            __riscv_vse16_v_f16m4(out0 + c2, __riscv_vfncvt_f_f_w_f16m4(v, vl), vl);
            c2 += (int)vl;
        }
        if (iprof)
        {
            const double tD = ime2_i8_now();
            #pragma omp critical
            {
                static double sA2 = 0, sB2 = 0, sC2 = 0;
                static int cnt2 = 0;
                sA2 += tB - tA;
                sB2 += tC - tB;
                sC2 += tD - tC;
                cnt2++;
                if (cnt2 == 196)
                    fprintf(stderr, "[i8-prof] packing-free path per call: quantize=%.1fus kernel=%.1fus convert=%.1fus total=%.1fus\n",
                            sA2 / cnt2 * 1000, sB2 / cnt2 * 1000, sC2 / cnt2 * 1000, (sA2 + sB2 + sC2) / cnt2 * 1000);
            }
        }
        return 0;
    }

    ncnn::Mat AT;
    if (ime2_pack_A(A, AT, M, K, nT, opt.workspace_allocator) != 0)
        return -100;

    const ime2_fp16* bt = (const ime2_fp16*)BT;
    const ime2_fp16* at = (const ime2_fp16*)AT;
    ime2_fp16* out = (ime2_fp16*)top_blob;

    /* Decode (M=1) fast path: GEMV kernel with 8 accumulation chains; p must be 1 (M<packn) */
    if (M == 1 && p == 1)
    {
        /* The guard and the kernel are two separate omp regions, and the runtime may
         * swap workers: at the entry of the execution region each worker self-checks
         * (mask ⊆ A100 set); after the barrier the whole team starts together or gives
         * up. */
        cpu_set_t a100set;
        int team_bad = !ime2_a100_set(nT, &a100set);
        if (!team_bad)
        {
            #pragma omp parallel num_threads(nT)
            {
                if (!ime2_worker_in_set(&a100set))
                {
                    #pragma omp atomic write
                    team_bad = 1;
                }
                #pragma omp barrier
                if (!team_bad)
                {
                    #pragma omp for
                    for (int j = 0; j < nt; j += 4)
                    {
                        const int jn = nt - j < 4 ? nt - j : 4;
                        float row[32];
                        if (jn == 4)
                        {
                            ime2_gemv_m1_x4(at, bt + (size_t)j * ldt, ldt, row, kt);
                        }
                        else
                        {
                            for (int t = 0; t < jn; t++)
                            {
                                float tile[64];
                                ime2_tile_8x8(at, bt + (size_t)(j + t) * ldt, tile, kt);
                                for (int c = 0; c < 8; c++)
                                    row[t * 8 + c] = tile[c];
                            }
                        }
                        /* 64 fp32 -> fp16, contiguous 128B store (vl=64: the e32m2 limit at VLEN=1024 is exactly 64) */
                        /* jn counts tiles, so jn*8 rounds the width up to a multiple of 8.
                        * The real output has only N columns; without this cap the store
                        * below writes up to seven elements past top_blob and, since the
                        * buffers are adjacent, can overwrite allocator bookkeeping. */
                        const int n = ((j + jn) * 8 < N) ? jn * 8 : (N - j * 8);
                        int c = 0;
                        while (c < n)
                        {
                            size_t vl = __riscv_vsetvl_e32m2(n - c);
                            vfloat32m2_t v = __riscv_vle32_v_f32m2(row + c, vl);
                            if (alpha != 1.f)
                                v = __riscv_vfmul_vf_f32m2(v, alpha, vl);
                            __riscv_vse16_v_f16m1(out + (size_t)j * 8 + c, __riscv_vfncvt_f_f_w_f16m1(v, vl), vl);
                            c += (int)vl;
                        }
                    }
                }
            }
        }
        if (team_bad)
            return -99;
        return 0;
    }

    (void)opt;

    /* Two-level blocking: partition (i,j) into IB x JB output blocks (each block 8x8
     * 16x16 tiles = 128x128 outputs); inside a block, loop j then i, so the A block
     * (128 rows x K) and the B block (128 columns x K) both stay resident in L2 and
     * are reused by the block's IB*JB tile computations.
     * Background: without blocking, every j rescans the whole A (1MB x nt/2 times =
     * hundreds of MB of L2 traffic); measured GEMM was only 58 GFLOP/s/core ≈ 6% of
     * the hardware IME2 capability (8 cores concurrently measured 925 GFLOP/s/core).
     * Parallelism is partitioned by block (collapse(2), outer loop jb); each thread
     * touches only its own column of blocks. */
    const int IB = 8, JB = 8; // unit: 16x16 tile pairs (i/j step 2)
    /* mt/nt count 8-wide tiles, so the number of 16-wide tile PAIRS is a ceiling:
     * with an odd mt the last tile still has to be computed, and M <= 8 must still
     * produce one pair (mt == 1 gives mt/2 == 0 and the block loop would never
     * run). Floor division here silently dropped those output rows/columns. */
    const int mti = (mt + 1) / 2;
    const int nti = (nt + 1) / 2;
    const int nib = (mti + IB - 1) / IB;
    const int njb = (nti + JB - 1) / JB;
    /* Heterogeneous dual-cluster work split (A100 IME2 + X100 RVV) was experimented with and rejected — a platform limitation, not a code problem:
                                                                                                                                                                                     *  (1) the two clusters do run concurrently without interference (two processes pinned separately: A100 374 + X100 120 = 490 t/s, potential +31%);
                                                                                                                                                                                     *  (2) but /proc/set_ai_thread registration (the prerequisite for getting VLEN=1024 on A100) **confines the process to the A100 cluster**:
                                                                                                                                                                                     *      measured: with a mixed run all threads land on cpu8-15; taskset to 8-15 directly without registration fails outright.
                                                                                                                                                                                     * So "dual-cluster work split within a single process" is infeasible on this platform; realizing the +31% requires a **multi-process** scheme.
                                                                                                                                                                                     * See REPORT 7.34; the X100 RVV tile kernel (ime2_rvv_tile_8x8, verified by differential testing) is kept for future use. */
    /* Same as the M=1 region: self-check inside the execution region (the guard and
     * the kernel are two regions, the runtime may swap workers); after the barrier
     * the whole team starts together or gives up. */
    cpu_set_t a100set2;
    int team_bad2 = !ime2_a100_set(nT, &a100set2);
    if (!team_bad2)
    {
        #pragma omp parallel num_threads(nT)
        {
            if (!ime2_worker_in_set(&a100set2))
            {
                #pragma omp atomic write
                team_bad2 = 1;
            }
            #pragma omp barrier
            if (!team_bad2)
            {
                #pragma omp for collapse(2)
                for (int jbb = 0; jbb < njb; jbb++)
                {
                    for (int ibb = 0; ibb < nib; ibb++)
                    {
                        ime2_compute_block_16x16(at, bt, out, ibb, jbb, mt, nt, M, N, ldat, ldt, kt, p, alpha);
                    }
                }
            }
        }
    }
    (void)nt;
    if (team_bad2)
        return -99;
    return 0;
}

} // namespace ncnn_ime2

#endif // NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2 && __riscv_v
#endif // GEMM_RISCV_IME2_H
