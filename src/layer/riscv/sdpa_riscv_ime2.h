// SpacemiT K3 A100 IME2 (smt.vfwmadot) kernels for SDPA prefill.
//
// 与 gemm_riscv_ime2.h 同源同约束：
//   - 只能在 A100 簇 (VLEN=1024) 执行；X100 上 vfwmadot 是 SIGILL。
//   - 进程须整体钉在 A100（ai-run / /proc/set_ai_thread）。
// 差别：这里的输入是 fp32（SDPA 的 q/k/v 都是 fp32 Mat），打包时转 fp16；
//       输出写 fp32（注意力分数/输出要接 softmax 和后续 fp32 图）。
//
// 本文件由 sdpa_riscv_zfh.cpp #include（zfh 变体 TU 带 xsmtvdotii 汇编支持）。

#ifndef SDPA_RISCV_IME2_H
#define SDPA_RISCV_IME2_H

#include "platform.h"
#include <float.h>
#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
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

/* ---------------- 运行时探测（A100?） ---------------- */

/* 跳转上下文必须是**线程局部**的：SIGILL 处理器是进程级的，若在探测窗口内
 * 另一个线程因自己的原因收到 SIGILL，处理器会在那个线程上运行；若无条件
 * siglongjmp 到本线程保存的 jb，就是跨线程 longjmp（UB，会同时毁掉两个线程
 * 的栈）。非探测线程的信号要交还给此前的处置。g_sdpa_ime2_old 只在探测锁内
 * 安装/恢复，读取它的异线程处理器与安装之间不存在长时间窗口。 */
static __thread sigjmp_buf tl_sdpa_ime2_jb;
static __thread int tl_sdpa_ime2_armed = 0;
static struct sigaction g_sdpa_ime2_old;

static void sdpa_ime2_sigill(int sig, siginfo_t* info, void* ucontext)
{
    if (tl_sdpa_ime2_armed)
        siglongjmp(tl_sdpa_ime2_jb, 1);

    /* 不是探测线程的 SIGILL：恢复旧处置后重发，交给内核完整交付。
     * 直接 C 调用旧处理器会绕过内核的信号交付语义——SA_RESETHAND 的一次性
     * 复位、SA_ONSTACK 的栈切换、sa_mask 的屏蔽集全都会被丢掉；而且
     * SIG_DFL(0)/SIG_IGN(1) 与处理器指针共享存储，根本不能当函数调。
     * 恢复+重发是唯一对所有处置都正确的做法。 */
    (void)info;
    (void)ucontext;
    sigaction(SIGILL, &g_sdpa_ime2_old, NULL);
    raise(sig);
}

#if defined __linux__
/* 异构双簇（K3：A100 的 VLEN=1024，X100 的 VLEN=512）下，"探测时所在的核对"与
 * "前向时 OpenMP 工人实际落到的核对"可能不同：前者恰好是 A100 就把 true 缓存
 * 下来，后者落到 X100 就吃 SIGILL。所以探测要覆盖 affinity mask 里的**每一个**
 * 核对：逐个把调用线程移过去读 vlenb，128B（1024bit）才算数。 */
static int sdpa_ime2_probe_affinity(int nthreads)
{
    cpu_set_t cs;
    if (sched_getaffinity(0, sizeof(cs), &cs) != 0)
        return 0; /* 无法确认亲和性：保守关闭 */
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
            all_ok = 0; /* 移不过去就无法确认那个核对 */
            break;
        }
        unsigned long vlenb = 0;
        __asm__ volatile("csrr %0, vlenb"
                         : "=r"(vlenb));
        if (vlenb != 128)
            all_ok = 0;
    }
    sched_setaffinity(0, sizeof(orig), &orig); /* 恢复 */

    /* (a2) OpenMP 工人可能有**逐线程**绑定（OMP_PROC_BIND / GOMP_CPU_AFFINITY /
     * 宿主运行时），上面的掩码遍历只看调用线程，看不到它们。而且 forward 的
     * 并行区都带 num_threads(opt.num_threads)：若它大于默认团队，运行时会给
     * forward 增派**从未被探测过**的工人。所以这里用与 forward 完全相同的
     * 团队大小开区域。
     * 单读一次 vlenb 也不够：工人的掩码若同时覆盖两簇，探测时恰好落在 A100
     * 会通过，forward 时调度器却可能把它挪到 X100。所以每个工人要遍历**自己**
     * 的 affinity mask 里的每一个核对，逐核确认 vlenb==128（探测完恢复工人
     * 原绑定）。任一工人的掩码里混有 X100 核对就保守关闭 IME2。 */
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
                sched_setaffinity(0, sizeof(worig), &worig); /* 恢复工人原绑定 */
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

static int sdpa_ime2_probe_locked(int nthreads)
{
    /* (a) 亲和性：本进程所有允许核对 + forward 团队的每个工人的 vlenb 都必须是 1024bit */
    if (!sdpa_ime2_probe_affinity(nthreads))
        return 0;

    /* (b) 指令本身：SIGILL 保护下执行一条零向量 vfwmadot。
     * 处理器是进程级的，但跳转上下文是**本线程**的 tl_sdpa_ime2_jb：别的线程
     * 在此窗口内收到 SIGILL 会被 sdpa_ime2_sigill 转交给原处置，不会跨线程跳转。 */
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = sdpa_ime2_sigill;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGILL, &sa, &g_sdpa_ime2_old);

    int ok = 0;
    if (sigsetjmp(tl_sdpa_ime2_jb, 1) == 0)
    {
        /* 上下文就绪后才武装：异步 SIGILL 先于武装到来会走转交路径 */
        tl_sdpa_ime2_armed = 1;
        __asm__ volatile(
            "vsetvli t0, zero, e32, m2, tu, mu\n\t" /* e32m2 下清零累加器 v16:v17 */
            "vmv.v.i v16, 0\n\t"
            "vsetvli t0, zero, e16, m1, tu, mu\n\t" /* 回到 LMUL=1 再执行 vfwmadot */
            "vmv.v.i v0, 0\n\t"
            "vmv.v.i v1, 0\n\t"
            "smt.vfwmadot v16, v0, v1\n\t"
            :
            :
            : "t0", "v0", "v1", "v16", "v17", "memory");
        ok = 1;
    }
    /* 两条返回路径都在这里解除武装（longjmp 回来时跳过 if 体直接到这） */
    tl_sdpa_ime2_armed = 0;
    /* 仅当当前处理器仍是我们的时才恢复旧处置：若探测窗口内发生过转发，
     * 当前已是旧处置（或已被内核按 RESETHAND 复位成 DFL）——再装回去会把
     * 一次性处理器错误地重新武装。 */
    {
        struct sigaction cur;
        memset(&cur, 0, sizeof cur);
        sigaction(SIGILL, NULL, &cur);
        if (cur.sa_sigaction == sdpa_ime2_sigill)
            sigaction(SIGILL, &g_sdpa_ime2_old, NULL);
    }
    return ok;
}

/* 返回 1 = 当前 affinity 上下文的所有允许核对都能执行 smt.vfwmadot；0 = 不可。
 * 探测会临时替换进程级 SIGILL 处理器，两个线程并发进入会互相覆盖安装/恢复，
 * 因此整个探测用互斥锁串行化（跳转上下文本身是线程局部的，见上）。
 * 注意不做进程级缓存：affinity 是逐线程的，探测结果只对本次调用时的 affinity
 * 上下文有效。若进程先把某线程绑到 A100 簇建了管线、再绑到 X100 簇建第二个
 * 管线，缓存会把只对前者成立的答案错误地复用到后者（X100 上 forward 即 SIGILL）。
 * nthreads 必须与 forward 的 num_threads 一致：探测的团队就是执行内核的团队。
 * 探测只在建管线时调用一次，重新探测的开销可忽略。 */
static int ime2_probe(int nthreads)
{
    static ncnn::Mutex lock;
    ncnn::MutexLockGuard g(lock);
    return sdpa_ime2_probe_locked(nthreads);
}

/* 一次性测定本进程当前允许的核对里哪些是 A100（vlenb==128）。硬件属性，
 * 进程级缓存；若测定时 mask 只覆盖部分核对，得到的集合是真实 A100 集合的
 * 子集——偏差方向保守（只会误关 IME2，不会误放）。 */
#if defined __linux__
static int sdpa_ime2_a100_set(int nthreads, cpu_set_t* out)
{
    static ncnn::Mutex lock;
    ncnn::MutexLockGuard g(lock);
    static int probed = 0;
    static cpu_set_t cached;
    if (probed)
    {
        *out = cached;
        return 1;
    }
    /* 1) 收集并集（区域内只做 sched_getaffinity，无特权指令，随处可跑）。
     * 必须看团队所有工人的掩码并集：OMP_PROC_BIND / GOMP_CPU_AFFINITY 下
     * 初始线程的掩码常只有自己的 place，单看它会把其他 A100 工人误判为
     * 不可用，等于静默关闭 IME2（性能回归）。 */
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
        return 0;
    /* 2) 调用线程逐核走一遍并集，记下 vlenb==128 的（走完恢复原绑定） */
    cpu_set_t orig;
    if (sched_getaffinity(0, sizeof(orig), &orig) != 0)
        return 0;
    CPU_ZERO(&cached);
    for (int c = 0; c < CPU_SETSIZE; c++)
    {
        if (!CPU_ISSET(c, &uni))
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
    probed = 1;
    *out = cached;
    return 1;
}
#endif

/* forward 入口的轻量校验：本次 forward 的团队**不可能**落到非 A100 核上。
 * 建管线探测（ime2_probe）只覆盖建管线那一刻的上下文：调用方之后可能用
 * set_cpu_thread_affinity() 改了绑定，或用更大的 num_threads 调 forward
 * （运行时增派未探测的工人）。而且只看工人"此刻"的 vlenb 也不够：掩码跨簇
 * 的工人可能采样时在 A100、内核运行中被迁到 X100。所以这里逐工人检查其
 * affinity mask 是否为 A100 集合的子集（子集则调度器无处可迁）。每个工人
 * 一次 sched_getaffinity + 位测试，µs 级，相对每层 SDPA 可忽略。
 * 返回 0 时调用方回落非 IME2 路径，结果语义不变。 */
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
    return ok;
#else
    (void)nthreads;
    return 0;
#endif
}

/* fp16 打包值域检查：|v| > 65504 的有限值打包成 ±inf，而上游 fp32 路径保持
 * 有限。IME2 预填充在打包前对所有输入扫一遍（在任何状态改动之前），越界则
 * 调用方整体回落上游 GEMM 组合路径。逐 channel 只扫有效元素（cstep 填充区是
 * 未初始化垃圾，不能误伤）。max 是精确运算，顺序无关，可用无序归约。 */
static int ime2_fits_fp16(const ncnn::Mat& m)
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
        while (i < n)
        {
            const size_t vl = __riscv_vsetvl_e32m1(n - i);
            vfloat32m1_t v = __riscv_vle32_v_f32m1(p + i, vl);
            vfloat32m1_t vabs = __riscv_vfabs_v_f32m1(v, vl);
            vmax = __riscv_vfmax_vv_f32m1_tu(vmax, vmax, vabs, vl);
            /* 非零值才参与下限：0 是合法内容，不算下溢。零 lane 置 FLT_MAX
             * （vfmerge：零 lane 选后者），再普通 _tu min。 */
            vbool32_t iszero = __riscv_vmfeq_vf_f32m1_b32(v, 0.f, vl);
            vfloat32m1_t vsel = __riscv_vfmerge_vfm_f32m1(vabs, FLT_MAX, iszero, vl);
            vmin = __riscv_vfmin_vv_f32m1_tu(vmin, vmin, vsel, vl);
            i += vl;
        }
        const float mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(vmax, __riscv_vfmv_v_f_f32m1(0.f, 1), vlmax));
        if (mx > 65504.f)
            return 0;
        /* 下溢到零：|v| < 2^-25 ≈ 2.98e-8 的有限值打包成 +0。Q/K 里的这种值
         * 会被 scale 放大成不可忽略的点积损失（scale 在 fp32 结果上施加，
         * 救不回来），同溢出一样回落。 */
        const float mn = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(vmin, __riscv_vfmv_v_f_f32m1(FLT_MAX, 1), vlmax));
        if (mn < 2.98e-8f)
            return 0;
    }
    return 1;
}

/* ---------------- 打包：fp32 -> 8x8 fp16 tile 连续布局 ---------------- */

/* src: [R, K] fp32 行主序（行距 ld 元素）-> tiles [rt][kt][64] fp16 */
static int ime2_pack_fp32(const float* src, size_t ld, ncnn::Mat& T, int R, int K, ncnn::Allocator* alloc)
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
                // 快路径：整块在界内，显式 RVV（vl=8：f32m2 取 8 个 -> vfncvt -> f16m1 存 16B）
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

/* 转置打包：src 是 [K, R] fp32 行主序（行距 ld），产出 B[R,K] 的 tiles，
 * 即 tile 元素 B[r][k] = src[k*ld + r]。用于 P·V 里的 V^T。 */
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
                // 转置快路径：源按列跨步（步长 ld），用 vlse32 一条指令取 8 个
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

/* ---------------- 内层内核（与 gemm 版相同的 vfwmadot tile） ---------------- */

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

/* ---------------- fp32 输出 GEMM（单线程，供 per-head 调用） ---------------- */

/* C[M,N] fp32（行距 ldc）= alpha * (A[M,K] · B[N,K]^T)。AT/BT 为打包好的 tiles。 */
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
 * Flash 式 SDPA（官方 spine-triton ops/08 的 online-softmax  recipe 移植）
 * 流式扫 K/V 块：QK^T(IME2) → 行 max/exp/sum → rescale acc → P·V(IME2 累加)
 * 全程 f32 状态（m_i / l_i / acc），仅 P 转 fp16 喂第二个 GEMM。
 * ================================================================== */

/* ---- k+1 软件流水 tile 内核（与 gemm_riscv_ime2.h 同款，显式指针以便跨步） ---- */

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
        "beqz %[k], 2f\n\t" /* kt==1 时零次循环，直接进尾声（否则 do-while 会越界读） */
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

/* 16x16（2x2 个 8x8 tile）流水版；A0/A1/B0/B1 显式传入（支持 B 跨步取 k 块） */
static void ime2_tile_16x16p(const ime2_fp16* A0, const ime2_fp16* A1,
                             const ime2_fp16* B0, const ime2_fp16* B1,
                             float* C00, float* C10, float* C01, float* C11, int kt)
{
    /* K 方向展开 2 步 + 寄存器轮转：去掉旧版每步 4 条 vmv（它们在 MAC 的依赖关键路径上）。
     * 微基准流式取数：210 -> 373 GFLOP/s/核（+77%，见 REPORT 7.17/7.18）。 */
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

/* 流水 GEMM 驱动：kt 维可跨步（at_stride/bt_stride = 各自 tile 行距，单位元素）。
 * beta==0: C = alpha·A·B^T；beta==1: C += A·B^T（alpha 视为 1）。 */
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
                        /* 快路径：整行 8 列都在界内 —— 显式 RVV（vl=8，e32m1）
                         * flash 分块下 GEMM 很小，写回占比高，必须走定长向量存取。 */
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

/* flash 分块尺寸：BM=64 行 Q / BN=128 列 KV（官方建议 tile ≤8192 元素；
 * S=64×128×4B=32KB、acc=64×256×4B=64KB 栈上驻留，L1/L2 友好） */
#define IME2_FLASH_BM    32
#define IME2_FLASH_BN    512
#define IME2_FLASH_MAXEV 256

/* ---------------- RVV 向量辅助（VLEN=1024；显式 vsetvl，杜绝 VL 越界） ---------------- */

#include <riscv_vector.h>

/* s[j] = exp(s[j] - mx)，j < n。x≤0 恒成立（mx 是行 max）。
 * 2^k 整数位构造 + [-ln2/2, ln2/2] 上 5 阶 Taylor（最大相对误差 ~2e-6，
 * 远低于 fp16 打包舍入 5e-4）；x < -87 钳位（exp 对 fp32 已下溢为 0）。 */
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
        x = __riscv_vfmax_vf_f32m8(x, -87.0f, vl);
        vint32m8_t ni = __riscv_vfcvt_x_f_v_i32m8(__riscv_vfmul_vf_f32m8(x, log2e, vl), vl); /* 默认 RNE */
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
        __riscv_vse32_v_f32m8(s + j, y, vl);
        j += vl;
    }
}

/* 行 max（含已加 mask 的分数） */
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

/* 行求和（无序归约；只进 l_i 归一化，不进逐元素结果） */
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

/* acc 行 *= alpha */
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

/* 行加（mask）：a[j] += b[j] */
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

/* 单头 flash 前向：QT/KT/VT 为打包 tiles（Q 已预缩放），mask 可为 0（非 causal 时必传）。
 * Out: [L, Ev] fp32 行距 ldo。返回 0 成功；-1 表示 Ev 超栈上预算（调用方走旧路径）。 */
static int ime2_flash_head_tiles(const ime2_fp16* QT, const ime2_fp16* KT, const ime2_fp16* VT,
                                 const float* mask, size_t ldm, float* Out, size_t ldo,
                                 int L, int dst, int E, int Ev, int past, int causal, float scale,
                                 ncnn::Mat& PT, ncnn::Allocator* alloc)
{
    if (Ev > IME2_FLASH_MAXEV)
        return -1;

    const int ktE = (E + 7) / 8;
    const int ktD = (dst + 7) / 8;
    /* 形状感知分块：BN=512 是为长序列（≥512）选的；序列很短时，softmax/P 打包/写回
     * 会按 512 列做大量无用功（115 token 时 QK 只有 13.5 MFLOP，却仍按 512 列处理）。
     * 按实际长度选块大小，保持连续（供 PT 的 tile 布局使用）。 */
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

        /* causal：第 m0..m0+rows-1 行的可见右端 = past+m0+rows-1（含），块只需扫到这里 */
        const int nend = causal ? (past + m0 + rows < dst ? past + m0 + rows : dst) : dst;

        for (int n0 = 0; n0 < nend; n0 += BN)
        {
            const int cols = BN < nend - n0 ? BN : nend - n0;

            /* S = scale · Q_blk · K_blk^T：QT **不**预乘 scale（fp16 打包前乘
             * scale 会把本来可表示的 Q 值溢出成 inf），改在 fp32 累加结果的
             * 写回上乘 alpha=scale —— 与非 flash 的 IME2 路径及上游回退一致。 */
            ime2_gemm_fp32out_p(QT + (size_t)(m0 / 8) * ktE * 64,
                                KT + (size_t)(n0 / 8) * ktE * 64,
                                S, BN, rows, cols, E, scale, 0,
                                (size_t)ktE * 64, (size_t)ktE * 64);

            /* 仅对角区域（或任意 mask 的兜底）需要加 mask */
            if (!causal || n0 + cols > past + m0)
            {
                for (int r = 0; r < rows; r++)
                    ime2_row_add(S + r * BN, mask + (size_t)(m0 + r) * ldm + n0, cols);
            }

            /* online softmax：行 max → exp → 行 sum → rescale 系数（RVV 向量化） */
            for (int r = 0; r < rows; r++)
            {
                float* s = S + r * BN;
                const float bmax = ime2_row_max(s, cols);
                const float mx = mi[r] > bmax ? mi[r] : bmax;
                const float alpha = expf(mi[r] - mx); /* 首块 mi=-FLT_MAX → alpha=0 */
                ime2_row_scale(acc + r * Ev, Ev, alpha);
                ime2_row_exp(s, cols, mx);
                li[r] = li[r] * alpha + ime2_row_sum(s, cols);
                mi[r] = mx;
            }

            /* P 块转 fp16 tiles（零填充到 8 的倍数；多填的 K 列对应 P=0，PV 无贡献） */
            if (ime2_pack_fp32(S, BN, PT, rows, cols, alloc) != 0)
                return -100;

            /* acc += P · V_blk（V^T tiles 的 k 块偏移 n0/8，行距 ktD） */
            const int ktP = (cols + 7) / 8;
            ime2_gemm_fp32out_p((const ime2_fp16*)PT, VT + (size_t)(n0 / 8) * 64,
                                acc, Ev, rows, Ev, ktP * 8, 1.f, 1,
                                (size_t)ktP * 64, (size_t)ktD * 64);
        }

        /* 扫完才归一化（一次乘法） */
        for (int r = 0; r < rows; r++)
        {
            const float inv = 1.f / li[r];
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
