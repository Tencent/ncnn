// SpacemiT K3 A100 IME2 (smt.vfwmadot) fp16 GEMM path for ncnn Gemm layer.
//
// 背景：
//   - K3 的 A100 簇 (cpu8-15, VLEN=1024) 支持 IME2 指令 smt.vfwmadot:
//       C[8x8]fp32 += A[8x8]fp16 x B^T[8x8]fp16   (即 C[i][j] += sum_k A[i][k]*B[j][k])
//   - 这正是 ncnn Gemm 在 transB=1 时的语义。
//   - 单 A100 核实测 ~77 GFLOP/s fp16（见 k3-ncnn/ime/ime2_gemm_fp16_2x2.c）。
//
// 约束（务必遵守）：
//   - 只能在 A100 簇上运行（X100 上 vfwmadot 是 SIGILL）。
//     进程必须整体钉在 A100（例如 ai-run），因为 ncnn 加载期按 vlenb 打包权重，
//     跨簇运行会静默算错。
//   - 仅支持：transA=0, transB=1, 无 C 项, alpha=1, 无 output_transpose/N1M。
//     不满足时回退到 ncnn 原有路径。
//
// 本文件由 gemm_riscv.cpp 在末尾 #include（沿用 ncnn 的 gemm_fp16s.h 风格）。

#ifndef GEMM_RISCV_IME2_H
#define GEMM_RISCV_IME2_H

#include "platform.h"
#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <string.h>
#if defined __linux__
#include <sched.h>
#endif
#include <stdio.h>

#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2 && __riscv_v

namespace ncnn_ime2 {

typedef __fp16 ime2_fp16;

/* ---------------- 运行时探测 ---------------- */

/* 跳转上下文必须是**线程局部**的：SIGILL 处理器是进程级的，若在探测窗口内
 * 另一个线程因自己的原因收到 SIGILL，处理器会在那个线程上运行；若无条件
 * siglongjmp 到本线程保存的 jb，就是跨线程 longjmp（UB，会同时毁掉两个线程
 * 的栈）。非探测线程的信号要交还给此前的处置。g_ime2_old 只在探测锁内
 * 安装/恢复，读取它的异线程处理器与安装之间不存在长时间窗口。 */
static __thread sigjmp_buf tl_ime2_jb;
static __thread int tl_ime2_armed = 0;
static struct sigaction g_ime2_old;

static void ime2_sigill(int sig, siginfo_t* info, void* ucontext)
{
    if (tl_ime2_armed)
        siglongjmp(tl_ime2_jb, 1);

    /* 不是探测线程的 SIGILL：按旧处置转交。
     * - SIG_DFL/SIG_IGN：恢复后重发（它们不读 siginfo，合成信号无损失）；
     * - 自定义处理器：直接以**原始** siginfo/ucontext 调用——raise 会合成
     *   新信号，丢掉 si_code/si_addr，模拟器类回调拿不到故障指令地址。
     *   同时手动补齐内核语义：SA_RESETHAND 先把处置复位成 DFL（内核会在
     *   交付前做同样的事）；sa_mask 里的信号在回调期间屏蔽（当前线程本就
     *   处于 SIGILL 上下文，SIGILL 自身已被内核阻塞，除非 SA_NODEFER）。
     *   SA_ONSTACK 的栈切换无法在转交时补齐，记录在案（SIGILL 用它极罕见）。 */
    if (g_ime2_old.sa_handler == SIG_DFL || g_ime2_old.sa_handler == SIG_IGN)
    {
        sigaction(SIGILL, &g_ime2_old, NULL);
        raise(sig);
        return;
    }
    if (g_ime2_old.sa_flags & SA_RESETHAND)
    {
        struct sigaction dfl;
        memset(&dfl, 0, sizeof dfl);
        dfl.sa_handler = SIG_DFL;
        sigaction(SIGILL, &dfl, NULL);
    }
    sigset_t oldmask;
    pthread_sigmask(SIG_BLOCK, &g_ime2_old.sa_mask, &oldmask);
    if (g_ime2_old.sa_flags & SA_SIGINFO)
        g_ime2_old.sa_sigaction(sig, info, ucontext);
    else
        g_ime2_old.sa_handler(sig);
    pthread_sigmask(SIG_SETMASK, &oldmask, NULL);
}

#if defined __linux__
/* 异构双簇（K3：A100 的 VLEN=1024，X100 的 VLEN=512）下，"探测时所在的核对"与
 * "前向时 OpenMP 工人实际落到的核对"可能不同：前者恰好是 A100 就把 true 缓存
 * 下来，后者落到 X100 就吃 SIGILL。所以探测要覆盖 affinity mask 里的**每一个**
 * 核对：逐个把调用线程移过去读 vlenb，128B（1024bit）才算数。 */
static int ime2_probe_affinity(int nthreads)
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
static int ime2_probe_affinity(int /*nthreads*/)
{
    return 0;
}
#endif // __linux__

static int ime2_probe_locked(int nthreads)
{
    /* (a) 亲和性：本进程所有允许核对 + forward 团队的每个工人的 vlenb 都必须是 1024bit */
    if (!ime2_probe_affinity(nthreads))
        return 0;

    /* (b) 指令本身：SIGILL 保护下执行一条零向量 vfwmadot。
     * 处理器是进程级的，但跳转上下文是**本线程**的 tl_ime2_jb：别的线程在
     * 此窗口内收到 SIGILL 会被 ime2_sigill 转交给原处置，不会跨线程跳转。 */
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = ime2_sigill;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGILL, &sa, &g_ime2_old);

    int ok = 0;
    if (sigsetjmp(tl_ime2_jb, 1) == 0)
    {
        /* 上下文就绪后才武装：异步 SIGILL 先于武装到来会走转交路径 */
        tl_ime2_armed = 1;
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
    tl_ime2_armed = 0;
    /* 仅当当前处理器仍是我们的时才恢复旧处置：若探测窗口内发生过转发
     * （恢复旧处置+重发），当前的已经是旧处置（或已被内核按 RESETHAND
     * 复位成 DFL）——此时再装回去会把一次性处理器错误地重新武装。 */
    {
        struct sigaction cur;
        memset(&cur, 0, sizeof cur);
        sigaction(SIGILL, NULL, &cur);
        if (cur.sa_sigaction == ime2_sigill)
            sigaction(SIGILL, &g_ime2_old, NULL);
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
    return ime2_probe_locked(nthreads);
}

/* 一次性测定团队可能落到的核对里哪些是 A100（vlenb==128）。硬件属性，
 * 进程级缓存。测定范围必须取**团队所有工人掩码的并集**，而不能只看调用
 * 线程：OMP_PROC_BIND / GOMP_CPU_AFFINITY 下初始线程的掩码常只有自己的
 * place，单看它会把其他 A100 工人误判为不可用，等于静默关闭 IME2（性能
 * 回归）。若测定时团队未覆盖全部 A100 核对，得到的集合是真实集合的子集
 * ——偏差方向保守（只会误关，不会误放）。 */
#if defined __linux__
static int ime2_a100_set(int nthreads, cpu_set_t* out)
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
    /* 1) 收集并集（区域内只做 sched_getaffinity，无特权指令，随处可跑） */
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
 * 一次 sched_getaffinity + 位测试，µs 级，相对每层 GEMM 可忽略。
 * 返回 0 时调用方回落非 IME2 路径，结果语义不变。 */
static int ime2_team_ok(int nthreads)
{
#if defined __linux__
    cpu_set_t a100;
    if (!ime2_a100_set(nthreads, &a100))
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

/* 权重值域检查：|w| > 65504 打包成 ±inf 后，IME2 路径（含 VLEN 无关回落——
 * 它也从 tile 读权重）就此失真。这种层不武装 IME2（回落到上游管线；上游
 * fp16s 存储对越界权重的语义由上游自担）。fp16/fp32 源都要扫。 */
static int ime2_weights_fit_fp16(const ncnn::Mat& B)
{
    const size_t n = B.total();
    if (B.elembits() == 16)
    {
        /* fp16 源的有限值天然在界内（至多 65504）；±inf 打包后仍 inf，
         * 与上游 fp16s 语义一致，不视为越界。 */
        return 1;
    }
    const float* p = (const float*)B;
    size_t i = 0;
    const size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vmax = __riscv_vfmv_v_f_f32m1(0.f, vlmax);
    while (i < n)
    {
        const size_t vl = __riscv_vsetvl_e32m1(n - i);
        vmax = __riscv_vfmax_vv_f32m1_tu(vmax, vmax, __riscv_vfabs_v_f32m1(__riscv_vle32_v_f32m1(p + i, vl), vl), vl);
        i += vl;
    }
    const float mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(vmax, __riscv_vfmv_v_f_f32m1(0.f, 1), vlmax));
    return mx <= 65504.f ? 1 : 0;
}

/* ---------------- 打包 ---------------- */

/* B: [N,K]（fp16 或 fp32, elempack=1）-> 8x8 tile 连续布局 [nt][kt][64] fp16 */
/* tile 行距填充：tile 之间原本相隔 kt*64 个 fp16（K=1024 时正好 16KB，2 的幂），
 * 相邻 tile 行会映射到同一 L1 组造成冲突缺失。微基准实测：行距 +256B 后内核
 * 200.9 -> 252.9 GFLOP/s/核（+26%）；+16/32/64B 无效或更差（见 REPORT 7.16）。 */
#define IME2_TILE_PAD    128 /* 单位：fp16 元素（= 256B） */
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

/* A: [M,K] fp16（elempack 任意）-> 8x8 tile 连续布局 [mt][kt][64] fp16
 * ncnn 打包布局：元素 (m,k) 在 ((m/p)*K + k)*p + m%p */
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
    /* 3-D A 的通道间行距（按元素计）是 cstep*p：Mat 把每面 padding 到 16B，
     * cstep 记录的是 pack-group 数。例如 elempack==1 且 K==9 时 cstep=16，
     * 按 K*p 推进就会从第二行起读进 padding。与上游 RVV packer 同一惯例。 */
    const size_t as = A.dims == 3 ? (size_t)A.cstep * p : (size_t)K * p;

    // 按 tile 行并行：prefill 下 pack_A 是纯数据搬运（每次调用 ~1MB），
    // 单线程时约占 GEMM 墙钟的 6%（196 次调用累计 ~78ms）。
    /* M=1（解码）时 mt=1：为一行数据开 8 线程的 omp 区域是纯开销。
     * 实测（NCNN_IME2_I8_PROF）驱动前缀 105µs/调用 × 196 次 = 20.6ms/token（占解码 24%），
     * 主要就来自这里。 ⇒ 只有 tile 行数足够多时才并行。
     * 注：早先加过同样的保护，但当时被一次"有后台进程并发"的污染测量误判为无收益而回退。 */
    /* 按**工作量**（tile 行数 × k 块数）决定是否并行，而不是只看 mt：
     * M=1（解码）时 1×kt ≈ 128 ⇒ 串行；M=27（短预填充）时 4×128 = 512 ⇒ 应当并行
     * （曾用 mt>=8 判定，把 27-token 预填充的打包也串行化，实测掉 14%。） */
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
            // 快路径：整块在界内。
            // p == 1 时一行的 8 个 k 连续，直接拷贝 16 字节。
            // p > 1 时 ncnn 的布局是 [m/p][k][p]：同一 m 行相邻 k 之间隔着 p 个 fp16，
            // 紧邻的 8 个 fp16 是同一 k 列上相邻的 pack lane，也就是**其他 m 行**，
            // 因此必须按 stride p 逐个 gather，不能整块 memcpy。
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

/* ---------------- 内层内核 ---------------- */

/* 单个 8x8 输出 tile：Ctmp(64 fp32) = At(8x8) x Bt(8x8)^T，沿 K 循环 kt 次 */
/* X100（VLEN=256）专用 8x8 输出 tile 内核：纯 RVV 内建，不用 IME2 指令。
 * 已由 bench/ime2_tile_verify.cpp 与"按打包布局逐元素算的标量参考"对拍通过：
 *   标量参考 = IME2 内核 = 本内核（max|d| 分别 1.5e-8 / 5.2e-8）
 * 结论：本内核的 tile 索引与 IME2 内核一致；双簇分工此前的失败在"接线"而非内核。
 * 目前未接线（接线需要把 16x16 块体抽成具名函数，见 REPORT 7.33）。 */
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
    /* 流水版：预取 k+1 步的 A/B，同时算当前步 —— 隐藏 L2 取数延迟。
     * 实测同结构比非流水快 ~2x（微基准 119->195 GFLOP/s/核）。 */
    int k = kt - 1;
    __asm__ volatile(
        "vsetvli t0, zero, e32, m2, tu, mu\n\t"
        "vmv.v.i v16, 0\n\t"
        "vsetvli t0, zero, e16, m1, tu, mu\n\t"
        "vle16.v v2, (%[A])\n\t"
        "vle16.v v8, (%[B])\n\t"
        "addi %[A], %[A], 128\n\t"
        "addi %[B], %[B], 128\n\t"
        "beqz %[k], 2f\n\t" /* kt==1 保护 */
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
    /* 16x16 输出 = 4 个 8x8 MICRO tile；K 方向按 2 步展开 + 寄存器轮转：
     * 每一步的取数提前一步发出，且**不再用 vmv 把预取寄存器搬到当前寄存器**
     * （旧版每步 4 条 vmv 正好落在 MAC 的依赖关键路径上）。
     * 微基准（K=1024）：填充前 199 / 仅填充 210 / 填充+展开2 **373** GFLOP/s/核。 */
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
        /* 相位 A：取 k+1 -> v12..v15；同时对 k 做 4 条 MAC */
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
        /* 相位 B：取 k+2 -> v2..v10；同时对 k+1 做 4 条 MAC */
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
        /* kt 为奇数时，最后一步的数据已在 v2/v4/v8/v10 */
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

/* 2x4 寄存器分块：一次算 16(M)x32(N)。k+1 预取流水（全部 vmv 在 8 个 MAC 之后，
 * 独立预装载寄存器 —— 早前移 vmv 会让 c2/c3 列用到下一步的 A 操作数，已对拍验证）。 */
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
        "beqz %[k], 2f\n\t" /* kt==1 保护 */
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

/* ---------------- 完整 GEMM ---------------- */

/* M=1（解码 GEMV）专用内核：一次 k 循环并行推进 4 个 N-tile（32 列输出），
 * 4 条独立累加链 + k+1 软件预取（A 与 4 路 B 双缓冲）。
 * 背景：M=1 时 8x8 内核对 512 列权重只能跑 11.9 GB/s，而 M=8（4 链）能到 23.6 GB/s ——
 * 解码 GEMM 卡在 vfwmadot 取数延迟上，这里用多链 + 预取把延迟藏掉。
 * 结果只取每个 8x8 tile 的第 0 行（M=1），写进 row[32]（tile t -> row[t*8..t*8+8)）。 */
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
    /* scale 必须用 fp32 存：amax/127 在 amax < ~7.6e-6 时会下溢成 fp16 的 0，
     * 而面板是按 fp32 的 sc 量化的非零值——scale 读回 0 会把整列点积抹成 0。 */
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

/* 把一行 fp16 激活量化成 int8（整行一个 scale），供整数域点积使用。 */
static float ime2_quantize_a_int8(const ime2_fp16* a, int K, signed char* a8, int nT)
{
    /* K 通常只有 1024：开 omp 区域（每次调用 2 个）比串行做还慢，
     * 而解码每 token 有 ~196 次 Gemm 调用 ⇒ 这里必须串行。 */
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

/* 整数域 M=1 GEMV：每列一个 int32 点积（128 个 MAC / 向量指令），最后乘两侧 scale。 */
static void ime2_gemv_m1_i8c(const signed char* a8, float a_scale,
                             const signed char* W, const float* wscale,
                             float* out, int n0, int n1, int K)
{
    /* 冷缓存微基准（bench/i8_gemv_cold_bench.c，50MB 工作集远超 LLC）：
     *   e8m1（旧）12.99 GB/s → e8m2 16.54 → e8m2+预取 16.99 GB/s（+31%）。
     * 解码时每层的 int8 面板只读一遍、总量 0.46GB 全走 DRAM，所以必须按"冷"条件调优。*/
    const size_t vlmax = __riscv_vsetvlmax_e32m8();
    for (int n = n0; n < n1; n++)
    {
        const signed char* w = W + (size_t)n * K;
        vint32m8_t acc = __riscv_vmv_v_x_i32m8(0, vlmax);
        int k = 0;
        while (k < K)
        {
            const size_t vl = __riscv_vsetvl_e8m2((size_t)(K - k));
            __builtin_prefetch(w + k + 512); /* 提前 512B 拉入缓存（zicbop） */
            __builtin_prefetch(w + k + 576);
            vint8m2_t av = __riscv_vle8_v_i8m2(a8 + k, vl);
            vint8m2_t wv = __riscv_vle8_v_i8m2(w + k, vl);
            vint16m4_t a16 = __riscv_vwadd_vx_i16m4(av, 0, vl);
            vint16m4_t w16 = __riscv_vwadd_vx_i16m4(wv, 0, vl);
            /* _tu：K 不是 256 的整数倍（VLEN=1024 上 e32m8 的 vlmax）时最后一个
             * 短 vl 的 vwmacc 尾部是 agnostic，而下面的归约读整个 vlmax 个 lane。 */
            acc = __riscv_vwmacc_vv_i32m8_tu(acc, a16, w16, vl);
            k += (int)vl;
        }
        const int isum = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m8_i32m1(acc, __riscv_vmv_v_x_i32m1(0, 1), vlmax));
        out[n] = a_scale * wscale[n] * (float)isum;
    }
}

static void ime2_gemv_m1_x4(const ime2_fp16* At, const ime2_fp16* B0, size_t bt_stride, float* row, int kt)
{
    /* K 方向展开 2 步 + 寄存器轮转：去掉旧版每步 5 条 vmv（它们位于 MAC 的依赖关键路径上）。
     * 与 16x16 内核同一手法（见 REPORT 7.17）：微基准里流式取数下 +77%。 */
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
        /* 相位 A：取 k+1 -> v3/v12..v15；同时对 k 做 4 条 MAC */
        "vle16.v v3, (%[A])\n\t"
        "vle16.v v12, (%[B0])\n\t vle16.v v13, (%[B1])\n\t vle16.v v14, (%[B2])\n\t vle16.v v15, (%[B3])\n\t"
        "smt.vfwmadot v24, v2, v8\n\t"
        "smt.vfwmadot v26, v2, v9\n\t"
        "smt.vfwmadot v28, v2, v10\n\t"
        "smt.vfwmadot v30, v2, v11\n\t"
        "addi %[A], %[A], 128\n\t"
        "addi %[B0], %[B0], 128\n\t addi %[B1], %[B1], 128\n\t"
        "addi %[B2], %[B2], 128\n\t addi %[B3], %[B3], 128\n\t"
        /* 相位 B：取 k+2 -> v2/v8..v11；同时对 k+1 做 4 条 MAC */
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
        /* 只取每个 8x8 tile 的第 0 行 */
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

/* 一个 16x16 输出块（IB x JB 个 tile 对）的计算 + 写回。
 * 抽成具名函数的原因：单簇路径与双簇（A100+X100）路径必须共用同一份实现。 */
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

            /* 散射 + fp32->fp16 转换，写进 ncnn 打包布局。
             * p==8（A100 fp16 常态）：同一 8 行 tile 恰是一个 elempack 组，
             * 沿 m 方向 16B 连续写（vlse32 跨步读临时 tile + vfncvt + vse16），
             * 行基址只算一次（除掉每元素 gm/p、gm%p 的除法）。
             * p==1：沿 n 方向行内连续写。 */
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
                                /* vl=8 显式（VLEN=1024 下 vsetvli zero 会是 32，越界！） */
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

    /* ---- int8 权重模式 + M=1：**GEMV 免打包** ----
     * A 是 [1,K] 连续的 fp16 行，整数域内核只需要"量化后的激活 + 每列权重"，
     * 完全不需要 IME2 的 tile 打包。实测 pack_A 在 M=1 时要 105µs/调用
     * （196 次/token = 20.6ms，占解码约 24%），这里直接省掉。 */
    /* 注意 A.elempack == 1 是必要条件：打包布局（elempack = 8/16）下同一行的 K 个值
     * 是跨步存放的（[m/p][k][p]），直接当连续数组读会算错。打包输入仍走下面的通用路径。 */
    if (!BT_i8c.empty() && !BT_wscale.empty() && M == 1 && p == 1 && A.elembits() == 16 && A.elempack == 1)
    {
        const signed char* w8c = (const signed char*)BT_i8c;
        const float* wsc = (const float*)BT_wscale;
        static const int IME2_I8_STK = 8192;
        signed char a8stk[IME2_I8_STK];
        float ostk[IME2_I8_STK];
        std::vector<signed char> a8heap;
        std::vector<float> oheap;
        signed char* a8p = a8stk;
        float* op = ostk;
        if (K > IME2_I8_STK)
        {
            a8heap.resize(K);
            a8p = a8heap.data();
        }
        if (N > IME2_I8_STK)
        {
            oheap.resize(N);
            op = oheap.data();
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
                    fprintf(stderr, "[i8-prof] 免打包路径 每调用: 量化=%.1fus 内核=%.1fus 转换=%.1fus 合计=%.1fus\n",
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

    /* 解码（M=1）快速路径：8 条累加链的 GEMV 内核，p 必为 1（M<packn） */
    if (M == 1 && p == 1)
    {
        #pragma omp parallel for num_threads(nT)
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
            /* 64 个 fp32 -> fp16 连续写 128B（vl=64：VLEN=1024 下 e32m2 上限正好 64） */
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
        return 0;
    }

    (void)opt;

    /* 两级分块：把 (i,j) 划成 IB x JB 的输出块（每块 8x8 个 16x16 tile = 128x128 输出），
     * 块内先 j 后 i 双重循环，使 A 块（128 行 x K）与 B 块（128 列 x K）都能驻留 L2 并被
     * 块内 IB*JB 次 tile 计算复用。
     * 背景：无分块时每个 j 都要重扫整个 A（1MB x nt/2 次 = 上百 MB 的 L2 流量），
     * 实测 GEMM 只有 58 GFLOP/s/核 ≈ 硬件 IME2 能力的 6%（8 核并发实测 925 GFLOP/s/核）。
     * 并行按块划分（collapse(2) 外层为 jb），每个线程只碰自己那一列块。 */
    const int IB = 8, JB = 8; // 单位是 16x16 tile 对（i/j 步长 2）
    /* mt/nt count 8-wide tiles, so the number of 16-wide tile PAIRS is a ceiling:
     * with an odd mt the last tile still has to be computed, and M <= 8 must still
     * produce one pair (mt == 1 gives mt/2 == 0 and the block loop would never
     * run). Floor division here silently dropped those output rows/columns. */
    const int mti = (mt + 1) / 2;
    const int nti = (nt + 1) / 2;
    const int nib = (mti + IB - 1) / IB;
    const int njb = (nti + JB - 1) / JB;
    /* 异构双簇分工（A100 IME2 + X100 RVV）已实验并否决 —— 平台限制，不是代码问题：
                                                                                                                                                                             *  (1) 两簇并发确实互不干扰（两个进程分别绑核：A100 374 + X100 120 = 490 t/s，潜力 +31%）；
                                                                                                                                                                             *  (2) 但 /proc/set_ai_thread 注册（A100 上拿到 VLEN=1024 的前提）会把进程**限制在 A100 簇内**：
                                                                                                                                                                             *      实测混合运行时的所有线程都落在 cpu8-15；不用注册直接 taskset 到 8-15 则直接失败。
                                                                                                                                                                             * 所以"单进程内双簇分工"在本平台上不可行，要兑现 +31% 只能走**多进程**方案。
                                                                                                                                                                             * 详见 REPORT 7.34；X100 的 RVV tile 内核（ime2_rvv_tile_8x8，已对拍通过）保留备用。 */
    #pragma omp parallel for num_threads(nT) collapse(2)
    for (int jbb = 0; jbb < njb; jbb++)
    {
        for (int ibb = 0; ibb < nib; ibb++)
        {
            ime2_compute_block_16x16(at, bt, out, ibb, jbb, mt, nt, M, N, ldat, ldt, kt, p, alpha);
        }
    }
    (void)nt;
    return 0;
}

} // namespace ncnn_ime2

#endif // NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2 && __riscv_v
#endif // GEMM_RISCV_IME2_H
