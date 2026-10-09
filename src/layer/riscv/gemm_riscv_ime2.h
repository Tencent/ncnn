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

#include <math.h>
#include <setjmp.h>
#include <signal.h>
#include <string.h>
#include <stdio.h>

#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2 && __riscv_v

namespace ncnn_ime2 {

typedef __fp16 ime2_fp16;

/* ---------------- 运行时探测 ---------------- */

static sigjmp_buf g_ime2_jb;
static void ime2_sigill(int)
{
    siglongjmp(g_ime2_jb, 1);
}

/* 返回 1 = 当前核可执行 smt.vfwmadot（A100）；0 = 不可（X100 或其它） */
static int ime2_probe(void)
{
    static int done = 0, result = 0;
    if (done)
        return result;
    done = 1;

    unsigned long vlenb = 0;
    __asm__ volatile("csrr %0, vlenb"
                     : "=r"(vlenb));
    if (vlenb != 128) /* IME2 的 fp16 矩阵单元只在 VLEN=1024 的 A100 上 */
        return 0;

    struct sigaction sa, old;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = ime2_sigill;
    sigaction(SIGILL, &sa, &old);

    if (sigsetjmp(g_ime2_jb, 1) == 0)
    {
        __asm__ volatile(
            "vsetvli t0, zero, e32, m2, tu, mu\n\t" /* e32m2 下清零累加器 v16:v17 */
            "vmv.v.i v16, 0\n\t"
            "vsetvli t0, zero, e16, m1, tu, mu\n\t" /* 必须回到 LMUL=1 再执行 vfwmadot */
            "vmv.v.i v0, 0\n\t"
            "vmv.v.i v1, 0\n\t"
            "smt.vfwmadot v16, v0, v1\n\t"
            :
            :
            : "t0", "v0", "v1", "v16", "v17", "memory");
        result = 1;
    }
    sigaction(SIGILL, &old, NULL);
    return result;
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
static int ime2_pack_A(const ncnn::Mat& A, ncnn::Mat& AT, int M, int K, int nT = 1)
{
    const int mt = (M + 7) / 8, kt = (K + 7) / 8;
    const size_t ldat = IME2_TILE_LD(kt);
    AT.create((size_t)mt * ldat, (size_t)2u, 1, (ncnn::Allocator*)0);
    if (AT.empty())
        return -100;
    ime2_fp16* d = (ime2_fp16*)AT;

    const ime2_fp16* a = (const ime2_fp16*)A;
    const int p = A.elempack;

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
            // 快路径：整块在界内 —— 每行 16 字节直接拷贝。
            // 任意 elempack 都适用：tile 行 r 对应的 8 个 fp16 在 ncnn 打包布局中同样连续，
            // 源地址按 (gm/p, gm%p) 逐行计算即可。
            if (gm0 + 8 <= M && gk0 + 8 <= K)
            {
                for (int r = 0; r < 8; r++)
                {
                    const int gm = gm0 + r;
                    const ime2_fp16* src = (p == 1) ? (a + (size_t)gm * K + gk0)
                                           : (a + ((size_t)(gm / p) * K + gk0) * p + gm % p);
                    memcpy(t + r * 8, src, 16);
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
                                v = a[(size_t)gm * K + gk];
                            else
                                v = a[((size_t)(gm / p) * K + gk) * p + gm % p];
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
    BT_wscale.create((size_t)N, (size_t)2u, 1, (ncnn::Allocator*)0);
    if (BT_i8.empty() || BT_wscale.empty())
        return;

    signed char* dst = (signed char*)BT_i8;
    ime2_fp16* ws = (ime2_fp16*)BT_wscale;

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
        ws[n] = (ime2_fp16)sc;
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
                             const signed char* W, const ime2_fp16* wscale,
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
            acc = __riscv_vwmacc_vv_i32m8(acc, a16, w16, vl);
            k += (int)vl;
        }
        const int isum = __riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m8_i32m1(acc, __riscv_vmv_v_x_i32m1(0, 1), vlmax));
        out[n] = a_scale * (float)wscale[n] * (float)isum;
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
        const ime2_fp16* wsc = (const ime2_fp16*)BT_wscale;
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
    if (ime2_pack_A(A, AT, M, K, nT) != 0)
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
            const int n = jn * 8;
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
    const int nib = (mt / 2 + IB - 1) / IB;
    const int njb = (nt / 2 + JB - 1) / JB;
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
