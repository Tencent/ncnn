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

#include <float.h>
#include <math.h>
#include <chrono>
#include <setjmp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2 && __riscv_v

namespace ncnn_ime2_sdpa {

typedef __fp16 ime2_fp16;

/* ---------------- 运行时探测（A100?） ---------------- */

static sigjmp_buf g_sdpa_ime2_jb;
static void sdpa_ime2_sigill(int)
{
    siglongjmp(g_sdpa_ime2_jb, 1);
}

static int ime2_probe(void)
{
    static int done = 0, result = 0;
    if (done)
        return result;
    done = 1;

    unsigned long vlenb = 0;
    __asm__ volatile("csrr %0, vlenb"
                     : "=r"(vlenb));
    if (vlenb != 128) /* IME2 fp16 矩阵单元只在 VLEN=1024 的 A100 上 */
        return 0;

    struct sigaction sa, old;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = sdpa_ime2_sigill;
    sigaction(SIGILL, &sa, &old);

    if (sigsetjmp(g_sdpa_ime2_jb, 1) == 0)
    {
        /* 真正探测：执行一条 vfwmadot（零向量，无副作用） */
        __asm__ volatile(
            "vsetvli t0, zero, e16, m1, tu, mu\n\t"
            "vmv.v.i v2, 0\n\t"
            "vmv.v.i v8, 0\n\t"
            "smt.vfwmadot v16, v2, v8\n\t"
            :
            :
            : "t0", "v2", "v8", "v16", "v17");
        result = 1;
    }
    else
    {
        result = 0;
    }

    sigaction(SIGILL, &old, 0);
    return result;
}

/* ---------------- 打包：fp32 -> 8x8 fp16 tile 连续布局 ---------------- */

/* src: [R, K] fp32 行主序（行距 ld 元素）-> tiles [rt][kt][64] fp16 */
static int ime2_pack_fp32(const float* src, size_t ld, ncnn::Mat& T, int R, int K, ncnn::Allocator* alloc)
{
    const int rt = (R + 7) / 8, kt = (K + 7) / 8;
    T.create((size_t)rt * kt * 64, (size_t)2u, 1, alloc);
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
    T.create((size_t)rt * kt * 64, (size_t)2u, 1, alloc);
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

/* 带预缩放的打包：fp32 src × scale → fp16 tiles（scale 在 fp32 侧乘，避免双舍入） */
static int ime2_pack_fp32_scale(const float* src, size_t ld, ncnn::Mat& T, int R, int K, float scale, ncnn::Allocator* alloc)
{
    const int rt = (R + 7) / 8, kt = (K + 7) / 8;
    T.create((size_t)rt * kt * 64, (size_t)2u, 1, alloc);
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
                const float* s0 = src + (size_t)gr0 * ld + gk0;
                for (int r = 0; r < 8; r++)
                {
                    const float* s = s0 + (size_t)r * ld;
                    ime2_fp16* d = t + r * 8;
                    for (int c = 0; c < 8; c++)
                        d[c] = (ime2_fp16)(s[c] * scale);
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
                        t[r * 8 + c] = (ime2_fp16)(v * scale);
                    }
            }
        }
    }
    return 0;
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
/* 诊断计时（NCNN_SDPA_PROF=1 时由 sdpa_riscv_zfh.cpp 打印） */
static inline double ime2_prof_now()
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static double g_prof_qk = 0, g_prof_softmax = 0, g_prof_ppack = 0, g_prof_pv = 0, g_prof_store = 0, g_prof_pack = 0;

static int ime2_flash_head_tiles(const ime2_fp16* QT, const ime2_fp16* KT, const ime2_fp16* VT,
                                 const float* mask, size_t ldm, float* Out, size_t ldo,
                                 int L, int dst, int E, int Ev, int past, int causal,
                                 ncnn::Mat& PT, ncnn::Allocator* alloc)
{
    if (Ev > IME2_FLASH_MAXEV)
        return -1;
    const bool prof = (getenv("NCNN_SDPA_PROF") != nullptr);
    struct
    {
        double qk, sm, pp, pv, st;
    } local = {0, 0, 0, 0, 0};
    ncnn::Mat& profPT = PT;
    (void)profPT;

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

            /* S = Q_blk · K_blk^T（QT 已含 scale，alpha=1 覆盖写） */
            const double t_qk0 = prof ? ime2_prof_now() : 0.0;
            ime2_gemm_fp32out_p(QT + (size_t)(m0 / 8) * ktE * 64,
                                KT + (size_t)(n0 / 8) * ktE * 64,
                                S, BN, rows, cols, E, 1.f, 0,
                                (size_t)ktE * 64, (size_t)ktE * 64);

            if (prof) local.qk += ime2_prof_now() - t_qk0;

            /* 仅对角区域（或任意 mask 的兜底）需要加 mask */
            if (!causal || n0 + cols > past + m0)
            {
                for (int r = 0; r < rows; r++)
                    ime2_row_add(S + r * BN, mask + (size_t)(m0 + r) * ldm + n0, cols);
            }

            /* online softmax：行 max → exp → 行 sum → rescale 系数（RVV 向量化） */
            const double t_sm0 = prof ? ime2_prof_now() : 0.0;
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

            if (prof) local.sm += ime2_prof_now() - t_sm0;

            /* P 块转 fp16 tiles（零填充到 8 的倍数；多填的 K 列对应 P=0，PV 无贡献） */
            const double t_pp0 = prof ? ime2_prof_now() : 0.0;
            if (ime2_pack_fp32(S, BN, PT, rows, cols, alloc) != 0)
                return -100;
            if (prof) local.pp += ime2_prof_now() - t_pp0;

            /* acc += P · V_blk（V^T tiles 的 k 块偏移 n0/8，行距 ktD） */
            const int ktP = (cols + 7) / 8;
            if (getenv("IME2_FLASH_DEBUG"))
                fprintf(stderr, "[flash] m0=%d n0=%d rows=%d cols=%d ktP=%d ktD=%d E=%d Ev=%d L=%d dst=%d QT=%p KT=%p VT=%p PT=%p\n",
                        m0, n0, rows, cols, ktP, ktD, E, Ev, L, dst, (const void*)QT, (const void*)KT, (const void*)VT, (const void*)PT);
            const double t_pv0 = prof ? ime2_prof_now() : 0.0;
            ime2_gemm_fp32out_p((const ime2_fp16*)PT, VT + (size_t)(n0 / 8) * 64,
                                acc, Ev, rows, Ev, ktP * 8, 1.f, 1,
                                (size_t)ktP * 64, (size_t)ktD * 64);
            if (prof) local.pv += ime2_prof_now() - t_pv0;
        }

        /* 扫完才归一化（一次乘法） */
        const double t_st0 = prof ? ime2_prof_now() : 0.0;
        for (int r = 0; r < rows; r++)
        {
            const float inv = 1.f / li[r];
            ime2_row_scale(acc + r * Ev, Ev, inv);
            const float* arow = acc + r * Ev;
            float* o = Out + (size_t)(m0 + r) * ldo;
            for (int e = 0; e < Ev; e++)
                o[e] = arow[e];
        }
        if (prof) local.st += ime2_prof_now() - t_st0;
    }
    if (prof)
    {
        #pragma omp atomic
        g_prof_qk += local.qk;
        #pragma omp atomic
        g_prof_softmax += local.sm;
        #pragma omp atomic
        g_prof_ppack += local.pp;
        #pragma omp atomic
        g_prof_pv += local.pv;
        #pragma omp atomic
        g_prof_store += local.st;
    }
    return 0;
}

} // namespace ncnn_ime2_sdpa

#endif // NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2 && __riscv_v
#endif // SDPA_RISCV_IME2_H
