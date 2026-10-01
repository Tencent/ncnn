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

#include <math.h>
#include <setjmp.h>
#include <signal.h>
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
        for (int j = 0; j < kt; j++)
        {
            ime2_fp16* t = dst + ((size_t)i * kt + j) * 64;
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                {
                    const int gr = i * 8 + r, gk = j * 8 + c;
                    float v = 0.f;
                    if (gr < R && gk < K)
                        v = src[(size_t)gr * ld + gk];
                    t[r * 8 + c] = (ime2_fp16)v;
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
        for (int j = 0; j < kt; j++)
        {
            ime2_fp16* t = dst + ((size_t)i * kt + j) * 64;
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++)
                {
                    const int gr = i * 8 + r, gk = j * 8 + c;
                    float v = 0.f;
                    if (gr < R && gk < K)
                        v = src[(size_t)gk * ld + gr];
                    t[r * 8 + c] = (ime2_fp16)v;
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

} // namespace ncnn_ime2_sdpa

#endif // NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2 && __riscv_v
#endif // SDPA_RISCV_IME2_H
