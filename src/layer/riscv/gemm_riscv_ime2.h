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
static int ime2_pack_B(const ncnn::Mat& B, ncnn::Mat& BT, int N, int K)
{
    const int nt = (N + 7) / 8, kt = (K + 7) / 8;
    BT.create((size_t)nt * kt * 64, (size_t)2u, 1, (ncnn::Allocator*)0);
    if (BT.empty())
        return -100;
    ime2_fp16* dst = (ime2_fp16*)BT;

    const int is_fp16 = B.elembits() == 16;

    for (int i = 0; i < nt; i++)
        for (int j = 0; j < kt; j++)
        {
            ime2_fp16* t = dst + ((size_t)i * kt + j) * 64;
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
static int ime2_pack_A(const ncnn::Mat& A, ncnn::Mat& AT, int M, int K)
{
    const int mt = (M + 7) / 8, kt = (K + 7) / 8;
    AT.create((size_t)mt * kt * 64, (size_t)2u, 1, (ncnn::Allocator*)0);
    if (AT.empty())
        return -100;
    ime2_fp16* d = (ime2_fp16*)AT;

    const ime2_fp16* a = (const ime2_fp16*)A;
    const int p = A.elempack;

    for (int i = 0; i < mt; i++)
    {
        const int gm0 = i * 8;
        for (int j = 0; j < kt; j++)
        {
            ime2_fp16* t = d + ((size_t)i * kt + j) * 64;
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
static void ime2_tile_8x8(const ime2_fp16 *At, const ime2_fp16 *Bt, float *Ctmp, int kt)
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
		"smt.vfwmadot v16, v2, v8\n\t"
		"vsetvli t0, zero, e32, m2, tu, mu\n\t"
		"vse32.v v16, (%[C])\n\t"
		: [A] "+r"(At), [B] "+r"(Bt), [k] "+r"(k)
		: [C] "r"(Ctmp)
		: "t0", "v2", "v8", "v12", "v13", "v16", "v17", "memory", "cc");
}

static void ime2_tile_16x16(const ime2_fp16 *At, const ime2_fp16 *Bt,
                            float *C00, float *C10, float *C01, float *C11, int kt)
{
	const ime2_fp16 *A1 = At + (size_t)kt * 64;
	const ime2_fp16 *B1 = Bt + (size_t)kt * 64;
	int k = kt - 1;
	__asm__ volatile(
		"vsetvli t0, zero, e32, m2, tu, mu\n\t"
		"vmv.v.i v16, 0\n\t vmv.v.i v18, 0\n\t vmv.v.i v20, 0\n\t vmv.v.i v22, 0\n\t"
		"vsetvli t0, zero, e16, m1, tu, mu\n\t"
		"vle16.v v2, (%[A0])\n\t vle16.v v8, (%[B0])\n\t vle16.v v4, (%[A1])\n\t vle16.v v10, (%[B1])\n\t"
		"addi %[A0], %[A0], 128\n\t addi %[A1], %[A1], 128\n\t addi %[B0], %[B0], 128\n\t addi %[B1], %[B1], 128\n\t"
		"1:\n\t"
		"vle16.v v12, (%[A0])\n\t"
		"smt.vfwmadot v16, v2, v8\n\t"
		"vle16.v v13, (%[B0])\n\t"
		"smt.vfwmadot v18, v4, v8\n\t"
		"vle16.v v14, (%[A1])\n\t"
		"smt.vfwmadot v20, v2, v10\n\t"
		"vle16.v v15, (%[B1])\n\t"
		"smt.vfwmadot v22, v4, v10\n\t"
		"vmv.v.v v2, v12\n\t vmv.v.v v8, v13\n\t vmv.v.v v4, v14\n\t vmv.v.v v10, v15\n\t"
		"addi %[A0], %[A0], 128\n\t addi %[A1], %[A1], 128\n\t"
		"addi %[B0], %[B0], 128\n\t addi %[B1], %[B1], 128\n\t"
		"addi %[k], %[k], -1\n\t bnez %[k], 1b\n\t"
		"smt.vfwmadot v16, v2, v8\n\t smt.vfwmadot v18, v4, v8\n\t"
		"smt.vfwmadot v20, v2, v10\n\t smt.vfwmadot v22, v4, v10\n\t"
		"vsetvli t0, zero, e32, m2, tu, mu\n\t"
		"vse32.v v16, (%[C00])\n\t vse32.v v18, (%[C10])\n\t vse32.v v20, (%[C01])\n\t vse32.v v22, (%[C11])\n\t"
		: [A0] "+r"(At), [A1] "+r"(A1), [B0] "+r"(Bt), [B1] "+r"(B1), [k] "+r"(k)
		: [C00] "r"(C00), [C10] "r"(C10), [C01] "r"(C01), [C11] "r"(C11)
		: "t0", "v2", "v4", "v8", "v10", "v12", "v13", "v14", "v15",
		  "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
		  "memory", "cc");
}

/* 2x4 寄存器分块：一次算 16(M)x32(N)。k+1 预取流水（全部 vmv 在 8 个 MAC 之后，
 * 独立预装载寄存器 —— 早前移 vmv 会让 c2/c3 列用到下一步的 A 操作数，已对拍验证）。 */
static void ime2_tile_16x32(const ime2_fp16 *At, const ime2_fp16 *Bt,
                            float *C00, float *C10,
                            float *C01, float *C11,
                            float *C02, float *C12,
                            float *C03, float *C13, int kt)
{
	const ime2_fp16 *A1 = At + (size_t)kt * 64;
	const ime2_fp16 *B1 = Bt + (size_t)kt * 64, *B2 = B1 + (size_t)kt * 64, *B3 = B2 + (size_t)kt * 64;
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

/* C[out_elempack 感知的 ncnn 打包输出] = A[M,K] x BT[N,K]^T，alpha=1 无 C 项。
 * top_blob 已按 w=N, h=M/p, elempack=p 创建好。 */
static int gemm_ime2_fp16(const ncnn::Mat& A, const ncnn::Mat& BT, ncnn::Mat& top_blob,
                          int M, int N, int K, float alpha, int out_elempack, int nT, const ncnn::Option& opt)
{
    const int mt = (M + 7) / 8, nt = (N + 7) / 8, kt = (K + 7) / 8;
    const int p = out_elempack;

    ncnn::Mat AT;
    if (ime2_pack_A(A, AT, M, K) != 0)
        return -100;

    const ime2_fp16* bt = (const ime2_fp16*)BT;
    const ime2_fp16* at = (const ime2_fp16*)AT;
    ime2_fp16* out = (ime2_fp16*)top_blob;

    (void)opt;

    #pragma omp parallel for num_threads(nT) collapse(2)
    for (int i = 0; i < mt; i += 2) {
        for (int j = 0; j < nt; j += 2) {
            float t00[64], t10[64], t01[64], t11[64];

            const bool full2x2 = (i + 1 < mt) && (j + 1 < nt);

            if (full2x2) {
                ime2_tile_16x16(at + (size_t)i * kt * 64, bt + (size_t)j * kt * 64,
                                t00, t10, t01, t11, kt);
            } else {
                ime2_tile_8x8(at + (size_t)i * kt * 64, bt + (size_t)j * kt * 64, t00, kt);
                if (j + 1 < nt)
                    ime2_tile_8x8(at + (size_t)i * kt * 64, bt + (size_t)(j + 1) * kt * 64, t01, kt);
                if (i + 1 < mt)
                    ime2_tile_8x8(at + (size_t)(i + 1) * kt * 64, bt + (size_t)j * kt * 64, t10, kt);
                if (i + 1 < mt && j + 1 < nt)
                    ime2_tile_8x8(at + (size_t)(i + 1) * kt * 64, bt + (size_t)(j + 1) * kt * 64, t11, kt);
            }

            /* 散射 + fp32->fp16 转换，写进 ncnn 打包布局 */
            for (int r = 0; r < 8; r++)
                for (int c = 0; c < 8; c++) {
                    const float *tiles[4] = { t00, t01, t10, t11 };
                    const int dm[4] = { 0, 0, 8, 8 };
                    const int dn[4] = { 0, 8, 0, 8 };
                    for (int q = 0; q < 4; q++) {
                        const int gm = i * 8 + dm[q] + r;
                        const int gn = j * 8 + dn[q] + c;
                        if (gm >= M || gn >= N)
                            continue;
                        const float v = tiles[q][r * 8 + c] * alpha;
                        if (p == 1)
                            out[(size_t)gm * N + gn] = (ime2_fp16)v;
                        else
                            out[((size_t)(gm / p) * N + gn) * p + gm % p] = (ime2_fp16)v;
                    }
                }
        }
    }
    (void)nt;
    return 0;
}

} // namespace ncnn_ime2

#endif // NCNN_ZFH && NCNN_RISCV_SPACEMIT_IME2 && __riscv_v
#endif // GEMM_RISCV_IME2_H
