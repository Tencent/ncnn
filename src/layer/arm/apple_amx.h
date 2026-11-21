// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_APPLE_AMX_H
#define NCNN_APPLE_AMX_H

// Apple's undocumented AMX matrix coprocessor instructions
// https://github.com/corsix/amx
//
// The instructions are encoded as a64 reserved instructions, see aarch64.h in the reference repo

#if defined __APPLE__ && __aarch64__

#include <stddef.h>
#include <stdint.h>

#define NCNN_AMX_NOP_OP_IMM5(op, imm5) \
    __asm__ __volatile__("nop\nnop\nnop\n.word (0x201000 + (%0 << 5) + %1)" : : "i"(op), "i"(imm5) : "memory")

#define NCNN_AMX_OP_GPR(op, gpr) \
    __asm__ __volatile__(".word (0x201000 + (%0 << 5) + 0%1 - ((0%1 >> 4) * 6))" : : "i"(op), "r"((uint64_t)(gpr)) : "memory")

#define NCNN_AMX_LDX(gpr)  NCNN_AMX_OP_GPR( 0, gpr)
#define NCNN_AMX_LDY(gpr)  NCNN_AMX_OP_GPR( 1, gpr)
#define NCNN_AMX_STX(gpr)  NCNN_AMX_OP_GPR( 2, gpr)
#define NCNN_AMX_STY(gpr)  NCNN_AMX_OP_GPR( 3, gpr)
#define NCNN_AMX_LDZ(gpr)  NCNN_AMX_OP_GPR( 4, gpr)
#define NCNN_AMX_STZ(gpr)  NCNN_AMX_OP_GPR( 5, gpr)
#define NCNN_AMX_FMA32(gpr) NCNN_AMX_OP_GPR(12, gpr)
#define NCNN_AMX_FMA16(gpr) NCNN_AMX_OP_GPR(15, gpr)
#define NCNN_AMX_MATINT(gpr) NCNN_AMX_OP_GPR(20, gpr)
#define NCNN_AMX_SET()     NCNN_AMX_NOP_OP_IMM5(17, 0)
#define NCNN_AMX_CLR()     NCNN_AMX_NOP_OP_IMM5(17, 1)

// fma32 matrix mode
// z[j][i] += x[i] * y[j]
// x = 16 consecutive f32 of B row k (along N)
// y = 16 consecutive f32 of A row k (along M)
// z row j maps to physical Z row j*4 when the z row operand is 0
// skip z input on the first k so the accumulator does not need explicit clearing
static inline void ncnn_apple_amx_gemm_tile16(const float* A, int lda, const float* B, int ldb, int m0, int n0, int K)
{
    for (int k = 0; k < K; k++)
    {
        NCNN_AMX_LDX((uint64_t)(B + (size_t)k * ldb + n0));
        NCNN_AMX_LDY((uint64_t)(A + (size_t)k * lda + m0));
        NCNN_AMX_FMA32(k == 0 ? (1ull << 27) : 0);
    }
}

// store z row j to 64 consecutive bytes
static inline void ncnn_apple_amx_store_row16(int j, float* dst)
{
    NCNN_AMX_STZ((uint64_t)dst + ((uint64_t)(j * 4) << 56));
}

// C[M][N] = A[K][M]^T * B[K][N]
// M and N must be multiples of 16
static inline void ncnn_apple_amx_gemm_f32(const float* A, int lda, const float* B, int ldb, float* C, int ldc, int M, int N, int K)
{
    NCNN_AMX_SET();

    for (int m0 = 0; m0 < M; m0 += 16)
    {
        for (int n0 = 0; n0 < N; n0 += 16)
        {
            ncnn_apple_amx_gemm_tile16(A, lda, B, ldb, m0, n0, K);

            for (int j = 0; j < 16; j++)
            {
                ncnn_apple_amx_store_row16(j, C + (size_t)(m0 + j) * ldc + n0);
            }
        }
    }

    NCNN_AMX_CLR();
}

#endif // defined __APPLE__ && __aarch64__

#endif // NCNN_APPLE_AMX_H
