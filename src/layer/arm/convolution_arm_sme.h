// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_CONVOLUTION_ARM_SME_H
#define NCNN_CONVOLUTION_ARM_SME_H

#include "platform.h"

#if NCNN_ARM_SME

#include <stddef.h>

// implemented in convolution_arm_sme.cpp, the only file built with sme enabled
// the kernels run in streaming mode, they are declared without any sme attribute here so that
// this file stays a plain aapcs64 translation unit: whenever the compiler can see that a callee
// needs streaming mode it books the vector length sized register state in the caller prologue
// with cntd or rdsvl, and apple silicon implements no non-streaming sve, so those instructions
// trap before the mode is entered. the transition is therefore made here, in plain asm.
// C[m][n] = sum_k A[k][m] * B[k][n], the output is channel major with out_cstep = N
extern "C" void ncnn_sme_conv1x1s1_f32(const float* A, int lda, const float* B, int ldb, float* outptr, size_t out_cstep, int m_begin, int m_end, int N, int K, const float* bias);

// same as above but writing the 4 channel packed layout with out_cstep = N * 4
// m must be a multiple of 4, the epilogue uses st4, call it only when cpu_support_arm_sme2()
extern "C" void ncnn_sme_conv1x1s1_pack4_f32(const float* A, int lda, const float* B, int ldb, float* outptr, size_t out_cstep, int m_begin, int m_end, int N, int K, const float* bias);

// the za storage is enabled together with streaming mode and both are left disabled on exit,
// smstart za, sm and smstop za, sm are written as words because this file is built without sme
// support and not every assembler accepts those mnemonics for a target that does not advertise sme
#define NCNN_SME_SMSTART() __asm__ __volatile__(".word 0xd503477f" ::: "memory")
#define NCNN_SME_SMSTOP() __asm__ __volatile__(".word 0xd503467f" ::: "memory")

// streaming mode discards v8-v15, the fpsimd callee-saved registers, so they are saved here in a
// fixed size slot of the plain aapcs64 frame and restored once the kernel has written its result
struct ncnn_sme_fpsimd_state
{
    float v[32];
} __attribute__((aligned(16)));

struct ncnn_sme_streaming_guard
{
    ncnn_sme_fpsimd_state fpsimd;

    ncnn_sme_streaming_guard()
    {
        void* p = &fpsimd;
        __asm__ __volatile__(
            "stp q8, q9, [%[p]]\n"
            "stp q10, q11, [%[p], #32]\n"
            "stp q12, q13, [%[p], #64]\n"
            "stp q14, q15, [%[p], #96]\n"
            :
            : [p] "r"(p)
            : "memory");

        NCNN_SME_SMSTART();
    }

    ~ncnn_sme_streaming_guard()
    {
        NCNN_SME_SMSTOP();

        void* p = &fpsimd;
        __asm__ __volatile__(
            "ldp q8, q9, [%[p]]\n"
            "ldp q10, q11, [%[p], #32]\n"
            "ldp q12, q13, [%[p], #64]\n"
            "ldp q14, q15, [%[p], #96]\n"
            :
            : [p] "r"(p)
            : "memory");
    }

private:
    ncnn_sme_streaming_guard(const ncnn_sme_streaming_guard&);
    ncnn_sme_streaming_guard& operator=(const ncnn_sme_streaming_guard&);
};

static inline void ncnn_conv1x1s1_sme_f32(const float* A, int lda, const float* B, int ldb, float* outptr, size_t out_cstep, int m_begin, int m_end, int N, int K, const float* bias)
{
    ncnn_sme_streaming_guard guard;

    ncnn_sme_conv1x1s1_f32(A, lda, B, ldb, outptr, out_cstep, m_begin, m_end, N, K, bias);
}

static inline void ncnn_conv1x1s1_sme_pack4_f32(const float* A, int lda, const float* B, int ldb, float* outptr, size_t out_cstep, int m_begin, int m_end, int N, int K, const float* bias)
{
    ncnn_sme_streaming_guard guard;

    ncnn_sme_conv1x1s1_pack4_f32(A, lda, B, ldb, outptr, out_cstep, m_begin, m_end, N, K, bias);
}

#endif // NCNN_ARM_SME

#endif // NCNN_CONVOLUTION_ARM_SME_H
