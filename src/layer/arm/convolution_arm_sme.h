// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_CONVOLUTION_ARM_SME_H
#define NCNN_CONVOLUTION_ARM_SME_H

#include "platform.h"

#if NCNN_ARM_SME

#include <stddef.h>

// implemented in convolution_arm_sme.cpp, the only file built with sme enabled
// the declarations carry no streaming attribute on purpose so that callers emit a plain call
// the kernels save and restore v8-v15 themselves, streaming mode discards them
// C[m][n] = sum_k A[k][m] * B[k][n], the output is channel major with out_cstep = N
extern "C" void ncnn_conv1x1s1_sme_f32(const float* A, int lda, const float* B, int ldb, float* outptr, size_t out_cstep, int m_begin, int m_end, int N, int K, const float* bias);

// same as above but writing the 4 channel packed layout with out_cstep = N * 4
// m must be a multiple of 4, the epilogue uses st4, call it only when cpu_support_arm_sme2()
extern "C" void ncnn_conv1x1s1_sme_pack4_f32(const float* A, int lda, const float* B, int ldb, float* outptr, size_t out_cstep, int m_begin, int m_end, int N, int K, const float* bias);

#endif // NCNN_ARM_SME

#endif // NCNN_CONVOLUTION_ARM_SME_H
