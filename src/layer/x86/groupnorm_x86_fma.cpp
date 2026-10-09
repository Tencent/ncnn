// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "groupnorm_bf16s.h"
#endif // NCNN_BF16
#include "groupnorm_fp32.h"

#if NCNN_BF16
void groupnorm_bf16s_fma(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, int elempack, size_t cstep)
{
    groupnorm_bf16s(ptr, gamma_ptr, beta_ptr, eps, channels, size, elempack, cstep);
}
#endif // NCNN_BF16

void groupnorm_fma(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int channels, int size, int elempack, size_t cstep)
{
    groupnorm(ptr, gamma_ptr, beta_ptr, eps, channels, size, elempack, cstep);
}
} // namespace ncnn
