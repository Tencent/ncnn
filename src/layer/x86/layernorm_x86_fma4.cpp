// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "layernorm_bf16s.h"
#endif // NCNN_BF16
#include "layernorm_fp32.h"

#if NCNN_BF16
void layernorm_bf16s_sse_fma4(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    layernorm_bf16s_sse(ptr, gamma_ptr, beta_ptr, eps, elemcount, elempack);
}
#endif // NCNN_BF16

void layernorm_fma4(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    layernorm(ptr, gamma_ptr, beta_ptr, eps, elemcount, elempack);
}
} // namespace ncnn
