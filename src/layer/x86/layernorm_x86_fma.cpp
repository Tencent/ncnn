// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "layernorm_bf16s.h"
#include "layernorm_fp32.h"

void layernorm_bf16s_sse_fma(unsigned short* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    layernorm_bf16s_sse(ptr, gamma_ptr, beta_ptr, eps, elemcount, elempack);
}

void layernorm_fma(float* ptr, const float* gamma_ptr, const float* beta_ptr, float eps, int elemcount, int elempack)
{
    layernorm(ptr, gamma_ptr, beta_ptr, eps, elemcount, elempack);
}
} // namespace ncnn
