// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "rmsnorm_bf16s.h"

void rmsnorm_bf16s_sse_avx512bf16(unsigned short* ptr, const float* gamma_ptr, float eps, int elemcount, int elempack)
{
    rmsnorm_bf16s_sse(ptr, gamma_ptr, eps, elemcount, elempack);
}

} // namespace ncnn
