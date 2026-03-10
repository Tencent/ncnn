// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "scale_bf16s.h"

void scale_bf16s_sse_avx512bf16(unsigned short* ptr, const float* scale, const float* bias, int size, int elempack)
{
    scale_bf16s_sse(ptr, scale, bias, size, elempack);
}

void scale_bf16s_no_bias_sse_avx512bf16(unsigned short* ptr, const float* scale, int size, int elempack)
{
    scale_bf16s_no_bias_sse(ptr, scale, size, elempack);
}

} // namespace ncnn
