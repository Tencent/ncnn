// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "softmax_bf16s.h"

void softmax_bf16s_to_fp32_avx512bf16(const unsigned short* src, float* dst, int size)
{
    softmax_bf16s_to_fp32(src, dst, size);
}

void softmax_fp32_to_bf16s_avx512bf16(const float* src, unsigned short* dst, int size)
{
    softmax_fp32_to_bf16s(src, dst, size);
}

} // namespace ncnn
