// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "batchnorm_bf16s.h"

void batchnorm_bf16s_sse_avx512bf16(unsigned short* ptr, const float* a, const float* b, int size, int elempack)
{
    batchnorm_bf16s_sse(ptr, a, b, size, elempack);
}

void batchnorm_bf16s_per_element_sse_avx512bf16(unsigned short* ptr, const float* a, const float* b, int size)
{
    batchnorm_bf16s_per_element_sse(ptr, a, b, size);
}

} // namespace ncnn
