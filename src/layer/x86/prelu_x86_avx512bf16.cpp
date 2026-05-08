// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "prelu_bf16s.h"

void prelu_bf16s_sse_avx512bf16(unsigned short* ptr, const float* slope, int size, int elempack)
{
    prelu_bf16s_sse(ptr, slope, size, elempack);
}

void prelu_bf16s_per_element_sse_avx512bf16(unsigned short* ptr, const float* slope, int size, int num_threads)
{
    prelu_bf16s_per_element_sse(ptr, slope, size, num_threads);
}

void prelu_bf16s_single_slope_sse_avx512bf16(unsigned short* ptr, float slope, int size, int num_threads)
{
    prelu_bf16s_single_slope_sse(ptr, slope, size, num_threads);
}

} // namespace ncnn
