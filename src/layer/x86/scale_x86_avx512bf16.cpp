// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "scale_bf16s.h"

void scale_bf16s_avx512bf16(unsigned short* ptr, const float* scale, const float* bias, int size, int elempack)
{
    scale_bf16s(ptr, scale, bias, size, elempack);
}

void scale_bf16s_no_bias_avx512bf16(unsigned short* ptr, const float* scale, int size, int elempack)
{
    scale_bf16s_no_bias(ptr, scale, size, elempack);
}

void scale_bf16s_per_element_avx512bf16(unsigned short* ptr, const float* scale, const float* bias, int size, int num_threads)
{
    scale_bf16s_per_element(ptr, scale, bias, size, num_threads);
}

void scale_bf16s_no_bias_per_element_avx512bf16(unsigned short* ptr, const float* scale, int size, int num_threads)
{
    scale_bf16s_no_bias_per_element(ptr, scale, size, num_threads);
}

} // namespace ncnn
