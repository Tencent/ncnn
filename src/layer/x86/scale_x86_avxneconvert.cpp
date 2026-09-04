// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#if NCNN_BF16

#include "scale_bf16s.h"

void scale_bf16s_avxneconvert(unsigned short* ptr, const float* scale, const float* bias, int size, int elempack)
{
    scale_bf16s(ptr, scale, bias, size, elempack);
}

void scale_bf16s_no_bias_avxneconvert(unsigned short* ptr, const float* scale, int size, int elempack)
{
    scale_bf16s_no_bias(ptr, scale, size, elempack);
}

void scale_bf16s_per_element_avxneconvert(unsigned short* ptr, const float* scale, const float* bias, int size, int num_threads)
{
    scale_bf16s_per_element(ptr, scale, bias, size, num_threads);
}

void scale_bf16s_no_bias_per_element_avxneconvert(unsigned short* ptr, const float* scale, int size, int num_threads)
{
    scale_bf16s_no_bias_per_element(ptr, scale, size, num_threads);
}

#endif // NCNN_BF16

} // namespace ncnn
