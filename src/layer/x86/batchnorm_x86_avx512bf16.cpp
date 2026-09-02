// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "batchnorm_bf16s.h"

void batchnorm_bf16s_avx512bf16(unsigned short* ptr, const float* a, const float* b, int size, int elempack)
{
    batchnorm_bf16s(ptr, a, b, size, elempack);
}

void batchnorm_bf16s_per_element_avx512bf16(unsigned short* ptr, const float* a, const float* b, int size, int num_threads)
{
    batchnorm_bf16s_per_element(ptr, a, b, size, num_threads);
}

} // namespace ncnn
