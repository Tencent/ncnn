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

} // namespace ncnn
