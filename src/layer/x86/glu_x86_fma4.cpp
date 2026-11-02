// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "sse_mathfun.h"
#include "avx_mathfun.h"

namespace ncnn {

#include "glu_fp32.h"

void glu_fma4(const float* input, const float* gate, float* output, int size)
{
    glu(input, gate, output, size);
}

void glu_pack_fma4(const float* input, const float* gate, float* output, int size, int elempack, int out_elempack)
{
    glu_pack(input, gate, output, size, elempack, out_elempack);
}

} // namespace ncnn
