// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "glu_x86.h"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#if __AVX512F__
#include "avx512_mathfun.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__

#include "cpu.h"

namespace ncnn {

#include "glu_fp32.h"

GLU_x86::GLU_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int GLU_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    return glu_fp32(bottom_blob, top_blob, axis, opt);
}

} // namespace ncnn
