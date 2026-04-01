// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gelu_x86.h"

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

#include "x86_usability.h"

#include "cpu.h"
#include "mat.h"

namespace ncnn {

#include "gelu_bf16s.h"

void gelu_bf16s_avx512bf16(Mat& a, int fast_gelu, const Option& opt)
{
    gelu_bf16s(a, fast_gelu, opt);
}

} // namespace ncnn
