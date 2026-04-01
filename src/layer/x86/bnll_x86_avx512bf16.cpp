// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "bnll_x86.h"

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

#include "bnll_bf16s.h"

void bnll_bf16s_avx512bf16(Mat& a, const Option& opt)
{
    bnll_bf16s(a, opt);
}

} // namespace ncnn
