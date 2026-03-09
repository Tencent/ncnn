// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "sigmoid_x86.h"

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

#include "sigmoid_bf16s.h"

int Sigmoid_x86::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    return sigmoid_bf16s(bottom_top_blob, opt);
}

} // namespace ncnn
