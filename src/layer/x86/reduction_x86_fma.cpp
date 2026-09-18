// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "reduction_x86.h"

#include "cpu.h"

#include <float.h>
#include <math.h>

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

namespace ncnn {

#include "reduction_fp32.h"

int reduction_fp32_fma(const Mat& a, Mat& b, bool reduce_w, bool reduce_h, bool reduce_d, bool reduce_c, int operation, const Option& opt)
{
    return reduction_fp32(a, b, reduce_w, reduce_h, reduce_d, reduce_c, operation, opt);
}

} // namespace ncnn
