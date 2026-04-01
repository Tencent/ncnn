// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "dropout_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"

#include "cpu.h"
#include "mat.h"

namespace ncnn {

#include "dropout_bf16s.h"

void dropout_bf16s_avx512bf16(Mat& a, float scale, const Option& opt)
{
    dropout_bf16s(a, scale, opt);
}

} // namespace ncnn
