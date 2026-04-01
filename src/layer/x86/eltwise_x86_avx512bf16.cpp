// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "eltwise_x86.h"

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

#include "eltwise_bf16s.h"

void eltwise_bf16s_avx512bf16(const std::vector<Mat>& bottom_blobs, Mat& top_blob, int op_type, const Mat& coeffs, const Option& opt)
{
    eltwise_bf16s(bottom_blobs, top_blob, op_type, coeffs, opt);
}

} // namespace ncnn
