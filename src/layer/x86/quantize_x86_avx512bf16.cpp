// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "quantize_x86.h"

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

#include "quantize_bf16s.h"

void quantize_forward_bf16s_avx512bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& scale_data, int scale_data_size, const Option& opt)
{
    quantize_forward_bf16s(bottom_blob, top_blob, scale_data, scale_data_size, opt);
}

} // namespace ncnn
