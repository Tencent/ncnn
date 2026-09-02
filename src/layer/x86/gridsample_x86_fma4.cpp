// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gridsample_x86.h"

#if __SSE2__
#include <emmintrin.h>
#include "sse_mathfun.h"
#if __AVX__
#include <immintrin.h>
#include "avx_mathfun.h"
#endif // __AVX__
#endif // __SSE2__

#include "cpu.h"
#include "x86_usability.h"

namespace ncnn {

#include "gridsample_compute_blob.h"
#include "gridsample_bilinear_apply_interpolation.h"
#include "gridsample_bicubic_apply_interpolation.h"
#include "gridsample_nearest_apply_interpolation.h"
#include "gridsample_fp32.h"

int gridsample_fp32_fma4(const Mat& bottom_blob, const Mat& grid, Mat& top_blob, int sample_type, int padding_mode, int align_corner, int permute_fusion, const Option& opt)
{
    return gridsample_fp32(bottom_blob, grid, top_blob, sample_type, padding_mode, align_corner, permute_fusion, opt);
}

} // namespace ncnn
