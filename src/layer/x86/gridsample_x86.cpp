// Copyright 2023 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "gridsample_x86.h"

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

namespace ncnn {

#include "gridsample_compute_blob.h"
#include "gridsample_bilinear_apply_interpolation.h"
#include "gridsample_bicubic_apply_interpolation.h"
#include "gridsample_nearest_apply_interpolation.h"
#include "gridsample_fp32.h"

GridSample_x86::GridSample_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

int GridSample_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    return gridsample_fp32(bottom_blobs, top_blobs, sample_type, padding_mode, align_corner, permute_fusion, opt);
}

} // namespace ncnn
