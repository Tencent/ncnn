// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "normalize_x86.h"

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif // __AVX__
#endif // __SSE2__

#include "x86_usability.h"
#include "cpu.h"

namespace ncnn {

Normalize_x86::Normalize_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
}

#include "normalize_fp32.h"

int Normalize_x86::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    return normalize_fp32(bottom_top_blob, scale_data, across_spatial, across_channel, channel_shared, eps, eps_mode, opt);
}

} // namespace ncnn
