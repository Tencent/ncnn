// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "interp_bf16s.h"

void interp_forward_bf16s_sse_avx512bf16(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt, int resize_type, int align_corner, float height_scale, float width_scale, int output_height, int output_width, int has_size_expr)
{
    interp_forward_bf16s_sse(bottom_blobs, top_blobs, opt, resize_type, align_corner, height_scale, width_scale, output_height, output_width, has_size_expr);
}

} // namespace ncnn
