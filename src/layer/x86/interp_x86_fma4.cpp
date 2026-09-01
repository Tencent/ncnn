// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "interp_bilinear.h"
#include "interp_bicubic.h"

#include "interp_bicubic_pack4.h"
#include "interp_bilinear_pack4.h"

#include "interp_bicubic_pack8.h"
#include "interp_bilinear_pack8.h"

#include "interp_fp32.h"
#if NCNN_BF16
#include "interp_bf16s.h"
#endif // NCNN_BF16

int interp_forward_fma4(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, int outw, int outh, int resize_type, int align_corner, float height_scale, float width_scale, int output_height, int output_width, int has_size_expr, const Option& opt)
{
    return interp_forward(bottom_blobs, top_blobs, outw, outh, resize_type, align_corner, height_scale, width_scale, output_height, output_width, has_size_expr, opt);
}

#if NCNN_BF16
void interp_forward_bf16s_sse_fma4(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt, int resize_type, int align_corner, float height_scale, float width_scale, int output_height, int output_width, int has_size_expr)
{
    interp_forward_bf16s_sse(bottom_blobs, top_blobs, opt, resize_type, align_corner, height_scale, width_scale, output_height, output_width, has_size_expr);
}
#endif // NCNN_BF16

} // namespace ncnn
