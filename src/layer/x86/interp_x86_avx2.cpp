// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "interp_bilinear.h"
#include "interp_bicubic.h"

#if NCNN_BF16
#include "interp_bf16s.h"
#endif // NCNN_BF16

void resize_bilinear_image_avx2(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
}

void resize_bicubic_image_avx2(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    resize_bicubic_image(src, dst, alpha, xofs, beta, yofs);
}

#if NCNN_BF16
void interp_bf16s_avx2(const Mat& bottom_blob, Mat& top_blob, int resize_type, int align_corner, float height_scale, float width_scale, int output_height, int output_width, int has_size_expr, const Option& opt)
{
    interp_bf16s(bottom_blob, top_blob, resize_type, align_corner, height_scale, width_scale, output_height, output_width, has_size_expr, opt);
}
#endif // NCNN_BF16

} // namespace ncnn
