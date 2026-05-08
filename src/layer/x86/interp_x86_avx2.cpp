// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "interp_bilinear.h"
#include "interp_bicubic.h"

void resize_bilinear_image_avx2(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    resize_bilinear_image(src, dst, alpha, xofs, beta, yofs);
}

void resize_bicubic_image_avx2(const Mat& src, Mat& dst, float* alpha, int* xofs, float* beta, int* yofs)
{
    resize_bicubic_image(src, dst, alpha, xofs, beta, yofs);
}

} // namespace ncnn
