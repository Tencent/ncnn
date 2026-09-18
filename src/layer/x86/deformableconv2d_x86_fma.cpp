// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deformableconv2d_x86.h"

#include <immintrin.h>

#include "cpu.h"
#include "x86_activation.h"
#include "x86_usability.h"

namespace ncnn {

#include "deformableconv2d_im2col.h"
#include "deformableconv2d_packed.h"

void deformableconv2d_im2col_fma(const Mat& bottom_blob, const Mat& offset_unpacked, const Mat& mask_unpacked, Mat& bottom_im2col, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int pad_left, int pad_top, int outw, int outh, int has_mask, const Option& opt)
{
    deformableconv2d_im2col(bottom_blob, offset_unpacked, mask_unpacked, bottom_im2col, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_left, pad_top, outw, outh, has_mask, opt);
}

void deformableconv2d_packed_fma(const std::vector<Mat>& bottom_blobs, Mat& top_blob, const Mat& weight_data_packed, const Mat& bias_data, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int pad_left, int pad_top, int activation_type, const Mat& activation_params, const Option& opt)
{
    deformableconv2d_packed(bottom_blobs, top_blob, weight_data_packed, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, pad_left, pad_top, activation_type, activation_params, opt);
}

} // namespace ncnn
