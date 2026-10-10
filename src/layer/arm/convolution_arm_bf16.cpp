// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"

#include "arm_usability.h"

namespace ncnn {

#if NCNN_BF16
#include "convolution_im2col_gemm_bf16s.h"

void convolution_im2col_gemm_transform_kernel_bf16s_bf16(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    convolution_im2col_gemm_transform_kernel_bf16s(kernel, AT, inch, outch, kernel_w, kernel_h, opt);
}

int convolution_im2col_gemm_bf16s_bf16(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, const Mat& bias, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    return convolution_im2col_gemm_bf16s(bottom_blob, top_blob, AT, bias, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, nT, opt);
}
#endif // NCNN_BF16

} // namespace ncnn
