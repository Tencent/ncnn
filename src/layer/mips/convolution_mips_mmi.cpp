// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "mat.h"
#if __mips_loongson_mmi
#include "loongson_mmi.h"
#endif // __mips_loongson_mmi

namespace ncnn {

#include "convolution_im2col_gemm_int8.h"
#include "convolution_winograd_transform_int8.h"
#include "convolution_winograd_dot_int8.h"
#include "convolution_3x3_int8.h"

void convolution_gemm_transB_packed_tile_int8_loongson_mmi(const Mat& AT_tile, const Mat& BT_tile, Mat& topT_tile, int i, int max_ii, int j, int max_jj, int k, int max_kk)
{
    convolution_gemm_transB_packed_tile_int8(AT_tile, BT_tile, topT_tile, i, max_ii, j, max_jj, k, max_kk);
}

void convolution_im2col_gemm_transform_kernel_int8_loongson_mmi(const Mat& kernel, Mat& AT, int inch, int outch, int kernel_w, int kernel_h, const Option& opt)
{
    convolution_im2col_gemm_transform_kernel_int8(kernel, AT, inch, outch, kernel_w, kernel_h, opt);
}

void conv3x3s1_winograd43_transform_kernel_int8_loongson_mmi(const Mat& kernel, Mat& kernel_tm_packed, int inch, int outch, const Option& opt)
{
    conv3x3s1_winograd43_transform_kernel_int8_msa(kernel, kernel_tm_packed, inch, outch, opt);
}

void convolution_winograd_dot_int8_loongson_mmi(Mat& bottom_blob_tm, int outch, const Mat& kernel_tm, Mat& top_blob_tm, const Option& opt)
{
    convolution_winograd_dot_int8_msa(bottom_blob_tm, outch, kernel_tm, top_blob_tm, opt);
}

} // namespace ncnn
