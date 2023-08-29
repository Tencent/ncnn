// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "cpu.h"
#include "mat.h"
#include "x86_usability.h"

namespace ncnn {

#include "convolution_packed_int8.h"
#include "convolution_im2col_gemm_int8.h"
#include "convolution_3x3_pack8to1_int8.h"
#include "convolution_3x3_pack8to4_int8.h"

// packed
void convolution_packed_int8_avxvnni(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data_tm, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, const Option& opt)
{
    convolution_packed_int8(bottom_blob, top_blob, weight_data_tm, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);
}

// gemm
void convolution_im2col_gemm_int8_avxvnni(const Mat& bottom_blob, Mat& top_blob, const Mat& AT, int kernel_w, int kernel_h, int dilation_w, int dilation_h, int stride_w, int stride_h, int nT, const Option& opt)
{
    convolution_im2col_gemm_int8(bottom_blob, top_blob, AT, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, nT, opt);
}

// winograd
void conv3x3s1_winograd43_transform_kernel_pack8to1_int8_sse_avxvnni(const Mat& kernel, Mat& kernel_tm, int inch, int outch, const Option& opt)
{
    conv3x3s1_winograd43_transform_kernel_pack8to1_int8_sse(kernel, kernel_tm, inch, outch, opt);
}

void conv3x3s1_winograd43_pack8to1_int8_sse_avxvnni(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    conv3x3s1_winograd43_pack8to1_int8_sse(bottom_blob, top_blob, kernel, opt);
}

void conv3x3s1_winograd43_transform_kernel_pack8to4_int8_sse_avxvnni(const Mat& kernel, Mat& kernel_tm, int inch, int outch, const Option& opt)
{
    conv3x3s1_winograd43_transform_kernel_pack8to4_int8_sse(kernel, kernel_tm, inch, outch, opt);
}

void conv3x3s1_winograd43_pack8to4_int8_sse_avxvnni(const Mat& bottom_blob, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    conv3x3s1_winograd43_pack8to4_int8_sse(bottom_blob, top_blob, kernel, opt);
}

} // namespace ncnn
