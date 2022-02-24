// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

namespace ncnn {

#include "convolution_sgemm_int8.h"
#include "convolution_sgemm_pack1to4_int8.h"
#include "convolution_sgemm_pack8to1_int8.h"
#include "convolution_sgemm_pack8to4_int8.h"

// pack1
void im2col_sgemm_int8_neon_arm82dot(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    im2col_sgemm_int8_neon(bottom_im2col, top_blob, kernel, opt);
}

void convolution_im2col_sgemm_transform_kernel_int8_neon_arm82dot(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    convolution_im2col_sgemm_transform_kernel_int8_neon(kernel, kernel_tm, inch, outch, kernel_w, kernel_h);
}

// pack1to4
void im2col_sgemm_pack1to4_int8_neon_arm82dot(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    im2col_sgemm_pack1to4_int8_neon(bottom_im2col, top_blob, kernel, opt);
}

void convolution_im2col_sgemm_transform_kernel_pack1to4_int8_neon_arm82dot(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    convolution_im2col_sgemm_transform_kernel_pack1to4_int8_neon(kernel, kernel_tm, inch, outch, kernel_w, kernel_h);
}

// pack8to1
void im2col_sgemm_pack8to1_int8_neon_arm82dot(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    im2col_sgemm_pack8to1_int8_neon(bottom_im2col, top_blob, kernel, opt);
}

void convolution_im2col_sgemm_transform_kernel_pack8to1_int8_neon_arm82dot(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    convolution_im2col_sgemm_transform_kernel_pack8to1_int8_neon(kernel, kernel_tm, inch, outch, kernel_w, kernel_h);
}

// pack8to4
void im2col_sgemm_pack8to4_int8_neon_arm82dot(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    im2col_sgemm_pack8to4_int8_neon(bottom_im2col, top_blob, kernel, opt);
}

void convolution_im2col_sgemm_transform_kernel_pack8to4_int8_neon_arm82dot(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    convolution_im2col_sgemm_transform_kernel_pack8to4_int8_neon(kernel, kernel_tm, inch, outch, kernel_w, kernel_h);
}

} // namespace ncnn
