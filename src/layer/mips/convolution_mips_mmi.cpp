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
#if __mips_loongson_mmi
#include "loongson_mmi.h"
#endif // __mips_loongson_mmi

namespace ncnn {

#include "convolution_sgemm_int8.h"
#include "convolution_winograd_transform_int8.h"
#include "convolution_winograd_dot_int8.h"
#include "convolution_3x3_int8.h"

// pack1
void im2col_sgemm_int8_loongson_mmi(const Mat& bottom_im2col, Mat& top_blob, const Mat& kernel, const Option& opt)
{
    im2col_sgemm_int8_msa(bottom_im2col, top_blob, kernel, opt);
}

void convolution_im2col_sgemm_transform_kernel_int8_loongson_mmi(const Mat& kernel, Mat& kernel_tm, int inch, int outch, int kernel_w, int kernel_h)
{
    convolution_im2col_sgemm_transform_kernel_int8_msa(kernel, kernel_tm, inch, outch, kernel_w, kernel_h);
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
