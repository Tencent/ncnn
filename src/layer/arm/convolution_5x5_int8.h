// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
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

static void conv5x5s1_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int kernel_w = 5;
    int kernel_h = 5;

    int stride_w = 1;
    int stride_h = 1;

    conv_im2col_sgemm_int8_neon(bottom_blob, top_blob, _kernel, kernel_w, kernel_h, stride_w, stride_h, opt);
}

static void conv5x5s2_int8_neon(const Mat &bottom_blob, Mat &top_blob, const Mat &_kernel, const Option& opt)
{
    int kernel_w = 5;
    int kernel_h = 5;

    int stride_w = 2;
    int stride_h = 2;

    conv_im2col_sgemm_int8_neon(bottom_blob, top_blob, _kernel, kernel_w, kernel_h, stride_w, stride_h, opt);
}
