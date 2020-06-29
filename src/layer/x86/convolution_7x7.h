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

static void conv7x7s1_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int kernel_w = 7;
    int kernel_h = 7;

    int stride_w = 1;
    int stride_h = 1;

    conv_im2col_sgemm_sse(bottom_blob, top_blob, _kernel, _bias, kernel_w, kernel_h, stride_w, stride_h, opt);
}

static void conv7x7s2_sse(const Mat& bottom_blob, Mat& top_blob, const Mat& _kernel, const Mat& _bias, const Option& opt)
{
    int kernel_w = 7;
    int kernel_h = 7;

    int stride_w = 2;
    int stride_h = 2;

    conv_im2col_sgemm_sse(bottom_blob, top_blob, _kernel, _bias, kernel_w, kernel_h, stride_w, stride_h, opt);
}