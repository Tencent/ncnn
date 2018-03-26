// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "convolution_x86.h"

namespace ncnn {

#include "convolution_1x1.h"
#include "convolution_3x3.h"
#include "convolution_5x5.h"

DEFINE_LAYER_CREATOR(Convolution_x86)

int Convolution_x86::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    // convolv with NxN kernel
    // value = value + bias

    if (bottom_blob.dims != 3)
    {
        return Convolution::forward(bottom_blob, top_blob);
    }

    if (kernel_w != kernel_h || stride_w != stride_h)
    {
        return Convolution::forward(bottom_blob, top_blob);
    }

    const int kernel_size = kernel_w;
    const int stride = stride_w;

    if (kernel_size > 5 || stride > 5 || dilation_w != 1 || dilation_h != 1)
    {
        return Convolution::forward(bottom_blob, top_blob);
    }

    typedef void (*conv_func)(const Mat&, Mat&, const Mat&, const Mat&);

    // kernel_size x stride
    conv_func conv_func_table[5][5] =
    {
        {
            conv1x1s1_sse,
            conv1x1s2_sse,
            0,
            0,
            0
        }, // kernel_size = 1
        {
            0,
            0,
            0,
            0,
            0
        }, // kernel_size = 2
        {
            conv3x3s1_sse,
            0,
            0,
            0,
            0
        }, // kernel_size = 3
        {
            0,
            0,
            0,
            0,
            0
        }, // kernel_size = 4
        {
            conv5x5s1_sse,
            0,
            0,
            0,
            0
        }  // kernel_size = 5
    };

    conv_func conv = conv_func_table[kernel_size-1][stride-1];
    if (!conv)
    {
        return Convolution::forward(bottom_blob, top_blob);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;

    Mat bottom_blob_bordered = bottom_blob;
    if (pad_w > 0 || pad_h > 0)
    {
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }
    else if (pad_w == -233 && pad_h == -233)
    {
        int wpad = kernel_size + (w - 1) / stride * stride - w;
        int hpad = kernel_size + (h - 1) / stride * stride - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_size) / stride + 1;
    int outh = (h - kernel_size) / stride + 1;

    top_blob.create(outw, outh, num_output);
    if (top_blob.empty())
        return -100;

    conv(bottom_blob_bordered, top_blob, weight_data, bias_data);

    return 0;
}

} // namespace ncnn
