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

#include "deconvolution_arm.h"

namespace ncnn {

#include "deconvolution_4x4.h"
#include "deconvolution_3x3.h"

DEFINE_LAYER_CREATOR(Deconvolution_arm)

int Deconvolution_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // deconvolv with NxN kernel
    // value = value + bias

    if (kernel_w != kernel_h || stride_w != stride_h)
    {
        return Deconvolution::forward(bottom_blob, top_blob, opt);
    }

    const int kernel_size = kernel_w;
    const int stride = stride_w;

    if ((kernel_size != 3 && kernel_size != 4) || stride > 2 || dilation_w != 1 || dilation_h != 1)
    {
        return Deconvolution::forward(bottom_blob, top_blob, opt);
    }

    typedef void (*deconv_func)(const Mat&, Mat&, const Mat&, const Mat&, const Option&);

    // kernel_size x stride
    deconv_func deconv_func_table[2][2] =
    {
        {
            deconv3x3s1_neon,
            deconv3x3s2_neon
        },  // kernel_size = 3
        {
            deconv4x4s1_neon,
            deconv4x4s2_neon
        }   // kernel_size = 4
    };

    deconv_func deconv = deconv_func_table[kernel_size-3][stride-1];
    if (!deconv)
    {
        return Deconvolution::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    int outw = (w - 1) * stride + kernel_size;
    int outh = (h - 1) * stride + kernel_size;

    Mat top_blob_bordered;
    if (pad_w > 0 || pad_h > 0)
    {
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.workspace_allocator);
        if (top_blob_bordered.empty())
            return -100;
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output, elemsize, opt.blob_allocator);
        if (top_blob_bordered.empty())
            return -100;
    }

    deconv(bottom_blob, top_blob_bordered, weight_data, bias_data, opt);

    if (pad_w > 0 || pad_h > 0)
    {
        copy_cut_border(top_blob_bordered, top_blob, pad_h, pad_h, pad_w, pad_w, opt.blob_allocator, opt.num_threads);
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }
    else
    {
        top_blob = top_blob_bordered;
    }

    return 0;
}

} // namespace ncnn
