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

#include "deconvolution_3x3.h"

DEFINE_LAYER_CREATOR(Deconvolution_arm)

int Deconvolution_arm::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    if (kernel_size != 3 || stride != 1 || dilation != 1)
    {
        return Deconvolution::forward(bottom_blob, top_blob);
    }

    typedef void (*deconv_func)(const Mat&, Mat&, const Mat&, const Mat&);

    deconv_func deconv = deconv3x3s1_neon;

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int outw = (w - 1) * stride + kernel_size;
    int outh = (h - 1) * stride + kernel_size;

    Mat top_blob_bordered;
    top_blob_bordered.create(outw, outh, num_output);
    if (top_blob_bordered.empty())
        return -100;

    deconv(bottom_blob, top_blob_bordered, weight_data, bias_data);

    top_blob = top_blob_bordered;

    if (pad > 0)
    {
        copy_cut_border(top_blob_bordered, top_blob, pad, pad, pad, pad);
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }

    return 0;
}

} // namespace ncnn
