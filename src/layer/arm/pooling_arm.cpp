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

#include "pooling_arm.h"

namespace ncnn {

#include "pooling_2x2.h"
#include "pooling_3x3.h"

DEFINE_LAYER_CREATOR(Pooling_arm)

int Pooling_arm::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    // max value in NxN window
    // avg value in NxN window

    if (kernel_w != kernel_h || stride_w != stride_h)
    {
        return Pooling::forward(bottom_blob, top_blob);
    }

    const int kernel_size = kernel_w;
    const int stride = stride_w;

    if (pooling_type != PoolMethod_MAX || stride != 2 || global_pooling == 1)
    {
        return Pooling::forward(bottom_blob, top_blob);
    }

    if (kernel_size != 2 && kernel_size != 3)
    {
        return Pooling::forward(bottom_blob, top_blob);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

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
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    int wtail = (w - kernel_w) % stride_w;
    int htail = (h - kernel_h) % stride_h;
    if ((pad_w == -233 && pad_h == -233) || (pad_w == -2333 && pad_h == -2333))
    {
        wtail = 0;
        htail = 0;
    }
    if (wtail != 0 || htail != 0)
    {
        int wtailpad = 0;
        int htailpad = 0;
        if (wtail != 0)
            wtailpad = kernel_w - wtail;
        if (htail != 0)
            htailpad = kernel_h - htail;

        Mat bottom_blob_bordered2;
        copy_make_border(bottom_blob_bordered, bottom_blob_bordered2, 0, htailpad, 0, wtailpad, BORDER_REPLICATE, 0.f);
        if (bottom_blob_bordered2.empty())
            return -100;

        bottom_blob_bordered = bottom_blob_bordered2;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;

        if (wtail != 0)
            outw += 1;
        if (htail != 0)
            outh += 1;
    }

    top_blob.create(outw, outh, channels);
    if (top_blob.empty())
        return -100;

    if (kernel_size == 2)
        pooling2x2s2_max_neon(bottom_blob_bordered, top_blob);
    if (kernel_size == 3)
        pooling3x3s2_max_neon(bottom_blob_bordered, top_blob);

    return 0;
}

} // namespace ncnn
