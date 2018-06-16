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

#include "convolution_arm.h"

namespace ncnn {

#include "convolution_1x1.h"
#include "convolution_2x2.h"
#include "convolution_3x3.h"
#include "convolution_4x4.h"
#include "convolution_5x5.h"
#include "convolution_7x7.h"

DEFINE_LAYER_CREATOR(Convolution_arm)

int Convolution_arm::load_param(const ParamDict& pd)
{
    int ret = Convolution::load_param(pd);
    if (ret != 0)
        return ret;

    use_winograd3x3 = false;

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        int num_input = weight_data_size / 9 / num_output;
        // winograd is slow on small channel count
        if (num_input >= 16 && num_output >= 16)
            use_winograd3x3 = true;
    }

    return 0;
}

int Convolution_arm::load_model(const ModelBin& mb)
{
    int ret = Convolution::load_model(mb);
    if (ret != 0)
        return ret;

    if (use_winograd3x3)
    {
        int num_input = weight_data_size / 9 / num_output;
//         conv3x3s1_winograd64_transform_kernel_neon(weight_data, weight_3x3_winograd64_data, num_input, num_output);
        conv3x3s1_winograd64_transform_kernel_neon5(weight_data, weight_3x3_winograd64_data, num_input, num_output);
    }

    return 0;
}

int Convolution_arm::forwardDilation(const Mat& bottom_blob, Mat& top_blob, conv_func conv) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    const int kernel_size = kernel_w;
    const int stride = stride_w;
    const int dilation = dilation_w;
    const int kernel_extent = dilation * (kernel_size - 1) + 1;

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
        int wpad = kernel_extent + (w - 1) / stride * stride - w;
        int hpad = kernel_extent + (h - 1) / stride * stride - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_extent) / stride + 1;
    int outh = (h - kernel_extent) / stride + 1;

    top_blob.create(outw, outh, num_output);
    if (top_blob.empty())
        return -100;

    // Make (dilation * dilation) batches
    Mat inner_bottom_blob;
    Mat inner_top_blob;
    for (int x = 0; x < dilation; x ++)
    {
        for (int y = 0; y < dilation; y ++)
        {
            int inner_w = (w - y + dilation - 1) / dilation;
            int inner_h = (h - x + dilation - 1) / dilation;

            int inner_outw = (inner_w - kernel_size) / stride + 1;
            int inner_outh = (inner_h - kernel_size) / stride + 1;

            if (inner_bottom_blob.w != inner_w || inner_bottom_blob.h != inner_h)
            {
                inner_bottom_blob.create(inner_w, inner_h, bottom_blob.c);

                if (inner_bottom_blob.empty())
                {
                    return -100;
                }
            }

            if (inner_top_blob.w != inner_outw || inner_top_blob.h != inner_outh)
            {
                inner_top_blob.create(inner_outw, inner_outh, num_output);

                if (inner_top_blob.empty())
                {
                    return -100;
                }
            }

            #pragma omp parallel for
            for (int c = 0; c < bottom_blob.c; c ++)
            {
                float *outptr = (float *) inner_bottom_blob.channel(c);

                for (int i = 0; i < inner_h; i ++)
                {
                    const float *ptr = (const float *) bottom_blob_bordered.channel(c) + dilation * i * w + x * w + y;
                    for (int j = 0; j < inner_w; j ++)
                    {
                        outptr[j] = ptr[j*dilation];
                    }
                    outptr += inner_w;
                }
            }

            conv(inner_bottom_blob, inner_top_blob, weight_data, bias_data);

            #pragma omp parallel for
            for (int c = 0; c < num_output; c ++)
            {
                float *outptr = (float *) top_blob.channel(c) + x * outw + y;
                for (int i = 0; i < inner_outh; i ++)
                {
                    const float *ptr = (const float *) inner_top_blob.channel(c) + i * inner_outw;
                    for (int j = 0; j < inner_outw; j ++)
                    {
                        outptr[j*dilation] = ptr[j];
                    }
                    outptr += dilation * outw;
                }
            }
        }
    }

    return 0;
}

int Convolution_arm::forward(const Mat& bottom_blob, Mat& top_blob) const
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

    if (kernel_size > 7 || stride > 4 || dilation_w != dilation_h)
    {
        return Convolution::forward(bottom_blob, top_blob);
    }

    typedef void (*conv_func)(const Mat&, Mat&, const Mat&, const Mat&);

    // kernel_size x stride
    conv_func conv_func_table[7][4] =
    {
        {
            conv1x1s1_neon,
            conv1x1s2_neon,
            0,
            0
        }, // kernel_size = 1
        {
            conv2x2s1_neon,
            0,
            0,
            0
        }, // kernel_size = 2
        {
            conv3x3s1_neon,
            conv3x3s2_neon,
            0,
            0
        }, // kernel_size = 3
        {
            0,
            0,
            0,
            conv4x4s4_neon
        }, // kernel_size = 4
        {
            conv5x5s1_neon,
            conv5x5s2_neon,
            0,
            0
        }, // kernel_size = 5
        {
            0,
            0,
            0,
            0
        }, // kernel_size = 6
        {
            conv7x7s1_neon,
            conv7x7s2_neon,
            0,
            0
        }  // kernel_size = 7
    };

    conv_func conv = conv_func_table[kernel_size-1][stride-1];
    if (!conv)
    {
        return Convolution::forward(bottom_blob, top_blob);
    }

    if (dilation_w != 1)
    {
        return forwardDilation(bottom_blob, top_blob, conv);
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

    if (use_winograd3x3 && w <= 120 && h <= 120)
    {
//         conv3x3s1_winograd64_neon4(bottom_blob_bordered, top_blob, weight_3x3_winograd64_data, bias_data);
        conv3x3s1_winograd64_neon5(bottom_blob_bordered, top_blob, weight_3x3_winograd64_data, bias_data);
    }
    else
        conv(bottom_blob_bordered, top_blob, weight_data, bias_data);

    return 0;
}

} // namespace ncnn
