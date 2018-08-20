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

#include "convolution_1x1_int8.h"
#include "convolution_3x3_int8.h"

DEFINE_LAYER_CREATOR(Convolution_x86)

int Convolution_x86::forwardDilation(const Mat& bottom_blob, Mat& top_blob, conv_func conv, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_size = kernel_w;
    const int stride = stride_w;
    const int dilation = dilation_w;
    const int kernel_extent = dilation * (kernel_size - 1) + 1;

    Mat bottom_blob_bordered = bottom_blob;
    if (pad_w > 0 || pad_h > 0)
    {
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
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
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_extent) / stride + 1;
    int outh = (h - kernel_extent) / stride + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
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

            inner_bottom_blob.create(inner_w, inner_h, bottom_blob.c, elemsize, opt.workspace_allocator);
            if (inner_bottom_blob.empty())
                return -100;

            inner_top_blob.create(inner_outw, inner_outh, num_output, elemsize, opt.workspace_allocator);
            if (inner_top_blob.empty())
                return -100;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int c = 0; c < bottom_blob.c; c ++)
            {
                float *outptr = inner_bottom_blob.channel(c);

                for (int i = 0; i < inner_h; i ++)
                {
                    const float* ptr = (const float *)bottom_blob_bordered.channel(c) + dilation * i * w + x * w + y;
                    for (int j = 0; j < inner_w; j ++)
                    {
                        outptr[j] = ptr[j*dilation];
                    }
                    outptr += inner_w;
                }
            }

            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = inner_top_blob.allocator;
            conv(inner_bottom_blob, inner_top_blob, weight_data, bias_data, opt_g);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int c = 0; c < num_output; c ++)
            {
                float *outptr = (float *)top_blob.channel(c) + x * outw + y;
                for (int i = 0; i < inner_outh; i ++)
                {
                    const float* ptr = (const float *)inner_top_blob.channel(c) + i * inner_outw;
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

int Convolution_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

    if (bottom_blob.dims != 3)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    if (kernel_w != kernel_h || stride_w != stride_h)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    const int kernel_size = kernel_w;
    const int stride = stride_w;

    if (kernel_size > 5 || stride > 5 || dilation_w != dilation_h)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    typedef void (*conv_func)(const Mat&, Mat&, const Mat&, const Mat&, const Option&);

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

    typedef void (*conv_int8_func)(const Mat&, Mat&, const Mat&, const Option&);

    // kernel_size x stride
    conv_int8_func conv_int8_func_table[5][5] =
    {
        {
            conv1x1s1_int8_sse,
            conv1x1s2_int8_sse,
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
            conv3x3s1_int8_sse,
            conv3x3s2_int8_sse,
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
            0,
            0,
            0,
            0,
            0
        }  // kernel_size = 5
    };

    conv_func conv = 0;
    conv_int8_func conv_int8 = 0;

    if (use_int8_inference)
    {
        conv_int8 = conv_int8_func_table[kernel_size-1][stride-1];
        if (!conv_int8)
        {
            return Convolution::forward(bottom_blob, top_blob, opt);
        }
    }
    else
    {
        conv = conv_func_table[kernel_size-1][stride-1];
        if (!conv)
        {
            return Convolution::forward(bottom_blob, top_blob, opt);
        }

        if (dilation_w != 1)
        {
            if (stride != 1)
                return Convolution::forward(bottom_blob, top_blob, opt);

            return forwardDilation(bottom_blob, top_blob, conv, opt);
        }
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    Mat bottom_blob_unbordered = bottom_blob;
    if (use_int8_inference && elemsize != 1)
    {
        Mat bottom_blob_int8;
        bottom_blob_int8.create(w, h, channels, (size_t)1u, opt.workspace_allocator);
        if (bottom_blob_int8.empty())
            return -100;

        // quantize, scale and round to nearest
        {
            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = bottom_blob_int8.allocator;

            quantize->forward(bottom_blob, bottom_blob_int8, opt_g);
        }

        bottom_blob_unbordered = bottom_blob_int8;
    }

    Mat bottom_blob_bordered = bottom_blob_unbordered;
    if (pad_w > 0 || pad_h > 0)
    {
        copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, pad_h, pad_h, pad_w, pad_w, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
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
            copy_make_border(bottom_blob_unbordered, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f, opt.workspace_allocator, opt.num_threads);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_size) / stride + 1;
    int outh = (h - kernel_size) / stride + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (use_int8_inference)
    {
        conv_int8(bottom_blob_bordered, top_blob, weight_data, opt);

        // dequantize, reverse scale inplace
        {
            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = top_blob.allocator;

            dequantize->forward_inplace(top_blob, opt_g);
        }

        return 0;
    }

    conv(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

    return 0;
}

} // namespace ncnn
