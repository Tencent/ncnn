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

#include "convolutiondepthwise_x86.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "layer_type.h"

namespace ncnn {

#include "convolutiondepthwise_3x3.h"

DEFINE_LAYER_CREATOR(ConvolutionDepthWise_x86)

int ConvolutionDepthWise_x86::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    // convolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    if (channels % group != 0 || num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

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
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, 0.f);
            if (bottom_blob_bordered.empty())
                return -100;
        }

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, num_output);
    if (top_blob.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // depth-wise
    if (channels == group && group == num_output)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1)
        {
            if (stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_sse(bottom_blob_bordered, top_blob, weight_data, bias_data);
                return 0;
            }
            else if (stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_sse(bottom_blob_bordered, top_blob, weight_data, bias_data);
                return 0;
            }
        }

#ifdef _OPENMP
        int nested_current = omp_get_nested();
        omp_set_nested(0);
#endif

        #pragma omp parallel for
        for (int g=0; g<group; g++)
        {
            Mat bottom_blob_bordered_g(w, h, 1, bottom_blob_bordered.channel(g));
            Mat top_blob_g(outw, outh, 1, top_blob.channel(g));
            Mat weight_data_g(maxk, (void*)((const float*)weight_data + maxk * g));
            Mat bias_data_g;
            if (bias_term)
                bias_data_g = Mat(1, (void*)((const float*)bias_data + g));

            // call Convolution
            ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Convolution);

            // set param
            ncnn::ParamDict pd;
            pd.set(0, 1);// num_output
            pd.set(1, kernel_w);
            pd.set(11, kernel_h);
            pd.set(2, dilation_w);
            pd.set(12, dilation_h);
            pd.set(3, stride_w);
            pd.set(13, stride_h);
            pd.set(4, 0);// pad_w
            pd.set(14, 0);// pad_h
            pd.set(5, bias_term);
            pd.set(6, maxk);// weight_data_size

            op->load_param(pd);

            // set weights
            ncnn::Mat weights[2];
            weights[0] = weight_data_g;
            weights[1] = bias_data_g;

            op->load_model(ModelBinFromMatArray(weights));

            // forward
            op->forward(bottom_blob_bordered_g, top_blob_g);

            delete op;
        }

#ifdef _OPENMP
        omp_set_nested(nested_current);
#endif
        return 0;
    }

    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    for (int g=0; g<group; g++)
    {
        Mat bottom_blob_bordered_g(w, h, channels_g, bottom_blob_bordered.channel(channels_g * g));
        Mat top_blob_g(outw, outh, num_output_g, top_blob.channel(num_output_g * g));
        Mat weight_data_g(maxk * channels_g * num_output_g, (void*)((const float*)weight_data + maxk * channels_g * num_output_g * g));
        Mat bias_data_g;
        if (bias_term)
            bias_data_g = Mat(num_output_g, (void*)((const float*)bias_data + num_output_g * g));

        // call Convolution
        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Convolution);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output_g);// num_output
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, dilation_w);
        pd.set(12, dilation_h);
        pd.set(3, stride_w);
        pd.set(13, stride_h);
        pd.set(4, 0);// pad_w
        pd.set(14, 0);// pad_h
        pd.set(5, bias_term);
        pd.set(6, maxk * channels_g * num_output_g);// weight_data_size

        op->load_param(pd);

        // set weights
        ncnn::Mat weights[2];
        weights[0] = weight_data_g;
        weights[1] = bias_data_g;

        op->load_model(ModelBinFromMatArray(weights));

        // forward
        op->forward(bottom_blob_bordered_g, top_blob_g);

        delete op;
    }

    return 0;
}

} // namespace ncnn
