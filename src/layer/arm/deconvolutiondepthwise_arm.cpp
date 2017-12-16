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

#include "deconvolutiondepthwise_arm.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ncnn {

#include "deconvolution_3x3.h"
#include "deconvolution_4x4.h"

DEFINE_LAYER_CREATOR(DeconvolutionDepthWise_arm)

int DeconvolutionDepthWise_arm::forward(const Mat& bottom_blob, Mat& top_blob) const
{
    // convolv with NxN kernel
    // value = value + bias

    if (kernel_w != kernel_h || stride_w != stride_h)
    {
        return DeconvolutionDepthWise::forward(bottom_blob, top_blob);
    }

    const int kernel_size = kernel_w;
    const int stride = stride_w;

    if ((kernel_size != 3 && kernel_size != 4) || stride > 2 || dilation_w != 1 || dilation_h != 1)
    {
        return DeconvolutionDepthWise::forward(bottom_blob, top_blob);
    }

    typedef void (*deconv_func)(const Mat&, Mat&, const Mat&, const Mat&);

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
        return DeconvolutionDepthWise::forward(bottom_blob, top_blob);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    int outw = (w - 1) * stride + kernel_size;
    int outh = (h - 1) * stride + kernel_size;

    Mat top_blob_bordered(outw, outh, num_output);
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_size * kernel_size;

    // depth-wise
    if (channels == group && group == num_output)
    {
#ifdef _OPENMP
        int nested_current = omp_get_nested();
        omp_set_nested(0);
#endif

        #pragma omp parallel for
        for (int g=0; g<group; g++)
        {
            Mat top_blob_bordered_g = top_blob_bordered.channel(g);
            Mat bottom_blob_g = bottom_blob.channel(g);
            Mat weight_data_g(maxk, (float*)(weight_data + maxk * g));

            Mat bias_data_g;
            if (bias_term)
                bias_data_g = Mat(1, (float*)(bias_data + g));

            deconv(bottom_blob_g, top_blob_bordered_g, weight_data_g, bias_data_g);
        }

#ifdef _OPENMP
        omp_set_nested(nested_current);
#endif
    } else {
        const int channels_g = channels / group;
        const int num_output_g = num_output / group;

        for (int g=0; g<group; g++)
        {
            Mat top_blob_bordered_g(outw, outh, num_output_g, top_blob_bordered.channel(num_output_g * g));
            Mat bottom_blob_g(w, h, channels_g, bottom_blob.channel(channels_g * g).data);
            Mat weight_data_g(maxk * channels_g * num_output_g, (float*)(weight_data + maxk * channels_g * num_output_g * g));
            Mat bias_data_g;
            if (bias_term)
                bias_data_g = Mat(num_output_g, (float*)(bias_data + num_output_g * g));

            deconv(bottom_blob_g, top_blob_bordered_g, weight_data_g, bias_data_g);
        }
    }

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

}// namespace ncnn
