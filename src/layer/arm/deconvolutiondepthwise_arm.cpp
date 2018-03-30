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

#include "layer_type.h"

namespace ncnn {

DEFINE_LAYER_CREATOR(DeconvolutionDepthWise_arm)

int DeconvolutionDepthWise_arm::forward(const Mat& bottom_blob, Mat& top_blob) const
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

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;

    Mat top_blob_bordered = top_blob;
    top_blob_bordered.create(outw, outh, num_output);
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

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
            Mat bottom_blob_g(w, h, 1, bottom_blob.channel(g).data);
            Mat top_blob_bordered_g(outw, outh, 1, top_blob_bordered.channel(g));
            Mat weight_data_g(maxk, (void*)((const float*)weight_data + maxk * g));

            Mat bias_data_g;
            if (bias_term)
                bias_data_g = Mat(1, (void*)((const float*)bias_data + g));

            // call Deconvolution
            ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Deconvolution);

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
            op->forward(bottom_blob_g, top_blob_bordered_g);

            delete op;
        }

#ifdef _OPENMP
        omp_set_nested(nested_current);
#endif
    }
    else
    {
        const int channels_g = channels / group;
        const int num_output_g = num_output / group;

        for (int g=0; g<group; g++)
        {
            Mat bottom_blob_g(w, h, channels_g, bottom_blob.channel(channels_g * g).data);
            Mat top_blob_bordered_g(outw, outh, num_output_g, top_blob_bordered.channel(num_output_g * g));
            Mat weight_data_g(maxk * channels_g * num_output_g, (void*)((const float*)weight_data + maxk * channels_g * num_output_g * g));
            Mat bias_data_g;
            if (bias_term)
                bias_data_g = Mat(num_output_g, (void*)((const float*)bias_data + num_output_g * g));

            // call Deconvolution
            ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Deconvolution);

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
            op->forward(bottom_blob_g, top_blob_bordered_g);

            delete op;
        }
    }

    top_blob = top_blob_bordered;

    if (pad_w > 0 || pad_h > 0)
    {
        copy_cut_border(top_blob_bordered, top_blob, pad_h, pad_h, pad_w, pad_w);
        if (top_blob.empty())
            return -100;

        outw = top_blob.w;
        outh = top_blob.h;
    }

    return 0;

}

} // namespace ncnn
