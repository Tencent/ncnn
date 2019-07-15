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
#include "layer_type.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

namespace ncnn {

DEFINE_LAYER_CREATOR(DeconvolutionDepthWise_arm)

DeconvolutionDepthWise_arm::DeconvolutionDepthWise_arm()
{
    activation = 0;
}

int DeconvolutionDepthWise_arm::create_pipeline(const Option& opt)
{
    if (activation_type == 1)
    {
        activation = ncnn::create_layer(ncnn::LayerType::ReLU);

        ncnn::ParamDict pd;
        activation->load_param(pd);
    }
    else if (activation_type == 2)
    {
        activation = ncnn::create_layer(ncnn::LayerType::ReLU);

        ncnn::ParamDict pd;
        pd.set(0, activation_params[0]);// slope
        activation->load_param(pd);
    }
    else if (activation_type == 3)
    {
        activation = ncnn::create_layer(ncnn::LayerType::Clip);

        ncnn::ParamDict pd;
        pd.set(0, activation_params[0]);// min
        pd.set(1, activation_params[1]);// max
        activation->load_param(pd);
    }
    else if (activation_type == 4)
    {
        activation = ncnn::create_layer(ncnn::LayerType::Sigmoid);

        ncnn::ParamDict pd;
        activation->load_param(pd);
    }

    if (activation)
    {
        Option opt_cpu = opt;
        opt_cpu.use_vulkan_compute = false;
        activation->create_pipeline(opt_cpu);
    }

    return 0;
}

int DeconvolutionDepthWise_arm::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        Option opt_cpu = opt;
        opt_cpu.use_vulkan_compute = false;
        activation->destroy_pipeline(opt_cpu);
        delete activation;
        activation = 0;
    }

    return 0;
}

int DeconvolutionDepthWise_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    if (channels % group != 0 || num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;

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

    const int maxk = kernel_w * kernel_h;

    // depth-wise
    if (channels == group && group == num_output)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g=0; g<group; g++)
        {
            Mat bottom_blob_g = bottom_blob.channel_range(g, 1);
            Mat top_blob_bordered_g = top_blob_bordered.channel_range(g, 1);
            Mat weight_data_g = weight_data.range(maxk * g, maxk);

            Mat bias_data_g;
            if (bias_term)
                bias_data_g = bias_data.range(g, 1);

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

            ncnn::Option opt_g = opt;
            opt_g.num_threads = 1;
            opt_g.blob_allocator = top_blob_bordered.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);

            delete op;
        }
    }
    else
    {
        const int channels_g = channels / group;
        const int num_output_g = num_output / group;

        for (int g=0; g<group; g++)
        {
            Mat bottom_blob_g = bottom_blob.channel_range(channels_g * g, channels_g);
            Mat top_blob_bordered_g = top_blob_bordered.channel_range(num_output_g * g, num_output_g);
            Mat weight_data_g = weight_data.range(maxk * channels_g * num_output_g * g, maxk * channels_g * num_output_g);
            Mat bias_data_g;
            if (bias_term)
                bias_data_g = bias_data.range(num_output_g * g, num_output_g);

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

            ncnn::Option opt_g = opt;
            opt_g.blob_allocator = top_blob_bordered.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);

            delete op;
        }
    }

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

    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;
}

} // namespace ncnn
