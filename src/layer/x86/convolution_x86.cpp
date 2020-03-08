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

#include "platform.h"
#if __SSE2__
#include <emmintrin.h>
#endif
#if __AVX__
#include <immintrin.h>
#endif

#include "layer_type.h"
#include "benchmark.h"

namespace ncnn {

#include "convolution_sgemm.h"
#include "convolution_1x1.h"
#include "convolution_3x3.h"
#include "convolution_5x5.h"
#include "convolution_7x7.h"
#include "convolution_sgemm_int8.h"
#include "convolution_1x1_int8.h"
#include "convolution_3x3_int8.h"

DEFINE_LAYER_CREATOR(Convolution_x86)

Convolution_x86::Convolution_x86()
{
    activation = 0;
    convolution_dilation1 = 0;
}

int Convolution_x86::create_pipeline(const Option& opt)
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
        if (use_int8_requantize)
        {
            pd.set(0, activation_params[0] * top_blob_int8_scale);// min
            pd.set(1, activation_params[1] * top_blob_int8_scale);// max
        }else{
            pd.set(0, activation_params[0]);// min
            pd.set(1, activation_params[1]);// max
        }

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
        activation->create_pipeline(opt);
    }

    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8_x86(opt);
    }

    int kernel_size = kernel_w * kernel_h;
    int num_input = weight_data_size / kernel_size / num_output;

    use_winograd3x3 = false;

    if (kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    {
        convolution_dilation1 = ncnn::create_layer(ncnn::LayerType::Convolution);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output);// num_output
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, 1);
        pd.set(12, 1);
        pd.set(3, 1);// stride_w
        pd.set(13, 1);// stride_h
        pd.set(4, 0);// pad_w
        pd.set(14, 0);// pad_h
        pd.set(5, bias_term);
        pd.set(6, weight_data_size);

        convolution_dilation1->load_param(pd);

        // set weights
        if (bias_term)
        {
            ncnn::Mat weights[2];
            weights[0] = weight_data;
            weights[1] = bias_data;

            convolution_dilation1->load_model(ModelBinFromMatArray(weights));
        }
        else
        {
            ncnn::Mat weights[1];
            weights[0] = weight_data;

            convolution_dilation1->load_model(ModelBinFromMatArray(weights));
        }

        convolution_dilation1->create_pipeline(opt);
    }
    else if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1
        && num_input >= 16 && num_output >= 16)
    {
        // winograd is slow on small channel count
        use_winograd3x3 = true;

        conv3x3s1_winograd23_transform_kernel_sse(weight_data, weight_3x3_winograd23_data, num_input, num_output);
//         conv3x3s1_winograd43_transform_kernel_sse(weight_data, weight_3x3_winograd43_data, num_input, num_output);

        // for small size
        conv_im2col_sgemm_transform_kernel_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_size);
    }
    else
    {
        conv_im2col_sgemm_transform_kernel_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_size);
    }

    return 0;
}

int Convolution_x86::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    if (convolution_dilation1)
    {
        convolution_dilation1->destroy_pipeline(opt);
        delete convolution_dilation1;
        convolution_dilation1 = 0;
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

    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8_x86(bottom_blob, top_blob, opt);
    }

    if ((dilation_w > 1 || dilation_h > 1) && (stride_w > 1 || stride_h > 1))
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    if ((dilation_w > 1 || dilation_h > 1) && dilation_w != dilation_h)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    // float32
    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    {
        return forwardDilation_x86(bottom_blob_bordered, top_blob, opt);
    }

    if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
    {
        if (use_winograd3x3 && outw >= 8 && outh >=8)
        {
            conv3x3s1_winograd23_sse(bottom_blob_bordered, top_blob, weight_3x3_winograd23_data, bias_data, opt);
//             conv3x3s1_winograd43_sse(bottom_blob_bordered, top_blob, weight_3x3_winograd43_data, bias_data, opt);
        }
        else
        {
            conv_im2col_sgemm_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, opt);
        }

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
    }
    else
    {
//         conv1x1s1_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);
//         conv1x1s2_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);
//         conv3x3s1_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);
//         conv3x3s2_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);
//         conv5x5s1_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

        conv_im2col_sgemm_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, opt);

        if (activation)
        {
            activation->forward_inplace(top_blob, opt);
        }
    }

    return 0;
}

int Convolution_x86::create_pipeline_int8_x86(const Option& opt)
{
    int kernel_size = kernel_w * kernel_h;
    int num_input = weight_data_size / kernel_size / num_output;

    use_winograd3x3_int8 = false;

    if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1
        && num_input >= 16 && num_output >= 16)
    {
        // winograd is slow on small channel count
        use_winograd3x3_int8 = true;

        conv3x3s1_winograd23_transform_kernel_int8_sse(weight_data, weight_3x3_winograd23_data_int8, num_input, num_output);
//         conv3x3s1_winograd43_transform_kernel_int8_sse(weight_data, weight_3x3_winograd23_data_int8, num_input, num_output);
    }
    else
    {
        // TODO offline transform weight
    }

    return 0;
}

int Convolution_x86::forward_int8_x86(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    if (dilation_w > 1 || dilation_h > 1)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_unbordered = bottom_blob;
    if (elemsize != 1)
    {
        Option opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;

        quantize_float32_to_int8(bottom_blob, bottom_blob_unbordered, bottom_blob_int8_scale, opt_g);
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob_unbordered, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    // int8
    size_t out_elemsize = use_int8_requantize ? 1u : 4u;

    top_blob.create(outw, outh, num_output, out_elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // int8
    if (use_int8_requantize)
    {
        Mat top_blob_tm;
        top_blob_tm.create(outw, outh, num_output, (size_t)4u, opt.workspace_allocator);
        if (top_blob_tm.empty())
            return -100;

        if (use_winograd3x3_int8)
        {
            conv3x3s1_winograd23_int8_sse(bottom_blob_bordered, top_blob_tm, weight_3x3_winograd23_data_int8, opt);
//             conv3x3s1_winograd43_int8_sse(bottom_blob_bordered, top_blob_tm, weight_3x3_winograd23_data_int8, opt);

            // requantize, reverse scale inplace
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<num_output; p++)
            {
                Option opt_g = opt;
                opt_g.num_threads = 1;
                opt_g.blob_allocator = top_blob.allocator;

                Mat top_blob_tm_g = top_blob_tm.channel_range(p, 1);
                Mat top_blob_g = top_blob.channel_range(p, 1);

                // requantize and relu
                float scale_in;
                if (weight_data_int8_scales[p] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[p]);

                float scale_out = top_blob_int8_scale;//FIXME load param

                requantize_int8_to_int8(top_blob_tm_g, top_blob_g, scale_in, scale_out, bias_term ? (const float*)bias_data + p : 0, bias_term ? 1 : 0, 0, opt_g);
            }
        }
        else
        {
            std::vector<float> requantize_scales;
            for (int p=0; p<num_output; p++)
            {
                float scale_in;
                if (weight_data_int8_scales[p] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[p]);

                float scale_out = top_blob_int8_scale;

                requantize_scales.push_back(scale_in);
                requantize_scales.push_back(scale_out);
            }

            conv_im2col_sgemm_int8_requant_sse(bottom_blob_bordered, top_blob, weight_data, kernel_w, kernel_h, stride_w, stride_h, bias_data, requantize_scales, opt);
        }
    }
    else
    {
        if (use_winograd3x3_int8)
        {
            conv3x3s1_winograd23_int8_sse(bottom_blob_bordered, top_blob, weight_3x3_winograd23_data_int8, opt);
//             conv3x3s1_winograd43_int8_sse(bottom_blob_bordered, top_blob, weight_3x3_winograd23_data_int8, opt);

            // dequantize, reverse scale inplace
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p=0; p<num_output; p++)
            {
                Option opt_g = opt;
                opt_g.num_threads = 1;
                opt_g.blob_allocator = top_blob.allocator;

                Mat top_blob_g = top_blob.channel_range(p, 1);

                // dequantize
                float scale_in;
                if (weight_data_int8_scales[p] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[p]);

                dequantize_int32_to_float32(top_blob_g, scale_in, bias_term ? (const float*)bias_data + p : 0, bias_term ? 1 : 0, opt_g);
            }
        }
        else
        {
            std::vector<float> dequantize_scales;
            for (int p=0; p<num_output; p++)
            {
                float scale_in;
                if (weight_data_int8_scales[p] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (bottom_blob_int8_scale * weight_data_int8_scales[p]);

                dequantize_scales.push_back(scale_in);
            }

            conv_im2col_sgemm_int8_dequant_sse(bottom_blob_bordered, top_blob, weight_data, kernel_w, kernel_h, stride_w, stride_h, bias_data, dequantize_scales, opt);
        }
    }

    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;
}

int Convolution_x86::forwardDilation_x86(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_size = kernel_w;
    const int stride = stride_w;
    const int dilation = dilation_w;
    const int kernel_extent = dilation * (kernel_size - 1) + 1;

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
                    const float* ptr = (const float *)bottom_blob.channel(c) + dilation * i * w + x * w + y;
                    for (int j = 0; j < inner_w; j ++)
                    {
                        outptr[j] = ptr[j*dilation];
                    }
                    outptr += inner_w;
                }
            }

            Option opt_g = opt;
            opt_g.blob_allocator = inner_top_blob.allocator;
            convolution_dilation1->forward(inner_bottom_blob, inner_top_blob, opt_g);

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

    if (activation)
    {
        activation->forward_inplace(top_blob, opt);
    }

    return 0;
}

} // namespace ncnn
