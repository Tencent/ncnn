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

#include <emmintrin.h>
#include "sse_activation.h"
#include "sse_usability.h"

#if __AVX__
#include <immintrin.h>
#include "avx_activation.h"
#include "avx_usability.h"
#endif

#include "layer_type.h"

namespace ncnn {
#if __AVX__
#include "convolutiondepthwise_3x3_pack8_fp16.h"
#include "convolutiondepthwise_3x3_pack8.h"
#include "convolutiondepthwise_5x5_pack8.h"
#endif
#include "convolutiondepthwise_3x3.h"
#include "convolutiondepthwise_3x3_int8.h"

ConvolutionDepthWise_x86::ConvolutionDepthWise_x86()
{
    support_packing = true;
#if __AVX__
    support_weight_fp16_storage = true;
#endif
    activation = 0;
}

int ConvolutionDepthWise_x86::create_pipeline(const Option& opt)
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
        pd.set(0, activation_params[0]); // slope
        activation->load_param(pd);
    }
    else if (activation_type == 3)
    {
        activation = ncnn::create_layer(ncnn::LayerType::Clip);

        ncnn::ParamDict pd;
        if (use_int8_requantize)
        {
            pd.set(0, activation_params[0] * top_blob_int8_scale); // min
            pd.set(1, activation_params[1] * top_blob_int8_scale); // max
        }
        else
        {
            pd.set(0, activation_params[0]); // min
            pd.set(1, activation_params[1]); // max
        }

        activation->load_param(pd);
    }
    else if (activation_type == 4)
    {
        activation = ncnn::create_layer(ncnn::LayerType::Sigmoid);

        ncnn::ParamDict pd;
        activation->load_param(pd);
    }
    else if (activation_type == 5)
    {
        activation = ncnn::create_layer(ncnn::LayerType::Mish);

        ncnn::ParamDict pd;
        activation->load_param(pd);
    }
    else if (activation_type == 5)
    {
        activation = ncnn::create_layer(ncnn::LayerType::Mish);

        ncnn::ParamDict pd;
        activation->load_param(pd);
    }
    if (activation)
    {
        activation->create_pipeline(opt);
    }

    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        support_packing = false;

        return create_pipeline_int8_x86(opt);
    }

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;
        if (opt.use_packing_layout)
        {
#if __AVX__
            elempack = channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4 : 1;
#else
            elempack = channels % 4 == 0 ? 4 : 1;
#endif
        }

#if __AVX__
        // pack8
        if (elempack == 8)
        {
            if (opt.use_weight_fp16_storage && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                Mat weight_data_r2 = weight_data.reshape(maxk, group);
                Mat weight_data_tmp;
                convert_packing(weight_data_r2, weight_data_tmp, 8);
                ncnn::cast_float32_to_float16(weight_data_tmp, weight_data_packed, opt);
                return 0;
            }
            if (opt.use_weight_fp16_storage && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                Mat weight_data_r2 = weight_data.reshape(maxk, group);
                Mat weight_data_tmp;
                convert_packing(weight_data_r2, weight_data_tmp, 8);
                ncnn::cast_float32_to_float16(weight_data_tmp, weight_data_packed, opt);
                return 0;
            }

            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_packed, 8);

            return 0;
        }
#endif // __AVX__

        // pack4
        if (elempack == 4)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_packed, 4);

            return 0;
        }

        if (elempack == 1)
        {
            // depth-wise specific
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                return 0;
            }
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                return 0;
            }
        }
    }

    // group convolution
    create_group_ops(opt);

    return 0;
}

int ConvolutionDepthWise_x86::create_group_ops(const Option& opt)
{
    // create Convolution op for each group
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    for (int i = 0; i < (int)group_ops.size(); i++)
        delete group_ops[i];

    group_ops.clear();

    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    group_ops.resize(group);

    for (int g = 0; g < group; g++)
    {
        Mat weight_data_g = weight_data.range(maxk * channels_g * num_output_g * g, maxk * channels_g * num_output_g);
        Mat bias_data_g;
        if (bias_term)
            bias_data_g = bias_data.range(num_output_g * g, num_output_g);

        ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Convolution);

        // FIXME
        // ((ncnn::Convolution*)op)->use_int8_requantize = use_int8_requantize;

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output_g); // num_output
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, dilation_w);
        pd.set(12, dilation_h);
        pd.set(3, stride_w);
        pd.set(13, stride_h);
        pd.set(4, 0);  // pad_w
        pd.set(14, 0); // pad_h
        pd.set(5, bias_term);
        pd.set(6, maxk * channels_g * num_output_g); // weight_data_size
        pd.set(8, int8_scale_term);
        pd.set(9, activation_type);
        pd.set(10, activation_params);

        op->load_param(pd);

        // set weights
        if (bias_term)
        {
            ncnn::Mat weights[4];
            weights[0] = weight_data_g;
            weights[1] = bias_data_g;

            if (int8_scale_term)
            {
                Mat weight_data_int8_scales_g(num_output_g);
                weight_data_int8_scales_g.fill(weight_data_int8_scales[g]);
                weights[2] = weight_data_int8_scales_g;
                weights[3] = bottom_blob_int8_scales.range(g, 1);
            }

            op->load_model(ModelBinFromMatArray(weights));
        }
        else
        {
            ncnn::Mat weights[3];
            weights[0] = weight_data_g;

            if (int8_scale_term)
            {
                Mat weight_data_int8_scales_g(num_output_g);
                weight_data_int8_scales_g.fill(weight_data_int8_scales[g]);
                weights[1] = weight_data_int8_scales_g;
                weights[2] = bottom_blob_int8_scales.range(g, 1);
            }

            op->load_model(ModelBinFromMatArray(weights));
        }

        op->create_pipeline(opt);

        group_ops[g] = op;
    }

    return 0;
}

int ConvolutionDepthWise_x86::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

    for (int i = 0; i < (int)group_ops.size(); i++)
    {
        group_ops[i]->destroy_pipeline(opt);
        delete group_ops[i];
    }
    group_ops.clear();

    return 0;
}

int ConvolutionDepthWise_x86::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8_x86(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

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
    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
#if __AVX__
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // fprintf(stderr, "Depthwise kernel %d x %d elempack=%d group=%d channels = %d stride = %d x %d  \n",kernel_w,kernel_h,elempack,group,channels,stride_w,stride_h );

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
#if __AVX__
        if (elempack == 8)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                if (opt.use_weight_fp16_storage)
                {
                    convdw3x3s1_fp16_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);
                }
                else
                {
                    convdw3x3s1_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);
                }

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                if (opt.use_weight_fp16_storage)
                {
                    convdw3x3s2_fp16_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);
                }
                else
                {
                    convdw3x3s2_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);
                }

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw5x5s1_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw5x5s2_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            else
            {
                const int maxk = kernel_w * kernel_h;

                // kernel offsets
                std::vector<int> _space_ofs(maxk);
                int* space_ofs = &_space_ofs[0];
                {
                    int p1 = 0;
                    int p2 = 0;
                    int gap = w * dilation_h - kernel_w * dilation_w;
                    for (int i = 0; i < kernel_h; i++)
                    {
                        for (int j = 0; j < kernel_w; j++)
                        {
                            space_ofs[p1] = p2;
                            p1++;
                            p2 += dilation_w;
                        }
                        p2 += gap;
                    }
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    float* outptr = top_blob.channel(g);
                    const float* kptr = (const float*)weight_data_packed + maxk * g * 8;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            __m256 _sum = _mm256_set1_ps(0.f);

                            if (bias_term)
                            {
                                _sum = _mm256_loadu_ps(((const float*)bias_data) + g * 8);
                            }

                            const float* sptr = m.row(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                __m256 _val = _mm256_loadu_ps(sptr + space_ofs[k] * 8);
                                __m256 _w = _mm256_loadu_ps(kptr + k * 8);
                                _sum = _mm256_fmadd_ps(_val, _w, _sum);
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params);

                            _mm256_storeu_ps(outptr + j * 8, _sum);
                        }

                        outptr += outw * 8;
                    }
                }

                return 0;
            }
        }
#endif // __AVX__

        if (elempack == 4)
        {
            {
                const int maxk = kernel_w * kernel_h;

                // kernel offsets
                std::vector<int> _space_ofs(maxk);
                int* space_ofs = &_space_ofs[0];
                {
                    int p1 = 0;
                    int p2 = 0;
                    int gap = w * dilation_h - kernel_w * dilation_w;
                    for (int i = 0; i < kernel_h; i++)
                    {
                        for (int j = 0; j < kernel_w; j++)
                        {
                            space_ofs[p1] = p2;
                            p1++;
                            p2 += dilation_w;
                        }
                        p2 += gap;
                    }
                }

                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    float* outptr = top_blob.channel(g);
                    const float* kptr = (const float*)weight_data_packed + maxk * g * 4;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            __m128 _sum = _mm_set1_ps(0.f);

                            if (bias_term)
                            {
                                _sum = _mm_loadu_ps(((const float*)bias_data) + g * 4);
                            }

                            const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                __m128 _val = _mm_loadu_ps(sptr + space_ofs[k] * 4);
                                __m128 _w = _mm_loadu_ps(kptr + k * 4);
                                _sum = _mm_add_ps(_mm_mul_ps(_val, _w), _sum);
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params);

                            _mm_storeu_ps(outptr + j * 4, _sum);
                        }

                        outptr += outw * 4;
                    }
                }

                return 0;
            }
        }

        if (elempack == 1)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
        }
    }

    // group convolution
    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    int g_elempack = 1;
    int out_g_elempack = 1;
    if (opt.use_packing_layout)
    {
#if __AVX__
        g_elempack = channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
        out_g_elempack = num_output_g % 8 == 0 ? 8 : num_output_g % 4 == 0 ? 4 : 1;
#else
        g_elempack = channels_g % 4 == 0 ? 4 : 1;
        out_g_elempack = num_output_g % 4 == 0 ? 4 : 1;
#endif
    }

    // unpacking
    Mat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack > g_elempack)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, g_elempack, opt_p);
    }

    Mat top_blob_unpacked = top_blob;
    if (out_g_elempack < out_elempack)
    {
        top_blob_unpacked.create(outw, outh, num_output / out_g_elempack, out_elemsize / out_elempack * out_g_elempack, out_g_elempack, opt.workspace_allocator);
        if (top_blob_unpacked.empty())
            return -100;
    }

    for (int g = 0; g < group; g++)
    {
        const Mat bottom_blob_bordered_g = bottom_blob_bordered_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
        Mat top_blob_g = top_blob_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

        const ncnn::Layer* op = group_ops[g];

        Option opt_g = opt;
        opt_g.blob_allocator = top_blob_unpacked.allocator;

        // forward
        op->forward(bottom_blob_bordered_g, top_blob_g, opt_g);
    }

    // packing
    if (out_g_elempack < out_elempack)
    {
        convert_packing(top_blob_unpacked, top_blob, out_elempack, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}

int ConvolutionDepthWise_x86::create_pipeline_int8_x86(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                return 0;
            }
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                return 0;
            }
        }
    }

    // group convolution
    create_group_ops(opt);

    return 0;
}

int ConvolutionDepthWise_x86::forward_int8_x86(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_unbordered = bottom_blob;
    if (elemsize != 1)
    {
        bottom_blob_unbordered.create(w, h, channels, (size_t)1u, opt.workspace_allocator);
        if (bottom_blob_unbordered.empty())
            return -100;

        const int channels_g = channels / group;

        // quantize, scale and round to nearest
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < group; g++)
        {
            Option opt_g = opt;
            opt_g.num_threads = 1;
            opt_g.blob_allocator = bottom_blob_unbordered.allocator;

            const Mat bottom_blob_g = bottom_blob.channel_range(channels_g * g, channels_g);
            Mat bottom_blob_int8_g = bottom_blob_unbordered.channel_range(channels_g * g, channels_g);

            quantize_float32_to_int8(bottom_blob_g, bottom_blob_int8_g, bottom_blob_int8_scales[g], opt_g);
        }
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

    // depth-wise
    if (channels == group && group == num_output)
    {
        if (use_int8_requantize)
        {
            std::vector<float> requantize_scales;
            for (int g = 0; g < group; g++)
            {
                float scale_in;
                if (weight_data_int8_scales[g] == 0)
                    scale_in = 0;
                else
                    scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                float scale_out = top_blob_int8_scale;

                requantize_scales.push_back(scale_in);
                requantize_scales.push_back(scale_out);
            }

            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_int8_requant_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, requantize_scales, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_int8_requant_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, requantize_scales, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
        }
        else
        {
            std::vector<float> dequantize_scales;
            for (int g = 0; g < group; g++)
            {
                float top_rescale = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                dequantize_scales.push_back(top_rescale);
            }

            if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
            {
                convdw3x3s1_int8_dequant_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, dequantize_scales, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            if (kernel_w == 3 && kernel_h == 3 && stride_w == 2 && stride_h == 2 && dilation_w == 1 && dilation_h == 1)
            {
                convdw3x3s2_int8_dequant_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, dequantize_scales, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
        }
    }

    // group convolution
    const int channels_g = channels / group;
    const int num_output_g = num_output / group;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        const Mat bottom_blob_bordered_g = bottom_blob_bordered.channel_range(channels_g * g, channels_g);
        Mat top_blob_g = top_blob.channel_range(num_output_g * g, num_output_g);

        const ncnn::Layer* op = group_ops[g];

        Option opt_g = opt;
        opt_g.blob_allocator = top_blob.allocator;

        // forward
        op->forward(bottom_blob_bordered_g, top_blob_g, opt_g);
    }

    return 0;
}

} // namespace ncnn
