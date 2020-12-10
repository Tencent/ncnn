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

#if __SSE2__
#include <emmintrin.h>
#include "sse_activation.h"
#include "sse_usability.h"

#if __AVX__
#include <immintrin.h>
#include "avx_activation.h"
#include "avx_usability.h"
#endif
#endif // __SSE2__

#include "benchmark.h"
#include "layer_type.h"

namespace ncnn {

#include "convolution_sgemm.h"
#include "convolution_sgemm_int8.h"
#if __SSE2__
#if __AVX__
#include "convolution_3x3_pack1to8.h"
#include "convolution_3x3_pack8to1.h"
#include "convolution_3x3_pack8.h"
#include "convolution_2x2_pack8.h"
#include "convolution_2x2_pack8_fp16.h"
#include "convolution_1x1_pack8.h"
#include "convolution_1x1_pack8_fp16.h"
#endif
#endif // __SSE2__

#include "convolution_1x1.h"
#include "convolution_1x1_int8.h"
#include "convolution_3x3.h"
#include "convolution_3x3_int8.h"
#include "convolution_5x5.h"
#include "convolution_7x7.h"

Convolution_x86::Convolution_x86()
{
#if __SSE2__
    support_packing = true;
#if __AVX__
    support_weight_fp16_storage = true;
#endif
#endif // __SSE2__

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

    if (activation)
    {
        activation->create_pipeline(opt);
    }

    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        support_packing = false;
        return create_pipeline_int8_x86(opt);
    }

    int kernel_size = kernel_w * kernel_h;
    int num_input = weight_data_size / kernel_size / num_output;

    use_winograd3x3 = false;

    if (!opt.use_packing_layout && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    {
        convolution_dilation1 = ncnn::create_layer(ncnn::LayerType::Convolution);

        // set param
        ncnn::ParamDict pd;
        pd.set(0, num_output); // num_output
        pd.set(1, kernel_w);
        pd.set(11, kernel_h);
        pd.set(2, 1);
        pd.set(12, 1);
        pd.set(3, 1);  // stride_w
        pd.set(13, 1); // stride_h
        pd.set(4, 0);  // pad_w
        pd.set(14, 0); // pad_h
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

        return 0;
    }

    int elempack = 1;
    int out_elempack = 1;

#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX__
        elempack = num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        elempack = num_input % 4 == 0 ? 4 : 1;
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        if (opt.use_winograd_convolution && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && num_input >= 16 && num_output >= 16)
        {
            // winograd is slow on small channel count
            use_winograd3x3 = true;

            conv3x3s1_winograd23_transform_kernel_sse(weight_data, weight_3x3_winograd23_data, num_input, num_output);
            // conv3x3s1_winograd43_transform_kernel_sse(weight_data, weight_3x3_winograd43_data, num_input, num_output);

            // for small size
            conv_im2col_sgemm_transform_kernel_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_size);
        }
        else
        {
            conv_im2col_sgemm_transform_kernel_sse(weight_data, weight_sgemm_data, num_input, num_output, kernel_size);
        }

        return 0;
    }

    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        weight_data_packed.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            Mat g0 = weight_data_packed.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                float* g00 = g0.row(p / elempack);

                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                            g00[0] = k00[k];

                            g00++;
                        }
                    }
                }
            }
        }
    }

#if __AVX__
    // pack8
    if (elempack == 8 && out_elempack == 8)
    {
        if (opt.use_weight_fp16_storage && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_fp16_pack8_avx(weight_data, weight_data_packed, num_input, num_output);
        }
        else if (opt.use_weight_fp16_storage && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s1_sgemm_transform_kernel_fp16_pack8_avx(weight_data, weight_data_packed, num_input, num_output);
        }
        else if (opt.use_weight_fp16_storage && kernel_w == 2 && kernel_h == 2 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv2x2s1_weight_fp16_pack8_avx(weight_data, weight_data_packed, num_input, num_output);
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_transform_kernel_pack8_avx(weight_data, weight_data_packed, num_input, num_output);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_transform_kernel_pack8_avx(weight_data, weight_data_packed, num_input, num_output);
        }
    }
#endif

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

    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8_x86(bottom_blob, top_blob, opt);
    }

    if (bottom_blob.dims != 3)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    if (!opt.use_packing_layout && (dilation_w > 1 || dilation_h > 1) && (stride_w > 1 || stride_h > 1))
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
    }

    if (!opt.use_packing_layout && (dilation_w > 1 || dilation_h > 1) && dilation_w != dilation_h)
    {
        return Convolution::forward(bottom_blob, top_blob, opt);
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
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX__
        out_elempack = num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#else
        out_elempack = num_output % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (!opt.use_packing_layout && kernel_w == kernel_h && dilation_w != 1 && dilation_h == dilation_w && stride_w == 1 && stride_h == 1)
    {
        if (outw >= dilation_w && outh >= dilation_h)
        {
            return forwardDilation_x86(bottom_blob_bordered, top_blob, opt);
        }
    }

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

#if __SSE2__
#if __AVX__
    if (elempack == 8 && out_elempack == 8)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (opt.use_weight_fp16_storage)
            {
                conv1x1s1_sgemm_fp16_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);
            }
            else
            {
                conv1x1s1_sgemm_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            if (opt.use_weight_fp16_storage)
            {
                conv1x1s2_fp16_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);
            }
            else
            {
                conv1x1s2_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);
            }
            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }

        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_winograd64_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 2 && kernel_h == 2 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (opt.use_weight_fp16_storage)
            {
                conv2x2s1_fp16_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);
            }
            else
            {
                conv2x2s1_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        __m256 _sum = _mm256_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm256_loadu_ps(((const float*)bias_data) + p * 8);
                        }

                        const float* kptr = (const float*)weight_data_packed + maxk * channels * p * 64;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                __m256 _val0 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8));
                                __m256 _val1 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 1);
                                __m256 _val2 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 2);
                                __m256 _val3 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 3);
                                __m256 _val4 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 4);
                                __m256 _val5 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 5);
                                __m256 _val6 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 6);
                                __m256 _val7 = _mm256_broadcast_ss((sptr + space_ofs[k] * 8) + 7);

                                __m256 _w0 = _mm256_loadu_ps(kptr);
                                __m256 _mul0 = _mm256_mul_ps(_val0, _w0);
                                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                                __m256 _mul1 = _mm256_mul_ps(_val1, _w1);
                                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                                __m256 _mul2 = _mm256_mul_ps(_val2, _w2);
                                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                                __m256 _mul3 = _mm256_mul_ps(_val3, _w3);
                                __m256 _w4 = _mm256_loadu_ps(kptr + 32);
                                __m256 _mul4 = _mm256_mul_ps(_val4, _w4);
                                __m256 _w5 = _mm256_loadu_ps(kptr + 40);
                                __m256 _mul5 = _mm256_mul_ps(_val5, _w5);
                                __m256 _w6 = _mm256_loadu_ps(kptr + 48);
                                __m256 _mul6 = _mm256_mul_ps(_val6, _w6);
                                __m256 _w7 = _mm256_loadu_ps(kptr + 56);
                                __m256 _mul7 = _mm256_mul_ps(_val7, _w7);
                                __m256 _sum01 = _mm256_add_ps(_mul0, _mul1);
                                __m256 _sum23 = _mm256_add_ps(_mul2, _mul3);
                                __m256 _sum45 = _mm256_add_ps(_mul4, _mul5);
                                __m256 _sum67 = _mm256_add_ps(_mul6, _mul7);
                                __m256 _sum_lo = _mm256_add_ps(_sum01, _sum23);
                                __m256 _sum_hi = _mm256_add_ps(_sum45, _sum67);
                                __m256 _sum_all = _mm256_add_ps(_sum_lo, _sum_hi);
                                _sum = _mm256_add_ps(_sum_all, _sum);

                                kptr += 64;
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        _mm256_storeu_ps(outptr + j * 8, _sum);
                    }

                    outptr += outw * 8;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 8)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1to8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1to8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        __m256 _sum = _mm256_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm256_loadu_ps(((const float*)bias_data) + p * 8);
                        }

                        const float* kptr = (const float*)weight_data_packed + maxk * channels * p * 8;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++) // 29.23
                            {
                                __m256 _val = _mm256_set1_ps(sptr[space_ofs[k]]);
                                __m256 _w = _mm256_loadu_ps(kptr);
                                _sum = _mm256_fmadd_ps(_val, _w, _sum);

                                kptr += 8;
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        _mm256_storeu_ps(outptr + j * 8, _sum);
                    }

                    outptr += outw * 8;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 8)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        __m256 _sum = _mm256_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm256_loadu_ps((const float*)bias_data + p * 8);
                        }

                        const float* kptr = weight_data_packed.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                __m256 _val0 = _mm256_broadcast_ss((sptr + space_ofs[k] * 4));
                                __m256 _val1 = _mm256_broadcast_ss((sptr + space_ofs[k] * 4) + 1);
                                __m256 _val2 = _mm256_broadcast_ss((sptr + space_ofs[k] * 4) + 2);
                                __m256 _val3 = _mm256_broadcast_ss((sptr + space_ofs[k] * 4) + 3);

                                __m256 _w0 = _mm256_loadu_ps(kptr);
                                _sum = _mm256_fmadd_ps(_val0, _w0, _sum);
                                __m256 _w1 = _mm256_loadu_ps(kptr + 8);
                                _sum = _mm256_fmadd_ps(_val1, _w1, _sum);
                                __m256 _w2 = _mm256_loadu_ps(kptr + 16);
                                _sum = _mm256_fmadd_ps(_val2, _w2, _sum);
                                __m256 _w3 = _mm256_loadu_ps(kptr + 24);
                                _sum = _mm256_fmadd_ps(_val3, _w3, _sum);

                                kptr += 32;
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        _mm256_storeu_ps(outptr + j * 8, _sum);
                    }

                    outptr += outw * 8;
                }
            }
        }
    }

    if (elempack == 8 && out_elempack == 1)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack8to1_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const float* kptr = (const float*)weight_data_packed + maxk * channels * p * 8;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++) // 29.23
                            {
                                __m256 _val = _mm256_loadu_ps(sptr + (space_ofs[k] * 8));
                                __m256 _w = _mm256_loadu_ps(kptr);
                                __m256 _s8 = _mm256_mul_ps(_val, _w);
                                sum += _mm256_reduce_add_ps(_s8); // dot
                                kptr += 8;
                            }
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    if (elempack == 8 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        __m128 _sum = _mm_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm_loadu_ps((const float*)bias_data + p * 4);
                        }

                        const float* kptr = weight_data_packed.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                __m128 _val0 = _mm_broadcast_ss((sptr + space_ofs[k] * 8));
                                __m128 _val1 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 1);
                                __m128 _val2 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 2);
                                __m128 _val3 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 3);
                                __m128 _val4 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 4);
                                __m128 _val5 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 5);
                                __m128 _val6 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 6);
                                __m128 _val7 = _mm_broadcast_ss((sptr + space_ofs[k] * 8) + 7);

                                __m128 _w0 = _mm_loadu_ps(kptr);
                                _sum = _mm_fmadd_ps(_val0, _w0, _sum);
                                __m128 _w1 = _mm_loadu_ps(kptr + 4);
                                _sum = _mm_fmadd_ps(_val1, _w1, _sum);
                                __m128 _w2 = _mm_loadu_ps(kptr + 8);
                                _sum = _mm_fmadd_ps(_val2, _w2, _sum);
                                __m128 _w3 = _mm_loadu_ps(kptr + 12);
                                _sum = _mm_fmadd_ps(_val3, _w3, _sum);
                                __m128 _w4 = _mm_loadu_ps(kptr + 16);
                                _sum = _mm_fmadd_ps(_val4, _w4, _sum);
                                __m128 _w5 = _mm_loadu_ps(kptr + 20);
                                _sum = _mm_fmadd_ps(_val5, _w5, _sum);
                                __m128 _w6 = _mm_loadu_ps(kptr + 24);
                                _sum = _mm_fmadd_ps(_val6, _w6, _sum);
                                __m128 _w7 = _mm_loadu_ps(kptr + 28);
                                _sum = _mm_fmadd_ps(_val7, _w7, _sum);

                                kptr += 32;
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        _mm_storeu_ps(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
    }
#endif

    if (elempack == 4 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        __m128 _sum = _mm_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm_loadu_ps((const float*)bias_data + p * 4);
                        }

                        const float* kptr = weight_data_packed.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                __m128 _val0 = _mm_set1_ps(sptr[space_ofs[k] * 4]);
                                __m128 _val1 = _mm_set1_ps(sptr[space_ofs[k] * 4 + 1]);
                                __m128 _val2 = _mm_set1_ps(sptr[space_ofs[k] * 4 + 2]);
                                __m128 _val3 = _mm_set1_ps(sptr[space_ofs[k] * 4 + 3]);

                                __m128 _w0 = _mm_loadu_ps(kptr);
                                _sum = _mm_add_ps(_mm_mul_ps(_val0, _w0), _sum);
                                __m128 _w1 = _mm_loadu_ps(kptr + 4);
                                _sum = _mm_add_ps(_mm_mul_ps(_val1, _w1), _sum);
                                __m128 _w2 = _mm_loadu_ps(kptr + 8);
                                _sum = _mm_add_ps(_mm_mul_ps(_val2, _w2), _sum);
                                __m128 _w3 = _mm_loadu_ps(kptr + 12);
                                _sum = _mm_add_ps(_mm_mul_ps(_val3, _w3), _sum);

                                kptr += 16;
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        _mm_storeu_ps(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 1 && out_elempack == 4)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output / out_elempack; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        __m128 _sum = _mm_set1_ps(0.f);

                        if (bias_term)
                        {
                            _sum = _mm_loadu_ps((const float*)bias_data + p * 4);
                        }

                        const float* kptr = weight_data_packed.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                __m128 _val = _mm_set1_ps(sptr[space_ofs[k]]);
                                __m128 _w = _mm_loadu_ps(kptr);
                                _sum = _mm_add_ps(_mm_mul_ps(_val, _w), _sum);

                                kptr += 4;
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        _mm_storeu_ps(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
    }

    if (elempack == 4 && out_elempack == 1)
    {
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const float* kptr = weight_data_packed.channel(p);

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                __m128 _val = _mm_loadu_ps(sptr + space_ofs[k] * 4);
                                __m128 _w = _mm_loadu_ps(kptr);
                                __m128 _s4 = _mm_mul_ps(_val, _w);
                                sum += _mm_reduce_add_ps(_s4); // dot

                                kptr += 4;
                            }
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }
#endif // __SSE2__

    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if (use_winograd3x3 && outw >= 8 && outh >= 8)
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
        else if (dilation_w == 1 && dilation_h == 1)
        {
            conv_im2col_sgemm_sse(bottom_blob_bordered, top_blob, weight_sgemm_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, opt);
            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const float* kptr = (const float*)weight_data + maxk * channels * p;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = sptr[space_ofs[k]];
                                float w = kptr[k];
                                sum += val * w;
                            }

                            kptr += maxk;
                        }

                        if (activation_type == 1)
                        {
                            sum = std::max(sum, 0.f);
                        }
                        else if (activation_type == 2)
                        {
                            float slope = activation_params[0];
                            sum = sum > 0.f ? sum : sum * slope;
                        }
                        else if (activation_type == 3)
                        {
                            float min = activation_params[0];
                            float max = activation_params[1];
                            if (sum < min)
                                sum = min;
                            if (sum > max)
                                sum = max;
                        }
                        else if (activation_type == 4)
                        {
                            sum = static_cast<float>(1.f / (1.f + exp(-sum)));
                        }
                        else if (activation_type == 5)
                        {
                            sum = static_cast<float>(sum * tanh(log(exp(sum) + 1.f)));
                        }

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
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
            for (int p = 0; p < num_output; p++)
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

                float scale_out = top_blob_int8_scale; //FIXME load param

                requantize_int8_to_int8(top_blob_tm_g, top_blob_g, scale_in, scale_out, bias_term ? (const float*)bias_data + p : 0, bias_term ? 1 : 0, 0, opt_g);
            }
        }
        else
        {
            std::vector<float> requantize_scales;
            for (int p = 0; p < num_output; p++)
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
            for (int p = 0; p < num_output; p++)
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
            for (int p = 0; p < num_output; p++)
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
    for (int x = 0; x < dilation; x++)
    {
        for (int y = 0; y < dilation; y++)
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
            for (int c = 0; c < bottom_blob.c; c++)
            {
                float* outptr = inner_bottom_blob.channel(c);

                for (int i = 0; i < inner_h; i++)
                {
                    const float* ptr = (const float*)bottom_blob.channel(c) + dilation * i * w + x * w + y;
                    for (int j = 0; j < inner_w; j++)
                    {
                        outptr[j] = ptr[j * dilation];
                    }
                    outptr += inner_w;
                }
            }

            Option opt_g = opt;
            opt_g.blob_allocator = inner_top_blob.allocator;
            convolution_dilation1->forward(inner_bottom_blob, inner_top_blob, opt_g);

            #pragma omp parallel for num_threads(opt.num_threads)
            for (int c = 0; c < num_output; c++)
            {
                float* outptr = (float*)top_blob.channel(c) + x * outw + y;
                for (int i = 0; i < inner_outh; i++)
                {
                    const float* ptr = (const float*)inner_top_blob.channel(c) + i * inner_outw;
                    for (int j = 0; j < inner_outw; j++)
                    {
                        outptr[j * dilation] = ptr[j];
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
