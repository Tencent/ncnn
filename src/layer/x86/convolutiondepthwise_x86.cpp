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

#if __SSE2__
#include <emmintrin.h>
#if __AVX__
#include <immintrin.h>
#endif
#endif // __SSE2__

#include "x86_activation.h"
#include "x86_usability.h"

#include "layer_type.h"

namespace ncnn {

#if __SSE2__
#include "convolutiondepthwise_3x3_pack4.h"
#include "convolutiondepthwise_5x5_pack4.h"
#if __AVX__
#include "convolutiondepthwise_3x3_pack8.h"
#include "convolutiondepthwise_5x5_pack8.h"
#if __AVX512F__
#include "convolutiondepthwise_3x3_pack16.h"
#include "convolutiondepthwise_5x5_pack16.h"
#endif // __AVX512F__
#endif // __AVX__
#endif // __SSE2__
#include "convolutiondepthwise_3x3.h"

#if NCNN_INT8
#include "convolutiondepthwise_3x3_int8.h"
#endif // NCNN_INT8

ConvolutionDepthWise_x86::ConvolutionDepthWise_x86()
{
#if __SSE2__
    support_packing = true;
#endif // __SSE2__
    activation = 0;
}

int ConvolutionDepthWise_x86::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8_x86(opt);
    }
#endif

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
#if __AVX512F__
            elempack = channels % 16 == 0 ? 16 : channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4 : 1;
#elif __AVX__
            elempack = channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4 : 1;
#else
            elempack = channels % 4 == 0 ? 4 : 1;
#endif
        }
#endif // __SSE2__

#if __SSE2__
#if __AVX__
        // pack16
        if (elempack == 16)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_packed, 16, opt);

            return 0;
        }
#if __AVX512F__
#endif // __AVX512F__

        // pack8
        if (elempack == 8)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_packed, 8, opt);

            return 0;
        }
#endif // __AVX__

        // pack4
        if (elempack == 4)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_packed, 4, opt);

            return 0;
        }
#endif // __SSE2__

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
            ncnn::Mat weights[5];
            weights[0] = weight_data_g;
            weights[1] = bias_data_g;

#if NCNN_INT8
            if (int8_scale_term)
            {
                Mat weight_data_int8_scales_g(num_output_g);
                weight_data_int8_scales_g.fill(weight_data_int8_scales[g]);
                weights[2] = weight_data_int8_scales_g;
                weights[3] = bottom_blob_int8_scales.range(g, 1);
            }
            if (int8_scale_term > 100)
            {
                weights[4] = top_blob_int8_scales.range(g, 1);
            }
#endif

            op->load_model(ModelBinFromMatArray(weights));
        }
        else
        {
            ncnn::Mat weights[4];
            weights[0] = weight_data_g;

#if NCNN_INT8
            if (int8_scale_term)
            {
                Mat weight_data_int8_scales_g(num_output_g);
                weight_data_int8_scales_g.fill(weight_data_int8_scales[g]);
                weights[1] = weight_data_int8_scales_g;
                weights[2] = bottom_blob_int8_scales.range(g, 1);
            }
            if (int8_scale_term > 100)
            {
                weights[3] = top_blob_int8_scales.range(g, 1);
            }
#endif

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
#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8_x86(bottom_blob, top_blob, opt);
    }
#endif

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
#if __AVX512F__
        out_elempack = num_output % 16 == 0 ? 16 : num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
#elif __AVX__
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

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
#if __SSE2__
#if __AVX__
#if __AVX512F__
        if (elempack == 16)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_pack16_avx512(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_pack16_avx512(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw5x5s1_pack16_avx512(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw5x5s2_pack16_avx512(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

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
                    const float* kptr = (const float*)weight_data_packed + maxk * g * 16;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            __m512 _sum = _mm512_set1_ps(0.f);

                            if (bias_term)
                            {
                                _sum = _mm512_loadu_ps(((const float*)bias_data) + g * 16);
                            }

                            const float* sptr = m.row(i * stride_h) + j * stride_w * 16;

                            for (int k = 0; k < maxk; k++)
                            {
                                __m512 _val = _mm512_loadu_ps(sptr + space_ofs[k] * 16);
                                __m512 _w = _mm512_loadu_ps(kptr + k * 16);
                                _sum = _mm512_fmadd_ps(_val, _w, _sum);
                            }

                            _mm512_storeu_ps(outptr, _sum);
                            outptr += 16;
                        }
                    }
                }

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
        }
#endif // __AVX512F__
        if (elempack == 8)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_pack8_avx(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

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
                                _sum = _mm256_comp_fmadd_ps(_val, _w, _sum);
                            }

                            _mm256_storeu_ps(outptr + j * 8, _sum);
                        }

                        outptr += outw * 8;
                    }
                }

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
        }
#endif // __AVX__

        if (elempack == 4)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_pack4_sse(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_pack4_sse(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw5x5s1_pack4_sse(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw5x5s2_pack4_sse(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
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

                            _sum = activation_sse(_sum, activation_type, activation_params);

                            _mm_storeu_ps(outptr + j * 4, _sum);
                        }

                        outptr += outw * 4;
                    }
                }

                return 0;
            }
        }
#endif // __SSE2__

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
#if __SSE2__
    if (opt.use_packing_layout)
    {
#if __AVX512F__
        g_elempack = channels_g % 16 == 0 ? 16 : channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
        out_g_elempack = num_output_g % 16 == 0 ? 16 : num_output_g % 8 == 0 ? 8 : num_output_g % 4 == 0 ? 4 : 1;
#elif __AVX__
        g_elempack = channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
        out_g_elempack = num_output_g % 8 == 0 ? 8 : num_output_g % 4 == 0 ? 4 : 1;
#else
        g_elempack = channels_g % 4 == 0 ? 4 : 1;
        out_g_elempack = num_output_g % 4 == 0 ? 4 : 1;
#endif
    }
#endif // __SSE2__

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

int ConvolutionDepthWise_x86::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _kernel_h = _weight_data.h;
    const int _num_output = _weight_data.c * _weight_data.elempack;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

    // weight_data_flattened as pack1
    weight_data_flattened.w *= weight_data_flattened.elempack;
    weight_data_flattened.elemsize /= weight_data_flattened.elempack;
    weight_data_flattened.elempack = 1;

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;

        // bias_data_flattened as pack1
        bias_data_flattened.w *= bias_data_flattened.elempack;
        bias_data_flattened.elemsize /= bias_data_flattened.elempack;
        bias_data_flattened.elempack = 1;
    }

    ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::ConvolutionDepthWise);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(11, _kernel_h);
    pd.set(2, dilation_w);
    pd.set(12, dilation_h);
    pd.set(3, stride_w);
    pd.set(13, stride_h);
    pd.set(4, pad_left);
    pd.set(15, pad_right);
    pd.set(14, pad_top);
    pd.set(16, pad_bottom);
    pd.set(18, pad_value);
    pd.set(5, bias_term);
    pd.set(6, weight_data_flattened.w);
    pd.set(7, group);
    pd.set(8, int8_scale_term);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    op->load_param(pd);

    ncnn::Mat weights[2];
    weights[0] = weight_data_flattened;
    weights[1] = bias_data_flattened;

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    op->forward(bottom_blob, top_blob, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

#if NCNN_INT8
int ConvolutionDepthWise_x86::create_pipeline_int8_x86(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
            elempack = channels % 8 == 0 ? 8 : 1;
        }
#endif // __SSE2__

        if (elempack == 8)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_int8, 8, opt);
        }

        return 0;
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
    int elempack = bottom_blob.elempack;

    int elembits = bottom_blob.elembits();

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_int8 = bottom_blob;
    if (elembits != 8)
    {
        const int channels_g = channels * elempack / group;

        Mat scales(channels * elempack);
        {
            float* ps = scales;
            for (int g = 0; g < group; g++)
            {
                float scale = bottom_blob_int8_scales[g];
                for (int q = 0; q < channels_g; q++)
                {
                    *ps++ = scale;
                }
            }
        }

        Option opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
        quantize_to_int8(bottom_blob, bottom_blob_int8, scales, opt_q);
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob_int8, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;
    channels = bottom_blob_bordered.c;
    elempack = bottom_blob_bordered.elempack;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
        int out_elempack = 1;
#if __SSE2__
        if (opt.use_packing_layout)
        {
            out_elempack = num_output % 8 == 0 ? 8 : 1;
        }
#endif // __SSE2__
        bool use_int8_requantize = int8_scale_term > 100;
        size_t out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;

        top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __SSE2__
        if (elempack == 8)
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
                    signed char* outptr_s8 = top_blob.channel(g);
                    float* outptr_f32 = top_blob.channel(g);
                    const signed char* kptr = (const signed char*)weight_data_int8 + maxk * g * 8;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            __m128i _sum0 = _mm_setzero_si128();
                            __m128i _sum1 = _mm_setzero_si128();

                            const signed char* sptr = m.row<const signed char>(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                // TODO use _mm_cvtepi8_epi16 on sse4.1
                                __m128i _val = _mm_loadl_epi64((const __m128i*)(sptr + space_ofs[k] * 8));
                                _val = _mm_unpacklo_epi8(_val, _mm_cmpgt_epi8(_mm_setzero_si128(), _val));

                                __m128i _w = _mm_loadl_epi64((const __m128i*)(kptr + k * 8));
                                _w = _mm_unpacklo_epi8(_w, _mm_cmpgt_epi8(_mm_setzero_si128(), _w));

                                __m128i _sl = _mm_mullo_epi16(_val, _w);
                                __m128i _sh = _mm_mulhi_epi16(_val, _w);
                                __m128i _s0 = _mm_unpacklo_epi16(_sl, _sh);
                                __m128i _s1 = _mm_unpackhi_epi16(_sl, _sh);

                                _sum0 = _mm_add_epi32(_sum0, _s0);
                                _sum1 = _mm_add_epi32(_sum1, _s1);
                            }

                            __m128 _scale_in0;
                            __m128 _scale_in1;
                            {
                                __m128 _bottom_blob_int8_scales0 = _mm_loadu_ps((const float*)bottom_blob_int8_scales + g * 8);
                                __m128 _bottom_blob_int8_scales1 = _mm_loadu_ps((const float*)bottom_blob_int8_scales + g * 8 + 4);
                                __m128 _weight_data_int8_scales0 = _mm_loadu_ps((const float*)weight_data_int8_scales + g * 8);
                                __m128 _weight_data_int8_scales1 = _mm_loadu_ps((const float*)weight_data_int8_scales + g * 8 + 4);
                                _scale_in0 = _mm_rcp_ps(_mm_mul_ps(_bottom_blob_int8_scales0, _weight_data_int8_scales0));
                                _scale_in1 = _mm_rcp_ps(_mm_mul_ps(_bottom_blob_int8_scales1, _weight_data_int8_scales1));

                                __m128 _m0 = _mm_cmpneq_ps(_weight_data_int8_scales0, _mm_setzero_ps());
                                __m128 _m1 = _mm_cmpneq_ps(_weight_data_int8_scales1, _mm_setzero_ps());
                                _scale_in0 = _mm_and_ps(_scale_in0, _m0);
                                _scale_in1 = _mm_and_ps(_scale_in1, _m1);
                            }

                            __m128 _sumfp32_0 = _mm_mul_ps(_mm_cvtepi32_ps(_sum0), _scale_in0);
                            __m128 _sumfp32_1 = _mm_mul_ps(_mm_cvtepi32_ps(_sum1), _scale_in1);

                            if (bias_term)
                            {
                                __m128 _bias0 = _mm_loadu_ps((const float*)bias_data + g * 8);
                                __m128 _bias1 = _mm_loadu_ps((const float*)bias_data + g * 8 + 4);
                                _sumfp32_0 = _mm_add_ps(_sumfp32_0, _bias0);
                                _sumfp32_1 = _mm_add_ps(_sumfp32_1, _bias1);
                            }

                            _sumfp32_0 = activation_sse(_sumfp32_0, activation_type, activation_params);
                            _sumfp32_1 = activation_sse(_sumfp32_1, activation_type, activation_params);

                            if (use_int8_requantize)
                            {
                                // requantize and relu
                                __m128 _scale_out0 = _mm_loadu_ps((const float*)top_blob_int8_scales + g * 8);
                                __m128 _scale_out1 = _mm_loadu_ps((const float*)top_blob_int8_scales + g * 8 + 4);
                                _sumfp32_0 = _mm_mul_ps(_sumfp32_0, _scale_out0);
                                _sumfp32_1 = _mm_mul_ps(_sumfp32_1, _scale_out1);
                                int64_t _sum8 = float2int8_sse(_sumfp32_0, _sumfp32_1);

                                *(int64_t*)outptr_s8 = _sum8;
                                outptr_s8 += 8;
                            }
                            else
                            {
                                // dequantize and relu
                                _mm_storeu_ps(outptr_f32, _sumfp32_0);
                                _mm_storeu_ps(outptr_f32 + 4, _sumfp32_1);
                                outptr_f32 += 8;
                            }
                        }
                    }
                }
            }
        }
#endif // __SSE2__

        if (elempack == 1)
        {
            if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1 && (activation_type == 0 || activation_type == 1))
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

                        float scale_out = top_blob_int8_scales[g];

                        requantize_scales.push_back(scale_in);
                        requantize_scales.push_back(scale_out);
                    }

                    convdw3x3s1_int8_requant_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, requantize_scales, opt);
                }
                else
                {
                    std::vector<float> dequantize_scales;
                    for (int g = 0; g < group; g++)
                    {
                        float top_rescale = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                        dequantize_scales.push_back(top_rescale);
                    }

                    convdw3x3s1_int8_dequant_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, dequantize_scales, opt);
                }

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2 && (activation_type == 0 || activation_type == 1))
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

                        float scale_out = top_blob_int8_scales[g];

                        requantize_scales.push_back(scale_in);
                        requantize_scales.push_back(scale_out);
                    }

                    convdw3x3s2_int8_requant_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, requantize_scales, opt);
                }
                else
                {
                    std::vector<float> dequantize_scales;
                    for (int g = 0; g < group; g++)
                    {
                        float top_rescale = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                        dequantize_scales.push_back(top_rescale);
                    }

                    convdw3x3s2_int8_dequant_sse(bottom_blob_bordered, top_blob, weight_data, bias_data, dequantize_scales, opt);
                }

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
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
                for (int g = 0; g < group; g++)
                {
                    signed char* outptr_s8 = top_blob.channel(g);
                    float* outptr_f32 = top_blob.channel(g);
                    const signed char* kptr = (const signed char*)weight_data + maxk * g;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            int sum = 0;

                            const signed char* sptr = m.row<const signed char>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                signed char val = sptr[space_ofs[k]];
                                signed char w = kptr[k];
                                sum += val * w;
                            }

                            float scale_in;
                            if (weight_data_int8_scales[g] == 0)
                                scale_in = 0;
                            else
                                scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                            float sumfp32 = sum * scale_in;

                            if (bias_term)
                                sumfp32 += bias_data[g];

                            sumfp32 = activation_ss(sumfp32, activation_type, activation_params);

                            if (use_int8_requantize)
                            {
                                // requantize
                                float scale_out = top_blob_int8_scales[g];
                                signed char sums8 = float2int8(sumfp32 * scale_out);
                                outptr_s8[0] = sums8;
                                outptr_s8 += 1;
                            }
                            else
                            {
                                // dequantize
                                outptr_f32[0] = sumfp32;
                                outptr_f32 += 1;
                            }
                        }
                    }
                }
            }
        }

        return 0;
    }

    bool use_int8_requantize = int8_scale_term > 100;
    int out_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
        if (use_int8_requantize)
            out_elempack = num_output % 8 == 0 ? 8 : 1;
        else
            out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __SSE2__
    size_t out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // group convolution
    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    int g_elempack = 1;
    int out_g_elempack = 1;
#if __SSE2__
    if (opt.use_packing_layout)
    {
        g_elempack = channels_g % 8 == 0 ? 8 : 1;
        if (use_int8_requantize)
            out_g_elempack = num_output_g % 8 == 0 ? 8 : 1;
        else
            out_g_elempack = num_output_g % 4 == 0 ? 4 : 1;
    }
#endif // __SSE2__

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

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int g = 0; g < group; g++)
    {
        const Mat bottom_blob_bordered_g = bottom_blob_bordered_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
        Mat top_blob_g = top_blob_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

        const ncnn::Layer* op = group_ops[g];

        Option opt_g = opt;
        opt_g.blob_allocator = top_blob.allocator;

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
#endif // NCNN_INT8

} // namespace ncnn
