// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "convolutiondepthwise_mips.h"

#include "layer_type.h"

#if __mips_msa
#include <msa.h>
#endif // __mips_msa

#include "mips_activation.h"
#include "mips_usability.h"

namespace ncnn {

#if __mips_msa
#include "convolutiondepthwise_3x3_pack4.h"
#include "convolutiondepthwise_5x5_pack4.h"
#endif // __mips_msa

ConvolutionDepthWise_mips::ConvolutionDepthWise_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa

    activation = 0;
}

int ConvolutionDepthWise_mips::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8_mips(opt);
    }
#endif

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;
#if __mips_msa
        if (opt.use_packing_layout)
        {
            elempack = channels % 4 == 0 ? 4 : 1;
        }
#endif

#if __mips_msa
        // pack4
        if (elempack == 4)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_packed, 4, opt);
        }
#endif // __mips_msa

        if (elempack == 1)
        {
        }

        return 0;
    }

    // group convolution
    create_group_ops(opt);

    return 0;
}

int ConvolutionDepthWise_mips::create_group_ops(const Option& opt)
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

int ConvolutionDepthWise_mips::destroy_pipeline(const Option& opt)
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

int ConvolutionDepthWise_mips::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8_mips(bottom_blob, top_blob, opt);
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
#if __mips_msa
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
#if __mips_msa
        if (elempack == 4)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_pack4_msa(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_pack4_msa(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw5x5s1_pack4_msa(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw5x5s2_pack4_msa(bottom_blob_bordered, top_blob, weight_data_packed, bias_data, opt);

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
                for (int g = 0; g < channels; g++)
                {
                    float* outptr = top_blob.channel(g);
                    const float* kptr = (const float*)weight_data_packed + maxk * g * 4;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            v4f32 _sum = (v4f32)__msa_fill_w(0);

                            if (bias_term)
                            {
                                _sum = (v4f32)__msa_ld_w((const float*)bias_data + g * 4, 0);
                            }

                            const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                v4f32 _val = (v4f32)__msa_ld_w(sptr + space_ofs[k] * 4, 0);
                                v4f32 _w = (v4f32)__msa_ld_w(kptr + k * 4, 0);
                                _sum = __msa_fmadd_w(_sum, _val, _w);
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params);

                            __msa_st_w((v4i32)_sum, outptr + j * 4, 0);
                        }

                        outptr += outw * 4;
                    }
                }
            }
        }
#endif // __mips_msa

        if (elempack == 1)
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
                for (int g = 0; g < group; g++)
                {
                    float* outptr = top_blob.channel(g);
                    const float* kptr = (const float*)weight_data + maxk * g;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                                sum = bias_data[g];

                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = (float)sptr[space_ofs[k]];
                                float w = (float)kptr[k];
                                sum += val * w;
                            }

                            sum = activation_ss(sum, activation_type, activation_params);

                            outptr[j] = sum;
                        }

                        outptr += outw;
                    }
                }
            }
        }

        return 0;
    }

    // group convolution
    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    int g_elempack = 1;
    int out_g_elempack = 1;
#if __mips_msa
    if (opt.use_packing_layout)
    {
        g_elempack = channels_g % 4 == 0 ? 4 : 1;
        out_g_elempack = num_output_g % 4 == 0 ? 4 : 1;
    }
#endif

    // unpacking
    Mat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack > g_elempack)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, 1, opt_p);
    }

    Mat top_blob_unpacked = top_blob;
    if (out_g_elempack < out_elempack)
    {
        top_blob_unpacked.create(outw, outh, num_output, out_elemsize / out_elempack, 1, opt.workspace_allocator);
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

int ConvolutionDepthWise_mips::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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
int ConvolutionDepthWise_mips::create_pipeline_int8_mips(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;
#if __mips_msa
        if (opt.use_packing_layout)
        {
            elempack = channels % 8 == 0 ? 8 : 1;
        }
#endif // __mips_msa

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

int ConvolutionDepthWise_mips::forward_int8_mips(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
#if __mips_msa
        if (opt.use_packing_layout)
        {
            out_elempack = num_output % 8 == 0 ? 8 : 1;
        }
#endif // __mips_msa
        bool use_int8_requantize = int8_scale_term > 100;
        size_t out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;

        top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __mips_msa
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
                            v4i32 _sum0 = __msa_fill_w(0);
                            v4i32 _sum1 = __msa_fill_w(0);

                            const signed char* sptr = m.row<const signed char>(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                v16i8 _val = __msa_ld_b(sptr + space_ofs[k] * 8, 0);
                                v8i16 _val16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_val, 0), _val);

                                v16i8 _w = __msa_ld_b(kptr + k * 8, 0);
                                v8i16 _w16 = (v8i16)__msa_ilvr_b(__msa_clti_s_b(_w, 0), _w);

                                v8i16 _s0 = __msa_mulv_h(_val16, _w16);
                                v8i16 _exts0 = __msa_clti_s_h(_s0, 0);
                                v4i32 _s0l = (v4i32)__msa_ilvr_h(_exts0, _s0);
                                v4i32 _s0h = (v4i32)__msa_ilvl_h(_exts0, _s0);

                                _sum0 = __msa_addv_w(_sum0, _s0l);
                                _sum1 = __msa_addv_w(_sum1, _s0h);
                            }

                            v4f32 _scale_in0;
                            v4f32 _scale_in1;
                            {
                                v4f32 _bottom_blob_int8_scales0 = (v4f32)__msa_ld_w((const float*)bottom_blob_int8_scales + g * 8, 0);
                                v4f32 _bottom_blob_int8_scales1 = (v4f32)__msa_ld_w((const float*)bottom_blob_int8_scales + g * 8 + 4, 0);
                                v4f32 _weight_data_int8_scales0 = (v4f32)__msa_ld_w((const float*)weight_data_int8_scales + g * 8, 0);
                                v4f32 _weight_data_int8_scales1 = (v4f32)__msa_ld_w((const float*)weight_data_int8_scales + g * 8 + 4, 0);
                                _scale_in0 = __msa_frcp_w(__msa_fmul_w(_bottom_blob_int8_scales0, _weight_data_int8_scales0));
                                _scale_in1 = __msa_frcp_w(__msa_fmul_w(_bottom_blob_int8_scales1, _weight_data_int8_scales1));

                                v4i32 _m0 = __msa_fcne_w(_weight_data_int8_scales0, __msa_fill_w_f32(0.f));
                                v4i32 _m1 = __msa_fcne_w(_weight_data_int8_scales1, __msa_fill_w_f32(0.f));
                                _scale_in0 = (v4f32)__msa_and_v((v16u8)_scale_in0, (v16u8)_m0);
                                _scale_in1 = (v4f32)__msa_and_v((v16u8)_scale_in1, (v16u8)_m1);
                            }

                            v4f32 _sumfp32_0 = __msa_fmul_w(__msa_ffint_s_w(_sum0), _scale_in0);
                            v4f32 _sumfp32_1 = __msa_fmul_w(__msa_ffint_s_w(_sum1), _scale_in1);

                            if (bias_term)
                            {
                                v4f32 _bias0 = (v4f32)__msa_ld_w((const float*)bias_data + g * 8, 0);
                                v4f32 _bias1 = (v4f32)__msa_ld_w((const float*)bias_data + g * 8 + 4, 0);
                                _sumfp32_0 = __msa_fadd_w(_sumfp32_0, _bias0);
                                _sumfp32_1 = __msa_fadd_w(_sumfp32_1, _bias1);
                            }

                            _sumfp32_0 = activation_ps(_sumfp32_0, activation_type, activation_params);
                            _sumfp32_1 = activation_ps(_sumfp32_1, activation_type, activation_params);

                            if (use_int8_requantize)
                            {
                                // requantize and relu
                                v4f32 _scale_out0 = (v4f32)__msa_ld_w((const float*)top_blob_int8_scales + g * 8, 0);
                                v4f32 _scale_out1 = (v4f32)__msa_ld_w((const float*)top_blob_int8_scales + g * 8 + 4, 0);
                                _sumfp32_0 = __msa_fmul_w(_sumfp32_0, _scale_out0);
                                _sumfp32_1 = __msa_fmul_w(_sumfp32_1, _scale_out1);
                                int64_t _sum8 = float2int8(_sumfp32_0, _sumfp32_1);

                                *(int64_t*)outptr_s8 = _sum8;
                                outptr_s8 += 8;
                            }
                            else
                            {
                                // dequantize and relu
                                __msa_st_w((v4i32)_sumfp32_0, outptr_f32, 0);
                                __msa_st_w((v4i32)_sumfp32_1, outptr_f32 + 4, 0);
                                outptr_f32 += 8;
                            }
                        }
                    }
                }
            }
        }
#endif // __mips_msa

        if (elempack == 1)
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
#if __mips_msa
    if (opt.use_packing_layout)
    {
        if (use_int8_requantize)
            out_elempack = num_output % 8 == 0 ? 8 : 1;
        else
            out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __mips_msa
    size_t out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // group convolution
    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    int g_elempack = 1;
    int out_g_elempack = 1;
#if __mips_msa
    if (opt.use_packing_layout)
    {
        g_elempack = channels_g % 8 == 0 ? 8 : 1;
        if (use_int8_requantize)
            out_g_elempack = num_output_g % 8 == 0 ? 8 : 1;
        else
            out_g_elempack = num_output_g % 4 == 0 ? 4 : 1;
    }
#endif // __mips_msa

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
