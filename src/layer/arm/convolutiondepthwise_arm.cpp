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

#include "convolutiondepthwise_arm.h"

#include "cpu.h"
#include "layer_type.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#include "convolutiondepthwise_3x3.h"
#include "convolutiondepthwise_5x5.h"

#if NCNN_INT8
#include "convolutiondepthwise_3x3_int8.h"
#endif // NCNN_INT8

#if __ARM_NEON
#include "convolutiondepthwise_3x3_pack4.h"
#include "convolutiondepthwise_5x5_pack4.h"

#if NCNN_BF16
#include "convolutiondepthwise_3x3_pack4_bf16s.h"
#include "convolutiondepthwise_5x5_pack4_bf16s.h"
#endif // NCNN_BF16

#if NCNN_INT8
#include "convolutiondepthwise_3x3_pack8_int8.h"
#endif // NCNN_INT8

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "convolutiondepthwise_3x3_fp16s.h"
#include "convolutiondepthwise_3x3_pack8_fp16s.h"
#include "convolutiondepthwise_5x5_pack8_fp16s.h"
#endif
#endif // __ARM_NEON

ConvolutionDepthWise_arm::ConvolutionDepthWise_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

#if NCNN_BF16
    support_bf16_storage = true;
#endif

    activation = 0;
}

int ConvolutionDepthWise_arm::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return create_pipeline_int8_arm(opt);
    }
#endif

    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;
#if __ARM_NEON
        if (opt.use_packing_layout)
        {
            elempack = channels % 4 == 0 ? 4 : 1;
        }
#endif // __ARM_NEON

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (opt.use_fp16_storage)
        {
            if (opt.use_packing_layout)
            {
                elempack = opt.use_fp16_arithmetic && channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4 : 1;
            }

            if (elempack == 8)
            {
                Mat weight_data_r2 = weight_data.reshape(maxk, group);
                Mat weight_data_r2_packed;
                convert_packing(weight_data_r2, weight_data_r2_packed, 8, opt);

                ncnn::cast_float32_to_float16(weight_data_r2_packed, weight_data_fp16, opt);
            }

            if (elempack == 4)
            {
                Mat weight_data_r2 = weight_data.reshape(maxk, group);
                Mat weight_data_r2_packed;
                convert_packing(weight_data_r2, weight_data_r2_packed, 4, opt);

                ncnn::cast_float32_to_float16(weight_data_r2_packed, weight_data_fp16, opt);
            }

            if (elempack == 1)
            {
                ncnn::cast_float32_to_float16(weight_data, weight_data_fp16, opt);
            }

            ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);

            return 0;
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if NCNN_BF16
        if (opt.use_bf16_storage)
        {
#if __ARM_NEON
            if (elempack == 4)
            {
                Mat weight_data_r2 = weight_data.reshape(maxk, group);
                convert_packing(weight_data_r2, weight_data_pack4, 4, opt);

                ncnn::cast_float32_to_bfloat16(weight_data_pack4, weight_data_pack4_bf16, opt);
            }
#endif // __ARM_NEON

            if (elempack == 1)
            {
                ncnn::cast_float32_to_bfloat16(weight_data, weight_data_bf16, opt);
            }

            return 0;
        }
#endif // NCNN_BF16

#if __ARM_NEON
        // pack4
        if (elempack == 4)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_pack4, 4, opt);

            return 0;
        }
#endif // __ARM_NEON

        if (elempack == 1)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                return 0;
            }
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                return 0;
            }
            if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                return 0;
            }
            if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                return 0;
            }
        }
    }

    // group convolution
    create_group_ops(opt);

    return 0;
}

int ConvolutionDepthWise_arm::create_group_ops(const Option& opt)
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

int ConvolutionDepthWise_arm::destroy_pipeline(const Option& opt)
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

int ConvolutionDepthWise_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8_arm(bottom_blob, top_blob, opt);
    }
#endif

    int elembits = bottom_blob.elembits();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

#if NCNN_BF16
    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blob, top_blob, opt);
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
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
#if __ARM_NEON
        if (elempack == 4)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw5x5s1_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw5x5s2_pack4_neon(bottom_blob_bordered, top_blob, weight_data_pack4, bias_data, opt);

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
                    const float* kptr = (const float*)weight_data_pack4 + maxk * g * 4;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float32x4_t _sum = vdupq_n_f32(0.f);

                            if (bias_term)
                            {
                                _sum = vld1q_f32(((const float*)bias_data) + g * 4);
                            }

                            const float* sptr = m.row(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vld1q_f32(sptr + space_ofs[k] * 4);
                                float32x4_t _w = vld1q_f32(kptr + k * 4);
                                _sum = vmlaq_f32(_sum, _val, _w);
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params);

                            vst1q_f32(outptr + j * 4, _sum);
                        }

                        outptr += outw * 4;
                    }
                }

                return 0;
            }
        }
#endif // __ARM_NEON

        if (elempack == 1)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw5x5s1_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }

                return 0;
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw5x5s2_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, opt);

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
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        g_elempack = channels_g % 4 == 0 ? 4 : 1;
        out_g_elempack = num_output_g % 4 == 0 ? 4 : 1;
    }
#endif

    // unpacking
    Mat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack == 4 && g_elempack == 1)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, 1, opt_p);
    }

    Mat top_blob_unpacked = top_blob;
    if (out_g_elempack == 1 && out_elempack == 4)
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
    if (out_g_elempack == 1 && out_elempack == 4)
    {
        convert_packing(top_blob_unpacked, top_blob, 4, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}

int ConvolutionDepthWise_arm::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
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

#if NCNN_ARM82
    if (opt.use_fp16_storage && cpu_support_arm_asimdhp() && weight_data_flattened.elembits() == 16)
    {
        Mat weight_data_flattened_fp32;
        cast_float16_to_float32(weight_data_flattened, weight_data_flattened_fp32, opt);
        weight_data_flattened = weight_data_flattened_fp32;
    }
#endif // NCNN_ARM82
#if NCNN_BF16
    if (opt.use_bf16_storage && weight_data_flattened.elembits() == 16)
    {
        Mat weight_data_flattened_fp32;
        cast_bfloat16_to_float32(weight_data_flattened, weight_data_flattened_fp32, opt);
        weight_data_flattened = weight_data_flattened_fp32;
    }
#endif // NCNN_BF16

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

#if NCNN_ARM82
        if (opt.use_fp16_storage && cpu_support_arm_asimdhp() && bias_data_flattened.elembits() == 16)
        {
            Mat bias_data_flattened_fp32;
            cast_float16_to_float32(bias_data_flattened, bias_data_flattened_fp32, opt);
            bias_data_flattened = bias_data_flattened_fp32;
        }
#endif // NCNN_ARM82
#if NCNN_BF16
        if (opt.use_bf16_storage && bias_data_flattened.elembits() == 16)
        {
            Mat bias_data_flattened_fp32;
            cast_bfloat16_to_float32(bias_data_flattened, bias_data_flattened_fp32, opt);
            bias_data_flattened = bias_data_flattened_fp32;
        }
#endif // NCNN_BF16

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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int ConvolutionDepthWise_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
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
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
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
                    __fp16* outptr = top_blob.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g * 4;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float32x4_t _sum = vdupq_n_f32(0.f);

                            if (bias_term)
                            {
                                _sum = vld1q_f32(((const float*)bias_data) + g * 4);
                            }

                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr + space_ofs[k] * 4));
                                float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr + k * 4));
                                _sum = vfmaq_f32(_sum, _val, _w);
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params);

                            vst1_f16(outptr + j * 4, vcvt_f16_f32(_sum));
                        }

                        outptr += outw * 4;
                    }
                }
            }
        }

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
                    __fp16* outptr = top_blob.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                                sum = bias_data[g];

                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = (float)sptr[space_ofs[k]];
                                float w = (float)kptr[k];
                                sum += val * w;
                            }

                            sum = activation_ss(sum, activation_type, activation_params);

                            outptr[j] = (__fp16)sum;
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

    int g_elempack = (opt.use_packing_layout && channels_g % 4 == 0) ? 4 : 1;
    int out_g_elempack = (opt.use_packing_layout && num_output_g % 4 == 0) ? 4 : 1;

    // unpacking
    Mat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack == 4 && g_elempack == 1)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, 1, opt_p);
    }

    Mat top_blob_unpacked = top_blob;
    if (out_g_elempack == 1 && out_elempack == 4)
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
    if (out_g_elempack == 1 && out_elempack == 4)
    {
        convert_packing(top_blob_unpacked, top_blob, 4, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}

int ConvolutionDepthWise_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
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
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
        if (elempack == 8)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw5x5s1_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw5x5s2_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

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
                    __fp16* outptr = top_blob.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g * 8;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                            if (bias_term)
                            {
                                _sum = vld1q_f16(((const __fp16*)bias_data_fp16) + g * 8);
                            }

                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                float16x8_t _val = vld1q_f16(sptr + space_ofs[k] * 8);
                                float16x8_t _w = vld1q_f16(kptr + k * 8);
                                _sum = vfmaq_f16(_sum, _val, _w);
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params);

                            vst1q_f16(outptr + j * 8, _sum);
                        }

                        outptr += outw * 8;
                    }
                }
            }
        }

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
                    __fp16* outptr = top_blob.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g * 4;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                            if (bias_term)
                            {
                                _sum = vld1_f16(((const __fp16*)bias_data_fp16) + g * 4);
                            }

                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float16x4_t _val = vld1_f16(sptr + space_ofs[k] * 4);
                                float16x4_t _w = vld1_f16(kptr + k * 4);
                                _sum = vfma_f16(_sum, _val, _w);
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params);

                            vst1_f16(outptr + j * 4, _sum);
                        }

                        outptr += outw * 4;
                    }
                }
            }
        }

        if (elempack == 1)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_fp16, bias_data_fp16, opt);

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
                    __fp16* outptr = top_blob.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                                sum = bias_data[g];

                            const __fp16* sptr = m.row<const __fp16>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                __fp16 val = sptr[space_ofs[k]];
                                __fp16 w = kptr[k];
                                sum += val * w;
                            }

                            sum = activation_ss(sum, activation_type, activation_params);

                            outptr[j] = (__fp16)sum;
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
    if (opt.use_packing_layout)
    {
        g_elempack = opt.use_fp16_arithmetic && channels_g % 8 == 0 ? 8 : channels_g % 4 == 0 ? 4 : 1;
        out_g_elempack = opt.use_fp16_arithmetic && num_output_g % 8 == 0 ? 8 : num_output_g % 4 == 0 ? 4 : 1;
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
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if NCNN_BF16
int ConvolutionDepthWise_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
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
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
#if __ARM_NEON
        if (elempack == 4)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw5x5s1_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, bias_data, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw5x5s2_pack4_bf16s_neon(bottom_blob_bordered, top_blob, weight_data_pack4_bf16, bias_data, opt);

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
                    unsigned short* outptr = top_blob.channel(g);
                    const unsigned short* kptr = (const unsigned short*)weight_data_pack4_bf16 + maxk * g * 4;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float32x4_t _sum = vdupq_n_f32(0.f);

                            if (bias_term)
                            {
                                _sum = vld1q_f32(((const float*)bias_data) + g * 4);
                            }

                            const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w * 4;

                            for (int k = 0; k < maxk; k++)
                            {
                                float32x4_t _val = vcvt_f32_bf16(vld1_u16(sptr + space_ofs[k] * 4));
                                float32x4_t _w = vcvt_f32_bf16(vld1_u16(kptr + k * 4));
                                _sum = vmlaq_f32(_sum, _val, _w);
                            }

                            _sum = activation_ps(_sum, activation_type, activation_params);

                            vst1_u16(outptr + j * 4, vcvt_bf16_f32(_sum));
                        }

                        outptr += outw * 4;
                    }
                }
            }

            return 0;
        }
#endif // __ARM_NEON

        if (elempack == 1)
        {
            //             if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            //             {
            //                 convdw3x3s1_neon(bottom_blob_bordered, top_blob, weight_data_bf16, bias_data, opt);
            //
            //                 if (activation)
            //                 {
            //                     activation->forward_inplace(top_blob, opt);
            //                 }
            //
            //                 return 0;
            //             }
            //             else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            //             {
            //                 convdw3x3s2_neon(bottom_blob_bordered, top_blob, weight_data_bf16, bias_data, opt);
            //
            //                 if (activation)
            //                 {
            //                     activation->forward_inplace(top_blob, opt);
            //                 }
            //
            //                 return 0;
            //             }
            //             else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            //             {
            //                 convdw5x5s1_neon(bottom_blob_bordered, top_blob, weight_data_bf16, bias_data, opt);
            //
            //                 if (activation)
            //                 {
            //                     activation->forward_inplace(top_blob, opt);
            //                 }
            //
            //                 return 0;
            //             }
            //             else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            //             {
            //                 convdw5x5s2_neon(bottom_blob_bordered, top_blob, weight_data_bf16, bias_data, opt);
            //
            //                 if (activation)
            //                 {
            //                     activation->forward_inplace(top_blob, opt);
            //                 }
            //
            //                 return 0;
            //             }
            //             else
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
                    unsigned short* outptr = top_blob.channel(g);
                    const unsigned short* kptr = (const unsigned short*)weight_data_bf16 + maxk * g;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                                sum = bias_data[g];

                            const unsigned short* sptr = m.row<const unsigned short>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = bfloat16_to_float32(sptr[space_ofs[k]]);
                                float w = bfloat16_to_float32(kptr[k]);
                                sum += val * w;
                            }

                            sum = activation_ss(sum, activation_type, activation_params);

                            outptr[j] = float32_to_bfloat16(sum);
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
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        g_elempack = channels_g % 4 == 0 ? 4 : 1;
        out_g_elempack = num_output_g % 4 == 0 ? 4 : 1;
    }
#endif

    // unpacking
    Mat bottom_blob_bordered_unpacked = bottom_blob_bordered;
    if (elempack == 4 && g_elempack == 1)
    {
        Option opt_p = opt;
        opt_p.blob_allocator = opt.workspace_allocator;
        convert_packing(bottom_blob_bordered, bottom_blob_bordered_unpacked, 1, opt_p);
    }

    Mat top_blob_unpacked = top_blob;
    if (out_g_elempack == 1 && out_elempack == 4)
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
    if (out_g_elempack == 1 && out_elempack == 4)
    {
        convert_packing(top_blob_unpacked, top_blob, 4, opt);
    }
    else
    {
        top_blob = top_blob_unpacked;
    }

    return 0;
}
#endif // NCNN_BF16

#if NCNN_INT8
int ConvolutionDepthWise_arm::create_pipeline_int8_arm(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;
#if __ARM_NEON
        if (opt.use_packing_layout)
        {
            elempack = channels % 8 == 0 ? 8 : 1;
        }
#endif // __ARM_NEON

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

int ConvolutionDepthWise_arm::forward_int8_arm(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
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
#if __ARM_NEON
        if (opt.use_packing_layout)
        {
            out_elempack = num_output % 8 == 0 ? 8 : 1;
        }
#endif // __ARM_NEON
        bool use_int8_requantize = int8_scale_term > 100;
        size_t out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (opt.use_fp16_storage)
        {
            out_elemsize = use_int8_requantize ? 1u * out_elempack : 2u * out_elempack;
        }
#endif
        if (opt.use_bf16_storage)
            out_elemsize = use_int8_requantize ? 1u * out_elempack : 2u * out_elempack;

        top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // TODO use fp16 / bf16
        out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;
        top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#if __ARM_NEON
        if (elempack == 8)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (activation_type == 0 || activation_type == 1))
            {
                Mat top_blob_int32;
                top_blob_int32.create(outw, outh, num_output / out_elempack, (size_t)4u * out_elempack, out_elempack, opt.workspace_allocator);
                if (top_blob_int32.empty())
                    return -100;

                convdw3x3s1_pack8_int8_neon(bottom_blob_bordered, top_blob_int32, weight_data_int8, opt);

                Mat scale_in_data(group);
                for (int g = 0; g < group; g++)
                {
                    // dequantize
                    float scale_in;
                    if (weight_data_int8_scales[g] == 0)
                        scale_in = 0;
                    else
                        scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                    scale_in_data[g] = scale_in;
                }

                if (use_int8_requantize)
                {
                    requantize_from_int32_to_int8(top_blob_int32, top_blob, scale_in_data, top_blob_int8_scales, bias_data, activation_type, activation_params, opt);
                }
                else
                {
                    dequantize_from_int32(top_blob_int32, top_blob, scale_in_data, bias_data, opt);

                    if (activation)
                    {
                        activation->forward_inplace(top_blob, opt);
                    }
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2 && (activation_type == 0 || activation_type == 1))
            {
                Mat top_blob_int32;
                top_blob_int32.create(outw, outh, num_output / out_elempack, (size_t)4u * out_elempack, out_elempack, opt.workspace_allocator);
                if (top_blob_int32.empty())
                    return -100;

                convdw3x3s2_pack8_int8_neon(bottom_blob_bordered, top_blob_int32, weight_data_int8, opt);

                Mat scale_in_data(group);
                for (int g = 0; g < group; g++)
                {
                    // dequantize
                    float scale_in;
                    if (weight_data_int8_scales[g] == 0)
                        scale_in = 0;
                    else
                        scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                    scale_in_data[g] = scale_in;
                }

                if (use_int8_requantize)
                {
                    requantize_from_int32_to_int8(top_blob_int32, top_blob, scale_in_data, top_blob_int8_scales, bias_data, activation_type, activation_params, opt);
                }
                else
                {
                    dequantize_from_int32(top_blob_int32, top_blob, scale_in_data, bias_data, opt);

                    if (activation)
                    {
                        activation->forward_inplace(top_blob, opt);
                    }
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
                    signed char* outptr_s8 = top_blob.channel(g);
                    float* outptr_f32 = top_blob.channel(g);
                    const signed char* kptr = (const signed char*)weight_data_int8 + maxk * g * 8;
                    const Mat m = bottom_blob_bordered.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            int32x4_t _sum0 = vdupq_n_s32(0);
                            int32x4_t _sum1 = vdupq_n_s32(0);

                            const signed char* sptr = m.row<const signed char>(i * stride_h) + j * stride_w * 8;

                            for (int k = 0; k < maxk; k++)
                            {
                                int8x8_t _val = vld1_s8(sptr + space_ofs[k] * 8);
                                int8x8_t _w = vld1_s8(kptr + k * 8);
                                int16x8_t _s0 = vmull_s8(_val, _w);
                                _sum0 = vaddw_s16(_sum0, vget_low_s16(_s0));
                                _sum1 = vaddw_s16(_sum1, vget_high_s16(_s0));
                            }

                            float32x4_t _scale_in0;
                            float32x4_t _scale_in1;
                            {
                                float32x4_t _bottom_blob_int8_scales0 = vld1q_f32((const float*)bottom_blob_int8_scales + g * 8);
                                float32x4_t _bottom_blob_int8_scales1 = vld1q_f32((const float*)bottom_blob_int8_scales + g * 8 + 4);
                                float32x4_t _weight_data_int8_scales0 = vld1q_f32((const float*)weight_data_int8_scales + g * 8);
                                float32x4_t _weight_data_int8_scales1 = vld1q_f32((const float*)weight_data_int8_scales + g * 8 + 4);
                                _scale_in0 = div_ps(vdupq_n_f32(1.f), vmulq_f32(_bottom_blob_int8_scales0, _weight_data_int8_scales0));
                                _scale_in1 = div_ps(vdupq_n_f32(1.f), vmulq_f32(_bottom_blob_int8_scales1, _weight_data_int8_scales1));

                                uint32x4_t _m0 = vmvnq_u32(vceqq_f32(_weight_data_int8_scales0, vdupq_n_f32(0.f)));
                                uint32x4_t _m1 = vmvnq_u32(vceqq_f32(_weight_data_int8_scales1, vdupq_n_f32(0.f)));
                                _scale_in0 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(_scale_in0), _m0));
                                _scale_in1 = vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(_scale_in1), _m1));
                            }

                            float32x4_t _sumfp32_0 = vmulq_f32(vcvtq_f32_s32(_sum0), _scale_in0);
                            float32x4_t _sumfp32_1 = vmulq_f32(vcvtq_f32_s32(_sum1), _scale_in1);

                            if (bias_term)
                            {
                                float32x4_t _bias0 = vld1q_f32((const float*)bias_data + g * 8);
                                float32x4_t _bias1 = vld1q_f32((const float*)bias_data + g * 8 + 4);
                                _sumfp32_0 = vaddq_f32(_sumfp32_0, _bias0);
                                _sumfp32_1 = vaddq_f32(_sumfp32_1, _bias1);
                            }

                            _sumfp32_0 = activation_ps(_sumfp32_0, activation_type, activation_params);
                            _sumfp32_1 = activation_ps(_sumfp32_1, activation_type, activation_params);

                            if (use_int8_requantize)
                            {
                                // requantize
                                float32x4_t _scale_out0 = vld1q_f32((const float*)top_blob_int8_scales + g * 8);
                                float32x4_t _scale_out1 = vld1q_f32((const float*)top_blob_int8_scales + g * 8 + 4);
                                int8x8_t _sum8 = float2int8(vmulq_f32(_sumfp32_0, _scale_out0), vmulq_f32(_sumfp32_1, _scale_out1));
                                vst1_s8(outptr_s8, _sum8);
                                outptr_s8 += 8;
                            }
                            else
                            {
                                // dequantize
                                vst1q_f32(outptr_f32, _sumfp32_0);
                                vst1q_f32(outptr_f32 + 4, _sumfp32_1);
                                outptr_f32 += 8;
                            }
                        }
                    }
                }
            }
        }
#endif // __ARM_NEON

        if (elempack == 1)
        {
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1 && (activation_type == 0 || activation_type == 1))
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

                    convdw3x3s1_int8_requant_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, requantize_scales, opt);
                }
                else
                {
                    Mat top_blob_int32;
                    top_blob_int32.create(outw, outh, num_output, (size_t)4u, opt.workspace_allocator);
                    if (top_blob_int32.empty())
                        return -100;

                    convdw3x3s1_int8_neon(bottom_blob_bordered, top_blob_int32, weight_data, opt);
                    //                 convdw3x3s1_int8_dequant_neon(bottom_blob_bordered, top_blob_int32, weight_data, bias_data, dequantize_scales, opt);

                    Mat scale_data(group);
                    for (int g = 0; g < group; g++)
                    {
                        // dequantize
                        float scale_in;
                        if (weight_data_int8_scales[g] == 0)
                            scale_in = 0;
                        else
                            scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                        scale_data[g] = scale_in;
                    }

                    dequantize_from_int32(top_blob_int32, top_blob, scale_data, bias_data, opt);
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

                    convdw3x3s2_int8_requant_neon(bottom_blob_bordered, top_blob, weight_data, bias_data, requantize_scales, opt);
                }
                else
                {
                    Mat top_blob_int32;
                    top_blob_int32.create(outw, outh, num_output, (size_t)4u, opt.workspace_allocator);
                    if (top_blob_int32.empty())
                        return -100;

                    convdw3x3s2_int8_neon(bottom_blob_bordered, top_blob_int32, weight_data, opt);
                    //                 convdw3x3s2_int8_dequant_neon(bottom_blob_bordered, top_blob_int32, weight_data, bias_data, dequantize_scales, opt);

                    Mat scale_data(group);
                    for (int g = 0; g < group; g++)
                    {
                        // dequantize
                        float scale_in;
                        if (weight_data_int8_scales[g] == 0)
                            scale_in = 0;
                        else
                            scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                        scale_data[g] = scale_in;
                    }

                    dequantize_from_int32(top_blob_int32, top_blob, scale_data, bias_data, opt);
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
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        if (use_int8_requantize)
            out_elempack = num_output % 8 == 0 ? 8 : 1;
        else
            out_elempack = num_output % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON
    size_t out_elemsize = use_int8_requantize ? 1u * out_elempack : 4u * out_elempack;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (opt.use_fp16_storage)
    {
        out_elemsize = use_int8_requantize ? 1u * out_elempack : 2u * out_elempack;
    }
#endif
    if (opt.use_bf16_storage)
        out_elemsize = use_int8_requantize ? 1u * out_elempack : 2u * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // group convolution
    const int channels_g = channels * elempack / group;
    const int num_output_g = num_output / group;

    int g_elempack = 1;
    int out_g_elempack = 1;
#if __ARM_NEON
    if (opt.use_packing_layout)
    {
        g_elempack = channels_g % 8 == 0 ? 8 : 1;
        if (use_int8_requantize)
            out_g_elempack = num_output_g % 8 == 0 ? 8 : 1;
        else
            out_g_elempack = num_output_g % 4 == 0 ? 4 : 1;
    }
#endif // __ARM_NEON

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
