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
#include "neon_mathfun.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "neon_mathfun_fp16s.h"
#endif
#endif // __ARM_NEON

#include "neon_activation.h"

namespace ncnn {

DeconvolutionDepthWise_arm::DeconvolutionDepthWise_arm()
{
#if __ARM_NEON
    support_packing = true;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    support_fp16_storage = true;
#endif
#endif // __ARM_NEON

    support_bf16_storage = true;
}

int DeconvolutionDepthWise_arm::create_pipeline(const Option& opt)
{
    // create Deconvolution op for each group
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = (support_packing && opt.use_packing_layout && channels % 4 == 0) ? 4 : 1;

        Mat weight_data_transposed(weight_data.w);
        {
            float* pt = weight_data_transposed;
            const float* p = weight_data;

            for (int i = 0; i < (channels / group) * (num_output / group) * group; i++)
            {
                for (int k = 0; k < maxk; k++)
                {
                    pt[maxk - 1 - k] = p[k];
                }

                p += maxk;
                pt += maxk;
            }
        }

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        if (opt.use_fp16_storage)
        {
            if (opt.use_packing_layout)
            {
                elempack = opt.use_fp16_arithmetic && channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4 : 1;
            }

            if (elempack == 8)
            {
                Mat weight_data_r2 = weight_data_transposed.reshape(maxk, group);
                Mat weight_data_r2_packed;
                convert_packing(weight_data_r2, weight_data_r2_packed, 8);

                ncnn::cast_float32_to_float16(weight_data_r2_packed, weight_data_fp16, opt);
            }

            if (elempack == 4)
            {
                Mat weight_data_r2 = weight_data_transposed.reshape(maxk, group);
                Mat weight_data_r2_packed;
                convert_packing(weight_data_r2, weight_data_r2_packed, 4);

                ncnn::cast_float32_to_float16(weight_data_r2_packed, weight_data_fp16, opt);
            }

            if (elempack == 1)
            {
                ncnn::cast_float32_to_float16(weight_data_transposed, weight_data_fp16, opt);
            }

            ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);

            return 0;
        }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

        if (opt.use_bf16_storage)
        {
#if __ARM_NEON
            if (elempack == 4)
            {
                Mat weight_data_r2 = weight_data_transposed.reshape(maxk, group);
                convert_packing(weight_data_r2, weight_data_pack4, 4);

                ncnn::cast_float32_to_bfloat16(weight_data_pack4, weight_data_bf16, opt);
            }
#endif // __ARM_NEON

            if (elempack == 1)
            {
                ncnn::cast_float32_to_bfloat16(weight_data_transposed, weight_data_bf16, opt);
            }

            return 0;
        }

#if __ARM_NEON
        // pack4
        if (elempack == 4)
        {
            Mat weight_data_r2 = weight_data_transposed.reshape(maxk, group);
            convert_packing(weight_data_r2, weight_data_pack4, 4);
        }
#endif // __ARM_NEON

        // pack1
        if (elempack == 1)
        {
            weight_data_pack1 = weight_data_transposed;
        }
    }
    else
    {
        // group deconvolution
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

            ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::Deconvolution);

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
            pd.set(9, activation_type);
            pd.set(10, activation_params);

            op->load_param(pd);

            // set weights
            if (bias_term)
            {
                ncnn::Mat weights[2];
                weights[0] = weight_data_g;
                weights[1] = bias_data_g;

                op->load_model(ModelBinFromMatArray(weights));
            }
            else
            {
                ncnn::Mat weights[1];
                weights[0] = weight_data_g;

                op->load_model(ModelBinFromMatArray(weights));
            }

            op->create_pipeline(opt);

            group_ops[g] = op;
        }
    }

    return 0;
}

int DeconvolutionDepthWise_arm::destroy_pipeline(const Option& opt)
{
    for (int i = 0; i < (int)group_ops.size(); i++)
    {
        group_ops[i]->destroy_pipeline(opt);
        delete group_ops[i];
    }
    group_ops.clear();

    return 0;
}

int DeconvolutionDepthWise_arm::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
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

    if (opt.use_bf16_storage && elembits == 16)
        return forward_bf16s(bottom_blob, top_blob, opt);

    // convolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = (support_packing && opt.use_packing_layout && num_output % 4 == 0) ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
#if __ARM_NEON
        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                float* outptr = top_blob_bordered.channel(g);
                const float* kptr = (const float*)weight_data_pack4 + maxk * g * 4;
                const Mat m = bottom_blob.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32((const float*)bias_data + g * 4);
                        }

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const float* sptr = m.row(sy) + sx * 4;

                                float32x4_t _val = vld1q_f32(sptr);

                                int k = y * kernel_w + x;

                                float32x4_t _w = vld1q_f32(kptr + k * 4);

                                _sum = vmlaq_f32(_sum, _val, _w);
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1q_f32(outptr + j * 4, _sum);
                    }

                    outptr += outw * 4;
                }
            }
        }
#endif // __ARM_NEON

        if (elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                float* outptr = top_blob_bordered.channel(g);
                const float* kptr = (const float*)weight_data_pack1 + maxk * g;
                const Mat m = bottom_blob.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[g];
                        }

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            const float* sptr = m.row(sy);

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                float val = sptr[sx];

                                int k = y * kernel_w + x;

                                float w = kptr[k];

                                sum += val * w;
                            }
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

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }
    else
    {
        // group deconvolution
        const int channels_g = channels * elempack / group;
        const int num_output_g = num_output / group;

        int g_elempack = (support_packing && opt.use_packing_layout && channels_g % 4 == 0) ? 4 : 1;
        int out_g_elempack = (support_packing && opt.use_packing_layout && num_output_g % 4 == 0) ? 4 : 1;

        // unpacking
        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack == 4 && g_elempack == 1)
        {
            Option opt_p = opt;
            opt_p.blob_allocator = opt.workspace_allocator;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_p);
        }

        Mat top_blob_bordered_unpacked = top_blob_bordered;
        if (out_g_elempack == 1 && out_elempack == 4)
        {
            top_blob_bordered_unpacked.create(outw, outh, num_output, out_elemsize / out_elempack, 1, opt.workspace_allocator);
            if (top_blob_bordered_unpacked.empty())
                return -100;
        }

        for (int g = 0; g < group; g++)
        {
            const Mat bottom_blob_g = bottom_blob_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
            Mat top_blob_bordered_g = top_blob_bordered_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

            const ncnn::Layer* op = group_ops[g];

            Option opt_g = opt;
            opt_g.blob_allocator = top_blob_bordered_unpacked.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);
        }

        // packing
        if (out_g_elempack == 1 && out_elempack == 4)
        {
            convert_packing(top_blob_bordered_unpacked, top_blob_bordered, 4, opt);
        }
        else
        {
            top_blob_bordered = top_blob_bordered_unpacked;
        }
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int DeconvolutionDepthWise_arm::forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = (support_packing && opt.use_packing_layout && num_output % 4 == 0) ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
        if (elempack == 4)
        {
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    __fp16* outptr = top_blob_bordered.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g * 4;
                    const Mat m = bottom_blob.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float32x4_t _sum = vdupq_n_f32(0.f);

                            if (bias_term)
                            {
                                _sum = vld1q_f32((const float*)bias_data + g * 4);
                            }

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                    float32x4_t _val = vcvt_f32_f16(vld1_f16(sptr));

                                    int k = y * kernel_w + x;

                                    float32x4_t _w = vcvt_f32_f16(vld1_f16(kptr + k * 4));

                                    _sum = vfmaq_f32(_sum, _val, _w);
                                }
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    __fp16* outptr = top_blob_bordered.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g;
                    const Mat m = bottom_blob.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                            {
                                sum = bias_data[g];
                            }

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    float val = (float)sptr[sx];

                                    int k = y * kernel_w + x;

                                    float w = (float)kptr[k];

                                    sum += val * w;
                                }
                            }

                            sum = activation_ss(sum, activation_type, activation_params);

                            outptr[j] = (__fp16)sum;
                        }

                        outptr += outw;
                    }
                }
            }
        }
    }
    else
    {
        // group deconvolution
        const int channels_g = channels * elempack / group;
        const int num_output_g = num_output / group;

        int g_elempack = (support_packing && opt.use_packing_layout && channels_g % 4 == 0) ? 4 : 1;
        int out_g_elempack = (support_packing && opt.use_packing_layout && num_output_g % 4 == 0) ? 4 : 1;

        // unpacking
        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack == 4 && g_elempack == 1)
        {
            Option opt_p = opt;
            opt_p.blob_allocator = opt.workspace_allocator;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_p);
        }

        Mat top_blob_bordered_unpacked = top_blob_bordered;
        if (out_g_elempack == 1 && out_elempack == 4)
        {
            top_blob_bordered_unpacked.create(outw, outh, num_output, out_elemsize / out_elempack, 1, opt.workspace_allocator);
            if (top_blob_bordered_unpacked.empty())
                return -100;
        }

        for (int g = 0; g < group; g++)
        {
            const Mat bottom_blob_g = bottom_blob_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
            Mat top_blob_bordered_g = top_blob_bordered_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

            const ncnn::Layer* op = group_ops[g];

            Option opt_g = opt;
            opt_g.blob_allocator = top_blob_bordered_unpacked.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);
        }

        // packing
        if (out_g_elempack == 1 && out_elempack == 4)
        {
            convert_packing(top_blob_bordered_unpacked, top_blob_bordered, 4, opt);
        }
        else
        {
            top_blob_bordered = top_blob_bordered_unpacked;
        }
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

int DeconvolutionDepthWise_arm::forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = 1;
    if (opt.use_packing_layout)
    {
        out_elempack = opt.use_fp16_arithmetic && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4 : 1;
    }
    size_t out_elemsize = elemsize / elempack * out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
        if (elempack == 8)
        {
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    __fp16* outptr = top_blob_bordered.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g * 8;
                    const Mat m = bottom_blob.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float16x8_t _sum = vdupq_n_f16((__fp16)0.f);

                            if (bias_term)
                            {
                                _sum = vld1q_f16((const __fp16*)bias_data_fp16 + g * 8);
                            }

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * 8;

                                    float16x8_t _val = vld1q_f16(sptr);

                                    int k = y * kernel_w + x;

                                    float16x8_t _w = vld1q_f16(kptr + k * 8);

                                    _sum = vfmaq_f16(_sum, _val, _w);
                                }
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
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    __fp16* outptr = top_blob_bordered.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g * 4;
                    const Mat m = bottom_blob.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float16x4_t _sum = vdup_n_f16((__fp16)0.f);

                            if (bias_term)
                            {
                                _sum = vld1_f16((const __fp16*)bias_data_fp16 + g * 4);
                            }

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    const __fp16* sptr = m.row<const __fp16>(sy) + sx * 4;

                                    float16x4_t _val = vld1_f16(sptr);

                                    int k = y * kernel_w + x;

                                    float16x4_t _w = vld1_f16(kptr + k * 4);

                                    _sum = vfma_f16(_sum, _val, _w);
                                }
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
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int g = 0; g < channels; g++)
                {
                    __fp16* outptr = top_blob_bordered.channel(g);
                    const __fp16* kptr = (const __fp16*)weight_data_fp16 + maxk * g;
                    const Mat m = bottom_blob.channel(g);

                    for (int i = 0; i < outh; i++)
                    {
                        for (int j = 0; j < outw; j++)
                        {
                            float sum = 0.f;

                            if (bias_term)
                            {
                                sum = bias_data[g];
                            }

                            for (int y = 0; y < kernel_h; y++)
                            {
                                int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                                if (sys < 0 || sys % stride_h != 0)
                                    continue;

                                int sy = sys / stride_h;
                                if (sy >= h)
                                    continue;

                                const __fp16* sptr = m.row<const __fp16>(sy);

                                for (int x = 0; x < kernel_w; x++)
                                {
                                    int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                    if (sxs < 0 || sxs % stride_w != 0)
                                        continue;

                                    int sx = sxs / stride_w;
                                    if (sx >= w)
                                        continue;

                                    __fp16 val = sptr[sx];

                                    int k = y * kernel_w + x;

                                    __fp16 w = kptr[k];

                                    sum += val * w;
                                }
                            }

                            sum = activation_ss(sum, activation_type, activation_params);

                            outptr[j] = (__fp16)sum;
                        }

                        outptr += outw;
                    }
                }
            }
        }
    }
    else
    {
        // group deconvolution
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
        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack > g_elempack)
        {
            Option opt_p = opt;
            opt_p.blob_allocator = opt.workspace_allocator;
            convert_packing(bottom_blob, bottom_blob_unpacked, g_elempack, opt_p);
        }

        Mat top_blob_bordered_unpacked = top_blob_bordered;
        if (out_g_elempack < out_elempack)
        {
            top_blob_bordered_unpacked.create(outw, outh, num_output / out_g_elempack, out_elemsize / out_elempack * out_g_elempack, out_g_elempack, opt.workspace_allocator);
            if (top_blob_bordered_unpacked.empty())
                return -100;
        }

        for (int g = 0; g < group; g++)
        {
            const Mat bottom_blob_g = bottom_blob_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
            Mat top_blob_bordered_g = top_blob_bordered_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

            const ncnn::Layer* op = group_ops[g];

            Option opt_g = opt;
            opt_g.blob_allocator = top_blob_bordered_unpacked.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);
        }

        // packing
        if (out_g_elempack < out_elempack)
        {
            convert_packing(top_blob_bordered_unpacked, top_blob_bordered, out_elempack, opt);
        }
        else
        {
            top_blob_bordered = top_blob_bordered_unpacked;
        }
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

int DeconvolutionDepthWise_arm::forward_bf16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    int outw = (w - 1) * stride_w + kernel_extent_w;
    int outh = (h - 1) * stride_h + kernel_extent_h;
    int out_elempack = opt.use_packing_layout && num_output % 4 == 0 ? 4 : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    Mat top_blob_bordered;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0 || output_pad_right > 0 || output_pad_bottom > 0 || (output_w > 0 && output_h > 0))
    {
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.workspace_allocator);
    }
    else
    {
        top_blob_bordered = top_blob;
        top_blob_bordered.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    }
    if (top_blob_bordered.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // depth-wise
    if (channels * elempack == group && group == num_output)
    {
#if __ARM_NEON
        if (elempack == 4)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                unsigned short* outptr = top_blob_bordered.channel(g);
                const unsigned short* kptr = (const unsigned short*)weight_data_bf16 + maxk * g * 4;
                const Mat m = bottom_blob.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float32x4_t _sum = vdupq_n_f32(0.f);

                        if (bias_term)
                        {
                            _sum = vld1q_f32((const float*)bias_data + g * 4);
                        }

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                const unsigned short* sptr = m.row<const unsigned short>(sy) + sx * 4;

                                float32x4_t _val = vcvt_f32_bf16(vld1_u16(sptr));

                                int k = y * kernel_w + x;

                                float32x4_t _w = vcvt_f32_bf16(vld1_u16(kptr + k * 4));

                                _sum = vmlaq_f32(_sum, _val, _w);
                            }
                        }

                        _sum = activation_ps(_sum, activation_type, activation_params);

                        vst1_u16(outptr + j * 4, vcvt_bf16_f32(_sum));
                    }

                    outptr += outw * 4;
                }
            }
        }
#endif // __ARM_NEON

        if (elempack == 1)
        {
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int g = 0; g < channels; g++)
            {
                unsigned short* outptr = top_blob_bordered.channel(g);
                const unsigned short* kptr = (const unsigned short*)weight_data_bf16 + maxk * g;
                const Mat m = bottom_blob.channel(g);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[g];
                        }

                        for (int y = 0; y < kernel_h; y++)
                        {
                            int sys = (i + y * dilation_h - (kernel_extent_h - 1));
                            if (sys < 0 || sys % stride_h != 0)
                                continue;

                            int sy = sys / stride_h;
                            if (sy >= h)
                                continue;

                            const unsigned short* sptr = m.row<const unsigned short>(sy);

                            for (int x = 0; x < kernel_w; x++)
                            {
                                int sxs = (j + x * dilation_w - (kernel_extent_w - 1));
                                if (sxs < 0 || sxs % stride_w != 0)
                                    continue;

                                int sx = sxs / stride_w;
                                if (sx >= w)
                                    continue;

                                float val = bfloat16_to_float32(sptr[sx]);

                                int k = y * kernel_w + x;

                                float w = bfloat16_to_float32(kptr[k]);

                                sum += val * w;
                            }
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

                        outptr[j] = float32_to_bfloat16(sum);
                    }

                    outptr += outw;
                }
            }
        }
    }
    else
    {
        // group deconvolution
        const int channels_g = channels * elempack / group;
        const int num_output_g = num_output / group;

        int g_elempack = (support_packing && opt.use_packing_layout && channels_g % 4 == 0) ? 4 : 1;
        int out_g_elempack = (support_packing && opt.use_packing_layout && num_output_g % 4 == 0) ? 4 : 1;

        // unpacking
        Mat bottom_blob_unpacked = bottom_blob;
        if (elempack == 4 && g_elempack == 1)
        {
            Option opt_p = opt;
            opt_p.blob_allocator = opt.workspace_allocator;
            convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt_p);
        }

        Mat top_blob_bordered_unpacked = top_blob_bordered;
        if (out_g_elempack == 1 && out_elempack == 4)
        {
            top_blob_bordered_unpacked.create(outw, outh, num_output, out_elemsize / out_elempack, 1, opt.workspace_allocator);
            if (top_blob_bordered_unpacked.empty())
                return -100;
        }

        for (int g = 0; g < group; g++)
        {
            const Mat bottom_blob_g = bottom_blob_unpacked.channel_range(channels_g * g / g_elempack, channels_g / g_elempack);
            Mat top_blob_bordered_g = top_blob_bordered_unpacked.channel_range(num_output_g * g / out_g_elempack, num_output_g / out_g_elempack);

            const ncnn::Layer* op = group_ops[g];

            Option opt_g = opt;
            opt_g.blob_allocator = top_blob_bordered_unpacked.allocator;

            // forward
            op->forward(bottom_blob_g, top_blob_bordered_g, opt_g);
        }

        // packing
        if (out_g_elempack == 1 && out_elempack == 4)
        {
            convert_packing(top_blob_bordered_unpacked, top_blob_bordered, 4, opt);
        }
        else
        {
            top_blob_bordered = top_blob_bordered_unpacked;
        }
    }

    cut_padding(top_blob_bordered, top_blob, opt);
    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
