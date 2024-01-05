// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

#include "arm_activation.h"
#include "arm_usability.h"

namespace ncnn {

#if NCNN_GNU_INLINE_ASM
#if __ARM_NEON
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "convolutiondepthwise_3x3_fp16s.h"
#include "convolutiondepthwise_3x3_pack8_fp16s.h"
#include "convolutiondepthwise_5x5_pack8_fp16s.h"
#endif
#endif // __ARM_NEON
#endif // NCNN_GNU_INLINE_ASM

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
int ConvolutionDepthWise_arm::create_pipeline_fp16s(const Option& opt)
{
    const int maxk = kernel_w * kernel_h;
    int channels = (weight_data_size / group) / maxk / (num_output / group) * group;

    // depth-wise
    if (channels == group && group == num_output)
    {
        int elempack = 1;

        if (opt.use_packing_layout)
        {
            elempack = opt.use_fp16_arithmetic && channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4 : 1;
        }

        if (elempack == 8)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            Mat weight_data_r2_packed;
            convert_packing(weight_data_r2, weight_data_r2_packed, 8, opt);

            ncnn::cast_float32_to_float16(weight_data_r2_packed, weight_data_tm, opt);
        }

        if (elempack == 4)
        {
            Mat weight_data_r2 = weight_data.reshape(maxk, group);
            Mat weight_data_r2_packed;
            convert_packing(weight_data_r2, weight_data_r2_packed, 4, opt);

            ncnn::cast_float32_to_float16(weight_data_r2_packed, weight_data_tm, opt);
        }

        if (elempack == 1)
        {
            ncnn::cast_float32_to_float16(weight_data, weight_data_tm, opt);
        }

        ncnn::cast_float32_to_float16(bias_data, bias_data_fp16, opt);

        weight_data.release();

        return 0;
    }

    // group convolution
    create_group_ops(opt);

    weight_data.release();

    return 0;
}

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
                    const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g * 4;
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
                    const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g;
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
#if NCNN_GNU_INLINE_ASM
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw5x5s1_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 5 && kernel_h == 5 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw5x5s2_pack8_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else
#endif // NCNN_GNU_INLINE_ASM
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
                    const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g * 8;
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

                            _sum = activation_ps_f16(_sum, activation_type, activation_params);

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
                    const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g * 4;
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

                            _sum = activation_ps_f16(_sum, activation_type, activation_params);

                            vst1_f16(outptr + j * 4, _sum);
                        }

                        outptr += outw * 4;
                    }
                }
            }
        }

        if (elempack == 1)
        {
#if NCNN_GNU_INLINE_ASM
            if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
            {
                convdw3x3s1_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
            {
                convdw3x3s2_fp16sa_neon(bottom_blob_bordered, top_blob, weight_data_tm, bias_data_fp16, opt);

                if (activation)
                {
                    activation->forward_inplace(top_blob, opt);
                }
            }
            else
#endif // NCNN_GNU_INLINE_ASM
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
                    const __fp16* kptr = (const __fp16*)weight_data_tm + maxk * g;
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

                            sum = activation_ss_f16(sum, activation_type, activation_params);

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

} // namespace ncnn
